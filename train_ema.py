#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import math
import time
import argparse
import logging
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Tuple
import glob, re
import math as _math


import torch
# torch.autograd.set_detect_anomaly(True)

import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torch.multiprocessing as mp
import ml_collections

from abx.data import dataset
from abx.model.abx import ScoreNetwork
from diffuser.full_diffuser import FullDiffuser
from abx.model.loss import AlphaFoldLoss
from abx.model.lr_schedulers import AlphaFoldLRScheduler


from abx.model.interface.energy_head import InterfaceEnergy
from abx.model.interface.physics_provider import EnergyTrainer


from abx.model.interface.tb_utils import (
    _tofloat,
    _add_hist,
    _add_scatter,
    _acc_pair,
    _triplet_ok,
    _margin_ok,
    _spearman,
    _pearson,
    
)


try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass

# --------------------------------------------------------
# 0.1 tqdm-friendly logging
# --------------------------------------------------------
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        tqdm.write(self.format(record), end="\n")


# --------------------------------------------------------
# 0.2 logging helpers
# --------------------------------------------------------
from logging.handlers import RotatingFileHandler

class RankFilter(logging.Filter):
    def __init__(self, rank: int):
        super().__init__()
        self.rank = rank
    def filter(self, record):
        record.rank = self.rank
        return True

def setup_root_logger(out_dir: str, rank: int):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(logging.DEBUG)

    rank_filter = RankFilter(rank)

    # terminal handler：非主进程尽量别刷屏
    term_hdl = TqdmLoggingHandler()
    term_hdl.setLevel(logging.INFO if rank in (-1, 0) else logging.WARNING)
    term_hdl.addFilter(rank_filter)
    term_hdl.setFormatter(logging.Formatter(
        "%(asctime)s [R%(rank)s] %(levelname)s %(message)s", "%H:%M:%S"
    ))
    logger.addHandler(term_hdl)

    # file handler：只让 rank0 写文件（避免多进程写同一文件炸裂）
    file_logger = logging.getLogger("file_only")
    file_logger.handlers.clear()
    file_logger.setLevel(logging.INFO)
    file_logger.propagate = False

    if rank in (-1, 0):
        file_name = "Backbone.log"
        file_hdl = RotatingFileHandler(
            filename=os.path.join(out_dir, file_name),
            maxBytes=10 * 1024 * 1024,
            backupCount=5,
            encoding="utf-8",
        )
        file_hdl.setLevel(logging.DEBUG)
        file_hdl.addFilter(rank_filter)
        file_hdl.setFormatter(logging.Formatter(
            "%(asctime)s [R%(rank)s] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(file_hdl)

        # file_only 也只在 rank0 绑定同一个 handler
        file_logger.addHandler(file_hdl)
    else:
        # 其他 rank 不写文件
        file_logger.addHandler(logging.NullHandler())

    return file_logger

class EMA:
    """
    模型参数的指数移动平均（包含偏差修正功能）。
    """
    def __init__(self, model: torch.nn.Module, decay: float, use_num_updates: bool = True):
        self.model = model
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.shadow = {}
        self.backup = {}
        # 用于偏差修正的计数器
        self.num_updates = 0 if use_num_updates else None
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        logging.info(f"[EMA] Registered {len(self.shadow)} parameters for EMA.")

    def update(self):
        # 动态计算当前step的decay值（偏差修正）
        decay = self.decay
        if self.use_num_updates and self.num_updates is not None:
            self.num_updates += 1
            # 在训练早期使用一个更小的decay值，以快速跟上参数变化
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))

        one_minus_decay = 1.0 - decay
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if name not in self.shadow:
                    # 惰性注册（首次出现时直接克隆进去，避免硬断言炸训练）
                    self.shadow[name] = param.data.detach().clone()
                    continue
                shadow_tensor = self.shadow[name].to(param.device)
                self.shadow[name] = (decay * shadow_tensor) + (one_minus_decay * param.detach())
                
        
    def apply_shadow(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    self.backup[name] = param.data.clone()
                    param.data.copy_( self.shadow[name].to(param.device) )

    def restore(self):
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    param.data.copy_( self.backup[name].to(param.device) )
            self.backup = {}

    def state_dict(self):
        return {
            'decay': self.decay,
            'num_updates': self.num_updates,
            'shadow': self.shadow
        }

    def load_state_dict(self, state_dict):
        # 升级load_state_dict以支持新旧两种格式
        if not isinstance(state_dict, dict) or 'shadow' not in state_dict:
            logging.warning("[EMA] Loading from legacy EMA state_dict format (shadow weights only).")
            shadow_state = state_dict
            self.num_updates = None # 标记为从旧格式加载，让外部逻辑处理
        else:
            shadow_state = state_dict['shadow']
            self.decay = state_dict.get('decay', self.decay)
            # 即使在新的state_dict中，也要允许num_updates可能不存在
            self.num_updates = state_dict.get('num_updates', None)

        model_keys = {name for name, p in self.model.named_parameters() if p.requires_grad}
        ckpt_keys = set(shadow_state.keys())
        missing_keys = model_keys - ckpt_keys
        unexpected_keys = ckpt_keys - model_keys
        if missing_keys:
            logging.warning(f"[EMA] Missing keys in checkpoint EMA state: {sorted(list(missing_keys))}")
        if unexpected_keys:
            logging.warning(f"[EMA] Unexpected keys in checkpoint EMA state: {sorted(list(unexpected_keys))}")

        for k in model_keys:
            if k in shadow_state:
                self.shadow[k] = shadow_state[k].clone()
        
        num_loaded = len(model_keys) - len(missing_keys)
        logging.info(f"[EMA] Loaded shadow weights for {num_loaded}/{len(model_keys)} parameters. Num_updates from ckpt: {self.num_updates}")


# --------------------------------------------------------
# 1. TrainerBase  (iteration-driven)
# --------------------------------------------------------
class TrainerBase:
    def __init__(self, cfg: dict, file_logger=None):
        self.cfg = cfg
        self.model = self.optimizer = self.scheduler = None
        self.train_loader = self.val_loader = None
        self.ema: Optional[EMA] = None # <-- [EMA]
        self.ema_energy: Optional[EMA] = None  # 能量头的 EMA 句柄
        self.loss_cfg: Optional[ml_collections.ConfigDict] = None # 用于存储详细的loss配置

        self.local_rank = -1
        self.global_step = 0
        self.best_metric = None
        
        #扩展 best_metric 和 prev_best_path 为按阶段记录的字典
        self.best_metrics_by_phase = {1: float('inf'), 2: float('inf'), 3: float('inf')}
        self.prev_best_paths = {1: None, 2: None, 3: None}
        
        # —— 额外追踪：按 DSM / Seq 的全局 best ckpt —— 
        self.best_DSM_metric = float('inf')
        self.best_seq_metric = float('inf')
        self.prev_best_DSM_path = None
        self.prev_best_seq_path = None
     
        # 缓存最近一次 _run_validation 的全量平均指标（含 Energy/*）
        self._run_validation_outputs = {}

        self.patience = cfg.get("patience", 0)
        self.no_improve = 0
        self.file_logger = file_logger  # 记录 file_logger
        self.energy_log_every = int(self.cfg.get("energy_log_every", 5))
        
        self._ret_final_for_energy = None
        
        # ---------- 运行目录 ----------
        if "run_dir" in cfg:
            self.save_root = cfg["run_dir"]         # ← 统一入口
        else:                                       # 防老脚本
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.save_root = os.path.join(cfg["output_dir"], cfg["mode"], ts)
            os.makedirs(self.save_root, exist_ok=True)

        self.writer: Optional[SummaryWriter] = None
        self.step_per_epoch = self.val_step_per_epoch = None

        # ---------- checkpoint 相关 ----------
        #self.prev_best_path = None
        self.prev_last_path = None
        self.prev_last_energy_path = None   # 能量 last 的上一份路径
        # —— 能量头：全局统一的 best，不分阶段 —— 
        self.best_energy_metric = float('inf')
        self.prev_best_energy_path = None

        self.resume_ckpt    = cfg.get("resume_checkpoint", "")
        self._just_loaded_step = 0
        #self._sync_prev_paths()

    def _elog(self, msg: str, level: str = "info"):
        if not self._is_main():
            return
        if hasattr(self, "energy_logger") and self.energy_logger:
            getattr(self.energy_logger, level)(msg)

    def _log_energy_metrics(self, energy_logs: dict, step: int, mode: str, *, aggregated: bool=False):
        """
        统一把能量头指标写入 TensorBoard，并写入 energy.log
        step: 全局步数
        mode: 'train' 或 'val'
        aggregated: 若为 True，表示这些指标已做过 DDP 聚合（只写一次标量，跳过向量图/相关性/直方图/散点/grad/lr）
        """
        # 仅主进程且 writer_energy 存在时写 TB；否则只写文本日志
        write_tb = self._is_main() and hasattr(self, "writer_energy") and (self.writer_energy is not None)
        pfx = "Energy" if mode == "train" else "EnergyVal"

        # ---------- 1) 基础标量 ----------
        # 注意：即使 aggregated=True，也应写这些标量
        if write_tb:
            base_tags = [
                "loss", "rank", "r_pred", "r_noise", "bce",
                "Eg", "Ep", "En",
                "rmsd_p", "rmsd_n"
            ]
                        
            for k in base_tags:
                if k in energy_logs:
                    try:
                        self.writer_energy.add_scalar(f"{pfx}/{k}", float(energy_logs[k]), step)
                    except Exception:
                        pass  # 防御：遇到非标量/NaN 不中断

            # 2) 间隙（gap）
            try:
                if ("Eg" in energy_logs) and ("Ep" in energy_logs):
                    gap_gp = float(energy_logs["Ep"]) - float(energy_logs["Eg"])
                    self.writer_energy.add_scalar(f"{pfx}/energy_gap_gt_pred", gap_gp, step)
            except Exception:
                pass
            try:
                if ("Ep" in energy_logs) and ("En" in energy_logs):
                    gap_pn = float(energy_logs["En"]) - float(energy_logs["Ep"])
                    self.writer_energy.add_scalar(f"{pfx}/energy_gap_pred_noisy", gap_pn, step)
            except Exception:
                pass

        # ---------- 3) 仅在非聚合场景下写向量相关指标/图 ----------
        if write_tb and (not aggregated):
            # 频率（与 train/val 独立控制）
            hist_every    = int(self.cfg.get("energy_hist_every"      if mode=="train" else "energy_hist_every_val",      200 if mode=="train" else 500))
            scatter_every = int(self.cfg.get("energy_scatter_every"   if mode=="train" else "energy_scatter_every_val",  1000 if mode=="train" else 2000))

            Egt = energy_logs.get("E_gt_vec", None)
            Epr = energy_logs.get("E_pred_vec", None)
            Enz = energy_logs.get("E_noisy_vec", None)
            Rpr = energy_logs.get("RMSD_pred_vec", None)
            Rnz = energy_logs.get("RMSD_noisy_vec", None)
            margin_m = float(energy_logs.get("margin", 0.0)) if ("margin" in energy_logs) else 0.0

            # 相关/排序准确率
            try:
                if (Egt is not None) and (Epr is not None):
                    self.writer_energy.add_scalar(f"{pfx}/acc_pair_gt_pred", _acc_pair(Egt, Epr), step)
            except Exception:
                pass
            try:
                if (Epr is not None) and (Enz is not None):
                    self.writer_energy.add_scalar(f"{pfx}/acc_pair_pred_noisy", _acc_pair(Epr, Enz), step)
            except Exception:
                pass
            try:
                if (Egt is not None) and (Enz is not None):
                    self.writer_energy.add_scalar(f"{pfx}/acc_pair_gt_noisy", _acc_pair(Egt, Enz), step)
            except Exception:
                pass
            try:
                if (Egt is not None) and (Epr is not None) and (Enz is not None):
                    self.writer_energy.add_scalar(f"{pfx}/triplet_ok_rate", _triplet_ok(Egt, Epr, Enz), step)
                    if margin_m > 0:
                        self.writer_energy.add_scalar(f"{pfx}/margin_ok_rate_gt_pred",   _margin_ok(Egt, Epr, margin_m), step)
                        self.writer_energy.add_scalar(f"{pfx}/margin_ok_rate_pred_noisy", _margin_ok(Epr, Enz, margin_m), step)
            except Exception:
                pass

            # 负相关越好
            try:
                if (Epr is not None) and (Rpr is not None):
                    self.writer_energy.add_scalar(f"{pfx}/spearman_r_energy_rmsd_pred", _spearman(Epr, Rpr), step)
                    self.writer_energy.add_scalar(f"{pfx}/pearson_r_energy_rmsd_pred",  _pearson(Epr, Rpr),  step)
            except Exception:
                pass

            # 直方图/散点
            try:
                if (step % hist_every == 0):
                    if Egt is not None: _add_hist(self.writer_energy, f"{pfx}/hist/Eg",    Egt, step)
                    if Epr is not None: _add_hist(self.writer_energy, f"{pfx}/hist/Ep",  Epr, step)
                    if Enz is not None: _add_hist(self.writer_energy, f"{pfx}/hist/En", Enz, step)
            except Exception:
                pass
            try:
                if (step % scatter_every == 0) and (Epr is not None) and (Rpr is not None):
                    _add_scatter(self.writer_energy, f"{pfx}/fig/scatter_E_vs_RMSD_pred", Rpr, Epr, step)
            except Exception:
                pass

        # ---------- 4) 数值健康度（仅训练 + 非聚合） ----------
        if write_tb and (mode == "train") and (not aggregated):
            try:
                self.writer_energy.add_scalar(f"{pfx}/grad_norm", getattr(self, "_last_energy_grad_norm", np.nan), step)
            except Exception:
                pass
            try:
                lr_val = (self.opt_energy.param_groups[0]["lr"] if getattr(self, "opt_energy", None) is not None else np.nan)
                self.writer_energy.add_scalar(f"{pfx}/lr", lr_val, step)
            except Exception:
                pass

        # ---------- 5) 文本日志（train 与 val 都写一行） ----------
        try:
            parts = []
            for alias, k in energy_logs.items():
                try:
                    parts.append(f"{alias}={_tofloat(energy_logs.get(alias, np.nan)):.3f}")
                except Exception:
                    pass

            head = "[ENERGY]" if mode == "train" else "[ENERGY-VAL]"
            log_line = f"{head} step={step} | " + " ".join(parts)
            self._elog(log_line)
        except Exception:
            pass

    def _purge_files(self, pattern: str):
        for p in glob.glob(pattern):
            try: os.remove(p)
            except OSError: pass

    def _sync_prev_energy_paths(self):
        ckpt_dir = os.path.join(self.save_root, "ckpt_energy")
        if not os.path.isdir(ckpt_dir): 
            self.prev_last_energy_path = None
            self.prev_best_energy_path = None
            return
        # last
        lasts = sorted(glob.glob(os.path.join(ckpt_dir, "energy_last_step*.pt")))
        self.prev_last_energy_path = lasts[-1] if lasts else None
        # best（取最小 loss）
        bests = []
        for p in glob.glob(os.path.join(ckpt_dir, "energy_best_step*.pt")):
            m = re.search(r"loss([0-9.]+)\.pt$", p)
            if m: bests.append((float(m.group(1)), p))
        self.prev_best_energy_path = min(bests)[1] if bests else None


    def _sync_prev_paths(self):
        ckpt_dir = os.path.join(self.save_root, "ckpt_main")
        if not os.path.isdir(ckpt_dir):
            return
        
        def parse_ckpt_info(path):
            basename = os.path.basename(path)
            try:
                match = re.search(r"step(\d+)(?:_loss([0-9]+(?:\.[0-9]+)?))?\.pt$", basename)
                if not match: return None
                step = int(match.group(1))
                loss = float(match.group(2)) if match.group(2) is not None else float('inf')
                return step, loss, path
            except: return None
        
         # 同步 last checkpoint
        last_ckpts = glob.glob(os.path.join(ckpt_dir, "last_step*.pt"))
        parsed_lasts = [info for p in last_ckpts if (info := parse_ckpt_info(p)) is not None]
        if parsed_lasts:
            self.prev_last_path = sorted(parsed_lasts, key=lambda x: x[0])[-1][2]

        # 同步分阶段的 best checkpoints
        all_best_found = {}
        for phase in [1, 2, 3]:
            phase_ckpts = glob.glob(os.path.join(ckpt_dir, f"best_phase{phase}_step*.pt"))
            parsed_bests = [info for p in phase_ckpts if (info := parse_ckpt_info(p)) is not None]
            if parsed_bests:
                best_in_phase = sorted(parsed_bests, key=lambda x: x[1])[0] # 按 loss 排序，找最小的
                _, loss, path = best_in_phase
                self.best_metrics_by_phase[phase] = loss
                self.prev_best_paths[phase] = path
                all_best_found[phase] = (path, loss)
                

        if self._is_main():
            logging.info(f"[Sync] Synced last checkpoint: {os.path.basename(self.prev_last_path) if self.prev_last_path else 'None'}")
            if all_best_found:
                logging.info("[Sync] Synced best checkpoints from disk (by parsing filenames):")
                for phase, (path, metric) in sorted(all_best_found.items()):
                    logging.info(f"       - Phase {phase}: best_metric={metric:.4f}, path={os.path.basename(path)}")
            
                # 确保从低到高依次检查（2会干掉1，3会干掉1/2）
            for ph in (1, 2, 3):
                if np.isfinite(self.best_metrics_by_phase.get(ph, float('inf'))):
                    self._enforce_phase_dominance(ph)     
                       
    def _maybe_load_ckpt(self, device):
        if not self.resume_ckpt:
            return
        
        logging.info(f"[Resume] Loading from checkpoint: {self.resume_ckpt}")
        ckpt = torch.load(self.resume_ckpt, map_location=device)

        # --- 模型 ---
        target = self.model.module if hasattr(self.model, "module") else self.model
        target.load_state_dict(ckpt["model"], strict=False)

        # --- 步数 & best (提前加载global_step，为EMA同步做准备) ---
        self.global_step  = int(ckpt.get("global_step", 0))
        last_metric = ckpt.get("metric", None)
        if self.best_metric is None and last_metric is not None:
            self.best_metric = float(last_metric)

        # --- [EMA] 加载 EMA 状态 ---
        if self.ema and "ema" in ckpt:
            try:
                self.ema.load_state_dict(ckpt["ema"])
                
                # --- 检查并同步EMA计步器，保证断点续训的兼容性 ---
                # 如果从旧的checkpoint恢复 (里面没有num_updates)，则将EMA的步数手动与全局步数同步
                if self.ema.num_updates is None and self.ema.use_num_updates:
                    self.ema.num_updates = self.global_step
                    logging.warning(f"[EMA Compatibility] Legacy EMA state detected. "
                                    f"Syncing EMA num_updates to global_step: {self.global_step}")
            except Exception as e:
                logging.warning(f"EMA state 未加载: {e}")

        # --- 优化器 / 调度器 ---
        try:  self.optimizer.load_state_dict(ckpt["optimizer"])
        except Exception as e: logging.warning(f"Opt state 未加载: {e}")

        if ckpt.get("scheduler") and self.scheduler:
            try:  self.scheduler.load_state_dict(ckpt["scheduler"])
            except Exception as e: logging.warning(f"Sch state 未加载: {e}")

        # --- 最后再同步一次磁盘路径 ---
        self._sync_prev_paths()
        if self._is_main() and self.best_metric is not None:
            logging.info(f"[Resume] restored best_metric: {self.best_metric:.4f}")
        # 恢复后立刻审计一次，确保 resume 场景目录一致
        self._audit_and_clean_best_main()
        self._audit_and_clean_best_energy()


    def _is_main(self) -> bool:
        return self.local_rank in (-1, 0)

    def _to_dev(self, obj, dev):
        if isinstance(obj, dict):
            return {k: self._to_dev(v, dev) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._to_dev(v, dev) for v in obj]
        if isinstance(obj, tuple):
            return tuple(self._to_dev(v, dev) for v in obj)
        return obj.to(dev) if hasattr(obj, "to") else obj


    def _remove_ckpt_safe(self, path: Optional[str], msg: str = ""):
        if path and os.path.exists(path):
            try:
                os.remove(path)
                if self._is_main():
                    self.file_logger.info(f"[Checkpoint] Removed {msg}: {os.path.basename(path)}")
            except OSError as e:
                logging.warning(f"[Checkpoint] Remove failed ({msg}): {e}")

    def _parse_loss_from_ckpt_path(self, path: str, default=float('inf')) -> float:
        # 兼容 best_phase{p}_step{n}_loss{val}.pt / last_step{n}_loss{val}.pt / energy_*.pt
        m = re.search(r"loss([0-9]+(?:\.[0-9]+)?)\.pt$", os.path.basename(path))
        try:
            return float(m.group(1)) if m else default
        except Exception:
            return default


    def _glob(self, pattern):  # 小工具
        try: return sorted(glob.glob(pattern))
        except: return []

    def _audit_and_clean_best_energy(self):
        ckpt_dir = os.path.join(self.save_root, "ckpt_energy")
        if not os.path.isdir(ckpt_dir):
            self.prev_last_energy_path = None
            self.prev_best_energy_path = None
            return

        # --- last: 只保留 step 最大的那份
        lasts = self._glob(os.path.join(ckpt_dir, "energy_last_step*.pt"))
        def _parse_step(p):
            m = re.search(r"step(\d+)", os.path.basename(p));  return int(m.group(1)) if m else -1
        if lasts:
            lasts = sorted(lasts, key=_parse_step)
            keep_last = lasts[-1]
            for p in lasts[:-1]:
                try: os.remove(p)
                except OSError: pass
            self.prev_last_energy_path = keep_last
        else:
            self.prev_last_energy_path = None

        # --- best: 只保留 loss 最小的那份
        bests = self._glob(os.path.join(ckpt_dir, "energy_best_step*.pt"))
        if bests:
            pairs = []
            for p in bests:
                loss = self._parse_loss_from_ckpt_path(p, default=float('inf'))
                pairs.append((loss, p))
            pairs.sort(key=lambda x: x[0])
            keep_best_loss, keep_best_path = pairs[0]
            for _, p in pairs[1:]:
                try: os.remove(p)
                except OSError: pass
            self.prev_best_energy_path = keep_best_path
            self.best_energy_metric = keep_best_loss
        else:
            self.prev_best_energy_path = None
            # 不改 best_energy_metric，让训练逻辑决定

    def _audit_and_clean_best_main(self):
        """
        主干 ckpt 审计：同阶段只留一份最优 best；跨阶段 dominance 生效；
        last 只留最新一份。
        """
        ckpt_dir = os.path.join(self.save_root, "ckpt_main")
        if not os.path.isdir(ckpt_dir): return

        # --- 清理 last：只留 step 最大
        lasts = self._glob(os.path.join(ckpt_dir, "last_step*.pt"))
        def _parse_step_loss(path):
            m = re.search(r"step(\d+)(?:_loss([0-9]+(?:\.[0-9]+)?))?(?=\.pt$)", os.path.basename(path))
            if not m: return (-1, float('inf'))
            step = int(m.group(1)); loss = float(m.group(2)) if m.group(2) else float('inf')
            return (step, loss)

        if lasts:
            lasts = sorted(lasts, key=lambda p: _parse_step_loss(p)[0])
            keep_last = lasts[-1]
            for p in lasts[:-1]:
                try: os.remove(p)
                except OSError: pass
            self.prev_last_path = keep_last
        else:
            self.prev_last_path = None

        # --- 分阶段 best 收集
        per_phase = {1: [], 2: [], 3: []}
        for ph in (1, 2, 3):
            for p in self._glob(os.path.join(ckpt_dir, f"best_phase{ph}_step*.pt")):
                loss = self._parse_loss_from_ckpt_path(p, default=float('inf'))
                per_phase[ph].append((loss, p))

        # --- 同阶段只留最优
        for ph in (1, 2, 3):
            entries = per_phase[ph]
            if not entries:
                self.prev_best_paths[ph] = None
                self.best_metrics_by_phase[ph] = float('inf')
                continue
            entries.sort(key=lambda x: x[0])  # 按 loss
            keep_loss, keep_path = entries[0]
            for _, p in entries[1:]:
                try: os.remove(p)
                except OSError: pass
            self.prev_best_paths[ph] = keep_path
            self.best_metrics_by_phase[ph] = keep_loss

        # --- 跨阶段 dominance：后期更优则删除早期
        # 以 2 支配 1、3 支配 1/2 的顺序执行
        def _rm(path):
            try:
                os.remove(path)
            except OSError:
                pass

        # phase 2 优于 1 → 删 1
        if np.isfinite(self.best_metrics_by_phase[2]) and \
        np.isfinite(self.best_metrics_by_phase[1]) and \
        (self.best_metrics_by_phase[2] < self.best_metrics_by_phase[1]) and \
        self.prev_best_paths[1]:
            _rm(self.prev_best_paths[1])
            self.prev_best_paths[1] = None
            self.best_metrics_by_phase[1] = float('inf')

        # phase 3 优于 1
        if np.isfinite(self.best_metrics_by_phase[3]) and \
        np.isfinite(self.best_metrics_by_phase[1]) and \
        (self.best_metrics_by_phase[3] < self.best_metrics_by_phase[1]) and \
        self.prev_best_paths[1]:
            _rm(self.prev_best_paths[1])
            self.prev_best_paths[1] = None
            self.best_metrics_by_phase[1] = float('inf')

        # phase 3 优于 2
        if np.isfinite(self.best_metrics_by_phase[3]) and \
        np.isfinite(self.best_metrics_by_phase[2]) and \
        (self.best_metrics_by_phase[3] < self.best_metrics_by_phase[2]) and \
        self.prev_best_paths[2]:
            _rm(self.prev_best_paths[2])
            self.prev_best_paths[2] = None
            self.best_metrics_by_phase[2] = float('inf')
        
        # --- DSM / Seq best：各自只保留一份，按 loss 最小 ---
        dsm_ckpts = self._glob(os.path.join(ckpt_dir, "best_DSM_step*.pt"))
        if dsm_ckpts:
            dsm_pairs = [(self._parse_loss_from_ckpt_path(p), p) for p in dsm_ckpts]
            dsm_pairs.sort(key=lambda x: x[0])
            keep_loss, keep_path = dsm_pairs[0]
            for _, p in dsm_pairs[1:]:
                try: os.remove(p)
                except OSError: pass
            self.best_DSM_metric = keep_loss
            self.prev_best_DSM_path = keep_path
        else:
            self.best_DSM_metric = float('inf')
            self.prev_best_DSM_path = None

        seq_ckpts = self._glob(os.path.join(ckpt_dir, "best_seq_step*.pt"))
        if seq_ckpts:
            seq_pairs = [(self._parse_loss_from_ckpt_path(p), p) for p in seq_ckpts]
            seq_pairs.sort(key=lambda x: x[0])
            keep_loss, keep_path = seq_pairs[0]
            for _, p in seq_pairs[1:]:
                try: os.remove(p)
                except OSError: pass
            self.best_seq_metric = keep_loss
            self.prev_best_seq_path = keep_path
        else:
            self.best_seq_metric = float('inf')
            self.prev_best_seq_path = None


    def _enforce_phase_dominance(self, current_phase: int):
        """
        若“当前阶段的 best”严格优于任意更早阶段的 best，则删除这些被支配的旧 best。
        仅 rank0 执行。
        """
        if not self._is_main():
            return
        cur_best = self.best_metrics_by_phase.get(current_phase, float('inf'))
        if not np.isfinite(cur_best):
            return
        for prev_phase in (1, 2, 3):
            if prev_phase >= current_phase:
                continue
            prev_best = self.best_metrics_by_phase.get(prev_phase, float('inf'))
            prev_path = self.prev_best_paths.get(prev_phase, None)
            # “后期更优 → 删前期”
            if np.isfinite(prev_best) and (cur_best < prev_best) and prev_path:
                self._remove_ckpt_safe(prev_path, msg=f"dominated best_phase{prev_phase}")
                self.prev_best_paths[prev_phase] = None
                self.best_metrics_by_phase[prev_phase] = float('inf')


    def _save_ckpt(self, tag: str, loss_val: float):
        if not self._is_main():
            return
    
        ckpt_dir = os.path.join(self.save_root, "ckpt_main")
        os.makedirs(ckpt_dir, exist_ok=True)
        
        # 先做一次主干审计，确保没有脏 best/last
        # if tag not in ("best_DSM", "best_seq"):
        #     self._audit_and_clean_best_main()
        #self._audit_and_clean_best_main()
        
        path_to_save = None # 先初始化

        if tag.startswith("best_phase"):
            try:
                phase = int(tag.split("best_phase")[1])
                fname = f"best_phase{phase}_step{self.global_step}_loss{loss_val:.4f}.pt"
                
                # 获取并删除旧的 best checkpoint
                path_to_remove = self.prev_best_paths.get(phase)
                if path_to_remove and os.path.exists(path_to_remove):
                    try:
                        os.remove(path_to_remove)
                        # 可以选择性地加一条日志，确认删除操作
                        self.file_logger.info(f"[Checkpoint] Removed old best: {os.path.basename(path_to_remove)}")
                    except OSError as e:
                        logging.warning(f"Failed to remove old best checkpoint: {e}")

                path_to_save = os.path.join(ckpt_dir, fname)
                self.prev_best_paths[phase] = path_to_save # 更新为新路径
                # 把 best_metrics_by_phase[phase] 置成这次的 loss_val
                self.best_metrics_by_phase[phase] = loss_val
                
                # === 保存本阶段新 best 之后，执行“跨阶段支配删除” ===
                self._enforce_phase_dominance(phase)
        
            except (ValueError, IndexError):
                logging.error(f"Invalid best tag: {tag}"); return
        
        elif tag == "last":  # 对 "last" 的处理也应该在这里
            ckpt_dir = os.path.join(self.save_root, "ckpt_main")
            os.makedirs(ckpt_dir, exist_ok=True)
            # 先清掉历史 last（跨 resume 的）
            self._purge_files(os.path.join(ckpt_dir, "last_step*.pt"))
            fname = f"last_step{self.global_step}_loss{loss_val:.4f}.pt"
            path_to_save = os.path.join(ckpt_dir, fname)
            self.prev_last_path = path_to_save
            
        elif tag == "best_DSM":
            fname = f"best_DSM_step{self.global_step}_loss{loss_val:.4f}.pt"
            # 先删旧的 best_DSM
            if self.prev_best_DSM_path and os.path.exists(self.prev_best_DSM_path):
                try:
                    os.remove(self.prev_best_DSM_path)
                    self.file_logger.info(
                        f"[Checkpoint] Removed old best_DSM: {os.path.basename(self.prev_best_DSM_path)}"
                    )
                except OSError as e:
                    logging.warning(f"Failed to remove old best_DSM checkpoint: {e}")
            path_to_save = os.path.join(ckpt_dir, fname)
            self.prev_best_DSM_path = path_to_save

        elif tag == "best_seq":
            fname = f"best_seq_step{self.global_step}_loss{loss_val:.4f}.pt"
            if self.prev_best_seq_path and os.path.exists(self.prev_best_seq_path):
                try:
                    os.remove(self.prev_best_seq_path)
                    self.file_logger.info(
                        f"[Checkpoint] Removed old best_seq: {os.path.basename(self.prev_best_seq_path)}"
                    )
                except OSError as e:
                    logging.warning(f"Failed to remove old best_seq checkpoint: {e}")
            path_to_save = os.path.join(ckpt_dir, fname)
            self.prev_best_seq_path = path_to_save
    
        
        else: # 如果有其他 tag 类型，可以在这里处理
            logging.warning(f"Unknown checkpoint tag: {tag}")
            return

        # --- 后续的保存逻辑完全不变 ---
        # if self.ema:
        #     self.ema.apply_shadow()

        hyperparams_to_save = {
            "lr": self.cfg.get("lr_max"), 
            "grad_clip": self.cfg.get("grad_clip"), 
            "use_ema": self.cfg.get("use_ema"),
            "ema_decay": self.cfg.get("ema_decay"), 
            "model_config": self.cfg.get("model_config"),
            "lr_warmup_steps": self.cfg.get("lr_warmup_steps")
        }
        
        # --- 生成要保存的权重：若有 EMA，合成一个不落地到活模型参数的 state_dict ---
        target_model = self.model.module if hasattr(self.model, "module") else self.model
        if self.ema:
            live_state = target_model.state_dict()
            ema_shadow = self.ema.shadow
            # 用 EMA 覆盖同名 key，不改活参数
            merged_state = {}
            for k, v in live_state.items():
                if (k in ema_shadow) and v.dtype.is_floating_point:
                    merged_state[k] = ema_shadow[k].to(v.device).clone()
                else:
                    merged_state[k] = v.clone()
        else:
            merged_state = target_model.state_dict()
            

        save_payload = {
            "global_step": self.global_step, 
            "metric": loss_val,
            "model": merged_state,
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict() if self.scheduler else None,
            "hyperparams": hyperparams_to_save,
        }
        if self.ema:
            save_payload["ema"] = self.ema.state_dict()

        torch.save(save_payload, path_to_save)

        # if self.ema:
        #     self.ema.restore()
            
        self.file_logger.info(f"[Checkpoint] saved {tag} → {os.path.basename(path_to_save)}")
    
    def _save_energy_ckpt(self, tag: str, energy_metric: float):
        if not (self._is_main() and self._energy_enabled_runtime):
            return

        ckpt_dir = os.path.join(self.save_root, "ckpt_energy")
        os.makedirs(ckpt_dir, exist_ok=True)
        # 存之前：先做一次能量头审计
        self._audit_and_clean_best_energy()
    
        # 保守：每次保存前，确保与磁盘同步一次（跨 resume 有用）
        self._sync_prev_energy_paths()

        if tag == "best":
            # 清库式：把所有 best 全删掉，再只存这一份
            for p in self._glob(os.path.join(ckpt_dir, "energy_best_step*.pt")):
                try: os.remove(p)
                except OSError: pass
            fname = f"energy_best_step{self.global_step}_loss{energy_metric:.4f}.pt"
            path_to_save = os.path.join(ckpt_dir, fname)
            self.prev_best_energy_path = path_to_save

        elif tag == "last":
            # last 采用“全清同类再保存”的策略
            self._purge_files(os.path.join(ckpt_dir, "energy_last_step*.pt"))
            fname = f"energy_last_step{self.global_step}_loss{energy_metric:.4f}.pt"
            path_to_save = os.path.join(ckpt_dir, fname)
            self.prev_last_energy_path = path_to_save
        else:
            logging.warning(f"Unknown energy checkpoint tag: {tag}")
            return

        eh = self.energy_head.module if hasattr(self.energy_head, "module") else self.energy_head
        if self.ema_energy:
            live_state_e = eh.state_dict()
            ema_shadow_e = self.ema_energy.shadow
            merged_e = {}
            for k, v in live_state_e.items():
                if (k in ema_shadow_e) and v.dtype.is_floating_point:
                    merged_e[k] = ema_shadow_e[k].to(v.device).clone()
                else:
                    merged_e[k] = v.clone()
        else:
            merged_e = (eh.state_dict())

        payload = {
            "global_step": self.global_step,
            "energy_model": merged_e,  # 直接保存合成后的 EMA 权重
            "optimizer": self.opt_energy.state_dict() if (self.opt_energy is not None) else None,
            "metric": energy_metric,
            "ema": (self.ema_energy.state_dict() if self.ema_energy else None),
        }
        torch.save(payload, path_to_save)
        self._elog(f"[Energy-CKPT] saved {tag} → {os.path.basename(path_to_save)}")

    def _maybe_load_energy_ckpt(self, device):
        path = self.cfg.get("resume_energy_checkpoint", "")
        if not (path and os.path.isfile(path)):
            return
        logging.info(f"[Energy-Resume] Loading energy checkpoint: {path}")
        ckpt = torch.load(path, map_location=device)

        eh = self.energy_head.module if hasattr(self.energy_head, "module") else self.energy_head
        try:
            eh.load_state_dict(ckpt["energy_model"], strict=False)
            logging.info("[Energy-Resume] energy_model loaded.")
        except Exception as e:
            logging.warning(f"[Energy-Resume] energy_model not loaded: {e}")

        if ("optimizer" in ckpt) and (ckpt["optimizer"] is not None) and (self.opt_energy is not None):
            try:
                self.opt_energy.load_state_dict(ckpt["optimizer"])
                logging.info("[Energy-Resume] optimizer loaded.")
            except Exception as e:
                logging.warning(f"[Energy-Resume] optimizer not loaded: {e}")

        # EMA
        if self.cfg.get("use_ema", False) and ckpt.get("ema"):
            if self.ema_energy is None:
                decay_e = (self.cfg.get("energy_ema_decay")
                        if self.cfg.get("energy_ema_decay") is not None
                        else self.cfg.get("ema_decay", 0.999))
                self.ema_energy = EMA(eh, decay=decay_e)
            try:
                self.ema_energy.load_state_dict(ckpt["ema"])
                # 兼容老 ckpt：如果 num_updates 丢失，用 global_step 兜底
                if (self.ema_energy.num_updates is None) and self.ema_energy.use_num_updates:
                    self.ema_energy.num_updates = int(ckpt.get("global_step", 0))
                logging.info("[Energy-Resume] EMA loaded.")
            except Exception as e:
                logging.warning(f"[Energy-Resume] EMA not loaded: {e}")



    def train(self, device_ids, local_rank):
        device = self._ddp_setup(device_ids, local_rank)
        self._maybe_load_ckpt(device)
        self._sync_prev_paths() # 在所有组件加载后，最终同步一次磁盘状态
        if self._is_main():
            self.writer = SummaryWriter(os.path.join(self.save_root, "tb_main"))
            self.writer_energy = SummaryWriter(os.path.join(self.save_root, "tb_energy"))       
            logging.info(f"[START] training for {self.cfg['max_iters']} iterations "
                         f"on {device_ids} (rank={local_rank})")
            # 开训前做一次全量审计
            self._audit_and_clean_best_main()
            self._audit_and_clean_best_energy()

        train_iter = iter(self.train_loader)
        tot_iters = self.cfg["max_iters"]
        val_freq  = self.cfg["val_freq"]
        log_every = self.cfg.get("log_steps", max(1, val_freq // 10))
        accum_steps = int(self.cfg.get("accumulation_steps", 1))
        accum_steps = max(1, accum_steps)
        if self._is_main():
            logging.info(f"[GA] accumulation_steps={accum_steps} (global_step counts optimizer updates)")

        pbar = tqdm(total=tot_iters, initial=self.global_step,
                    desc="Train", unit="step",
                    disable=not self._is_main(), dynamic_ncols=True)

        try:
            while self.global_step < tot_iters:
                # --- 每个“update step”开始时：清零主干梯度（只做一次） ---
                self.optimizer.zero_grad(set_to_none=True)
                if self._energy_enabled_runtime and (self.opt_energy is not None):
                    self.opt_energy.zero_grad(set_to_none=True)

                # 用于日志：累计 micro-batch 的 loss/breakdown（最后取平均）
                loss_accum = 0.0
                # 用于训练态能量日志：缓存一次“raw（含向量）”的能量输出
                last_energy_logs_raw = None
                breakdown_accum = {}
                
                # DDP: intermediate micro steps 用 no_sync，最后一次才同步 allreduce
                ddp_main = hasattr(self.model, "no_sync")  # DDP wrapper 才有
                ddp_energy = self._energy_enabled_runtime and hasattr(self.energy_head, "no_sync")

                for micro_idx in range(accum_steps):
                    try:
                        batch = next(train_iter)
                    except StopIteration:
                        train_iter = iter(self.train_loader)
                        batch = next(train_iter)

                    batch = self._to_dev(batch, device)
                    
                    # ---------- Energy plugin: lazy enable @ step ----------
                    if (micro_idx == 0) and (self.enable_energy
                        and (not self._energy_enabled_runtime)
                        and (self.global_step >= self.cfg.get("energy_start_step", 0))):
                        
                        # 解冻 —— 只解冻“能量头独有”的参数
                        for p in self.energy_head.parameters():
                            if id(p) in getattr(self, "_energy_shared_param_ids", set()):
                                continue
                            p.requires_grad = True
        
                        if self._is_main():
                            n_trainable = sum(p.numel() for p in self.energy_head.parameters() if p.requires_grad)
                            self._elog(f"[Energy] trainable params: {n_trainable/1e6:.2f}M")
        
                            # === 此时再包 DDP ===
                        if dist.is_initialized() and (self.local_rank != -1):
                            # 启用后必须真的有可训练参数，否则依旧无意义
                            if not any(p.requires_grad for p in self.energy_head.parameters()):
                                raise RuntimeError("energy_head still has no trainable parameters after unfreezing.")
                            self.energy_head.to(device)
                            if not isinstance(self.energy_head, torch.nn.parallel.DistributedDataParallel):
                                self.energy_head = torch.nn.parallel.DistributedDataParallel(
                                    self.energy_head,
                                    device_ids=[self.local_rank],
                                    output_device=self.local_rank,
                                    find_unused_parameters=True  # 能量头启用后参与反传，不需要 unused
                )
                
                        # 学习率/权重衰减：未指定则继承主干
                        eff_lr = self.cfg.get("lr_max", 1e-3)
                        eff_wd = (self.cfg["energy_weight_decay"]
                                if self.cfg["energy_weight_decay"] is not None
                                else self.cfg.get("weight_decay", 0.0))

                        eh = self.energy_head.module if hasattr(self.energy_head, "module") else self.energy_head
                        eh_params = list(eh.parameters())
                        
                        no_decay_e = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight"]
                        decay_e, no_decay_grp_e = [], []
                        _shared = getattr(self, "_energy_shared_param_ids", set())

                        for n, p in (self.energy_head.module if hasattr(self.energy_head,"module") else self.energy_head).named_parameters():
                            # 1) 排除共享到主干的参数（绝不能被能量头优化器管理）
                            if id(p) in _shared:
                                continue
                            # 2) 只纳入能量头“独有且可训练”的参数
                            if not p.requires_grad:
                                continue
                            if any(nd in n for nd in no_decay_e):
                                no_decay_grp_e.append(p)
                            else:
                                decay_e.append(p)

                        self.opt_energy = torch.optim.AdamW(
                            [{"params": decay_e, "weight_decay": eff_wd if eff_wd is not None else 0.01},
                            {"params": no_decay_grp_e, "weight_decay": 0.0}],
                            lr=eff_lr
                        )
                        # ====== 用 ratio 计算实际的 steps（内部使用）======
                        total_steps = int(self.cfg["max_iters"])

                        # 1) warmup
                        warmup_ratio = float(self.cfg.get("lr_warmup_ratio", 0.05))
                        warmup_ratio = max(0.0, min(warmup_ratio, 0.999))
                        warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

                        # 2) decay start
                        decay_start_ratio = float(self.cfg.get("lr_start_decay_ratio", 0.5))
                        decay_start_ratio = max(0.0, min(decay_start_ratio, 0.999))
                        decay_start_steps = max(warmup_steps + 1, int(round(total_steps * decay_start_ratio)))

                        # 3) decay interval
                        decay_every_ratio = float(self.cfg.get("lr_decay_every_ratio", 0.1))
                        decay_every_ratio = max(0.0, min(decay_every_ratio, 0.999))
                        decay_every_steps = max(1, int(round(total_steps * decay_every_ratio)))

                        # 如果 decay 起点超过总步数，向前拉一点，至少保证有衰减区间
                        if decay_start_steps >= total_steps:
                            decay_start_steps = max(warmup_steps + 1, total_steps - 1)
                            
                        self.sched_energy = AlphaFoldLRScheduler(
                            optimizer=self.opt_energy,
                            base_lr=self.cfg.get('lr_base', 0.0),
                            max_lr=self.cfg.get('lr_max', 1e-3),
                            warmup_no_steps=warmup_steps,
                            start_decay_after_n_steps=decay_start_steps,
                            decay_every_n_steps=decay_every_steps,
                            decay_factor=self.cfg.get('lr_decay_factor', 0.95),
                        )

                        # 构建“只算loss与日志
                        trainer_kwargs = {}
                        if self.cfg.get("energy_config"):
                            with open(self.cfg["energy_config"]) as ef:
                                ec = ml_collections.ConfigDict(json.load(ef))
                                trainer_kwargs.update(ec.get("interface_energy", {}).get("trainer_cfg", {}))
                                
                        # —— EMA for energy head：仅在启用瞬间创建/注册 —— 
                        if self.cfg.get("use_ema", False):
                            decay_e = self.cfg.get("energy_ema_decay")
                            # 这里一定要用“解包后”的模块（与 DDP 解耦），避免把 DDP wrapper 交给 EMA
                            eh_unwrapped = self.energy_head.module if hasattr(self.energy_head, "module") else self.energy_head
                            self.ema_energy = EMA(eh_unwrapped, decay=decay_e)
                            if self._is_main():
                                self._elog(f"[Energy-EMA] enabled with decay={decay_e}")
            
                        self.energy_trainer = EnergyTrainer(energy_model=self.energy_head, **trainer_kwargs)
                        self._energy_enabled_runtime = True
                        # 如指定了能量恢复 ckpt，这里加载（必须在 DDP 包裹 & 优化器构建之后）
                        self._maybe_load_energy_ckpt(device)


                    # ================= 第一段：只训练主干 =================
                    # ---------------- 主干 forward/backward（累计） ----------------
                    # 注意：要除以 accum_steps，保证梯度等价于“大batch求均值”
                    sync_ctx = torch.enable_grad()
                    if ddp_main and (micro_idx < accum_steps - 1):
                        sync_ctx = self.model.no_sync()

                    with sync_ctx:
                        loss, breakdown = self.forward_loss(batch, self.global_step, training=True)

                        # stability: 建议对“未缩放的 loss”做保护，然后再缩放
                        if loss.item() > 100.0:
                            self.file_logger.warning(
                                f"[STABILITY] Loss spike! update_step={self.global_step}, micro={micro_idx}, loss={loss.item():.2f} -> clamp"
                            )
                            loss = torch.clamp(loss, max=100.0)

                        (loss / accum_steps).backward()

                    loss_accum += float(loss.detach().item())

                    for k, v in breakdown.items():
                        breakdown_accum[k] = breakdown_accum.get(k, 0.0) + float(_tofloat(v))
                    
                    # ---------------- 能量头 forward/backward（累计） ----------------
                    if self._energy_enabled_runtime and (self.opt_energy is not None):
                        safe_out = getattr(self, "_ret_final_for_energy", None)
                        if safe_out is None:
                            raise RuntimeError("safe_out missing. forward_loss(training=True) must cache it.")
                        self._ret_final_for_energy = None

                        e_sync_ctx = torch.enable_grad()
                        if ddp_energy and (micro_idx < accum_steps - 1):
                            e_sync_ctx = self.energy_head.no_sync()

                        with e_sync_ctx:
                            energy_logs = self.energy_trainer.train_step(batch=batch, ret_final=safe_out)
                            loss_energy = energy_logs["loss"]

                            # 你 args 里有 energy_lambda，但当前训练里没用上：这里建议乘进去
                            energy_lambda = float(self.cfg.get("energy_lambda", 1.0))
                            (loss_energy * energy_lambda / accum_steps).backward()
                            if self._is_main():
                                def _detach_tree(x):
                                    if isinstance(x, dict):
                                        return {k: _detach_tree(v) for k, v in x.items()}
                                    if isinstance(x, (list, tuple)):
                                        return type(x)(_detach_tree(v) for v in x)
                                    if torch.is_tensor(x):
                                        return x.detach()
                                    return x
                                last_energy_logs_raw = _detach_tree(energy_logs)

                        # 把能量 logs 也累计进 breakdown_accum（用于最后 log/ckpt）
                        for k, v in energy_logs.items():
                            try:
                                breakdown_accum[f"Energy/{k}"] = breakdown_accum.get(f"Energy/{k}", 0.0) + float(_tofloat(v))
                            except Exception:
                                pass
                
                # ---------- micro loop 结束：做一次 clip + step（主干） ----------
                total_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.get("grad_clip", 1.0))
                self.optimizer.step()

                if self.scheduler and self.cfg["sched_freq"] == "batch":
                    self.scheduler.step()
                if self.ema:
                    self.ema.update()

                # ---------- energy optimizer：也只 step 一次 ----------
                if self._energy_enabled_runtime and (self.opt_energy is not None):
                    # 记录能量头 grad norm（step 前）
                    gn = 0.0; cnt = 0
                    for p in (self.energy_head.module if hasattr(self.energy_head,"module") else self.energy_head).parameters():
                        if p.requires_grad and (p.grad is not None):
                            g2 = float(p.grad.detach().float().norm(2).item() ** 2)
                            gn += g2; cnt += 1
                    self._last_energy_grad_norm = float(_math.sqrt(gn)) if cnt>0 else 0.0

                    torch.nn.utils.clip_grad_norm_(
                        [p for p in (self.energy_head.module if hasattr(self.energy_head,"module") else self.energy_head).parameters()
                        if p.requires_grad],
                        self.cfg.get("grad_clip", 1.0)
                    )
                    self.opt_energy.step()

                    if (self.sched_energy is not None) and self.cfg["sched_freq"] == "batch":
                        self.sched_energy.step()
                    if self.ema_energy is not None:
                        self.ema_energy.update()
                
                # ---------- 更新 global_step：现在它是“optimizer update step” ----------
                self.global_step += 1
                pbar.update(1)
                # ---------- 日志：对 breakdown_accum 做平均 ----------
                breakdown_mean = {k: (v / accum_steps) for k, v in breakdown_accum.items()}
                loss_mean = loss_accum / accum_steps

                if self._is_main() and self.global_step % log_every == 0:
                    self.writer.add_scalar("Loss/train", loss_mean, self.global_step)
                    self.writer.add_scalar("LR", self.optimizer.param_groups[0]['lr'], self.global_step)
                    self.writer.add_scalar("GradNorm", total_norm, self.global_step)
                    if torch.cuda.is_available():
                        self.writer.add_scalar("GPU/alloc_MB", torch.cuda.memory_allocated() / 1024 ** 2, self.global_step)
                    # 仅主干项写入主 TB
                    for k, v in breakdown_mean.items():
                        if not str(k).startswith("Energy/"):
                            self.writer.add_scalar(f"LossBreak/{k}", v, self.global_step)

                    # 仅主干项写入主 log
                    main_items = [f"{k}={_tofloat(v):.3f}" for k, v in breakdown_mean.items() if not str(k).startswith("Energy/")]
                    loss_str = " ".join(main_items)
                    self.file_logger.info(f"[TRAIN] step={self.global_step} | loss={loss_mean:.3f} | {loss_str}")
                    
                    pbar.set_postfix(loss=f"{loss_mean:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.2e}")
                    
                    diff = sum(_tofloat(breakdown_mean.get(k, 0.0)) for k in ['DSM','ce','elbo'])
                    geo  = sum(_tofloat(breakdown_mean.get(k, 0.0)) for k in ['fape','distogram','plddt'])
                    phys = _tofloat(breakdown_mean.get('violation', 0.0))
                    fpe_r3  = _tofloat(breakdown_mean.get('FPE_r3', 0.0))
                    fpe_so3 = _tofloat(breakdown_mean.get('FPE_so3', 0.0))

                    pbar.set_postfix_str(f"L={loss_mean:.3f} | D={diff:.3f} G={geo:.3f} P={phys:.3f} F_r3={fpe_r3:.3f} F_so3={fpe_so3:.3f}")

                # ================= 写能量头训练日志=================
                if self._energy_enabled_runtime and self._is_main() and (self.global_step % self.energy_log_every == 0):
                    if last_energy_logs_raw is not None:
                        self._log_energy_metrics(last_energy_logs_raw, step=self.global_step, mode="train", aggregated=False)

                if (self.cfg["save_interval"] > 0 and
                    self.global_step % self.cfg["save_interval"] == 0 and
                    self.global_step != 0):
                    self._save_ckpt("last", loss_mean)
                    e_metric = float(breakdown_mean.get("Energy/loss", 0.0)) if breakdown_mean else 0.0
                    self._save_energy_ckpt("last", e_metric)

                if (self.global_step % val_freq == 0 and self.global_step > 0) or (self.global_step == tot_iters):
                    # 1.  所有进程都必须调用 _run_validation 以进行同步
                    # val_metric 现在是所有进程上都相同的、聚合后的全局值
                    val_metric = self._run_validation(device)
                    
                    # 2.  只有主进程(rank 0)进行判断、调度、记录和保存
                    if self._is_main():
                        if self.scheduler and self.cfg.get("sched_freq") == "val":
                            self.scheduler.step(val_metric)

                        # 从 loss_cfg (在 build 中设置) 获取阶段信息
                        assert self.loss_cfg is not None, "self.loss_cfg was not set in build()"
                        phase1_end = self.loss_cfg.get("curriculum_phase1_steps", 20000)
                        phase2_end = self.loss_cfg.get("curriculum_phase2_steps", 160000)
                        if self.global_step >= phase2_end:
                            current_phase = 3
                        elif self.global_step >= phase1_end:
                            current_phase = 2
                        else:
                            current_phase = 1
                        
                        # 与对应阶段的最佳指标比较
                        is_improve = (val_metric < self.best_metrics_by_phase[current_phase])
                        
                        if is_improve:
                            self.best_metrics_by_phase[current_phase] = val_metric
                            self._save_ckpt(f"best_phase{current_phase}", val_metric)
                            # 日志记录也只在主进程执行
                            logging.info(f"[Best VAL] New best for Phase {current_phase}: {val_metric:.4f} at step {self.global_step}")
                            self.no_improve = 0
                        else:
                            # 只有在最后一个阶段才考虑 early stopping
                            if self.patience > 0 and current_phase == 3:
                                self.no_improve += val_freq
                        
                        # === best_DSM: 按 DSM 验证均值最小保存 ckpt（不分 phase）===
                        dsm_val = self._run_validation_outputs.get("DSM", None)
                        if dsm_val is not None:
                            dsm_float = _tofloat(dsm_val)
                            if np.isfinite(dsm_float) and dsm_float < self.best_DSM_metric:
                                self.best_DSM_metric = dsm_float
                                self._save_ckpt("best_DSM", self.best_DSM_metric)
                                logging.info(
                                    f"[Best DSM] New best DSM={self.best_DSM_metric:.4f} at step {self.global_step}"
                                )

                        # === best_seq: 这里定义为 ce + elbo（你以后要改成纯 ce，只改这一段就够了）===
                        ce_val   = self._run_validation_outputs.get("ce", None)
                        elbo_val = self._run_validation_outputs.get("elbo", None)
                        if (ce_val is not None) or (elbo_val is not None):
                            ce_f   = _tofloat(ce_val)   if ce_val   is not None else 0.0
                            elbo_f = _tofloat(elbo_val) if elbo_val is not None else 0.0
                            #seq_metric = ce_f + elbo_f
                            seq_metric = ce_f  # 仅 ce
                            if np.isfinite(seq_metric) and seq_metric < self.best_seq_metric:
                                self.best_seq_metric = seq_metric
                                self._save_ckpt("best_seq", self.best_seq_metric)
                                logging.info(
                                    f"[Best Seq] New best Seq={self.best_seq_metric:.4f} "
                                    f"(ce) at step {self.global_step}"
                                )

                        # === 能量 best（仅在能量已启用时；不分阶段；允许 0 触发） ===
                        if self._energy_enabled_runtime:
                            import math as _m
                            raw = self._run_validation_outputs.get("Energy/loss", None)
                            try:
                                val_energy = float(raw)
                            except (TypeError, ValueError):
                                val_energy = float('inf')

                            if not _m.isfinite(val_energy):
                                # 兜底：如果 eval_step 没返回 loss，就尝试分量近似
                                # 注意：验证路径里目前只回填了 loss/energy_*，没有 rank_*，因此大概率还是 inf。
                                approx_keys = ("Energy/rank", "Energy/r_pred", "Energy/r_noise", "Energy/bce")
                                approx_vals = [self._run_validation_outputs.get(k, float('inf')) for k in approx_keys]
                                if all(_m.isfinite(float(v)) for v in approx_vals):
                                    val_energy = float(sum(float(v) for v in approx_vals))

                            # 现在 val_energy 合法时（包含 0.0），触发 best
                            if _m.isfinite(val_energy) and (val_energy < self.best_energy_metric):
                                self.best_energy_metric = val_energy
                                self._save_energy_ckpt("best", val_energy)
                                self._elog(f"[Best VAL Energy] New best: {val_energy:.4f} @ step {self.global_step}")

                    # 3.  Early stopping 的同步
                    # 主进程的 no_improve 状态需要同步给其他进程
                    should_stop = torch.tensor(0, device=device)
                    if self._is_main():
                        if self.patience > 0 and self.no_improve >= self.patience:
                            should_stop.fill_(1)
                    
                    if dist.is_initialized():
                        dist.broadcast(should_stop, src=0)

                    if should_stop.item() == 1:
                        if self._is_main(): logging.info("Early stopping triggered. Terminating training.")
                        break # 所有进程都会收到信号并退出循环
        except Exception as e:
            logging.exception("Training terminated with exception.")
            raise e
        finally:
            pbar.close()
            if self._is_main():
                logging.info("[FINISH] training loop completed.")

    def _ddp_setup(self, device_ids, local_rank):
        self.local_rank = 0 if (local_rank is None or local_rank == -1) else int(local_rank)
        self.cfg['rank'] = self.local_rank
        # world_size：单卡=1，多卡=进程数
        self.cfg['world_size'] = max(1, int(os.environ.get("WORLD_SIZE", "1")))

        main_dev = local_rank if local_rank != -1 else device_ids[0]
        device = torch.device("cpu" if main_dev == -1 else f"cuda:{main_dev}")
        
        self.build()
        
        self.model.to(device)
        if hasattr(self, "energy_head") and (self.energy_head is not None):
            self.energy_head.to(device)
        
        if local_rank != -1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
        return device

    def _run_validation(self, device):
        if self.ema:
            self.ema.apply_shadow()
        self.model.eval()
        
        # —— 能量头验证也走 EMA，与主干保持一致 ——
        _used_energy_ema = False
        if self._energy_enabled_runtime and (self.ema_energy is not None):
            self.ema_energy.apply_shadow()
            _used_energy_ema = True
        if self._energy_enabled_runtime and hasattr(self, "energy_head") and (self.energy_head is not None):
            self.energy_head.eval()
            
        # 每个进程只初始化自己的容器
        local_metric_sum = 0.0
        local_loss_sum_dict = {}
        local_batch_count = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = self._to_dev(batch, device)
                loss, breakdown = self.forward_loss(batch, self.global_step, training=False)
                
                local_metric_sum += loss.item()
                local_batch_count += 1
                
                for k, v in breakdown.items():
                    local_loss_sum_dict[k] = local_loss_sum_dict.get(k, 0.0) + _tofloat(v)
        
        self.model.train()
        if self.ema:
            self.ema.restore()

        # —— 恢复能量头的原始权重，与主干一致 ——
        if _used_energy_ema:
            self.ema_energy.restore()
        if self._energy_enabled_runtime and hasattr(self, "energy_head") and (self.energy_head is not None):
            self.energy_head.train()
        
        #  聚合所有进程的结果
        if dist.is_initialized():
            # 1) 聚合 batch 计数
            local_tensor_counts = torch.tensor([local_batch_count], device=device, dtype=torch.float32)
            dist.all_reduce(local_tensor_counts, op=dist.ReduceOp.SUM)
            global_batch_count = int(local_tensor_counts.item())

            # 2) 聚合 loss 和
            local_tensor_metrics = torch.tensor([local_metric_sum], device=device, dtype=torch.float32)
            dist.all_reduce(local_tensor_metrics, op=dist.ReduceOp.SUM)
            global_metric_sum = float(local_tensor_metrics.item())

            # 3) 聚合 breakdown
            keys = sorted(local_loss_sum_dict.keys())
            if len(keys) == 0:
                global_metric = (global_metric_sum / global_batch_count) if global_batch_count > 0 else 0.0
                global_avg_loss_dict = {}
            else:
                local_breakdown_tensor = torch.tensor(
                    [float(local_loss_sum_dict[k]) for k in keys],
                    device=device, dtype=torch.float32
                )
                dist.all_reduce(local_breakdown_tensor, op=dist.ReduceOp.SUM)
                global_breakdown_sums = local_breakdown_tensor.detach().cpu().numpy()
                denom = max(global_batch_count, 1)
                global_metric = global_metric_sum / denom
                global_avg_loss_dict = {keys[i]: float(global_breakdown_sums[i]) / denom for i in range(len(keys))}
        else:
            global_metric = (local_metric_sum / local_batch_count) if local_batch_count > 0 else 0.0
            global_avg_loss_dict = {k: float(v) / max(local_batch_count, 1) for k, v in local_loss_sum_dict.items()}

            
        #  只在主进程记录日志
        if self._is_main(): 
            self.writer.add_scalar("Loss/val", global_metric, self.global_step)
            metrics_to_log = {f"phase_{k}": v for k, v in self.best_metrics_by_phase.items() if v != float('inf')}
            self.writer.add_scalars("BestMetricsByPhase", metrics_to_log, self.global_step)

            # —— 只把“主干项”写入 TensorBoard 和主干文件日志；过滤掉能量头项 —— 
            for k, v in global_avg_loss_dict.items():
                if not str(k).startswith("Energy/"):
                    self.writer.add_scalar(f"LossBreakValAvg/{k}", v, self.global_step)

            # 主干日志只打印主干项，避免出现 Energy/xxx
            main_items = [f"{k}={v:.3f}" for k, v in global_avg_loss_dict.items() if not str(k).startswith("Energy/")]
            loss_breakdown_str = " | ".join(main_items)
            self.file_logger.info(f"[VAL] step={self.global_step} | L={global_metric:.3f} | {loss_breakdown_str}")

            # === 只在这里（聚合后）写一次 能量头 VAL 的 TB & 文本日志 ===
            if self._energy_enabled_runtime:
                # 从全局平均字典抽取能量标量；注意去掉 "Energy/" 前缀
                def _get(key, default=np.nan):
                    v = global_avg_loss_dict.get(key, default)
                    try:
                        return float(v)
                    except Exception:
                        return default
                energy_logs_agg = {
                    "loss":      _get("Energy/loss"),
                    "rank":       _get("Energy/rank"),
                    "r_pred":  _get("Energy/r_pred"),
                    "r_noise": _get("Energy/r_noise"),
                    "bce":        _get("Energy/bce"),
                    "Eg":       _get("Energy/Eg"),
                    "Ep":     _get("Energy/Ep"),
                    "En":    _get("Energy/En"),
                    "rmsd_p":  _get("Energy/rmsd_p"),
                    "rmsd_n": _get("Energy/rmsd_n"),
                }
                # 统一用已经改造过的工具函数来写（一次性）
                self._log_energy_metrics(energy_logs_agg, step=self.global_step, mode="val", aggregated=True)


    
        # 缓存全量指标，供 train() 内保存 best 使用
        self._run_validation_outputs = dict(global_avg_loss_dict)
        self._run_validation_outputs['total_loss'] = global_metric
    
        # 返回全局唯一的、聚合后的验证指标
        return global_metric


    def build(self): ...
    def forward_loss(self, batch, global_step, training=True): ...


# --------------------------------------------------------
# 2. ABXTrainer
# --------------------------------------------------------
class ABXTrainer(TrainerBase):
    
    def build(self):
        with open(self.cfg["model_config"]) as f:
            mcfg = ml_collections.ConfigDict(json.load(f))
        with open(self._abs(self.cfg["model_features"])) as f:
            feats_cfg = json.load(f)

        self.loss_cfg = mcfg.loss  
        #dev_str = ("cpu" if self.cfg["gpus"][0] == -1 else f"cuda:{self.cfg['rank']}")       
        # 单卡时 rank==-1，此时应当指向 gpus[0]；多卡时用 local_rank
        if self.cfg["gpus"][0] == -1:
            dev_str = "cpu"
        elif self.local_rank == -1:
            dev_str = f"cuda:{self.cfg['gpus'][0]}"
        else:
            dev_str = f"cuda:{self.local_rank}"

        def _replace(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "device" and v == "%(device)s": obj[k] = dev_str
                    else: _replace(v)
            elif isinstance(obj, (list, tuple)):
                for v in obj: _replace(v)
        _replace(feats_cfg)

        self.diffuser = FullDiffuser.get(mcfg.diffuser)
        self.model    = ScoreNetwork(model_conf=mcfg.model, diffuser=self.diffuser)
        self.loss_fn  = AlphaFoldLoss(mcfg.loss)
        
        # ---------- Energy plugin: register (不启用，仅挂载到 Trainer) ----------
        self.enable_energy = bool(self.cfg.get("enable_energy", False))
        self._energy_enabled_runtime = False
        self.opt_energy = None
        self.energy_trainer = None
        self.energy_head = None
        self.energy_logger = logging.getLogger("energy")

        # 能量独立配置（可选）
        energy_cfg = {}
        if self.cfg.get("energy_config"):
            with open(self.cfg["energy_config"]) as ef:
                energy_cfg = ml_collections.ConfigDict(json.load(ef))

        if self.enable_energy:
            self.energy_head = InterfaceEnergy(energy_cfg['interface_energy']['repr_cfg'],
                                               energy_cfg['interface_energy']['readout'],
                                               energy_cfg['interface_energy']['derivative'],
                                               backbone=self.model.impl.diffusion_module.ScoreNetwork.embedding_and_seqformer_module
                                               )

            # 记录“主干表征”的参数 id 集合，用于识别共享参数
            _backbone = self.model.impl.diffusion_module.ScoreNetwork.embedding_and_seqformer_module
            self._energy_shared_param_ids = {id(p) for p in _backbone.parameters()}

            # 启用前先冻结 —— 只冻结“能量头独有”的参数，跳过共享到主干的参数
            for p in self.energy_head.parameters():
                if id(p) in self._energy_shared_param_ids:
                    # 这是主干的共享参数，绝不能在能量头这边改 requires_grad
                    continue
                p.requires_grad = False
    
            self._elog("[Plugin] Energy head registered. It will be activated at the specified start_step.")

        

        if self.cfg.get("use_ema", False):
            self.ema = EMA(self.model, decay=self.cfg.get("ema_decay", 0.999))

        train_names = self._read_idx(self.cfg["train_idx"])
        val_names   = self._read_idx(self.cfg["val_idx"])
       # bs_gpu = max(1, self.cfg["batch_size"] // max(1, len(self.cfg["gpus"])))
        bs_gpu = max(1, self.cfg["batch_size"])
        #r, w = self.cfg["rank"], self.cfg["world_size"]
        # 确保单卡 r=0, w=1；多卡用实际 local_rank / world_size
        r = 0 if (self.local_rank == -1) else self.local_rank
        w = 1 if (self.cfg.get('world_size', 1) <= 1) else self.cfg['world_size']
        
        accum_steps = int(self.cfg.get("accumulation_steps", 1))
        accum_steps = max(1, accum_steps)

        if self.cfg["max_iters"] <= 0:
            # 自动设置 max_iters
            TOTAL_SAMPLES = 10 * 320000  # 3_200_000
            effective_global_bs = bs_gpu * max(1, w) * accum_steps
            effective_global_bs = max(1, effective_global_bs)
            new_max_iters = max(1, TOTAL_SAMPLES // effective_global_bs)

            if self._is_main():
                logging.info(
                    f"[auto-max-iters] micro_bs={bs_gpu}, world_size={w}, accum_steps={accum_steps} "
                    f"=> effective_global_bs={effective_global_bs}, set max_iters={new_max_iters}"
                )
            self.cfg["max_iters"] = int(new_max_iters)
        
        _nw = int(self.cfg["num_workers"])
        _pw = (_nw > 0)  # 只有 >0 才开持久化 worker
        self.train_loader = dataset.load(
            self.cfg["data_dir"], train_names, feats_cfg, True,
            rank=r, world_size=w, batch_size=bs_gpu, num_workers=self.cfg["num_workers"],
            pin_memory=False, persistent_workers=_pw)

        self.val_loader = dataset.load(
            self.cfg["data_dir"], val_names, feats_cfg, True,
            rank=r, world_size=w, batch_size=bs_gpu, num_workers=self.cfg["num_workers"],
            pin_memory=False, persistent_workers=_pw)

        # 用 idx 列表长度按 world_size 与 batch_size 估算步数
        accum_steps = max(1, int(self.cfg.get("accumulation_steps", 1)))
        est_n_updates = math.ceil(len(train_names) / max(1, w) / (bs_gpu * accum_steps))
        est_n_val     = max(1, math.ceil(len(val_names) / max(1, w) / bs_gpu))  # val 通常不 accumulation
        logging.info(f"[sanity] rank={r}, world_size={w}, "
                     f"est_train_batches/epoch≈{est_n_updates}, est_val_batches/epoch≈{est_n_val}, "
                     f"train_idx={len(train_names)}, val_idx={len(val_names)}, bs_gpu={bs_gpu}")
        self.step_per_epoch     = est_n_updates
        self.val_step_per_epoch = est_n_val

        # AdamW + 分组：对 LayerNorm/偏置不做权重衰减
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight", "ln.weight"]
        decay_params, no_decay_params = [], []
        for n, p in self.model.named_parameters():
            if not p.requires_grad: continue
            if any(nd in n for nd in no_decay): no_decay_params.append(p)
            else: decay_params.append(p)
        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": self.cfg.get("weight_decay", 0.01)},
             {"params": no_decay_params, "weight_decay": 0.0}],
            lr=self.cfg.get("lr_max", 1e-3)
        )

        # ====== 用 ratio 计算实际的 steps（内部使用）======
        total_steps = int(self.cfg["max_iters"])
        if total_steps <= 0:
            raise ValueError(f"max_iters must be positive, got {total_steps}")

        # 1) warmup
        warmup_ratio = float(self.cfg.get("lr_warmup_ratio", 0.05))
        warmup_ratio = max(0.0, min(warmup_ratio, 0.999))
        warmup_steps = max(1, int(round(total_steps * warmup_ratio)))

        # 2) decay start
        decay_start_ratio = float(self.cfg.get("lr_start_decay_ratio", 0.5))
        decay_start_ratio = max(0.0, min(decay_start_ratio, 0.999))
        decay_start_steps = max(warmup_steps + 1, int(round(total_steps * decay_start_ratio)))

        # 3) decay interval
        decay_every_ratio = float(self.cfg.get("lr_decay_every_ratio", 0.1))
        decay_every_ratio = max(0.0, min(decay_every_ratio, 0.999))
        decay_every_steps = max(1, int(round(total_steps * decay_every_ratio)))

        # 如果 decay 起点超过总步数，向前拉一点，至少保证有衰减区间
        if decay_start_steps >= total_steps:
            decay_start_steps = max(warmup_steps + 1, total_steps - 1)
            
            
        self.scheduler = AlphaFoldLRScheduler(
            optimizer=self.optimizer,
            base_lr=self.cfg.get('lr_base', 0.0), 
            max_lr=self.cfg.get('lr_max', 1e-3),
            warmup_no_steps=warmup_steps,
            start_decay_after_n_steps=decay_start_steps,
            decay_every_n_steps=decay_every_steps,
            decay_factor=self.cfg.get('lr_decay_factor', 0.95)
        )
        # AlphaFoldLRScheduler 是按步更新的
        self.cfg["sched_freq"] = "batch" 
        total_steps = self.cfg['max_iters']
        if self._is_main():
            n_param = sum(p.numel() for p in self.model.parameters()) / 1e6
            logging.info(f"[build] steps/epoch≈{self.step_per_epoch}, total_steps={total_steps}, #param={n_param:.2f} M")
            if self.ema:
                logging.info(f"[build] EMA enabled with decay={self.ema.decay}")

    def forward_loss(self, batch, global_step, training=True):
        t = torch.rand(batch["rigids_t"].shape[0], device=batch["rigids_t"].device)
        batch["t"] = t
        batch["rot_score_scaling"], batch["trans_score_scaling"] = self.diffuser.score_scaling(t)
        out = self.model(batch, global_step)
        
        loss_main, breakdown = self.loss_fn(out, batch, global_step, _return_breakdown=True)
        
        # —— 能量头：训练/验证分路径 —— 
        if self.enable_energy and self._energy_enabled_runtime:
            if training:
                # 训练路径（方案B）：这里只做“安全缓存”，不做能量前/反向
                def _detach_tree(x):
                    if isinstance(x, dict): return {k: _detach_tree(v) for k, v in x.items()}
                    if isinstance(x, (list, tuple)): return type(x)(_detach_tree(v) for v in x)
                    return x.detach() if torch.is_tensor(x) else x

                safe_out = _detach_tree(out)
                # 把当步主干输出（已 detach）暂存到 Trainer，供第二段使用
                self._ret_final_for_energy = safe_out

                # 训练阶段总损失仅主干
                loss_total = loss_main

                # 训练态此处不写入 Energy/* breakdown（避免和第二段重复/混淆）
            else:
                # 验证路径：只做评估，不参与 loss/backward（保持原逻辑）
                with torch.no_grad():
                    energy_logs = self.energy_trainer.eval_step(batch=batch, ret_final=out)
                
                breakdown["Energy/loss"]      = float(energy_logs["loss"])
                breakdown["Energy/Eg"]       = float(energy_logs["Eg"])
                breakdown["Energy/Ep"]     = float(energy_logs["Ep"])
                breakdown["Energy/En"]    = float(energy_logs["En"])
                breakdown["Energy/rmsd_p"]  = float(energy_logs["rmsd_p"])
                breakdown["Energy/rank"]       = float(energy_logs["rank"])
                breakdown["Energy/r_pred"]  = float(energy_logs["r_pred"])
                breakdown["Energy/r_noise"] = float(energy_logs["r_noise"])
                breakdown["Energy/bce"]        = float(energy_logs["bce"])
                breakdown["Energy/rmsd_n"] = float(energy_logs["rmsd_n"])

                loss_total = loss_main  # 验证阶段总损失仅主干
        else:
            # 未启用能量头：总损失仅主干
            loss_total = loss_main

        return loss_total, breakdown


    @staticmethod
    def _read_idx(path):
        with open(path) as f:
            return [l.strip() for l in f if l.strip()]
    @staticmethod
    def _abs(p): return p if os.path.isabs(p) else os.path.abspath(p)


def parse_args():
    p = argparse.ArgumentParser("ABX trainer (iteration-driven)")
    p.add_argument("--model_config",    required=True)
    p.add_argument("--model_features", required=True)
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--train_idx",      required=True)
    p.add_argument("--val_idx",        required=True)
    p.add_argument("--output_dir",  required=True)
    p.add_argument("--batch_size",  type=int,   default=5)
    p.add_argument("--accumulation_steps", type=int, default=1,
               help="Number of gradient accumulation micro-steps per optimizer update.")

    
    # AlphaFoldLRScheduler 设计的参数
    p.add_argument("--lr_base", type=float, default=0.0, help="Base LR for warmup start.")
    p.add_argument("--lr_max", type=float, default=3e-4, help="Max LR (plateau).")
    # p.add_argument("--lr_warmup_steps", type=int, default=1000, help="Number of warmup steps.")
    # p.add_argument("--lr_start_decay_steps", type=int, default=50000, help="Step at which LR decay begins.")
    # p.add_argument("--lr_decay_every_steps", type=int, default=50000, help="Frequency of LR decay.")
        # ====== LR schedule by ratio（完全基于 max_iters 的比例）======
    p.add_argument("--lr_warmup_ratio", type=float, default=0.05,
                   help="Warmup steps as a fraction of max_iters, e.g., 0.05 = first 5% steps.")
    p.add_argument("--lr_start_decay_ratio", type=float, default=0.5,
                   help="When to start LR decay, as fraction of max_iters, e.g., 0.5 = after 50%.")
    p.add_argument("--lr_decay_every_ratio", type=float, default=0.1,
                   help="Decay interval as fraction of max_iters, e.g., 0.1 = every 10% of training.")
    
    p.add_argument("--lr_decay_factor", type=float, default=0.95, help="Multiplicative factor for LR decay.")

    p.add_argument("--max_iters",   type=int,   default=-1, help="total training iterations") # <=0 表示“自动计算”
    p.add_argument("--val_freq",    type=int,   default=1000, help="validation / checkpoint frequency")
    p.add_argument("--weight_decay",type=float, default=1e-4)
    p.add_argument("--grad_clip",   type=float, default=1.0)
    p.add_argument("--patience",    type=int,   default=0, help="early-stopping patience")
    p.add_argument("--num_workers", type=int,   default=2)
    p.add_argument("--save_interval", type=int, default=3000, help="last-checkpoint save interval")
    p.add_argument("--resume_checkpoint", type=str, default="", help="resume from checkpoint")
    p.add_argument("--use_ema", action="store_true", help="enable EMA")
    p.add_argument("--ema_decay", type=float, default=0.999, help="EMA decay rate")
    p.add_argument("--gpus",       type=int, nargs="+", required=True)
    p.add_argument("--local_rank", type=int, default=-1)
    p.add_argument('--mode', type=str, choices=['design', 'train', 'optimize', 'trajectory'], default='train')
    p.add_argument('--log_steps', type=int, default=5, help="log interval")
    
    # --- Energy plugin flags ---
    p.add_argument("--enable_energy", action="store_true", help="Enable the energy head plugin (disabled by default).")
    p.add_argument("--energy_start_step", type=int, default=10, help="Start training the energy head at or after this global step.")
    p.add_argument("--energy_lambda", type=float, default=1.0, help="Weighting factor (lambda) for the energy loss.")
    # p.add_argument("--energy_lr", type=float, default=1e-3, help="Learning rate for the energy head.")
    p.add_argument("--energy_weight_decay", type=float, default=1e-4, help="Weight decay for the energy head.")
    p.add_argument("--energy_log_every", type=int, default=5, help="Logging interval for energy-related metrics (in steps).")
    # --- 能量头独立配置 ---
    p.add_argument("--energy_config", type=str, default="", help="Path to a separate JSON config for the energy head.")
    # --- 能量头断点续训与EMA ---
    p.add_argument("--resume_energy_checkpoint", type=str, default="", help="Path to an energy checkpoint to resume the energy head, its optimizer, and EMA state.")
    p.add_argument("--energy_ema_decay", type=float, default=0.999, help="EMA decay for the energy head.")
    
    return p.parse_args()

    
def main():
    args = parse_args()

    # ---------- 1) 只初始化一次 DDP（用 env:// 最稳） ----------
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    is_ddp = world_size > 1

    if is_ddp:
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        local_rank = -1
        rank = 0

    args.local_rank = local_rank  # 统一使用这个
    # 注意：args.gpus 在 torchrun 场景下最好传 0..N-1（对应 CUDA_VISIBLE_DEVICES 内的序号）

    # ---------- 2) run_dir：仅 rank0 创建，然后 barrier ----------
    if args.resume_checkpoint:
        ckpt_dir = os.path.dirname(args.resume_checkpoint)
        run_dir = os.path.abspath(os.path.join(ckpt_dir, ".."))
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(args.output_dir, args.mode, ts)

    if local_rank in (-1, 0):
        os.makedirs(run_dir, exist_ok=True)
    if is_ddp:
        dist.barrier()

    # ---------- 3) 初始化 logger（此时 rank 已经正确） ----------
    file_logger = setup_root_logger(out_dir=run_dir, rank=local_rank)

    # energy logger：建议按 rank 分文件（你已经这么做了）
    energy_log_path = os.path.join(
        run_dir,
        ("energy.log" if local_rank in (-1, 0) else f"energy_rank{local_rank}.log")
    )
    energy_logger = logging.getLogger("energy")
    energy_logger.handlers.clear()
    energy_logger.setLevel(logging.INFO)
    energy_logger.propagate = False

    if local_rank in (-1, 0):
        energy_log_path = os.path.join(run_dir, "energy.log")
        eh = logging.FileHandler(energy_log_path, encoding="utf-8")
        eh.setFormatter(logging.Formatter(
            "%(asctime)s [R0] %(levelname)s %(message)s", "%Y-%m-%d %H:%M:%S"
        ))
        energy_logger.addHandler(eh)
    else:
        energy_logger.addHandler(logging.NullHandler())

    # （可选）如果你不想 rank*.log 里塞满 distributed barrier 的 INFO：
    logging.getLogger("torch.nn.parallel.distributed").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed.distributed_c10d").setLevel(logging.WARNING)

    # ---------- 4) 配置快照（保留你的逻辑） ----------
    try:
        import shutil, pathlib, json
        snap_dir = os.path.join(run_dir, "config_snapshot")
        os.makedirs(snap_dir, exist_ok=True)
        for _p in [args.model_config, args.model_features, args.energy_config]:
            if _p and os.path.isfile(_p):
                shutil.copy2(_p, os.path.join(snap_dir, pathlib.Path(_p).name))
        with open(os.path.join(run_dir, "cfg_runtime.json"), "w", encoding="utf-8") as fw:
            json.dump(vars(args), fw, indent=2, ensure_ascii=False)
    except Exception as _e:
        logging.warning(f"[ConfigSnapshot] failed: {_e}")

    # ---------- 5) 组装 cfg 并开训 ----------
    cfg = vars(args)
    cfg["run_dir"] = run_dir
    cfg["rank"] = rank
    cfg["world_size"] = world_size
    cfg["energy_log_path"] = energy_log_path

    trainer = ABXTrainer(cfg, file_logger)
    trainer.train(device_ids=args.gpus, local_rank=local_rank)

    if is_ddp:
        dist.destroy_process_group()



if __name__ == "__main__":
    main()