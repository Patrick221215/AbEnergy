import os
import argparse
import logging
from logging.handlers import QueueHandler, QueueListener
import functools
import json
import copy
import time
import csv
import math
import random
import hashlib
from datetime import datetime


import ml_collections
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm, trange

from abx.data import dataset
from abx.data.utils import save_pdb
from abx.common import residue_constants
from abx.common.utils import index_to_str_seq
from abx.model.abx import ScoreNetwork, get_prev
from abx.model.features import FeatureBuilder
from diffuser.full_diffuser import FullDiffuser


# ============================================================
# Seed / reproducibility helpers
# ============================================================

def parse_seed_args(seed, seed_list):
    """
    解析 --seed / --seed_list。

    约束：
    1. 不能同时指定 --seed 和 --seed_list，避免输出目录和随机性语义混乱。
    2. 不指定时返回 [None]，保持原 inference-abconf.py 的非受控随机行为。
    """
    if seed is not None and seed_list is not None:
        raise ValueError("Please specify only one of --seed or --seed_list, not both.")

    if seed_list is not None:
        raw = [x.strip() for x in seed_list.split(',') if x.strip()]
        if len(raw) == 0:
            raise ValueError("--seed_list is empty after parsing.")
        return [int(x) for x in raw]

    if seed is not None:
        return [int(seed)]

    return [None]


def seed_to_tag(seed):
    return f"seed-{seed}" if seed is not None else "seed-uncontrolled"


def stable_int_hash(*parts, mod=2**31 - 1):
    """
    用稳定哈希把 base_seed/sample_idx/batch_name 等组合成 PyTorch 可用的 int seed。
    不使用 Python 内置 hash，因为 Python hash 默认受 PYTHONHASHSEED 影响，不适合作为可复现实验种子。
    """
    s = "||".join([str(x) for x in parts])
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:16], 16) % mod


def derive_local_seed(base_seed, mode, sample_idx, batch_names, phase_tag=""):
    """
    从一个实验级 base_seed 派生出当前 batch 的局部 seed。

    为什么不直接每次 set_global_seed(base_seed + k)：
    多卡 DistributedDataset 下不同 rank 看到的 batch 不同；如果只用 sample_idx，
    不同 PDB 可能拿到相同随机流。加入 batch_names 后，单卡/多卡/重跑时更稳定。
    """
    if base_seed is None:
        return None
    if batch_names is None:
        batch_names = []
    if isinstance(batch_names, str):
        batch_names = [batch_names]
    return stable_int_hash(base_seed, mode, sample_idx, phase_tag, *batch_names)


def set_global_seed(seed, deterministic=False):
    """同时设置 random / numpy / torch / cuda 的随机种子。"""
    if seed is None:
        return

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    else:
        # 推理时通常不需要 benchmark；关闭它可以减少不同输入尺寸下的算法选择漂移。
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def use_distributed(args):
    return args.device == 'gpu' and len(args.gpu_list) > 1 and torch.cuda.is_available()


def forward_model(model, batch):
    """
    兼容两类模型 forward：
    1. 新版：model(batch, global_step=0)
    2. 旧版：model(batch)
    """
    try:
        return model(batch, global_step=0)
    except TypeError:
        return model(batch)


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



def log_setup(args):
    os.makedirs(args.run_dir, exist_ok=True)

    log_path = os.path.join(
        args.run_dir,
        f'{os.path.splitext(os.path.basename(__file__))[0]}.log'
    )

    handlers = [logging.FileHandler(log_path, encoding="utf-8")]

    def handler_apply(h, f, *arg):
        f(*arg)
        return h

    level = logging.DEBUG if args.verbose else logging.INFO
    handlers = [handler_apply(h, h.setLevel, level) for h in handlers]

    fmt = '%(asctime)-15s [%(levelname)s] (%(process)d-%(filename)s:%(lineno)d) %(message)s'
    handlers = [handler_apply(h, h.setFormatter, logging.Formatter(fmt)) for h in handlers]

    # 这行可以留，也可以删；留着也不会再有控制台输出，因为 handlers 里没有 StreamHandler
    logging.basicConfig(format=fmt, level=level, handlers=handlers, force=True)

    log_queue = mp.Queue(-1)
    return log_queue, handlers


# --- 其他函数 (worker_setup 到 sample_fn 前) 保持不变 ---
def worker_setup(rank, log_queue, args):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.addHandler(QueueHandler(log_queue))
    level = logging.DEBUG if args.verbose else logging.INFO
    logger.setLevel(level)

    if use_distributed(args):
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = str(args.master_port)
        world_size = len(args.gpu_list)
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)


def worker_cleanup(args):
    if use_distributed(args) and dist.is_initialized():
        dist.destroy_process_group()


def worker_device(rank, args):
    if args.device == 'gpu':
        return torch.device(f'cuda:{args.gpu_list[rank]}')
    return torch.device('cpu')


# The `worker_load` function remains based on inference-abconf.py:
# strict checkpoint['model'] loading is preserved because this project checkpoint format is already known.
def worker_load(rank, args):
    device = worker_device(rank, args)
    gpu_index = args.gpu_list[rank] if args.device == 'gpu' else None
    if args.device == 'gpu':
        torch.cuda.set_device(gpu_index)

    with open(args.model_config, 'r', encoding='utf-8') as f:
       config = json.loads(f.read())
       diff_feat = config['diffuser']
       config = ml_collections.ConfigDict(config)
    model_conf, diff_conf = config.model, config.diffuser
    diff_conf.so3.use_cached_score = True
    diffuser = FullDiffuser.get(diff_conf)
    
    logging.info(f"Loading checkpoint from: {args.model}")
    checkpoint = torch.load(args.model, map_location='cpu')
    
    model = ScoreNetwork(model_conf=model_conf, diffuser=diffuser)

    model.load_state_dict(checkpoint['model'], strict=True)
    logging.info("Loaded standard model weights as a base.")
    logging.info("[EMA-Inference] --use_ema was already applied during training.")

    with open(args.model_features, 'r', encoding='utf-8') as f:
        feats = json.loads(f.read())
        optimize_steps = None
        for i in range(len(feats)):
            feat_name, feat_args = feats[i]
            if 'device' in feat_args and feat_args['device'] == '%(device)s':
                feat_args['device'] = device
            if 'diffuse' in feat_name:
                feat_args.update({'diff_conf': diff_feat})
                if 'optimize_steps' in feat_args:
                    optimize_steps = feat_args.pop('optimize_steps')
                    
    model = model.to(device=device).eval()
    

    if use_distributed(args):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu_index])

    if args.mode in ['design', 'trajectory']:
        return feats, model, diffuser, config
    return feats, model, diffuser, config, optimize_steps

def postprocess_one(name, str_heavy_seq, str_light_seq, coord, args, pLDDT, antigen_data, time=None):
    pdb_filename = f'{name}@{time:.4f}.pdb' if time else f'{name}.pdb'
    pdb_file = os.path.join(args.output_dir, pdb_filename)
    heavy_chain, light_chain = name.split('_')[1], name.split('_')[2]
    save_pdb(str_heavy_seq, heavy_chain, str_light_seq, light_chain, coord, pdb_file, pLDDT, antigen_data)

def _extract_pred_energy(model_out: dict, batch_names) -> list:
    """
    model_out['heads']['interface_energy'] = {
        'E0_gt': E0,
        'pred': E_pred,
        'iface_logit': logit_pred,
        'rmsd': rmsd_pred,
        'forces': forces
    }
    这里只取 pred（E_pred），返回 python float list，长度=B
    """
    # import ipdb; ipdb.set_trace()
    aux = model_out["heads"]["interface_energy"]
    if (not isinstance(aux, dict)) or ("pred" not in aux):
        raise KeyError(f"interface_energy bad format: type={type(aux)}, keys={list(aux.keys()) if isinstance(aux, dict) else None}")

    E = aux["pred"]
    if not torch.is_tensor(E):
        raise TypeError(f"interface_energy['pred'] is not a tensor: {type(E)}")

    B = len(batch_names)

    # 统一成 [B]
    if E.ndim == 0:
        E = E.repeat(B)
    elif E.shape[0] != B:
        # 兼容奇怪形状：能 reshape 就 reshape，不行就聚合
        if E.numel() == B:
            E = E.view(B)
        else:
            E = E.view(B, -1).mean(dim=1)
    elif E.ndim > 1:
        E = E.view(B, -1).mean(dim=1)

    return E.detach().float().cpu().tolist()

def _append_energy_csv(csv_path: str, rows: list, sample_idx: int) -> None:
    """
    rows: list[(pdb_filename, energy_float)]
    sample_idx: 当前是第几次生成（例如 k+1）
    CSV 三列：pdb / success_idx(=sample_idx) / energy
    """
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    write_header = not os.path.exists(csv_path)

    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if write_header:
            w.writerow(["pdb", "success_idx", "energy"])
        for pdb_name, e in rows:
            if e is None or (isinstance(e, float) and (math.isnan(e) or math.isinf(e))):
                continue
            w.writerow([pdb_name, sample_idx, float(e)])


def postprocess_trajectory(batch, traj, args):
    for data_point in traj:
        pLDDT, seq, coords = data_point['pLDDT'], data_point['seq'], data_point['atom14_results']
        time_val = data_point.get('time') if len(traj) > 1 else None
        for i in range(len(batch['name'])):
            name = batch['name'][i]
            str_heavy_seq, str_light_seq = batch['str_heavy_seq'][i], batch['str_light_seq'][i]
            h_len, l_len = len(str_heavy_seq), len(str_light_seq)
            
            # --- 修改开始：复原绝对坐标系 ---
            # 1. 取出当前样本的平移中心，并 reshape 为 [1, 1, 3] 以便后续与 [L, 14, 3] 坐标张量相加
            bb_center = batch['bb_center'][i].cpu().numpy().reshape(1, 1, 3)
            
            # 2. 获取抗原截断后的相对坐标，加上 bb_center 还原真实的晶体绝对坐标
            ag_origin_coords = batch['antigen_origin_atom14_gt_positions'][i]
            if torch.is_tensor(ag_origin_coords):
                ag_origin_coords = ag_origin_coords.cpu().numpy()
            ag_abs_coords = ag_origin_coords + bb_center
            
            antigen_data = {
                'antigen_str_seq': batch['antigen_origin_str_seq'][i],
                'antigen_coords': ag_abs_coords,  # <--- 使用还原后的抗原绝对坐标
                'antigen_coord_mask': batch['antigen_origin_atom14_gt_exists'][i],
                'antigen_chain_ids': batch['antigen_origin_chain_ids'][i],
                'antigen_chains': list(name.split('_'))[-1]
            }
            
            str_heavy_seq_ = index_to_str_seq(seq[i, :h_len])
            str_light_seq_ = index_to_str_seq(seq[i, h_len:h_len+l_len])
            
            # # 3. 获取模型生成的抗体相对坐标，同样加上 bb_center 还原绝对坐标
            # coord_to_save = coords[i, :h_len+l_len] + bb_center  # <--- 使用还原后的抗体绝对坐标
            
            
            # ==========================================
            # 3. 抗体：获取预测相对坐标 -> 加回中心点 -> 动态清理幽灵原子
            # ==========================================
            pred_coords_np = coords[i, :h_len+l_len]
            if torch.is_tensor(pred_coords_np):
                pred_coords_np = pred_coords_np.cpu().numpy()
                
            coord_to_save = pred_coords_np + bb_center
            
            # 🌟 关键补丁：动态判断模型输出中原本为 [0,0,0] 的原子，把它们重置为 0
            ab_valid_mask = (np.abs(pred_coords_np).sum(axis=-1) > 1e-5)
            coord_to_save = coord_to_save * ab_valid_mask[..., None]
            # --- 修改结束 ---

            postprocess_one(name, str_heavy_seq_, str_light_seq_, coord_to_save, args, pLDDT[i], antigen_data, time_val)

            
def _set_t_feats(feats, diffuser, t, t_placeholder):
    feats['t'] = t * t_placeholder
    rot_score_scaling, trans_score_scaling = diffuser.score_scaling(feats['t'])
    feats['rot_score_scaling'], feats['trans_score_scaling'] = rot_score_scaling * t_placeholder, trans_score_scaling * t_placeholder
    return feats

def _self_conditioning(batch, model, config):
    # The model now requires `global_step`. For inference, we can pass a dummy value like 0.
    model_sc = forward_model(model, batch)
    batch.update(get_prev(batch, model_sc, config.model))
    return batch

def sample_fn(data_init, config, diffuser, model, args, num_t=100, min_t=0.01, center=True, self_condition=True, noise_scale=1.0, eps=1e-8):
    
    score_network_conf = config.model.heads.diffusion_module
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data_init.items()}
    diffuse_mask = (1 - batch['fixed_mask']) * batch['atom14_gt_exists'][..., 0]
    antibody_len = batch['anchor_flag'].shape[1]
    t_placeholder = torch.ones(batch['rigids_t'].shape[0], device=device, dtype=torch.float32)
    reverse_steps = np.linspace(min_t, 1.0, num_t)[::-1]
    dt = torch.tensor(1/num_t, device=device)
    opt_step = batch['t'][0].cpu().numpy()
    
    #if args.mode == 'optimize' and (opt_step := batch['t'][0].cpu().numpy()) < 1.0:
    if args.mode == 'optimize' and opt_step < 1.0:
        reverse_steps = reverse_steps[reverse_steps <= opt_step + eps]
    with torch.no_grad():
        traj = []
        if score_network_conf.embed.embed_self_conditioning and self_condition and len(reverse_steps) > 0:
            batch = _set_t_feats(batch, diffuser, reverse_steps[0], t_placeholder)
            batch = _self_conditioning(batch, model, config)
        is_main_process = not dist.is_initialized() or dist.get_rank() == 0
        sde_pbar = tqdm(reverse_steps, desc="Reverse SDE", leave=False, ncols=100, disable=not is_main_process)
        for t in sde_pbar:
            if t > min_t:
                t_ = torch.full((batch['rigids_t'].shape[0],), t, device=device)
                batch = _set_t_feats(batch, diffuser, t_, t_placeholder)
                # The model now requires `global_step`. For inference, we can pass a dummy value like 0.
                model_out = forward_model(model, batch)
                if score_network_conf.embed.embed_self_conditioning:
                    batch.update(get_prev(batch, model_out, config.model))
                
                rigids_t, seq_t = diffuser.reverse(
                    rigid_t=batch['rigids_t'], seq_t=batch['seq_t'],
                    rot_score=model_out['heads']['folding']['rot_score'],
                    trans_score=model_out['heads']['folding']['trans_score'],
                    logits_t=model_out['heads']['sequence_module']['logits'],
                    diffuse_mask=diffuse_mask, t=t_, dt=dt, center=center, noise_scale=noise_scale
                )
            else:
                # The model now requires `global_step`. For inference, we can pass a dummy value like 0.
                model_out = forward_model(model, batch)
                rigids_t, seq_t = model_out['heads']['folding']['rigids'], model_out['heads']['sequence_module']['seq_0']
            batch.update({'rigids_t': rigids_t, 'seq_t': seq_t})
            pLDDT = model_out['heads']['predicted_lddt']['pLDDT']
            pLDDT_item = torch.sum(pLDDT * diffuse_mask, dim=1) / torch.sum(diffuse_mask, dim=1)
            traj.append({
                'seq': torch.clamp(seq_t[:,:antibody_len], 0, 19).long().cpu().numpy(),
                'atom14_results': model_out['heads']['folding']['final_atom14_positions'][:,:antibody_len].cpu().numpy(),
                'pLDDT': torch.tile(pLDDT_item[:, None], (1, antibody_len)).cpu().numpy(),
                'time': t
            })
        if args.mode != 'trajectory':
            traj = [traj[-1]]
        postprocess_trajectory(data_init, traj, args)
        
        # ===== 新增：返回最后一步能量，用于外层写 CSV =====
        rows = []
        # import ipdb; ipdb.set_trace()
        try:
            names = batch["name"]  # list[str], length=B
            energy_list = _extract_pred_energy(model_out, names)  # pred energy
            for name, e in zip(names, energy_list):
                # 你们 design/optimize 模式下 postprocess 会保存成 name.pdb（因为 traj 长度=1，time_val=None）
                pdb_name = f"{name}.pdb"
                rows.append((pdb_name, e))
        except Exception as ex:
            logging.error(f"[EnergyCSV] extract energy failed: {ex}", exc_info=True)

        return rows
    


def _build_dataset_loader(args, rank, name_idx, feats):
    """保持 inference-abconf.py 的 Dataset/FeatureBuilder/DataLoader 逻辑。"""
    base_test_dataset = dataset.IgStructureDataset(
        data_dir=args.data_dir,
        name_idx=name_idx,
        is_training=False,
    )
    final_dataset = (
        dataset.DistributedDataset(dataset=base_test_dataset, rank=rank, word_size=len(args.gpu_list))
        if use_distributed(args) else base_test_dataset
    )
    feat_builder = FeatureBuilder(feats, is_training=False)
    collate_fn = functools.partial(final_dataset.collate_fn, feat_builder=feat_builder)
    return DataLoader(
        final_dataset,
        batch_size=args.batch_size,
        sampler=None,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
        pin_memory=False,
    )


def _write_energy_rows_distributed(rows, csv_path, sample_idx, is_main_process):
    """多卡时把各 rank 的 energy rows 汇总到 rank0；单卡时直接写。"""
    if dist.is_initialized():
        world = dist.get_world_size()
        gathered = [None for _ in range(world)] if is_main_process else None
        dist.gather_object(rows, gathered, dst=0)

        if is_main_process:
            all_rows = []
            for part in gathered:
                if part:
                    all_rows.extend(part)
            all_rows.sort(key=lambda x: x[0])
            _append_energy_csv(csv_path, all_rows, sample_idx=sample_idx)
    else:
        if is_main_process:
            rows.sort(key=lambda x: x[0])
            _append_energy_csv(csv_path, rows, sample_idx=sample_idx)


def inference(rank, log_queue, args):
    worker_setup(rank, log_queue, args)
    is_main_process = (rank == 0)

    # --- 1. 加载模型和配置：以 inference-abconf.py 为准 ---
    if args.mode == 'optimize':
        feats, model, diffuser, config, optimize_steps = worker_load(rank, args)
        if optimize_steps is None:
            raise ValueError("args.mode='optimize' but optimize_steps was not found in model_features diffuse feature args.")
    else:
        feats, model, diffuser, config = worker_load(rank, args)
    
    if is_main_process:
        logging.info('Feats: %s', [f[0] for f in feats])

    inference_step = config.diffuser.inference_step
    seed_values = args.seed_values
    multi_seed = len(seed_values) > 1
    num_samples = args.num_samples
    output_dir = args.run_dir

    with open(args.name_idx) as f:
        name_idx = [x.strip() for x in f if x.strip()]

    def inference_loop(
        current_args,
        data_loader,
        outer_pbar=None,
        records_acc=None,
        base_seed=None,
        sample_idx=None,
        phase_tag="",
    ):
        is_main_proc = not dist.is_initialized() or dist.get_rank() == 0

        # 重要：FeatureBuilder/collate_fn 里通常会构造扩散初始噪声。
        # 所以 seed 必须在 next(data_iter) 之前设置；否则只是在 batch 生成之后控 seed，
        # 不能真正控制初始 rigids_t / seq_t。
        data_iter = iter(data_loader)
        i = 0
        while True:
            collate_seed = derive_local_seed(
                base_seed=base_seed,
                mode=current_args.mode,
                sample_idx=sample_idx,
                batch_names=[f'rank-{rank}', f'batch-{i}'],
                phase_tag=f'{phase_tag}|collate',
            )
            set_global_seed(collate_seed, deterministic=current_args.deterministic)

            try:
                batch = next(data_iter)
            except StopIteration:
                break

            local_seed = derive_local_seed(
                base_seed=base_seed,
                mode=current_args.mode,
                sample_idx=sample_idx,
                batch_names=batch.get('name', []),
                phase_tag=phase_tag,
            )
            try:
                # 第二次设置 seed 控制 sample_fn 内 reverse SDE 的随机项；
                # 这里加入真实 batch_names，便于同一个 sample_idx 下不同 PDB 有不同随机流。
                set_global_seed(local_seed, deterministic=current_args.deterministic)

                if is_main_proc and outer_pbar:
                    postfix = {
                        'batch': i + 1,
                        'names': ','.join(batch.get('name', [])),
                    }
                    if local_seed is not None:
                        postfix['seed'] = local_seed
                    outer_pbar.set_postfix(**postfix)

                rows = sample_fn(batch, config, diffuser, model, current_args, num_t=inference_step)
                if records_acc is not None and rows:
                    records_acc.extend(rows)

                logging.info(
                    "sample=%s batch=%d names=%s base_seed=%s local_seed=%s out=%s",
                    str(sample_idx),
                    i + 1,
                    ','.join(batch.get('name', [])),
                    str(base_seed),
                    str(local_seed),
                    current_args.output_dir,
                )
            except Exception as e:
                logging.error(
                    "Failed to predict for batch %s | base_seed=%s | local_seed=%s | sample=%s | phase=%s: %s",
                    batch.get('name', 'N/A'),
                    str(base_seed),
                    str(local_seed),
                    str(sample_idx),
                    phase_tag,
                    str(e),
                    exc_info=True,
                )
            finally:
                i += 1

    if is_main_process:
        os.makedirs(output_dir, exist_ok=True)

    # --- 2. reference 只生成一次：仍放在 run_dir/reference，不按 seed 重复生成 ---
    if is_main_process:
        logging.info("Generating reference structures... (once for all seed groups/tasks)")
        ref_dir = os.path.join(output_dir, 'reference')
        os.makedirs(ref_dir, exist_ok=True)

        base_test_dataset = dataset.IgStructureDataset(data_dir=args.data_dir, name_idx=name_idx, is_training=False)
        feat_builder = FeatureBuilder(feats, is_training=False)
        collate_fn = functools.partial(base_test_dataset.collate_fn, feat_builder=feat_builder)
        ref_loader = DataLoader(base_test_dataset, batch_size=args.batch_size, collate_fn=collate_fn)

        for batch in tqdm(ref_loader, desc="Reference", disable=not is_main_process):
            args_copy = copy.deepcopy(args)
            args_copy.output_dir = ref_dir
            ref_data = {
                'atom14_results': batch['atom14_gt_positions'][:, :batch['anchor_flag'].shape[1]].cpu().numpy(),
                'seq': batch['seq'][:, :batch['anchor_flag'].shape[1]].cpu().numpy(),
                'pLDDT': np.full((len(batch['name']), batch['anchor_flag'].shape[1]), 100.0),
            }
            postprocess_trajectory(batch, [ref_data], args_copy)

    if use_distributed(args):
        dist.barrier()

    # --- 3. seed loop：新增功能；单 seed 时保持原目录层级，多 seed 时增加 seed-xxxx 子目录 ---
    for base_seed in seed_values:
        seed_tag = seed_to_tag(base_seed)
        seed_root = os.path.join(output_dir, seed_tag) if multi_seed else output_dir

        if is_main_process:
            os.makedirs(seed_root, exist_ok=True)
            logging.info("===== Running seed group: %s =====", seed_tag)

        if use_distributed(args):
            dist.barrier()

        if args.mode == 'optimize':
            if is_main_process:
                logging.info(f"Running in OPTIMIZE mode with steps: {optimize_steps}")
            
            for step in optimize_steps:
                if is_main_process:
                    logging.info("--------------------")
                    logging.info("Processing Optimize Step: %s under %s", str(step), seed_tag)
                
                current_feats = copy.deepcopy(feats)
                for i in range(len(current_feats)):
                    feat_name, feat_args = current_feats[i]
                    if 'diffuse' in feat_name:
                        feat_args['diff_conf'].update({'opt_step': step})

                test_loader = _build_dataset_loader(args, rank, name_idx, current_feats)

                opt_step_dir = os.path.join(seed_root, f'OPT-{step}')
                if is_main_process:
                    os.makedirs(opt_step_dir, exist_ok=True)
                energy_csv_path = os.path.join(opt_step_dir, "energy.csv")

                with trange(num_samples, desc=f"Sampling ({seed_tag}, step {step})", disable=not is_main_process, ncols=120) as p_sample:
                    for k in p_sample:
                        p_sample.set_description(f"Sampling [{k+1}/{num_samples}] ({seed_tag}, step {step})")
                        sample_dir = os.path.join(opt_step_dir, f'{k:04d}')
                        if is_main_process:
                            os.makedirs(sample_dir, exist_ok=True)
                        if use_distributed(args):
                            dist.barrier()
                        
                        args_copy = copy.deepcopy(args)
                        args_copy.output_dir = sample_dir

                        local_rows = []
                        inference_loop(
                            args_copy,
                            test_loader,
                            outer_pbar=p_sample,
                            records_acc=local_rows,
                            base_seed=base_seed,
                            sample_idx=k,
                            phase_tag=f"opt-step-{step}",
                        )
                        _write_energy_rows_distributed(local_rows, energy_csv_path, sample_idx=k, is_main_process=is_main_process)

        else:
            if is_main_process:
                logging.info("Running in %s mode under %s.", args.mode.upper(), seed_tag)

            test_loader = _build_dataset_loader(args, rank, name_idx, feats)
            energy_csv_path = os.path.join(seed_root, "energy.csv")

            with trange(num_samples, desc=f"Sampling ({seed_tag})", disable=not is_main_process, ncols=120) as p_sample:
                for k in p_sample:
                    p_sample.set_description(f"Sampling [{k+1}/{num_samples}] ({seed_tag})")
                    sample_dir = os.path.join(seed_root, f'{k:04d}')
                    if is_main_process:
                        os.makedirs(sample_dir, exist_ok=True)
                    if use_distributed(args):
                        dist.barrier()
                    
                    args_copy = copy.deepcopy(args)
                    args_copy.output_dir = sample_dir

                    local_rows = []
                    inference_loop(
                        args_copy,
                        test_loader,
                        outer_pbar=p_sample,
                        records_acc=local_rows,
                        base_seed=base_seed,
                        sample_idx=k,
                        phase_tag=args.mode,
                    )
                    _write_energy_rows_distributed(local_rows, energy_csv_path, sample_idx=k, is_main_process=is_main_process)

    worker_cleanup(args)
    
def main(args):
    args.seed_values = parse_seed_args(args.seed, args.seed_list)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    if len(args.seed_values) == 1:
        seed_tag = seed_to_tag(args.seed_values[0])
    else:
        seed_tag = f"seedlist-{len(args.seed_values)}"

    # 为了兼容原 inference-abconf.py：没有指定 seed/run_name 时仍使用纯 timestamp 目录。
    if args.run_name is not None:
        run_name = args.run_name
    elif args.seed is None and args.seed_list is None:
        run_name = timestamp
    else:
        run_name = f"{timestamp}_{seed_tag}_ns{args.num_samples}"

    run_dir = os.path.join(args.output_dir, args.mode, run_name)
    args.run_dir = run_dir
    os.makedirs(args.run_dir, exist_ok=True)

    manifest = {
        "run_dir": args.run_dir,
        "created_at": datetime.now().isoformat(),
        "mode": args.mode,
        "model": os.path.abspath(args.model),
        "model_features": os.path.abspath(args.model_features),
        "model_config": os.path.abspath(args.model_config),
        "name_idx": os.path.abspath(args.name_idx),
        "data_dir": os.path.abspath(args.data_dir),
        "batch_size": args.batch_size,
        "num_samples": args.num_samples,
        "use_ema": args.use_ema,
        "gpu_list": args.gpu_list,
        "device": args.device,
        "seed": args.seed,
        "seed_list": args.seed_list,
        "seed_values": args.seed_values,
        "deterministic": args.deterministic,
        "run_name": args.run_name,
        "master_addr": args.master_addr,
        "master_port": args.master_port,
    }
    save_json(manifest, os.path.join(args.run_dir, "manifest.json"))

    mp.set_start_method('spawn', force=True)
    log_queue, handlers = log_setup(args)
    listener = QueueListener(log_queue, *handlers, respect_handler_level=True)
    listener.start()
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(QueueHandler(log_queue))
    root_logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logging.info('-----------------')
    logging.info(f'All outputs for this run will be saved to: {args.run_dir}')
    logging.info('Initial Arguments: %s', args)
    logging.info('Parsed seed values: %s', args.seed_values)
    logging.info('-----------------')
    if use_distributed(args):
        mp.spawn(inference, args=(log_queue, args), nprocs=len(args.gpu_list), join=True)
    else:
        inference(0, log_queue, args)
    listener.stop()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_list', type=int, nargs='+', default=[0])
    parser.add_argument('--device', type=str, choices=['gpu', 'cpu'], default='gpu')
    parser.add_argument('--model', type=str, required=True, help="Path to the model checkpoint (.pt) file.")
    parser.add_argument('--model_features', type=str, required=True)
    parser.add_argument('--model_config', type=str, required=True)
    parser.add_argument('--name_idx', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--mode', type=str, choices=['design', 'optimize', 'trajectory'], default='design')
    parser.add_argument('--batch_size', type=int, default=1, help="Batch size PER GPU")
    parser.add_argument('--num_samples', type=int, default=100)
    
    parser.add_argument('--use_ema', action='store_true', 
                        help="If specified, use the EMA (Exponential Moving Average) weights from the checkpoint "
                             "for inference. Otherwise, the standard model weights are used.")

    # Reproducibility / multi-seed controls
    parser.add_argument('--seed', type=int, default=None, help='Run one controlled random seed.')
    parser.add_argument('--seed_list', type=str, default=None, help='Comma-separated seeds, e.g. 2024,2025,2026.')
    parser.add_argument('--deterministic', action='store_true', help='Enable deterministic torch algorithms when possible.')
    parser.add_argument('--run_name', type=str, default=None, help='Optional custom run directory name under output_dir/mode/.')

    # DDP convenience controls
    parser.add_argument('--master_addr', type=str, default='127.0.0.1')
    parser.add_argument('--master_port', type=int, default=29525)

    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()

    main(args)