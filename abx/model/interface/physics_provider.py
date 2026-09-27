import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
from torch_scatter import scatter

from .energy_head import InterfaceEnergy
from .guidance_utils import calculate_cdr_rmsd  



def build_energy_minibatch(batch: dict, ret_final: dict, device=None) -> dict:
    """
    只挑能量头实际会用到的字段；零拷贝（仅 .detach()），不做任何几何重建。
    """
    def _get(d, path, default=None):
        cur = d
        for k in path:
            if not isinstance(cur, dict) or (k not in cur):
                return default
            cur = cur[k]
        return cur

    minibatch = {}
    miniret = {}
    # ---- 直接来自 batch 的键 ----
    for k in [
        "residx", "mask", "anchor_flag", "fixed_mask", "chain_id", "cdr_def",
        "seq", "is_recycling", "rigids_0", "rigids_t",
        "seq_t",
        "atom14_gt_positions", "atom14_alt_gt_positions",
        "torsion_angles_sin_cos", "alt_torsion_angles_sin_cos",
        "atom14_gt_exists", "atom14_alt_gt_exists",
    ]:
        v = batch.get(k, None)
        if v is not None and hasattr(v, "detach"):
            v = v.detach()
        minibatch[k] = v

    # ---- 来自 ret_final 的键 ----
    miniret["x0_pred"]                 = _get(ret_final, ["heads","folding","rigids"], None)
    miniret["final_atom14_positions"]  = _get(ret_final, ["heads","folding","final_atom14_positions"], None)
    miniret["angles_sin_cos"]     = _get(ret_final, ["heads","folding","sidechains","angles_sin_cos"], None)
    miniret["atom14_atom_exists"]      = _get(ret_final, ["heads","folding","atom14_atom_exists"], None)
    miniret["seq_0"]              = _get(ret_final, ["heads","sequence_module","seq_0"], None)

    for k in ["x0_pred","final_atom14_positions","angles_sin_cos","atom14_atom_exists","seq_0"]:
        v = miniret[k]
        if v is not None and hasattr(v, "detach"):
            miniret[k] = v.detach()

    # ---- 统一设备（可选）----
    if device is not None:
        for k, v in minibatch.items():
            if hasattr(v, "to") and v.device != device:
                minibatch[k] = v.to(device, non_blocking=True)

        for k, v in miniret.items():
            if hasattr(v, "to") and v.device != device:
                miniret[k] = v.to(device, non_blocking=True)

    return minibatch, miniret


class EnergyTrainer(nn.Module):
    """
    我们的“能量判官”训练器。
    - 在线自监督：不用离线缓存
    - 损失 = ranking物理约束 + RMSD二元监督(BCE)
    """

    def __init__(
        self,
        energy_model: InterfaceEnergy,
        margin_pred: float = 0.5,
        margin_noise: float = 2.0,
        lambda_rank: float = 0.5,
        lambda_bce: float = 1.0,
        bce_temp: float = 10.0,
        rmsd_cutoff: float = 2.0,
        sharpness: float = 2.0,
    ):
        super().__init__()
        self.energy_model = energy_model
        # self.opt = torch.optim.Adam(self.energy_model.parameters(), lr=lr)

        # ranking 超参
        self.margin_pred = margin_pred     # 惩罚 pred 比 gt 还低能量
        self.margin_noise = margin_noise   # 惩罚 noisy 跟 gt 太接近

        self.bce_temp = bce_temp  # BCE 温度缩放参数
        # 损失加权
        self.lambda_rank = lambda_rank
        self.lambda_bce = lambda_bce

        # 标签的阈值
        self.rmsd_cutoff = rmsd_cutoff
        self.sharpness = sharpness
        
        # 不再直接用 float，而是可学习 log_temp
        self.log_bce_temp = nn.Parameter(torch.log(torch.tensor(float(bce_temp))))
        


    def _safe_pick(self, d, path, default=None):
        cur = d
        for k in path:
            if isinstance(cur, (list, tuple)):
                try:
                    cur = cur[k]
                except Exception:
                    return default
            elif isinstance(cur, dict):
                if k not in cur: 
                    return default
                cur = cur[k]
            else:
                return default
        return cur

    def _gather_triplet(self, batch, ret_final):
        # 1) rigids
        rigids_gt    = batch['rigids_0'].detach()
        rigids_noisy = batch['rigids_t'].detach()
        rigids_pred  = self._safe_pick(ret_final, ['x0_pred'])
        rigids_pred  = rigids_pred.detach()

        # 2) seq
        seq_gt    = batch.get('seq',    None)
        seq_t     = batch.get('seq_t',  None)
        seq_pred  = self._safe_pick(ret_final, ['seq_0'])
        # 兼容设备 + detach
        if seq_gt   is not None: seq_gt   = seq_gt.detach()
        if seq_t    is not None: seq_t    = seq_t.detach()
        if seq_pred is not None: seq_pred = seq_pred.detach()

        # 3) atom14 pos
        atom14_pos_gt    = batch.get('atom14_gt_positions', None)
        atom14_pos_noisy = batch.get('atom14_alt_gt_positions', None)
        atom14_pos_pred  = self._safe_pick(ret_final, ['final_atom14_positions'])
        if atom14_pos_gt    is not None: atom14_pos_gt    = atom14_pos_gt.detach()
        if atom14_pos_noisy is not None: atom14_pos_noisy = atom14_pos_noisy.detach()
        if atom14_pos_pred  is not None: atom14_pos_pred  = atom14_pos_pred.detach()

        # 4) torsion angles (sin,cos)
        angles_gt    = batch.get('torsion_angles_sin_cos', None)
        angles_noisy = batch.get('alt_torsion_angles_sin_cos', None)
        angles_pred  = self._safe_pick(ret_final, ['angles_sin_cos'])
        if angles_gt    is not None: angles_gt    = angles_gt.detach()
        if angles_noisy is not None: angles_noisy = angles_noisy.detach()
        if angles_pred  is not None: angles_pred  = angles_pred.detach()

        # 5) atom14 exists
        atom14_exists_gt    = batch.get('atom14_gt_exists', None)
        atom14_exists_noisy = batch.get('atom14_alt_gt_exists', None)
        atom14_exists_pred  = self._safe_pick(ret_final, ['atom14_atom_exists'])
        if atom14_exists_gt    is not None: atom14_exists_gt    = atom14_exists_gt.detach()
        if atom14_exists_noisy is not None: atom14_exists_noisy = atom14_exists_noisy.detach()
        if atom14_exists_pred  is not None: atom14_exists_pred  = atom14_exists_pred.detach()

        # —— 兜底策略：任何 *pred* 缺失，回退到 *gt*，保证 eval 不会崩，但会显著降低可分性 —— 
        if seq_pred           is None: seq_pred           = seq_gt
        if atom14_pos_pred    is None: atom14_pos_pred    = atom14_pos_gt
        if angles_pred        is None: angles_pred        = angles_gt
        if atom14_exists_pred is None: atom14_exists_pred = atom14_exists_gt
        if seq_t              is None: seq_t              = seq_gt  # noisy 序列缺失，回退
        
        return dict(
            seq_gt=seq_gt, seq_pred=seq_pred, seq_noisy=seq_t,
            rigids_gt=rigids_gt, rigids_pred=rigids_pred, rigids_noisy=rigids_noisy,
            atom14_pos_gt=atom14_pos_gt, atom14_pos_pred=atom14_pos_pred, atom14_pos_noisy=atom14_pos_noisy,
            angles_gt=angles_gt, angles_pred=angles_pred, angles_noisy=angles_noisy,
            atom14_exists_gt=atom14_exists_gt, atom14_exists_pred=atom14_exists_pred, atom14_exists_noisy=atom14_exists_noisy,
        )


    def soft_label_from_rmsd(self, r, center: float, sharpness: float):
        """
        r: [B]，CDR RMSD
        center: 阈值中心，比如 2.0 或 3.0
        sharpness: 越大，越接近硬阈值；越小，越平滑
        返回 ∈ (0,1) 的连连续标签
        """
        return torch.sigmoid((center - r) * sharpness)

    def _forward_and_losses(self, batch, ret_final):
        tri = self._gather_triplet(batch, ret_final)

        # 计算能量
        E_gt = self._compute_energy(
            seq=tri['seq_gt'], rigids7=tri['rigids_gt'],
            atom14_positions=tri['atom14_pos_gt'],
            angles_sin_cos=tri['angles_gt'],
            atom14_exists=tri['atom14_exists_gt'],
            batch=batch,
        )
        E_pred = self._compute_energy(
            seq=tri['seq_pred'], rigids7=tri['rigids_pred'],
            atom14_positions=tri['atom14_pos_pred'],
            angles_sin_cos=tri['angles_pred'],
            atom14_exists=tri['atom14_exists_pred'],
            batch=batch,
        )
        E_noisy = self._compute_energy(
            seq=tri['seq_noisy'], rigids7=tri['rigids_noisy'],
            atom14_positions=tri['atom14_pos_noisy'],
            angles_sin_cos=tri['angles_noisy'],
            atom14_exists=tri['atom14_exists_noisy'],
            batch=batch,
        )
        
        # -------------------------------------------------
        # 2) 对每条样本做“组内零均值”：(gt, pred, noisy) 看作一个组
        #  ，因为你一条样本本身就是一个 complex 的三种构象
        # -------------------------------------------------
        # E_all: [B, 3]，每行是 (E_gt, E_pred, E_noisy)
        E_all = torch.stack([E_gt, E_pred, E_noisy], dim=1)  # [B, 3]

        # 对每一行求均值，再减掉：E_centered 同样是 [B, 3]
        E_mean = E_all.mean(dim=1, keepdim=True)             # [B, 1]
        E_centered = E_all - E_mean                          # [B, 3]

        # 拆回三种构象的“去均值能量”
        E_gt_c, E_pred_c, E_noisy_c = E_centered.unbind(dim=1)  # 三个都是 [B]
        
        
        # Ranking
        def margin_rank_loss(bad, good, margin):
            return F.relu(good - bad + margin).mean()

        loss_rank_pred  = margin_rank_loss(E_pred_c,  E_gt_c,  self.margin_pred)
        loss_rank_noise = margin_rank_loss(E_noisy_c, E_gt_c,  self.margin_noise)
        loss_rank_pred_noisy = margin_rank_loss(E_noisy_c, E_pred_c, self.margin_pred)
        loss_rank = (loss_rank_pred + loss_rank_noise + loss_rank_pred_noisy) / 3.0

        # BCE (基于 CDR RMSD)
        rmsd_pred  = calculate_cdr_rmsd(tri['rigids_pred'],  tri['rigids_gt'], batch)
        rmsd_noisy = calculate_cdr_rmsd(tri['rigids_noisy'], tri['rigids_gt'], batch)
        
        # y_pred   = (rmsd_pred  < self.rmsd_cutoff).float()
        # y_noisy  = (rmsd_noisy < self.rmsd_cutoff).float()
        # y_gt     = torch.ones_like(y_pred)
        
        y_gt    = torch.ones_like(rmsd_pred)
        # gt 结构的 RMSD≈0，label≈1

        y_pred  = self.soft_label_from_rmsd(rmsd_pred,  center=self.rmsd_cutoff, sharpness=self.sharpness)
        y_noisy = self.soft_label_from_rmsd(rmsd_noisy, center=self.rmsd_cutoff, sharpness=self.sharpness)

        
        # 学习到的温度（限制在 [1, 50] 防止爆炸）
        bce_temp = self.log_bce_temp.exp().clamp(1.0, 50.0)
        
        logit_gt    = -E_gt_c    / bce_temp
        logit_pred  = -E_pred_c  / bce_temp
        logit_noisy = -E_noisy_c / bce_temp
        
        # 以 rmsd_cutoff 为中心，越接近 cutoff，权重越大
        # 例如：w = 1 + exp(-|RMSD - cutoff|)
        with torch.no_grad():
            w_pred  = torch.exp(- (rmsd_pred  - self.rmsd_cutoff).abs())
            w_noisy = torch.exp(- (rmsd_noisy - self.rmsd_cutoff).abs())
            # 归一化到均值 1 左右，防止整体 loss 变大
            w_pred  = w_pred  / (w_pred.mean()  + 1e-8)
            w_noisy = w_noisy / (w_noisy.mean() + 1e-8)

        bce_gt   = F.binary_cross_entropy_with_logits(logit_gt,   y_gt,   reduction="mean")
        bce_pred = (w_pred  * F.binary_cross_entropy_with_logits(logit_pred,  y_pred, reduction="none")).mean()
        bce_noisy= (w_noisy * F.binary_cross_entropy_with_logits(logit_noisy, y_noisy,reduction="none")).mean()
        loss_bce = (bce_gt + bce_pred + bce_noisy) / 3.0

        temp_reg = 1e-4 * (self.log_bce_temp ** 2)
        loss_total = self.lambda_rank * loss_rank + self.lambda_bce * loss_bce + temp_reg

        logs = {
            "loss":      loss_total,
            "rank":       loss_rank,
            "r_pred":  loss_rank_pred,
            "r_noise": loss_rank_noise,
            "bce":        loss_bce,
            "Eg":       E_gt.mean(),
            "Ep":     E_pred.mean(),
            "En":     E_noisy.mean(),
            "rmsd_p":  rmsd_pred.mean(),
            "rmsd_n": rmsd_noisy.mean(),
        }
        return logs


    def _compute_energy(
        self,
        seq: torch.Tensor,                     # [B, L, d_seq] 
        rigids7: torch.Tensor,                 # [B, L, 7]
        atom14_positions: torch.Tensor,        # [B, L, 14, 3]
        angles_sin_cos: torch.Tensor,          # [B, L, ..., 2] 
        atom14_exists: torch.Tensor,           # [B, L, 14]
        batch: Dict,
    ) -> torch.Tensor:
        """
        用 InterfaceEnergy 计算能量（不要求力）。
        """
        # 1) 原始能量
        energy_raw, _ = self.energy_model(
            seq=seq,
            rigids7=rigids7,
            atom14_positions=atom14_positions,
            angles_sin_cos=angles_sin_cos,
            atom14_exists=atom14_exists,
            batch=batch,
            return_force=False,
        )
        energy_raw = torch.nan_to_num(energy_raw, nan=0.0, posinf=1e6, neginf=-1e6)
        # e = 1000.0 * torch.tanh(energy_raw / 1000.0)  # [B]
        return energy_raw

    def train_step(self, batch, ret_final) -> Dict[str, torch.Tensor]:
        """
        训练期：与主干同步反传。注意：外部已对 ret_final 做 detach。
        这里仍仅对 energy_head 反向。
        """
        # 确保进入训练态（DDP 包裹时 .train() 调的是 wrapper）
        self.energy_model.train()
        minibatch, miniret = build_energy_minibatch(batch, ret_final, device=batch["rigids_0"].device)

        # 直接返回带图的张量；外层需要反传
        return self._forward_and_losses(minibatch, miniret)

    
    @torch.no_grad()
    def eval_step(self, batch, ret_final) -> Dict[str, float]:
        """验证期：与训练同口径，返回 float。"""
        was_training = self.energy_model.training
        self.energy_model.eval()

        minibatch, miniret = build_energy_minibatch(batch, ret_final, device=batch["rigids_0"].device)
        out = self._forward_and_losses(minibatch, miniret)
        logs = {k: (float(v.item()) if torch.is_tensor(v) else float(v)) for k, v in out.items()}

        if was_training:
            self.energy_model.train()
        return logs