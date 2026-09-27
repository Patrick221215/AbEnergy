# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import ml_collections
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from abx.common import residue_constants
from abx.model.rigid_utils import Rotation, Rigid
from abx.model.vector import Vec3Array, euclidean_distance
#from openfold.utils.geometry.vector import Vec3Array, euclidean_distance
from abx.model.utils import (
    tree_map,
    masked_mean,
    permute_final_dims,
    compute_backbone,
    torsion_angles_to_frames,
    frames_to_atom14_pos,
)
import logging
from abx.model.utils import tensor_tree_map

logger = logging.getLogger(__name__)



def get_rc_tensor(rc_np, seq):
    return torch.tensor(rc_np, device=seq.device)[seq]

def apply_mask(x_diff, x_fixed, diff_mask):
    return diff_mask * x_diff + (1 - diff_mask) * x_fixed
    
def softmax_cross_entropy(logits, labels):
    loss = -1 * torch.sum(
        labels * torch.nn.functional.log_softmax(logits, dim=-1),
        dim=-1,
    )
    return loss


def sigmoid_cross_entropy(logits, labels):
    logits_dtype = logits.dtype
    logits = logits.double()
    labels = labels.double()
    log_p = torch.nn.functional.logsigmoid(logits)
    # log_p = torch.log(torch.sigmoid(logits))
    log_not_p = torch.nn.functional.logsigmoid(-1 * logits)
    # log_not_p = torch.log(torch.sigmoid(-logits))
    loss = (-1. * labels) * log_p - (1. - labels) * log_not_p
    loss = loss.to(dtype=logits_dtype)
    return loss


def torsion_angle_loss(
    a,  # [*, N, 7, 2]
    a_gt,  # [*, N, 7, 2]
    a_alt_gt,  # [*, N, 7, 2]
):
    # [*, N, 7]
    norm = torch.norm(a, dim=-1)

    # [*, N, 7, 2]
    a = a / norm.unsqueeze(-1)

    # [*, N, 7]
    diff_norm_gt = torch.norm(a - a_gt, dim=-1)
    diff_norm_alt_gt = torch.norm(a - a_alt_gt, dim=-1)
    min_diff = torch.minimum(diff_norm_gt ** 2, diff_norm_alt_gt ** 2)

    # [*]
    l_torsion = torch.mean(min_diff, dim=(-1, -2))
    l_angle_norm = torch.mean(torch.abs(norm - 1), dim=(-1, -2))

    an_weight = 0.02
    return l_torsion + an_weight * l_angle_norm


def compute_fape(
    pred_frames: Rigid,
    target_frames: Rigid,
    frames_mask: torch.Tensor,
    pred_positions: torch.Tensor,
    target_positions: torch.Tensor,
    positions_mask: torch.Tensor,
    length_scale: float,
    pair_mask: Optional[torch.Tensor] = None,
    l1_clamp_distance: Optional[float] = None,
    eps=1e-8,
) -> torch.Tensor:
    """
        Computes FAPE loss.

        Args:
            pred_frames:
                [*, N_frames] Rigid object of predicted frames
            target_frames:
                [*, N_frames] Rigid object of ground truth frames
            frames_mask:
                [*, N_frames] binary mask for the frames
            pred_positions:
                [*, N_pts, 3] predicted atom positions
            target_positions:
                [*, N_pts, 3] ground truth positions
            positions_mask:
                [*, N_pts] positions mask
            length_scale:
                Length scale by which the loss is divided
            pair_mask:
                [*,  N_frames, N_pts] mask to use for
                separating intra- from inter-chain losses.
            l1_clamp_distance:
                Cutoff above which distance errors are disregarded
            eps:
                Small value used to regularize denominators
        Returns:
            [*] loss tensor
    """
    # [*, N_frames, N_pts, 3]
    local_pred_pos = pred_frames.invert()[..., None].apply(
        pred_positions[..., None, :, :],
    )
    local_target_pos = target_frames.invert()[..., None].apply(
        target_positions[..., None, :, :],
    )
    #import ipdb; ipdb.set_trace()
    error_dist = torch.sqrt(
        torch.sum((local_pred_pos - local_target_pos) ** 2, dim=-1) + eps
    )

    if l1_clamp_distance is not None:
        error_dist = torch.clamp(error_dist, min=0, max=l1_clamp_distance)

    normed_error = error_dist / length_scale
    normed_error = normed_error * frames_mask[..., None]
    normed_error = normed_error * positions_mask[..., None, :]

    if pair_mask is not None:
        normed_error = normed_error * pair_mask
        normed_error = torch.sum(normed_error, dim=(-1, -2))

        mask = frames_mask[..., None] * positions_mask[..., None, :] * pair_mask
        norm_factor = torch.sum(mask, dim=(-2, -1))

        normed_error = normed_error / (eps + norm_factor)
    else:
        # FP16-friendly averaging. Roughly equivalent to:
        #
        # norm_factor = (
        #     torch.sum(frames_mask, dim=-1) *
        #     torch.sum(positions_mask, dim=-1)
        # )
        # normed_error = torch.sum(normed_error, dim=(-1, -2)) / (eps + norm_factor)
        #
        # ("roughly" because eps is necessarily duplicated in the latter)
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = (
            normed_error / (eps + torch.sum(frames_mask, dim=-1))[..., None]
        )
        normed_error = torch.sum(normed_error, dim=-1)
        normed_error = normed_error / (eps + torch.sum(positions_mask, dim=-1))

    return normed_error


def backbone_loss(
    rigids_0: torch.Tensor,
    mask: torch.Tensor,
    traj: torch.Tensor,
    pair_mask: Optional[torch.Tensor] = None,
    use_clamped_fape: Optional[torch.Tensor] = None,
    clamp_distance: float = 10.0,
    loss_unit_distance: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    ### need to check if the traj belongs to 4*4 matrix or a tensor_7
    if traj.shape[-1] == 7:
        pred_aff = Rigid.from_tensor_7(traj)
    elif traj.shape[-1] == 4:
        pred_aff = Rigid.from_tensor_4x4(traj)

    pred_aff = Rigid(
        Rotation(rot_mats=pred_aff.get_rots().get_rot_mats(), quats=None),
        pred_aff.get_trans(),
    )

    # DISCREPANCY: DeepMind somehow gets a hold of a tensor_7 version of
    # backbone tensor, normalizes it, and then turns it back to a rotation
    # matrix. To avoid a potentially numerically unstable rotation matrix
    # to quaternion conversion, we just use the original rotation matrix
    # outright. This one hasn't been composed a bunch of times, though, so
    # it might be fine.
    gt_aff = Rigid.from_tensor_7(rigids_0)

    fape_loss = compute_fape(
        pred_aff,
        gt_aff[None],
        mask[None],
        pred_aff.get_trans(),
        gt_aff[None].get_trans(),
        mask[None],
        pair_mask=pair_mask,
        l1_clamp_distance=clamp_distance,
        length_scale=loss_unit_distance,
        eps=eps,
    )
    if use_clamped_fape is not None:
        unclamped_fape_loss = compute_fape(
            pred_aff,
            gt_aff[None],
            mask[None],
            pred_aff.get_trans(),
            gt_aff[None].get_trans(),
            mask[None],
            pair_mask=pair_mask,
            l1_clamp_distance=None,
            length_scale=loss_unit_distance,
            eps=eps,
        )

        fape_loss = fape_loss * use_clamped_fape + unclamped_fape_loss * (
            1 - use_clamped_fape
        )

    # Average over the batch dimension
    fape_loss = torch.mean(fape_loss)

    return fape_loss



def sidechain_loss(
    sidechain_frames: torch.Tensor,
    sidechain_atom_pos: torch.Tensor,
    rigidgroups_gt_frames: torch.Tensor,
    rigidgroups_alt_gt_frames: torch.Tensor,
    rigidgroups_gt_exists: torch.Tensor,
    renamed_atom14_gt_positions: torch.Tensor,
    renamed_atom14_gt_exists: torch.Tensor,
    alt_naming_is_better: torch.Tensor,
    fixed_mask: torch.Tensor,
    clamp_distance: float = 10.0,
    length_scale: float = 10.0,
    eps: float = 1e-4,
    **kwargs,
) -> torch.Tensor:
    
    diffuse_mask = 1 - fixed_mask
    rigidgroups_gt_frames = Rigid(Rotation(rot_mats=rigidgroups_gt_frames[0]), rigidgroups_gt_frames[1]).to_tensor_4x4()
    rigidgroups_gt_frames = rigidgroups_gt_frames * diffuse_mask[..., None, None, None] 
    rigidgroups_alt_gt_frames = Rigid(Rotation(rot_mats=rigidgroups_alt_gt_frames[0]), rigidgroups_alt_gt_frames[1]).to_tensor_4x4() 
    rigidgroups_alt_gt_frames = rigidgroups_alt_gt_frames * diffuse_mask[..., None, None, None] 
    renamed_gt_frames = (1.0 - alt_naming_is_better[..., None, None, None]) * rigidgroups_gt_frames + alt_naming_is_better[
                            ..., None, None, None] * rigidgroups_alt_gt_frames
    
    # Steamroll the inputs
    #sidechain_frames = sidechain_frames[-1]
    batch_dims = sidechain_frames.shape[:-4]
    sidechain_frames = sidechain_frames.view(*batch_dims, -1, 4, 4)
    sidechain_frames = Rigid.from_tensor_4x4(sidechain_frames)
    
    renamed_gt_frames = renamed_gt_frames.view(*batch_dims, -1, 4, 4)
    renamed_gt_frames = Rigid.from_tensor_4x4(renamed_gt_frames)
    rigidgroups_gt_exists = rigidgroups_gt_exists.reshape(*batch_dims, -1)
    #sidechain_atom_pos = sidechain_atom_pos[-1]
    sidechain_atom_pos = sidechain_atom_pos.view(*batch_dims, -1, 3)
    renamed_atom14_gt_positions = renamed_atom14_gt_positions.view(
        *batch_dims, -1, 3
    )
    renamed_atom14_gt_exists = renamed_atom14_gt_exists.view(*batch_dims, -1)
    #import ipdb; ipdb.set_trace()
    fape = compute_fape(
        sidechain_frames,
        renamed_gt_frames,
        rigidgroups_gt_exists,
        sidechain_atom_pos,
        renamed_atom14_gt_positions,
        renamed_atom14_gt_exists,
        pair_mask=None,
        l1_clamp_distance=clamp_distance,
        length_scale=length_scale,
        eps=eps,
    )

    return fape

def fape_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict,
) -> torch.Tensor:
    """
    Computes FAPE loss, with logic corrected by referencing the initial working version.
    This is the definitive, final fix.
    """
    # 1. Backbone FAPE (这部分逻辑保持不变，因为它是全局的，不涉及复杂的 sidechain 展平)
    pred_frames_bb = Rigid.from_tensor_7(out['heads']['folding']["rigids"])
    target_frames_bb = Rigid.from_tensor_7(batch["rigids_0"])
    backbone_rigid_mask = batch["mask"]
    
    pred_positions_ca = out['heads']['folding']['final_atom_positions'][..., residue_constants.atom_order["CA"], :]
    target_positions_ca = batch["renamed_atom14_gt_positions"][..., residue_constants.atom_order["CA"], :]
    positions_mask_ca = batch["renamed_atom14_gt_exists"][..., residue_constants.atom_order["CA"]]

    clamped_fape_loss = compute_fape(
        pred_frames=pred_frames_bb, target_frames=target_frames_bb, frames_mask=backbone_rigid_mask,
        pred_positions=pred_positions_ca, target_positions=target_positions_ca, positions_mask=positions_mask_ca,
        l1_clamp_distance=config.backbone.clamp_distance, length_scale=config.backbone.loss_unit_distance,
        eps=config.backbone.eps, pair_mask=None
    )
    unclamped_fape_loss = compute_fape(
        pred_frames=pred_frames_bb, target_frames=target_frames_bb, frames_mask=backbone_rigid_mask,
        pred_positions=pred_positions_ca, target_positions=target_positions_ca, positions_mask=positions_mask_ca,
        l1_clamp_distance=None, length_scale=config.backbone.loss_unit_distance,
        eps=config.backbone.eps, pair_mask=None
    )
    use_clamped_fape = config.backbone.get("use_clamped_fape", 0.0)
    backbone_fape_loss = torch.mean(
        clamped_fape_loss * use_clamped_fape + unclamped_fape_loss * (1 - use_clamped_fape)
    )

    # 2. Sidechain FAPE - [最终修正] 回归到不展平的直接调用方式
    diffuse_mask = (1 - batch["fixed_mask"]).float()
    
    # --- 获取预测的侧链数据 (不展平) ---
    sidechain_output = out['heads']['folding']['sidechains']
    last_layer_rots = sidechain_output['frames'][0][-1]
    last_layer_trans = sidechain_output['frames'][1][-1]
    sidechain_frames_pred = Rigid(rots=Rotation(rot_mats=last_layer_rots), trans=last_layer_trans)
    sidechain_pos_pred = sidechain_output['atom_pos'][-1] # 保持 [B, N, 14, 3]

    # --- 获取真值的侧链数据 (不展平) ---
    gt_sidechain_rots, gt_sidechain_trans = batch["rigidgroups_gt_frames"]
    alt_gt_sidechain_rots, alt_gt_sidechain_trans = batch["rigidgroups_alt_gt_frames"]
    alt_naming_is_better = batch["alt_naming_is_better"]
    renamed_gt_rots = (1.0 - alt_naming_is_better[..., None, None, None]) * gt_sidechain_rots + alt_naming_is_better[..., None, None, None] * alt_gt_sidechain_rots
    renamed_gt_trans = (1.0 - alt_naming_is_better[..., None, None]) * gt_sidechain_trans + alt_naming_is_better[..., None, None] * alt_gt_sidechain_trans
    renamed_gt_frames = Rigid(rots=Rotation(rot_mats=renamed_gt_rots), trans=renamed_gt_trans)

    # --- 准备掩码 (不展平) ---
    # `rigidgroups_gt_exists` 保持原始形状 [B, N, 8]
    # `diffuse_mask` 需要扩展以匹配
    sidechain_frames_mask = batch["rigidgroups_gt_exists"] * diffuse_mask.unsqueeze(-1)
    
    # `renamed_atom14_gt_exists` 来自 batch，保持原始形状 [B, N, 14]
    positions_mask = batch["renamed_atom14_gt_exists"]
    
    # --- 调用 compute_fape (所有输入都保持其多维结构) ---
    sidechain_fape_loss = compute_fape(
        pred_frames=sidechain_frames_pred,
        target_frames=renamed_gt_frames,
        frames_mask=sidechain_frames_mask,
        pred_positions=sidechain_pos_pred, # 传入 [B, N, 14, 3]
        target_positions=batch["renamed_atom14_gt_positions"], # 传入 [B, N, 14, 3]
        positions_mask=positions_mask, # 传入 [B, N, 14]
        l1_clamp_distance=config.sidechain.clamp_distance,
        length_scale=config.sidechain.length_scale,
        eps=config.sidechain.eps,
        pair_mask=None
    )
    sidechain_fape_loss = torch.mean(sidechain_fape_loss)

    # 3. 合并损失
    loss = (
        backbone_fape_loss * config.backbone.weight +
        sidechain_fape_loss * config.sidechain.weight
    )

    return loss



def energy_head_joint_loss(
    is_training: bool,
    batch: Dict[str, torch.Tensor],
    pred_trans_score: torch.Tensor,   # [B, N, 3]
    energy_pair: dict,                # {'E0_gt':[B], 'pred':[B], 'forces': {'trans':[B,N,3]}, ...}
    t: torch.Tensor,                  # [B]
    mask: torch.Tensor,               # [B, N]
    fixed_mask: torch.Tensor,         # [B, N] (1=fixed, 0=designable)
    config,
    eps: float = 1e-6
):
    
    losses = {}
    device = pred_trans_score.device
    dtype = pred_trans_score.dtype

    E0 = energy_pair["E0_gt"]
    Ep = energy_pair["pred"]
    Fx = energy_pair["forces"]["trans"]

    def stat_finite(x, name):
        xd = x.detach()
        finite = torch.isfinite(xd)
        rate = finite.float().mean()
        if finite.any():
            xf = xd[finite]
            return {
                f"{name}_finite_rate": rate.item(),
                f"{name}_maxabs_f": xf.abs().max().item(),
                f"{name}_min_f": xf.min().item(),
                f"{name}_max_f": xf.max().item(),
            }
        else:
            return {f"{name}_finite_rate": rate.item()}


    # print(stat_finite(E0, "E0_gt"), stat_finite(Ep, "Ep_pred"), stat_finite(Fx, "F_trans"))

    # ===========================
    # NEW: read diffusion-scale info from batch (single source of truth)
    # ===========================
    tss = batch.get("trans_score_scaling", None)   # [B] or None
    if tss is not None:
        tss = tss.to(device=device, dtype=dtype)
        # losses["eh_tss_mean"] = tss.detach().mean()
        # losses["eh_tss_min"]  = tss.detach().min()
        # losses["eh_tss_max"]  = tss.detach().max()

    s_gt = batch.get("trans_score", None)          # [B,N,3] or None (GT score from diffuser)

    # -------------------------
    # 1) time gating
    # -------------------------
    t_threshold = float(config.get("t_threshold", 0.5))
    gate_k = float(config.get("time_gate_k", 12.0))
    time_weight = torch.sigmoid(torch.tensor(gate_k, device=device, dtype=dtype) * (t_threshold - t))  # [B]

    # NEW: correctness/debug mode -> disable time gating to stabilize diagnostics
    if bool(config.get("disable_time_gating", False)):
        time_weight = torch.ones_like(t, device=device, dtype=dtype)

    # losses["eh_time_w_mean"] = time_weight.detach().mean()
    # losses["eh_time_w_min"] = time_weight.detach().min()
    # losses["eh_time_w_max"] = time_weight.detach().max()

    # ---- NEW: effective sample size (ESS) + t distribution ----
    w_det = time_weight.detach()
    ess = (w_det.sum() ** 2) / (w_det.pow(2).sum().clamp_min(eps))
    losses["eh_time_ess"] = ess
    # t_det = t.detach()
    # losses["eh_t_mean"] = t_det.mean()
    # losses["eh_t_p10"] = torch.quantile(t_det, torch.tensor(0.10, device=device, dtype=dtype))
    # losses["eh_t_p50"] = torch.quantile(t_det, torch.tensor(0.50, device=device, dtype=dtype))
    # losses["eh_t_p90"] = torch.quantile(t_det, torch.tensor(0.90, device=device, dtype=dtype))

    # -------------------------
    # 2) design mask
    # -------------------------
    design_2d = (mask.float() * (1.0 - fixed_mask.float())).to(dtype=dtype)    # [B,N]
    design_mask = design_2d.unsqueeze(-1)                                       # [B,N,1]
    L_design = design_2d.sum(dim=1).clamp_min(1.0)                              # [B]
    valid_design = (design_2d.sum(dim=1) > 0.5).to(dtype=dtype)                # [B]
    losses["eh_L_design_mean"] = (design_2d.sum(dim=1).detach()).mean()

    # -------------------------
    # helpers
    # -------------------------
    def _safe_weighted_mean(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # x: [B], w: [B]
        s = (x * w).sum()
        d = w.sum().clamp_min(eps)
        return s / d

    def _safe_quantile_1d(x_1d: torch.Tensor, q: float) -> torch.Tensor:
        # x_1d: [M]
        if x_1d.numel() == 0:
            return torch.tensor(0.0, device=device, dtype=dtype)
        return torch.quantile(x_1d, torch.tensor(q, device=device, dtype=dtype))

    def _drop_nonfinite_scalar(x: torch.Tensor, key: str) -> torch.Tensor:
        """
        If x is NaN/Inf, return constant 0 (no grad flows), and log a flag.
        Works for scalar tensors ().
        """
        finite = torch.isfinite(x)
        losses[f"{key}_isfinite"] = finite.detach().to(dtype=dtype)
        # if not finite -> choose a constant zero (detached) so backward is blocked
        zero = x.detach().new_zeros(())
        return torch.where(finite, x, zero)

    # ==========================================================
    # 3) FORCE / DIRECTION terms
    # ==========================================================
    loss_force = pred_trans_score.sum() * 0.0
    loss_force_cos_local = pred_trans_score.sum() * 0.0

    # diagnostics defaults
    losses["eh_force_norm"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_valid_force_rate"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_s_rms_mean"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_gradE_rms_mean"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_cos_p50"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_cos_p90"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_cos_pos_frac_raw"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_cos_local"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_cos_local_relu"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_loss_force"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_loss_force_cos_local"] = torch.tensor(0.0, device=device, dtype=dtype)
    losses["eh_scale_ratio"] = torch.tensor(1.0, device=device, dtype=dtype)  # 按你的标准：不再解释它
    losses["eh_coord_c"] = torch.tensor(float(config.get("coordinate_scaling", 1.0)), device=device, dtype=dtype)  # 仅记录，不参与任何计算

    if is_training:
        # score and gradE (same space, no coordinate_scaling involved)
        # score lives in SCALED coordinates (see R3Diffuser.forward_marginal): score = ∇_y log p(y,t)
        # NEW: optional apply trans_score_scaling inside loss (default OFF)
        apply_tss = bool(config.get("apply_trans_score_scaling_in_loss", False))
        s_raw = pred_trans_score
        if apply_tss and (tss is not None):
            s_raw = s_raw * tss[:, None, None]                 # [B,N,3]
        s = s_raw * design_mask                                 # [B,N,3]
        # losses["eh_apply_tss_in_loss"] = torch.tensor(float(apply_tss), device=device, dtype=dtype)

        # energy_head forces are typically w.r.t. UNscaled Å coordinates: f_x = -∇_x E
        f_x_raw = energy_pair["forces"]["trans"].to(dtype=dtype)
        # f_x_raw = torch.nan_to_num(f_x_raw, nan=0.0, posinf=0.0, neginf=0.0)

        # [NEW] clip by per-residue norm (safer than component clamp)
        force_clip = float(config.get("force_clip", 1e3))  # in Å-space units of your energy head
        f_norm = torch.linalg.vector_norm(f_x_raw, dim=-1, keepdim=True).clamp_min(eps)
        scale = (force_clip / f_norm).clamp_max(1.0)
        f_x_raw = f_x_raw * scale
        losses["eh_force_clip_frac"] = (scale < 0.999).float().mean().detach()

        f_x = f_x_raw * design_mask
        f_x = f_x.detach()

        gradE_x = -f_x                                          # [B,N,3]

        # ===================== coordinate scaling alignment =====================
        # y = c x  =>  ∇_y E = (1/c) ∇_x E  and  f_y = (1/c) f_x
        coord_c = float(config.get("coordinate_scaling", 1.0))
        if coord_c <= 0:
            raise ValueError(f"Invalid coordinate_scaling={coord_c}")

        gradE = gradE_x / coord_c # [B,N,3]  gradE_y
        f = f_x / coord_c         # [B,N,3]  f_y  (if you still need f later)
        # =================

        # per-residue norms
        s_n = torch.sqrt((s * s).sum(dim=-1) + eps)            # [B,N]
        g_n = torch.sqrt((gradE * gradE).sum(dim=-1) + eps)    # [B,N]

        gradE_dir = gradE
        g_n_dir = g_n
        g_unit = gradE_dir / (g_n_dir.unsqueeze(-1) + eps)
        s_unit = s     / (s_n.unsqueeze(-1) + eps)
        # ---------- [INSERT BEGIN] define rms + valid_force + margins (training-critical) ----------
        # RMS over design region (per-sample)  [B]
        s_rms = torch.sqrt(((s_n * s_n) * design_2d).sum(dim=1) / L_design + eps)
        g_rms = torch.sqrt(((g_n * g_n) * design_2d).sum(dim=1) / L_design + eps)

        # ===================== quantile-adaptive force thresholds =====================
        if bool(config.get("force_thr_adaptive", False)):
            q_min = float(config.get("force_min_q", 0.2))
            q_ref = float(config.get("force_ref_q", 0.6))
            pool = g_rms.detach()[torch.isfinite(g_rms.detach())]
            if pool.numel() > 0:
                fmin = torch.quantile(pool, torch.tensor(q_min, device=pool.device, dtype=pool.dtype))
                fref = torch.quantile(pool, torch.tensor(q_ref, device=pool.device, dtype=pool.dtype))
                # 合理下限/间隔保护
                fmin = fmin.clamp_min(float(config.get("force_min_floor", 1e-6)))
                fref = torch.maximum(fref, fmin * float(config.get("force_ref_min_ratio", 1.2)))
                force_min = float(fmin.item())
                force_ref = float(fref.item())
            else:
                force_min = float(config.get("force_min", 1e-4))
                force_ref = float(config.get("force_ref", 5.0 * force_min))
        else:
            force_min = float(config.get("force_min", 1e-4))
            force_ref = float(config.get("force_ref", 5.0 * force_min))

        force_w = ((g_rms - force_min) / (force_ref - force_min + eps)).clamp(0.0, 1.0)
        finite_g = torch.isfinite(g_rms)
        force_w = torch.where(finite_g, force_w, force_w.new_zeros(()))

        # 仍然记录一个“是否足够大”的 rate，但训练权重用 force_w
        valid_force = (force_w > 0).to(dtype=dtype)
        losses["eh_valid_force_rate"] = valid_force.detach().mean()

        losses["eh_s_rms_mean"]     = _safe_weighted_mean(s_rms.detach(), time_weight.detach())
        losses["eh_gradE_rms_mean"] = _safe_weighted_mean(g_rms.detach(), time_weight.detach())
        losses["eh_force_norm"]     = _safe_weighted_mean(g_rms.detach(), time_weight.detach())

        # margins used in hinges
        descent_margin = float(config.get("descent_margin", 0.01))
        cos_margin     = float(config.get("cos_margin", 0.2))
        
        cos_g_min = float(config.get("cos_g_min", 1e-4))
        cos_s_min = float(config.get("cos_s_min", 1e-4))

        global_g_min = float(config.get("global_g_min", cos_g_min))
        global_s_min = float(config.get("global_s_min", cos_s_min))

        if bool(config.get("global_thr_adaptive", False)):
            qg = float(config.get("global_g_q", 0.05))
            qs = float(config.get("global_s_q", 0.05))
            g_pool = g_n.detach()[ (design_2d > 0.5) & torch.isfinite(g_n.detach()) ]
            s_pool = s_n.detach()[ (design_2d > 0.5) & torch.isfinite(s_n.detach()) ]
            if g_pool.numel() > 0:
                global_g_min = float(torch.quantile(g_pool, torch.tensor(qg, device=device, dtype=dtype)).item())
            if s_pool.numel() > 0:
                global_s_min = float(torch.quantile(s_pool, torch.tensor(qs, device=device, dtype=dtype)).item())


        global_valid = (design_2d > 0.5) & (g_n > global_g_min) & (s_n > global_s_min)
        global_valid_f = global_valid.to(dtype=dtype)
        global_cnt_raw = global_valid_f.sum(dim=1)                       # [B]
        has_global = (global_cnt_raw > 0.5).to(dtype=dtype)
        global_cnt_safe = global_cnt_raw.clamp_min(1.0)

        losses["eh_global_cnt_mean"] = global_cnt_raw.detach().mean()
        losses["eh_global_cnt_frac_mean"] = (global_cnt_raw.detach() / L_design.detach().clamp_min(1.0)).mean()

        # --- base weight shared by force-type terms ---
        w_base = time_weight * valid_design * force_w            # [B]
        
        w_global = w_base * has_global                               # [B]
        w_global_sum = w_global.sum().clamp_min(eps)                 # scalar

        # average cosine over VALID positions only
        dot = (g_unit * s_unit).sum(dim=-1)                # [B,N]
        dot = torch.where(global_valid, dot, dot.new_zeros(()))
        cos_global = dot.sum(dim=1) / global_cnt_safe

        # want s to be in descent direction: cos_global <= -margin
        loss_force_per = F.relu(cos_global + descent_margin) * valid_design * force_w * has_global         # [B]

        # diagnostics (optional but strongly recommended)
        losses["eh_global_valid_rate"] = (has_global.detach() * time_weight.detach()).sum() / (time_weight.detach().sum().clamp_min(eps))

        # aggregate force loss: also gate by has_global
        loss_force = (loss_force_per * time_weight).sum() / (w_global_sum + eps)

        losses["eh_cos_global_mean"] = _safe_weighted_mean(cos_global.detach(), time_weight.detach())
        losses["eh_cos_global_p50"] = _safe_quantile_1d(cos_global.detach(), 0.50)
        losses["eh_cos_global_p90"] = _safe_quantile_1d(cos_global.detach(), 0.90)

        # ---- sign sanity check (force vs grad) ----
        f_n = torch.sqrt((f * f).sum(dim=-1) + eps)  # [B,N]
        f_unit = f / (f_n.unsqueeze(-1) + eps)

        dot_f = (f_unit * s_unit).sum(dim=-1)          # [B,N]
        dot_f = torch.where(global_valid, dot_f, dot_f.new_zeros(()))
        cos_force_global = dot_f.sum(dim=1) / global_cnt_safe

        losses["eh_cos_force_global_mean"] = _safe_weighted_mean(cos_force_global.detach(), time_weight.detach())
        losses["eh_cos_force_global_p50"]  = _safe_quantile_1d(cos_force_global.detach(), 0.50)
        losses["eh_cos_force_global_p90"]  = _safe_quantile_1d(cos_force_global.detach(), 0.90)
        
        # optional caps
        force_cap_w = float(config.get("force_cap_weight", 0.0))
        force_cap = float(config.get("force_cap", 5.0))
        loss_force_cap = torch.zeros_like(g_rms)
        if force_cap_w > 0:
            # soft & safe cap to avoid gradient explosion on extreme batches
            force_cap_max_excess = float(config.get("force_cap_max_excess", 50.0 * force_cap))
            excess = F.relu(g_rms - force_cap)                         # [B]
            excess = excess.clamp_max(force_cap_max_excess)            # hard stop
            # log penalty grows slowly; square on log is still bounded by clamp
            loss_force_cap = torch.log1p(excess / (force_cap + eps)).pow(2) * valid_design
            losses["eh_force_cap_excess_max"] = excess.detach().max()

        score_cap_w = float(config.get("score_cap_weight", 0.01))
        score_cap = float(config.get("score_cap", 50.0))
        loss_score_cap = torch.zeros_like(s_rms)
        if score_cap_w > 0:
            loss_score_cap = F.relu(s_rms - score_cap) * valid_design

        # aggregate force loss
        w_force = time_weight * valid_design * force_w  # [B]
        w_force_sum = w_force.sum().clamp_min(eps)

        if force_cap_w > 0:
            loss_force = loss_force + force_cap_w * (loss_force_cap * w_force).sum() / w_force_sum
        if score_cap_w > 0:
            loss_force = loss_force + score_cap_w * (loss_score_cap * w_force).sum() / w_force_sum

        # --- hard safety: drop each branch if non-finite (block backward) ---
        # loss_force           = _drop_nonfinite_scalar(loss_force,           "eh_loss_force")

        losses["eh_loss_force"] = loss_force.detach()
        losses["eh_force_norm"] = _safe_weighted_mean(g_rms.detach(), time_weight.detach())

        # ======================================================
        # (B) cosine hinge (local per-residue)
        #     关键：只在“向量足够大”的位置算 cosine，否则 eps 主导会把 cos 压扁到 0
        # ======================================================
        # local dot / cos
        dot_local = (gradE_dir * s).sum(dim=-1)                  # [B,N]
        denom = (g_n_dir * s_n).clamp_min(eps)                   # [B,N]
        cos_local = dot_local / denom                           # [-1,1] ideally

        # VALID positions for cosine:
        # - in design region
        # - both norms exceed small thresholds (avoid eps-dominated cos ~ 0)
        cos_g_min = float(config.get("cos_g_min", 1e-4))
        cos_s_min = float(config.get("cos_s_min", 1e-4))

        # ===================== quantile-adaptive cosine thresholds =====================
        if bool(config.get("cos_thr_adaptive", False)):
            qg = float(config.get("cos_g_q", 0.1))
            qs = float(config.get("cos_s_q", 0.1))
            g_pool = g_n.detach()[ (design_2d > 0.5) & torch.isfinite(g_n.detach()) ]
            s_pool = s_n.detach()[ (design_2d > 0.5) & torch.isfinite(s_n.detach()) ]
            if g_pool.numel() > 0:
                cos_g_min = float(torch.quantile(g_pool, torch.tensor(qg, device=device, dtype=dtype)).item())
                cos_g_min = max(cos_g_min, float(config.get("cos_g_min_floor", 1e-8)))
            if s_pool.numel() > 0:
                cos_s_min = float(torch.quantile(s_pool, torch.tensor(qs, device=device, dtype=dtype)).item())
                cos_s_min = max(cos_s_min, float(config.get("cos_s_min_floor", 1e-8)))

        cos_valid = (design_2d > 0.5) & (g_n > cos_g_min) & (s_n > cos_s_min)
        cos_valid_f = cos_valid.to(dtype=dtype)

        # cosine margin: require cos <= -margin (must descend with angle)
        cos_margin = float(config.get("cos_margin", 0.2))
        cos_local = dot_local / denom
        cos_local = torch.where(cos_valid, cos_local, cos_local.new_zeros(()))
        loss_cos_local = F.relu(cos_local + cos_margin) * cos_valid_f

        # per-sample average over valid positions (fallback to 0 if none)
        denom_pos_raw = cos_valid_f.sum(dim=1)                          # [B]
        has_local = (denom_pos_raw > 0.5).to(dtype=dtype)
        denom_pos_safe = denom_pos_raw.clamp_min(1.0)

        # 日志：raw 才是真计数；safe 只用于除法
        losses["eh_local_cnt_mean"] = denom_pos_raw.detach().mean()
        losses["eh_local_cnt_frac_mean"] = (denom_pos_raw.detach() / L_design.detach().clamp_min(1.0)).mean()

        loss_force_cos_local_per = (loss_cos_local.sum(dim=1) / denom_pos_safe)     # [B]
        loss_force_cos_local_per = loss_force_cos_local_per * valid_design * force_w * has_local

        w_local = w_base * has_local                                           # [B]  (w_base 在前面已定义)
        w_local_sum = w_local.sum().clamp_min(eps)

        loss_force_cos_local = (loss_force_cos_local_per * time_weight).sum() / (w_local_sum + eps)
        # loss_force_cos_local = _drop_nonfinite_scalar(loss_force_cos_local, "eh_loss_force_cos_local")
        losses["eh_loss_force_cos_local"] = loss_force_cos_local.detach()

        # diagnostics
        losses["eh_local_valid_rate"] = (has_local.detach() * time_weight.detach()).sum() / (time_weight.detach().sum().clamp_min(eps))

        # diagnostics for cosine distribution (flatten valid entries)
        cos_flat = cos_local[cos_valid & torch.isfinite(cos_local)].detach()
        losses["eh_cos_p50"] = _safe_quantile_1d(cos_flat, 0.50)
        losses["eh_cos_p90"] = _safe_quantile_1d(cos_flat, 0.90)

        # raw positive fraction (over valid positions only)
        if cos_flat.numel() > 0:
            losses["eh_cos_pos_frac_raw"] = (cos_flat > 0).float().mean()
        else:
            losses["eh_cos_pos_frac_raw"] = torch.tensor(0.0, device=device, dtype=dtype)

        # log mean cos and mean relu(cos) (over valid positions)
        if cos_flat.numel() > 0:
            losses["eh_cos_local"] = cos_flat.mean()
            losses["eh_cos_local_relu"] = F.relu(cos_flat).mean()
        else:
            losses["eh_cos_local"] = torch.tensor(0.0, device=device, dtype=dtype)
            losses["eh_cos_local_relu"] = torch.tensor(0.0, device=device, dtype=dtype)

    # ==========================================================
    # 3) ENERGY scalar term (Ep vs E0) with optional length normalization
    # ==========================================================
    E0 = energy_pair["E0_gt"].to(device=device, dtype=dtype).detach()  # [B]
    Ep = energy_pair["pred"].to(device=device, dtype=dtype)            # [B]

    # ---- per-sample finite mask (DO NOT nuke the whole batch) ----
    finite_E = torch.isfinite(E0) & torch.isfinite(Ep)                 # [B]
    E_abs_cap = float(config.get("E_abs_cap", 1e4))  # 先保守：1e4 或 1e5，别上来给 1e12
    finite_E = finite_E & (E0.abs() < E_abs_cap) & (Ep.abs() < E_abs_cap)
    losses["eh_E_cap_rate"] = ((E0.abs() >= E_abs_cap) | (Ep.abs() >= E_abs_cap)).float().mean()

    losses["eh_E_finite_rate"] = finite_E.float().mean()               # 0~1, 比原来的 0/1 更有用
    # losses["eh_E_nonfinite_rate"] = (1.0 - finite_E.float()).mean()
    
    # finite_E0 = torch.isfinite(E0)
    # finite_Ep = torch.isfinite(Ep)
    # # 这两条能迅速判断“到底是谁全坏”
    # losses["eh_E0_any_finite"] = (finite_E0.any()).float()
    # losses["eh_Ep_any_finite"] = (finite_Ep.any()).float()
    badE_rate = 1.0 - finite_E.float().mean()
    if badE_rate > 0.999:
        # 也可以只在 rank0 打印/每 K step 打印
        losses["eh_E_all_bad"] = torch.tensor(1.0, device=device, dtype=dtype)


    # 训练样本权重
    w_sample = time_weight * valid_design                              # [B]
    wE = w_sample * finite_E.float()                                   # [B]
    wE_sum_raw  = wE.sum()
    wE_sum_safe = wE_sum_raw.clamp_min(eps)
    bce_w = float(config.get("bce_weight", 0.0))
    scale_guard_w = float(config.get("scale_guard_weight", 0.0))

    loss_energy_scalar = pred_trans_score.sum() * 0.0
    loss_scale_guard   = pred_trans_score.sum() * 0.0
    loss_bce           = pred_trans_score.sum() * 0.0

    # 先打默认日志，防止缺 key
    losses["eh_loss_energy_scalar"] = loss_energy_scalar.detach()
    losses["eh_loss_scale_guard"]   = loss_scale_guard.detach()
    losses["eh_bce"]                = loss_bce.detach()

    # ---- ENERGY: only compute on finite samples ----
    if wE_sum_raw <= eps:
        # 没有任何可用样本：不要把 E0/Ep 均值硬写 0（那会误导），而是写 finite rate + 跳过
        losses["eh_loss_energy_scalar_isfinite"] = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        # optional length norm
        len_pow = float(config.get("energy_len_norm_power", 0.0))
        if len_pow > 0:
            denom_len = L_design.clamp_min(1.0).pow(len_pow)           # [B]
            E0n = (E0 / denom_len).detach()
            Epn = Ep / denom_len
        else:
            E0n, Epn = E0.detach(), Ep

        # ---- robust stats computed ONLY on finite subset ----
        E0n_f = E0n[finite_E]                                          # [M]
        Epn_f = Epn[finite_E]                                          # [M]

        e0_mu = E0n_f.median()
        mad = (E0n_f - e0_mu).abs().median()
        e0_std_raw = (1.4826 * mad).clamp_min(0.0)

        std_floor = float(config.get("energy_std_floor", 1e-3))
        e0_std = e0_std_raw.clamp_min(std_floor)

        # compute per-sample loss on finite subset
        z0_f = (E0n_f - e0_mu) / e0_std
        zp_f = (Epn_f - e0_mu) / e0_std
        loss_energy_z_f = F.smooth_l1_loss(zp_f, z0_f, reduction="none")    # [M]

        low_var_thr = float(config.get("energy_low_var_thr", 5e-4))
        use_abs = (e0_std_raw < low_var_thr).to(dtype=dtype)                # scalar
        loss_energy_abs_fallback = (
            F.smooth_l1_loss(Epn_f, E0n_f, beta=std_floor, reduction="none") / std_floor
        )
        loss_energy_z_f = (1 - use_abs) * loss_energy_z_f + use_abs * loss_energy_abs_fallback

        energy_z_w = float(config.get("energy_z_weight", 1.0))
        energy_abs_w = float(config.get("energy_abs_weight", 0.0))
        margin_good = float(config.get("margin_pred_vs_gt", 0.0))

        loss_energy_abs_f = loss_energy_z_f.new_zeros(loss_energy_z_f.shape)
        if energy_abs_w > 0:
            loss_energy_abs_f = F.relu((Epn_f - E0n_f) / (e0_std + eps) + margin_good)

        loss_energy_per_f = energy_z_w * loss_energy_z_f + energy_abs_w * loss_energy_abs_f  # [M]

        # weighted mean over finite samples
        wE_f = wE[finite_E]                                              # [M]
        loss_energy_scalar = (loss_energy_per_f * wE_f).sum() / (wE_f.sum().clamp_min(eps))
        # loss_energy_scalar = _drop_nonfinite_scalar(loss_energy_scalar, "eh_loss_energy_scalar")

        losses["eh_loss_energy_scalar"] = loss_energy_scalar.detach()
        losses["eh_loss_energy_scalar_isfinite"] = torch.tensor(1.0, device=device, dtype=dtype)

        # 这些统计也只统计 finite 子集，避免被 NaN 污染
        losses["eh_E0_mean"] = E0n_f.mean().detach()
        losses["eh_Ep_mean"] = Epn_f.mean().detach()
        losses["eh_E0_std"]  = E0n_f.std(unbiased=False).detach()
        losses["eh_Ep_std"]  = Epn_f.std(unbiased=False).detach()

        # ---- scale guard 也只在 finite 子集上算（否则 log/std 会炸）----
        loss_scale_guard = Ep.new_zeros(())
        if scale_guard_w > 0:
            Ep_mu = Epn_f.mean()
            Ep_std = Epn_f.std(unbiased=False).clamp_min(eps)

            min_std = float(config.get("min_Ep_std", 1e-3))
            max_std = float(config.get("max_Ep_std", 1e3))
            mean_cap = float(config.get("Ep_mean_abs_cap", 1e6))

            hi = F.relu(torch.log(Ep_std / (max_std + eps)))
            lo = F.relu(torch.log((min_std + eps) / Ep_std))
            mu = F.relu(torch.log((Ep_mu.abs() + eps) / (mean_cap + eps)))

            clamp_log = float(config.get("scale_guard_log_clamp", 20.0))
            hi = hi.clamp_max(clamp_log)
            lo = lo.clamp_max(clamp_log)
            mu = mu.clamp_max(clamp_log)

            loss_scale_guard = hi.pow(2) + lo.pow(2) + mu.pow(2)

    # loss_scale_guard = _drop_nonfinite_scalar(loss_scale_guard, "eh_loss_scale_guard")
    losses["eh_loss_scale_guard"] = loss_scale_guard.detach()

    # ==========================================================
    # 4) optional BCE (RMSD -> label)  [DECOUPLED FROM Ep/E0]
    # ==========================================================
    loss_bce = Ep.new_zeros(())
    if bce_w > 0 and ("iface_logit" in energy_pair) and ("rmsd" in energy_pair):
        iface_logit = energy_pair["iface_logit"]
        rmsd = energy_pair["rmsd"]
        rmsd_pos_thr = float(config.get("rmsd_pos_thr", 3.0))
        y = (rmsd <= rmsd_pos_thr).float()

        finite_bce = torch.isfinite(iface_logit) & torch.isfinite(rmsd)
        wB = w_sample * finite_bce.float()
        wB_sum_raw = wB.sum()
        wB_sum_safe = wB_sum_raw.clamp_min(eps)

        if wB_sum_raw > eps:
            pos_w = float(config.get("bce_pos_weight", 1.0))
            pos_weight = torch.tensor(pos_w, device=iface_logit.device, dtype=iface_logit.dtype)

            bce_per = F.binary_cross_entropy_with_logits(
                iface_logit, y, pos_weight=pos_weight, reduction="none"
            )
            loss_bce = (bce_per * wB).sum() / wB_sum_safe
            # loss_bce = _drop_nonfinite_scalar(loss_bce, "eh_bce")

            losses["eh_bce"] = loss_bce.detach()
            losses["eh_pos_rate"] = (y * wB).sum().detach() / wB_sum_safe
        else:
            losses["eh_bce"] = loss_bce.detach()

        losses["eh_bce_finite_rate"] = finite_bce.float().mean()

        
    # ==========================================================
    # 5) FINAL
    # ==========================================================
    w_force = float(config.get("w_force", 1.0))
    w_energy = float(config.get("w_energy", 0.1))
    w_force_cos = float(config.get("w_force_cos", 0.0))   # [NEW]

    final_loss = (
        w_force * loss_force
        + w_force_cos * loss_force_cos_local
        + w_energy * loss_energy_scalar
        + bce_w * loss_bce
        + scale_guard_w * loss_scale_guard
    )
    # drop non-finite final loss BEFORE soft cap
    # final_loss = _drop_nonfinite_scalar(final_loss, "eh_final_loss_pre_cap")
    # soft cap for numerical safety
    loss_cap = float(config.get("loss_cap", 10.0))
    final_loss = torch.where(
        final_loss <= loss_cap,
        final_loss,
        loss_cap + torch.log(final_loss - loss_cap + 1.0)
    )
    # drop non-finite final loss BEFORE soft cap
    # final_loss = _drop_nonfinite_scalar(final_loss, "eh_final_loss_post_cap")
    return final_loss, losses


def supervised_chi_loss(
    angles_sin_cos: torch.Tensor,
    unnormalized_angles_sin_cos: torch.Tensor,
    seq: torch.Tensor,
    seq_mask: torch.Tensor,
    chi_mask: torch.Tensor,
    chi_angles_sin_cos: torch.Tensor,
    chi_weight: float,
    angle_norm_weight: float,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    """
        Implements Algorithm 27 (torsionAngleLoss)

        Args:
            angles_sin_cos:
                [*, N, 7, 2] predicted angles
            unnormalized_angles_sin_cos:
                The same angles, but unnormalized
            seq:
                [*, N] residue indices
            seq_mask:
                [*, N] sequence mask
            chi_mask:
                [*, N, 7] angle mask
            chi_angles_sin_cos:
                [*, N, 7, 2] ground truth angles
            chi_weight:
                Weight for the angle component of the loss
            angle_norm_weight:
                Weight for the normalization component of the loss
        Returns:
            [*] loss tensor
    """
    pred_angles = angles_sin_cos[..., 3:, :]
    residue_type_one_hot = torch.nn.functional.one_hot(
        seq,
        residue_constants.restype_num + 1,
    )
    chi_pi_periodic = torch.einsum(
        "...ij,jk->ik",
        residue_type_one_hot.type(angles_sin_cos.dtype),
        angles_sin_cos.new_tensor(residue_constants.chi_pi_periodic),
    )

    true_chi = chi_angles_sin_cos[None]

    shifted_mask = (1 - 2 * chi_pi_periodic).unsqueeze(-1)
    true_chi_shifted = shifted_mask * true_chi
    sq_chi_error = torch.sum((true_chi - pred_angles) ** 2, dim=-1)
    sq_chi_error_shifted = torch.sum(
        (true_chi_shifted - pred_angles) ** 2, dim=-1
    )
    sq_chi_error = torch.minimum(sq_chi_error, sq_chi_error_shifted)

    # The ol' switcheroo
    sq_chi_error = sq_chi_error.permute(
        *range(len(sq_chi_error.shape))[1:-2], 0, -2, -1
    )

    sq_chi_loss = masked_mean(
        chi_mask[..., None, :, :], sq_chi_error, dim=(-1, -2, -3)
    )

    loss = chi_weight * sq_chi_loss

    angle_norm = torch.sqrt(
        torch.sum(unnormalized_angles_sin_cos ** 2, dim=-1) + eps
    )
    norm_error = torch.abs(angle_norm - 1.0)
    norm_error = norm_error.permute(
        *range(len(norm_error.shape))[1:-2], 0, -2, -1
    )
    angle_norm_loss = masked_mean(
        seq_mask[..., None, :, None], norm_error, dim=(-1, -2, -3)
    )

    loss = loss + angle_norm_weight * angle_norm_loss

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def compute_plddt(logits: torch.Tensor) -> torch.Tensor:
    num_bins = logits.shape[-1]
    bin_width = 1.0 / num_bins
    bounds = torch.arange(
        start=0.5 * bin_width, end=1.0, step=bin_width, device=logits.device
    )
    probs = torch.nn.functional.softmax(logits, dim=-1)
    pred_lddt_ca = torch.sum(
        probs * bounds.view(*((1,) * len(probs.shape[:-1])), *bounds.shape),
        dim=-1,
    )
    return pred_lddt_ca * 100


def lddt(
    all_atom_pred_pos: torch.Tensor,
    atom37_gt_positions: torch.Tensor,
    atom37_gt_exists: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    n = atom37_gt_exists.shape[-2]
    dmat_true = torch.sqrt(
        eps
        + torch.sum(
            (
                atom37_gt_positions[..., None, :]
                - atom37_gt_positions[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    dmat_pred = torch.sqrt(
        eps
        + torch.sum(
            (
                all_atom_pred_pos[..., None, :]
                - all_atom_pred_pos[..., None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    dists_to_score = (
        (dmat_true < cutoff)
        * atom37_gt_exists
        * permute_final_dims(atom37_gt_exists, (1, 0))
        * (1.0 - torch.eye(n, device=atom37_gt_exists.device))
    )

    dist_l1 = torch.abs(dmat_true - dmat_pred)

    score = (
        (dist_l1 < 0.5).type(dist_l1.dtype)
        + (dist_l1 < 1.0).type(dist_l1.dtype)
        + (dist_l1 < 2.0).type(dist_l1.dtype)
        + (dist_l1 < 4.0).type(dist_l1.dtype)
    )
    score = score * 0.25

    dims = (-1,) if per_residue else (-2, -1)
    norm = 1.0 / (eps + torch.sum(dists_to_score, dim=dims))
    score = norm * (eps + torch.sum(dists_to_score * score, dim=dims))

    return score


def lddt_ca(
    all_atom_pred_pos: torch.Tensor,
    atom37_gt_positions: torch.Tensor,
    atom37_gt_exists: torch.Tensor,
    cutoff: float = 15.0,
    eps: float = 1e-10,
    per_residue: bool = True,
) -> torch.Tensor:
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    atom37_gt_positions = atom37_gt_positions[..., ca_pos, :]
    atom37_gt_exists = atom37_gt_exists[..., ca_pos: (ca_pos + 1)]  # keep dim

    return lddt(
        all_atom_pred_pos,
        atom37_gt_positions,
        atom37_gt_exists,
        cutoff=cutoff,
        eps=eps,
        per_residue=per_residue,
    )

def lddt_loss(
    logits: torch.Tensor,
    all_atom_pred_pos: torch.Tensor,
    atom37_gt_positions: torch.Tensor,
    atom37_gt_exists: torch.Tensor,
    fixed_mask: torch.Tensor,
    cutoff: float = 15.0,
    no_bins: int = 50,
    eps: float = 1e-10,
    **kwargs,
) -> torch.Tensor:
    
    # 计算 lddt 分数 (这部分与 OpenFold 相同)
    ca_pos = residue_constants.atom_order["CA"]
    pred_pos_ca = all_atom_pred_pos[..., ca_pos, :]
    gt_pos_ca = atom37_gt_positions[..., ca_pos, :]
    gt_exists_ca = atom37_gt_exists[..., ca_pos: (ca_pos + 1)]

    score = lddt(
        pred_pos_ca,
        gt_pos_ca,
        gt_exists_ca,
        cutoff=cutoff,
        eps=eps
    )
    score = torch.nan_to_num(score)
    score = score.detach()

    bin_index = torch.floor(score * no_bins).long()
    bin_index = torch.clamp(bin_index, max=(no_bins - 1))
    lddt_ca_one_hot = torch.nn.functional.one_hot(bin_index, num_classes=no_bins)

    errors = softmax_cross_entropy(logits, lddt_ca_one_hot)
    
    # ✅ 关键修正：我们只关心CDR区域的误差，但用全局长度来归一化
    
    # diffuse_mask (1 for CDR, 0 for framework)
    diffuse_mask = (1 - fixed_mask).float() # [*, N]

    # 从真值中获取有效的C-alpha原子掩码
    ca_exists_mask = atom37_gt_exists[..., ca_pos] # [*, N]

    # 最终用于计算误差的掩码，只包含CDR中真实存在的C-alpha
    loss_mask = ca_exists_mask * diffuse_mask
    
    # 分子：只对CDR区域的误差求和
    loss_numerator = torch.sum(errors * loss_mask, dim=-1)
    
    # 分母：使用整条链上真实存在的C-alpha数量，保证分母稳定
    loss_denominator = torch.sum(ca_exists_mask, dim=-1)

    # 避免除以零
    stable_loss = loss_numerator / (loss_denominator + eps)

    # 对 batch 维度求平均
    loss = torch.mean(stable_loss)

    return loss

def distogram_loss(
    logits,
    pseudo_beta,
    pseudo_beta_mask,
    min_bin=2.3125,
    max_bin=21.6875,
    no_bins=64,
    eps=1e-6,
    **kwargs, # 接收但可能不使用 fixed_mask 等
):
    # 理由：Distogram评估的是残基间的距离分布，这是一个全局特征。
    # 为了数值稳定，归一化必须在整条链上进行。
    boundaries = torch.linspace(
        min_bin,
        max_bin,
        no_bins - 1,
        device=logits.device,
    )
    boundaries = boundaries ** 2

    dists = torch.sum(
        (pseudo_beta[..., None, :] - pseudo_beta[..., None, :, :]) ** 2,
        dim=-1,
        keepdims=True,
    )

    true_bins = torch.sum(dists > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits,
        torch.nn.functional.one_hot(true_bins, no_bins),
    )

    # ✅ 关键修正：square_mask 基于完整的 pseudo_beta_mask 构建，不进行任何局部裁剪。
    square_mask = pseudo_beta_mask[..., None] * pseudo_beta_mask[..., None, :]

    # ✅ 关键修正：归一化分母是基于完整的 square_mask 计算的，数值巨大且稳定。
    # 使用与 OpenFold 相同的 FP16 友好求和方式。
    denom = eps + torch.sum(square_mask, dim=(-1, -2))
    mean = torch.sum(errors * square_mask, dim=(-1, -2)) / denom
    
    # 对 batch 维度求平均
    mean = torch.mean(mean)

    return mean

def _calculate_bin_centers(boundaries: torch.Tensor):
    step = boundaries[1] - boundaries[0]
    bin_centers = boundaries + step / 2
    bin_centers = torch.cat(
        [bin_centers, (bin_centers[-1] + step).unsqueeze(-1)], dim=0
    )
    return bin_centers


def _calculate_expected_aligned_error(
    alignment_confidence_breaks: torch.Tensor,
    aligned_distance_error_probs: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    bin_centers = _calculate_bin_centers(alignment_confidence_breaks)
    return (
        torch.sum(aligned_distance_error_probs * bin_centers, dim=-1),
        bin_centers[-1],
    )


def compute_predicted_aligned_error(
    logits: torch.Tensor,
    max_bin: int = 31,
    no_bins: int = 64,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes aligned confidence metrics from logits.

    Args:
      logits: [*, num_res, num_res, num_bins] the logits output from
        PredictedAlignedErrorHead.
      max_bin: Maximum bin value
      no_bins: Number of bins
    Returns:
      aligned_confidence_probs: [*, num_res, num_res, num_bins] the predicted
        aligned error probabilities over bins for each residue pair.
      predicted_aligned_error: [*, num_res, num_res] the expected aligned distance
        error for each pair of residues.
      max_predicted_aligned_error: [*] the maximum predicted error possible.
    """
    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    aligned_confidence_probs = torch.nn.functional.softmax(logits, dim=-1)
    (
        predicted_aligned_error,
        max_predicted_aligned_error,
    ) = _calculate_expected_aligned_error(
        alignment_confidence_breaks=boundaries,
        aligned_distance_error_probs=aligned_confidence_probs,
    )

    return {
        "aligned_confidence_probs": aligned_confidence_probs,
        "predicted_aligned_error": predicted_aligned_error,
        "max_predicted_aligned_error": max_predicted_aligned_error,
    }


def compute_tm(
    logits: torch.Tensor,
    residue_weights: Optional[torch.Tensor] = None,
    asym_id: Optional[torch.Tensor] = None,
    interface: bool = False,
    max_bin: int = 31,
    no_bins: int = 64,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    if residue_weights is None:
        residue_weights = logits.new_ones(logits.shape[-2])

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )

    bin_centers = _calculate_bin_centers(boundaries)
    clipped_n = max(torch.sum(residue_weights), 19)

    d0 = 1.24 * (clipped_n - 15) ** (1.0 / 3) - 1.8

    probs = torch.nn.functional.softmax(logits, dim=-1)

    tm_per_bin = 1.0 / (1 + (bin_centers ** 2) / (d0 ** 2))
    predicted_tm_term = torch.sum(probs * tm_per_bin, dim=-1)

    n = residue_weights.shape[-1]
    pair_mask = residue_weights.new_ones((n, n), dtype=torch.int32)
    if interface and (asym_id is not None):
        if len(asym_id.shape) > 1:
            assert len(asym_id.shape) <= 2
            batch_size = asym_id.shape[0]
            pair_mask = residue_weights.new_ones((batch_size, n, n), dtype=torch.int32)
        pair_mask *= (asym_id[..., None] != asym_id[..., None, :]).to(dtype=pair_mask.dtype)

    predicted_tm_term *= pair_mask

    pair_residue_weights = pair_mask * (
        residue_weights[..., None, :] * residue_weights[..., :, None]
    )
    denom = eps + torch.sum(pair_residue_weights, dim=-1, keepdims=True)
    normed_residue_mask = pair_residue_weights / denom
    per_alignment = torch.sum(predicted_tm_term * normed_residue_mask, dim=-1)

    weighted = per_alignment * residue_weights

    argmax = (weighted == torch.max(weighted)).nonzero()[0]
    return per_alignment[tuple(argmax)]


def tm_loss(
    logits,
    final_affine_tensor,
    rigids_0,
    mask,
    resolution,
    max_bin=31,
    no_bins=64,
    min_resolution: float = 0.1,
    max_resolution: float = 3.0,
    eps=1e-8,
    **kwargs,
):
    # first check whether this is a tensor_7 or tensor_4*4
    if final_affine_tensor.shape[-1] == 7:
        pred_affine = Rigid.from_tensor_7(final_affine_tensor)
    elif final_affine_tensor.shape[-1] == 4:
        pred_affine = Rigid.from_tensor_4x4(final_affine_tensor)
    backbone_rigid = Rigid.from_tensor_4x4(rigids_0)

    def _points(affine):
        pts = affine.get_trans()[..., None, :, :]
        return affine.invert()[..., None].apply(pts)

    sq_diff = torch.sum(
        (_points(pred_affine) - _points(backbone_rigid)) ** 2, dim=-1
    )

    sq_diff = sq_diff.detach()

    boundaries = torch.linspace(
        0, max_bin, steps=(no_bins - 1), device=logits.device
    )
    boundaries = boundaries ** 2
    true_bins = torch.sum(sq_diff[..., None] > boundaries, dim=-1)

    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_bins, no_bins)
    )

    square_mask = (
        mask[..., None] * mask[..., None, :]
    )

    loss = torch.sum(errors * square_mask, dim=-1)
    scale = 0.5  # hack to help FP16 training along
    denom = eps + torch.sum(scale * square_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    # Average over the batch dimension
    loss = torch.mean(loss)

    return loss


def between_residue_bond_loss(
    pred_atom_positions: torch.Tensor,  # (*, N, 37/14, 3)
    pred_atom_mask: torch.Tensor,  # (*, N, 37/14)
    residx: torch.Tensor,  # (*, N)
    seq: torch.Tensor,  # (*, N)
    tolerance_factor_soft=12.0,
    tolerance_factor_hard=12.0,
    eps=1e-6,
) -> Dict[str, torch.Tensor]:
    """Flat-bottom loss to penalize structural violations between residues.

    This is a loss penalizing any violation of the geometry around the peptide
    bond between consecutive amino acids. This loss corresponds to
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 44, 45.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residx: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      seq: Amino acid type of given residue
      tolerance_factor_soft: soft tolerance factor measured in standard deviations
        of pdb distributions
      tolerance_factor_hard: hard tolerance factor measured in standard deviations
        of pdb distributions

    Returns:
      Dict containing:
        * 'c_n_loss_mean': Loss for peptide bond length violations
        * 'ca_c_n_loss_mean': Loss for violations of bond angle around C spanned
            by CA, C, N
        * 'c_n_ca_loss_mean': Loss for violations of bond angle around N spanned
            by C, N, CA
        * 'per_residue_loss_sum': sum of all losses for each residue
        * 'per_residue_violation_mask': mask denoting all residues with violation
            present.
    """
    
    # Get the positions of the relevant backbone atoms.
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    this_c_pos = pred_atom_positions[..., :-1, 2, :]
    this_c_mask = pred_atom_mask[..., :-1, 2]
    next_n_pos = pred_atom_positions[..., 1:, 0, :]
    next_n_mask = pred_atom_mask[..., 1:, 0]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residx[..., 1:] - residx[..., :-1]) == 1.0

    # Compute loss for the C--N bond.
    c_n_bond_length = torch.sqrt(
        eps + torch.sum((this_c_pos - next_n_pos) ** 2, dim=-1)
    )

    # The C-N bond to proline has slightly different length because of the ring.
    next_is_proline = seq[..., 1:] == residue_constants.resname_to_idx["PRO"]
    gt_length = (
                    ~next_is_proline
                ) * residue_constants.between_res_bond_length_c_n[
                    0
                ] + next_is_proline * residue_constants.between_res_bond_length_c_n[
                    1
                ]
    gt_stddev = (
                    ~next_is_proline
                ) * residue_constants.between_res_bond_length_stddev_c_n[
                    0
                ] + next_is_proline * residue_constants.between_res_bond_length_stddev_c_n[
                    1
                ]
    c_n_bond_length_error = torch.sqrt(eps + (c_n_bond_length - gt_length) ** 2)
    c_n_loss_per_residue = torch.nn.functional.relu(
        c_n_bond_length_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * has_no_gap_mask
    c_n_loss = torch.sum(mask * c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_violation_mask = mask * (
        c_n_bond_length_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute loss for the angles.
    ca_c_bond_length = torch.sqrt(
        eps + torch.sum((this_ca_pos - this_c_pos) ** 2, dim=-1)
    )
    n_ca_bond_length = torch.sqrt(
        eps + torch.sum((next_n_pos - next_ca_pos) ** 2, dim=-1)
    )

    c_ca_unit_vec = (this_ca_pos - this_c_pos) / ca_c_bond_length[..., None]
    c_n_unit_vec = (next_n_pos - this_c_pos) / c_n_bond_length[..., None]
    n_ca_unit_vec = (next_ca_pos - next_n_pos) / n_ca_bond_length[..., None]

    ca_c_n_cos_angle = torch.sum(c_ca_unit_vec * c_n_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_ca_c_n[0]
    gt_stddev = residue_constants.between_res_bond_length_stddev_c_n[0]
    ca_c_n_cos_angle_error = torch.sqrt(
        eps + (ca_c_n_cos_angle - gt_angle) ** 2
    )
    ca_c_n_loss_per_residue = torch.nn.functional.relu(
        ca_c_n_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_ca_mask * this_c_mask * next_n_mask * has_no_gap_mask
    ca_c_n_loss = torch.sum(mask * ca_c_n_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    ca_c_n_violation_mask = mask * (
        ca_c_n_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    c_n_ca_cos_angle = torch.sum((-c_n_unit_vec) * n_ca_unit_vec, dim=-1)
    gt_angle = residue_constants.between_res_cos_angles_c_n_ca[0]
    gt_stddev = residue_constants.between_res_cos_angles_c_n_ca[1]
    c_n_ca_cos_angle_error = torch.sqrt(
        eps + torch.square(c_n_ca_cos_angle - gt_angle)
    )
    c_n_ca_loss_per_residue = torch.nn.functional.relu(
        c_n_ca_cos_angle_error - tolerance_factor_soft * gt_stddev
    )
    mask = this_c_mask * next_n_mask * next_ca_mask * has_no_gap_mask

    c_n_ca_loss = torch.sum(mask * c_n_ca_loss_per_residue, dim=-1) / (
        torch.sum(mask, dim=-1) + eps
    )
    c_n_ca_violation_mask = mask * (
        c_n_ca_cos_angle_error > (tolerance_factor_hard * gt_stddev)
    )

    # Compute a per residue loss (equally distribute the loss to both
    # neighbouring residues).
    per_residue_loss_sum = (
        c_n_loss_per_residue + ca_c_n_loss_per_residue + c_n_ca_loss_per_residue
    )
    per_residue_loss_sum = 0.5 * (
        torch.nn.functional.pad(per_residue_loss_sum, (0, 1))
        + torch.nn.functional.pad(per_residue_loss_sum, (1, 0))
    )

    # Compute hard violations.
    violation_mask = torch.max(
        torch.stack(
            [c_n_violation_mask, ca_c_n_violation_mask, c_n_ca_violation_mask],
            dim=-2,
        ),
        dim=-2,
    )[0]
    violation_mask = torch.maximum(
        torch.nn.functional.pad(violation_mask, (0, 1)),
        torch.nn.functional.pad(violation_mask, (1, 0)),
    )

    return {
        "c_n_loss_mean": c_n_loss,
        "ca_c_n_loss_mean": ca_c_n_loss,
        "c_n_ca_loss_mean": c_n_ca_loss,
        "per_residue_loss_sum": per_residue_loss_sum,
        "per_residue_violation_mask": violation_mask,
    }


def between_residue_clash_loss(
    final_atom14_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_atom_radius: torch.Tensor,
    residx: torch.Tensor,
    asym_id: Optional[torch.Tensor] = None,
    overlap_tolerance_soft=1.5,
    overlap_tolerance_hard=1.5,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes between residues.

    This is a loss penalizing any steric clashes due to non bonded atoms in
    different peptides coming too close. This loss corresponds to the part with
    different residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
      final_atom14_positions: Predicted positions of atoms in
        global prediction frame
      atom14_atom_exists: Mask denoting whether atom at positions exists for given
        amino acid type
      atom14_atom_radius: Van der Waals radius for each atom.
      residx: Residue index for given amino acid.
      overlap_tolerance_soft: Soft tolerance factor.
      overlap_tolerance_hard: Hard tolerance factor.

    Returns:
      Dict containing:
        * 'mean_loss': average clash loss
        * 'per_atom_loss_sum': sum of all clash losses per atom, shape (N, 14)
        * 'per_atom_clash_mask': mask whether atom clashes with any other atom
            shape (N, 14)
    """
    fp_type = final_atom14_positions.dtype

    # Create the distance matrix.
    # (N, N, 14, 14)
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                final_atom14_positions[..., :, None, :, None, :]
                - final_atom14_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Create the mask for valid distances.
    # shape (N, N, 14, 14)
    dists_mask = (
        atom14_atom_exists[..., :, None, :, None]
        * atom14_atom_exists[..., None, :, None, :]
    ).type(fp_type)

    # Mask out all the duplicate entries in the lower triangular matrix.
    # Also mask out the diagonal (atom-pairs from the same residue) -- these atoms
    # are handled separately.
    dists_mask = dists_mask * (
        residx[..., :, None, None, None]
        < residx[..., None, :, None, None]
    )

    # Backbone C--N bond between subsequent residues is no clash.
    c_one_hot = torch.nn.functional.one_hot(
        residx.new_tensor(2, dtype=torch.long), num_classes=14
    )
    c_one_hot = c_one_hot.reshape(
        *((1,) * len(residx.shape[:-1])), *c_one_hot.shape
    )
    c_one_hot = c_one_hot.type(fp_type)

    n_one_hot = torch.nn.functional.one_hot(
        residx.new_tensor(0, dtype=torch.long), num_classes=14
    )
    n_one_hot = n_one_hot.reshape(
        *((1,) * len(residx.shape[:-1])), *n_one_hot.shape
    )
    n_one_hot = n_one_hot.type(fp_type)

    neighbour_mask = (residx[..., :, None] + 1) == residx[..., None, :]

    if asym_id is not None:
        neighbour_mask = neighbour_mask & (asym_id[..., :, None] == asym_id[..., None, :])

    neighbour_mask = neighbour_mask[..., None, None]

    c_n_bonds = (
        neighbour_mask
        * c_one_hot[..., None, None, :, None]
        * n_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - c_n_bonds)

    # Disulfide bridge between two cysteines is no clash.
    cys = residue_constants.restype_name_to_atom14_names["CYS"]
    cys_sg_idx_val = cys.index("SG")

    cys_sg_idx = residx.new_tensor(cys_sg_idx_val, dtype=torch.long)
    cys_sg_idx = cys_sg_idx.reshape(
        *((1,) * len(residx.shape[:-1])), 1
    ).squeeze(-1)
    cys_sg_one_hot = torch.nn.functional.one_hot(cys_sg_idx, num_classes=14).type(fp_type)

    disulfide_bonds = (
        cys_sg_one_hot[..., None, None, :, None]
        * cys_sg_one_hot[..., None, None, None, :]
    )
    dists_mask = dists_mask * (1.0 - disulfide_bonds)

    # Compute the lower bound for the allowed distances.
    # shape (N, N, 14, 14)
    dists_lower_bound = dists_mask * (
        atom14_atom_radius[..., :, None, :, None]
        + atom14_atom_radius[..., None, :, None, :]
    )

    # Compute the error.
    # shape (N, N, 14, 14)
    dists_to_low_error = dists_mask * torch.nn.functional.relu(
        dists_lower_bound - overlap_tolerance_soft - dists
    )

    # Compute the mean loss.
    # shape ()
    mean_loss = torch.sum(dists_to_low_error) / (1e-6 + torch.sum(dists_mask))

    # Compute the per atom loss sum.
    # shape (N, 14)
    per_atom_loss_sum = torch.sum(dists_to_low_error, dim=(-4, -2)) + torch.sum(
        dists_to_low_error, dim=(-3, -1)
    )

    # Compute the hard clash mask.
    # shape (N, N, 14, 14)
    clash_mask = dists_mask * (
        dists < (dists_lower_bound - overlap_tolerance_hard)
    )

    per_atom_num_clash = torch.sum(clash_mask, dim=(-4, -2)) + torch.sum(
        clash_mask, dim=(-3, -1)
    )

    # Compute the per atom clash.
    # shape (N, 14)
    per_atom_clash_mask = torch.maximum(
        torch.amax(clash_mask, dim=(-4, -2)),
        torch.amax(clash_mask, dim=(-3, -1)),
    )

    return {
        "mean_loss": mean_loss,  # shape ()
        "per_atom_loss_sum": per_atom_loss_sum,  # shape (N, 14)
        "per_atom_clash_mask": per_atom_clash_mask,  # shape (N, 14)
        "per_atom_num_clash": per_atom_num_clash,  # shape (N, 14)
    }

def within_residue_violations(
    final_atom14_positions: torch.Tensor,
    atom14_atom_exists: torch.Tensor,
    atom14_dists_lower_bound: torch.Tensor,
    atom14_dists_upper_bound: torch.Tensor,
    tighten_bounds_for_loss=0.0,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """Loss to penalize steric clashes within residues.

    This is a loss penalizing any steric violations or clashes of non-bonded atoms
    in a given peptide. This loss corresponds to the part with
    the same residues of
    Jumper et al. (2021) Suppl. Sec. 1.9.11, eq 46.

    Args:
        final_atom14_positions ([*, N, 14, 3]):
            Predicted positions of atoms in global prediction frame.
        atom14_atom_exists ([*, N, 14]):
            Mask denoting whether atom at positions exists for given
            amino acid type
        atom14_dists_lower_bound ([*, N, 14]):
            Lower bound on allowed distances.
        atom14_dists_upper_bound ([*, N, 14]):
            Upper bound on allowed distances
        tighten_bounds_for_loss ([*, N]):
            Extra factor to tighten loss

    Returns:
      Dict containing:
        * 'per_atom_loss_sum' ([*, N, 14]):
              sum of all clash losses per atom, shape
        * 'per_atom_clash_mask' ([*, N, 14]):
              mask whether atom clashes with any other atom shape
    """
    # Compute the mask for each residue.
    dists_masks = 1.0 - torch.eye(14, device=atom14_atom_exists.device)[None]
    dists_masks = dists_masks.reshape(
        *((1,) * len(atom14_atom_exists.shape[:-2])), *dists_masks.shape
    )
    dists_masks = (
        atom14_atom_exists[..., :, :, None]
        * atom14_atom_exists[..., :, None, :]
        * dists_masks
    )


    # Distance matrix
    dists = torch.sqrt(
        eps
        + torch.sum(
            (
                final_atom14_positions[..., :, :, None, :]
                - final_atom14_positions[..., :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    # Compute the loss.
    dists_to_low_error = torch.nn.functional.relu(
        atom14_dists_lower_bound + tighten_bounds_for_loss - dists
    )
    dists_to_high_error = torch.nn.functional.relu(
        dists - (atom14_dists_upper_bound - tighten_bounds_for_loss)
    )
    loss = dists_masks * (dists_to_low_error + dists_to_high_error)

    # Compute the per atom loss sum.
    per_atom_loss_sum = torch.sum(loss, dim=-2) + torch.sum(loss, dim=-1)

    # Compute the violations mask.
    violations = dists_masks * (
        (dists < atom14_dists_lower_bound) | (dists > atom14_dists_upper_bound)
    )

    per_atom_num_clash = torch.sum(violations, dim=-2) + torch.sum(violations, dim=-1)

    # Compute the per atom violations.
    per_atom_violations = torch.maximum(
        torch.max(violations, dim=-2)[0], torch.max(violations, axis=-1)[0]
    )

    return {
        "per_atom_loss_sum": per_atom_loss_sum,
        "per_atom_violations": per_atom_violations,
        "per_atom_num_clash": per_atom_num_clash
    }


def find_structural_violations(
    batch: Dict[str, torch.Tensor],
    final_atom14_positions: torch.Tensor,
    violation_tolerance_factor: float,
    clash_overlap_tolerance: float,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """Computes several checks for structural violations."""
    #import ipdb; ipdb.set_trace()
    # Compute between residue backbone violations of bonds and angles.
    connection_violations = between_residue_bond_loss(
        pred_atom_positions=final_atom14_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residx=batch["residx"],
        seq=batch["seq"],
        tolerance_factor_soft=violation_tolerance_factor,
        tolerance_factor_hard=violation_tolerance_factor,
    )

    # Compute the Van der Waals radius for every atom
    # (the first letter of the atom name is the element type).
    # Shape: (N, 14).
    atomtype_radius = [
        residue_constants.van_der_waals_radius[name[0]]
        for name in residue_constants.atom_types
    ]

    atomtype_radius = final_atom14_positions.new_tensor(atomtype_radius)

    # TODO: Consolidate monomer/multimer modes
    asym_id = batch.get("asym_id")
    if asym_id is not None:
        residx_atom14_to_atom37 = get_rc_tensor(
            residue_constants.RESTYPE_ATOM14_TO_ATOM37, batch["seq"]
        )
        atom14_atom_radius = (
            batch["atom14_atom_exists"]
            * atomtype_radius[residx_atom14_to_atom37.long()]
        )
    else:
        atom14_atom_radius = (
            batch["atom14_atom_exists"]
            * atomtype_radius[batch["residx_atom14_to_atom37"]]
        )

    # Compute the between residue clash loss.
    between_residue_clashes = between_residue_clash_loss(
        final_atom14_positions=final_atom14_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_atom_radius=atom14_atom_radius,
        residx=batch["residx"],
        asym_id=asym_id,
        overlap_tolerance_soft=clash_overlap_tolerance,
        overlap_tolerance_hard=clash_overlap_tolerance,
    )

    # Compute all within-residue violations (clashes,
    # bond length and angle violations).
    restype_atom14_bounds = residue_constants.make_atom14_dists_bounds(
        overlap_tolerance=clash_overlap_tolerance,
        bond_length_tolerance_factor=violation_tolerance_factor,
    )
    atom14_atom_exists = batch["atom14_atom_exists"]
    atom14_dists_lower_bound = final_atom14_positions.new_tensor(
        restype_atom14_bounds["lower_bound"]
    )[batch["seq"]]
    atom14_dists_upper_bound = final_atom14_positions.new_tensor(
        restype_atom14_bounds["upper_bound"]
    )[batch["seq"]]
    residue_violations = within_residue_violations(
        final_atom14_positions=final_atom14_positions,
        atom14_atom_exists=batch["atom14_atom_exists"],
        atom14_dists_lower_bound=atom14_dists_lower_bound,
        atom14_dists_upper_bound=atom14_dists_upper_bound,
        tighten_bounds_for_loss=0.0,
    )

    # Combine them to a single per-residue violation mask (used later for LDDT).
    per_residue_violations_mask = torch.max(
        torch.stack(
            [
                connection_violations["per_residue_violation_mask"],
                torch.max(
                    between_residue_clashes["per_atom_clash_mask"], dim=-1
                )[0],
                torch.max(residue_violations["per_atom_violations"], dim=-1)[0],
            ],
            dim=-1,
        ),
        dim=-1,
    )[0]

    return {
        "between_residues": {
            "bonds_c_n_loss_mean": connection_violations["c_n_loss_mean"],  # ()
            "angles_ca_c_n_loss_mean": connection_violations[
                "ca_c_n_loss_mean"
            ],  # ()
            "angles_c_n_ca_loss_mean": connection_violations[
                "c_n_ca_loss_mean"
            ],  # ()
            "connections_per_residue_loss_sum": connection_violations[
                "per_residue_loss_sum"
            ],  # (N)
            "connections_per_residue_violation_mask": connection_violations[
                "per_residue_violation_mask"
            ],  # (N)
            "clashes_mean_loss": between_residue_clashes["mean_loss"],  # ()
            "clashes_per_atom_loss_sum": between_residue_clashes[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "clashes_per_atom_clash_mask": between_residue_clashes[
                "per_atom_clash_mask"
            ],  # (N, 14)
            "clashes_per_atom_num_clash": between_residue_clashes[
                "per_atom_num_clash"
            ],  # (N, 14)
        },
        "within_residues": {
            "per_atom_loss_sum": residue_violations[
                "per_atom_loss_sum"
            ],  # (N, 14)
            "per_atom_violations": residue_violations[
                "per_atom_violations"
            ],  # (N, 14),
            "per_atom_num_clash": residue_violations[
                "per_atom_num_clash"
            ],  # (N, 14)
        },
        "total_per_residue_violations_mask": per_residue_violations_mask,  # (N)
    }


def find_structural_violations_np(
    batch: Dict[str, np.ndarray],
    final_atom14_positions: np.ndarray,
    config: ml_collections.ConfigDict,
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    final_atom14_positions = to_tensor(final_atom14_positions)

    out = find_structural_violations(batch, final_atom14_positions, **config)

    to_np = lambda x: np.array(x)
    np_out = tensor_tree_map(to_np, out)

    return np_out


def extreme_ca_ca_distance_violations(
    pred_atom_positions: torch.Tensor,  # (N, 37(14), 3)
    pred_atom_mask: torch.Tensor,  # (N, 37(14))
    residx: torch.Tensor,  # (N)
    max_angstrom_tolerance=1.5,
    eps=1e-6,
) -> torch.Tensor:
    """Counts residues whose Ca is a large distance from its neighbour.

    Measures the fraction of CA-CA pairs between consecutive amino acids that are
    more than 'max_angstrom_tolerance' apart.

    Args:
      pred_atom_positions: Atom positions in atom37/14 representation
      pred_atom_mask: Atom mask in atom37/14 representation
      residx: Residue index for given amino acid, this is assumed to be
        monotonically increasing.
      max_angstrom_tolerance: Maximum distance allowed to not count as violation.
    Returns:
      Fraction of consecutive CA-CA pairs with violation.
    """
    this_ca_pos = pred_atom_positions[..., :-1, 1, :]
    this_ca_mask = pred_atom_mask[..., :-1, 1]
    next_ca_pos = pred_atom_positions[..., 1:, 1, :]
    next_ca_mask = pred_atom_mask[..., 1:, 1]
    has_no_gap_mask = (residx[..., 1:] - residx[..., :-1]) == 1.0
    ca_ca_distance = torch.sqrt(
        eps + torch.sum((this_ca_pos - next_ca_pos) ** 2, dim=-1)
    )
    violations = (
                     ca_ca_distance - residue_constants.ca_ca
                 ) > max_angstrom_tolerance
    mask = this_ca_mask * next_ca_mask * has_no_gap_mask
    mean = masked_mean(mask, violations, -1)
    return mean


def compute_violation_metrics(
    batch: Dict[str, torch.Tensor],
    final_atom14_positions: torch.Tensor,  # (N, 14, 3)
    violations: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Compute several metrics to assess the structural violations."""
    ret = {}
    extreme_ca_ca_violations = extreme_ca_ca_distance_violations(
        pred_atom_positions=final_atom14_positions,
        pred_atom_mask=batch["atom14_atom_exists"],
        residx=batch["residx"],
    )
    ret["violations_extreme_ca_ca_distance"] = extreme_ca_ca_violations
    ret["violations_between_residue_bond"] = masked_mean(
        batch["seq_mask"],
        violations["between_residues"][
            "connections_per_residue_violation_mask"
        ],
        dim=-1,
    )
    ret["violations_between_residue_clash"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["between_residues"]["clashes_per_atom_clash_mask"],
            dim=-1,
        )[0],
        dim=-1,
    )
    ret["violations_within_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=torch.max(
            violations["within_residues"]["per_atom_violations"], dim=-1
        )[0],
        dim=-1,
    )
    ret["violations_per_residue"] = masked_mean(
        mask=batch["seq_mask"],
        value=violations["total_per_residue_violations_mask"],
        dim=-1,
    )
    return ret


def compute_violation_metrics_np(
    batch: Dict[str, np.ndarray],
    final_atom14_positions: np.ndarray,
    violations: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    to_tensor = lambda x: torch.tensor(x)
    batch = tree_map(to_tensor, batch, np.ndarray)
    final_atom14_positions = to_tensor(final_atom14_positions)
    violations = tree_map(to_tensor, violations, np.ndarray)

    out = compute_violation_metrics(batch, final_atom14_positions, violations)

    to_np = lambda x: np.array(x)
    return tree_map(to_np, out, torch.Tensor)


def violation_loss(
    violations: Dict[str, torch.Tensor],
    atom14_atom_exists: torch.Tensor,
    fixed_mask: torch.Tensor,
    average_clashes: bool = False,
    eps=1e-6,
    **kwargs,
) -> torch.Tensor:
    
    # diffuse_mask (1 for CDR, 0 for framework)
    diffuse_mask = (1 - fixed_mask).float() # [*, N]

    # 理由: 我们虽然全局计算了violation, 但在计算最终loss时，
    # 我们更关心发生在CDR区域的原子或与CDR区域相邻的原子产生的violation。
    # 这里我们对per-atom的clash loss进行加权。

    # 计算总的原子clash loss
    per_atom_clash_loss_sum = (
        violations["between_residues"]["clashes_per_atom_loss_sum"] +
        violations["within_residues"]["per_atom_loss_sum"]
    ) # Shape: [*, N, 14]

    # 只关注CDR区域原子的clash loss
    cdr_clash_loss = per_atom_clash_loss_sum * diffuse_mask[..., None]
    
    # 归一化：用总原子数
    num_total_atoms = torch.sum(atom14_atom_exists) + eps
    l_clash = torch.sum(cdr_clash_loss) / num_total_atoms

    # 键长键角loss是per-residue-pair的，它们本身已经是平均值，可以直接使用
    l_bond_angle = (
        violations["between_residues"]["bonds_c_n_loss_mean"] +
        violations["between_residues"]["angles_ca_c_n_loss_mean"] +
        violations["between_residues"]["angles_c_n_ca_loss_mean"]
    )

    loss = l_bond_angle + l_clash

    # 对 batch 维度求平均
    mean = torch.mean(loss)

    return mean


def elbo_ctmc_loss(
    logits: torch.Tensor,      # [B, N, S]  sequence head logits, p_theta(x0 | x_t)
    x_t: torch.Tensor,         # [B, N]     batch['seq_t']，当前 CTMC 状态 a
    qt0: torch.Tensor,         # [B, S, S]  q_t(x | x0)，行 x0, 列 x
    rate_matrix: torch.Tensor, # [B, S, S]  Q_t(a -> b)，行 from, 列 to
    mask: torch.Tensor,        # [B, N]
    fixed_mask: torch.Tensor,  # [B, N]     1=框架/抗原, 0=CDR
    eps: float = 1e-9,
    rate_normalize: bool = True,
) -> torch.Tensor:
    r"""
    CTMC ELBO (direction-correct, one-forward-pass, ABX-adapted).

    理论目标（对每个位置 a = x_t[i]）:
        L(a) =  ∑_{b≠a}  Ŝ(a -> b)
              - Z_t(a) * E_{b ~ r_t(·|a)} [ log Ŝ(b -> a) ]

      - Term1: outflow, 使用 Ŝ(a -> b)
      - Term2: inflow, 使用 Ŝ(b -> a)，方向与 τLDR / ABX 公式一致
      - 只做一次 forward：所有 Ŝ(· -> ·) 都用 p_theta(x0 | a) 构造
    """
    B, N, S = logits.shape
    device = logits.device

    # -------- 0. 预处理 --------
    x_t = torch.clamp(x_t, 0, S - 1).long()
    mask = mask.float()
    fixed_mask = fixed_mask.float()

    # p_theta(x0 | a)
    p0t = F.softmax(logits, dim=-1)              # [B, N, S_x0]

    # 索引辅助
    idx_b  = torch.arange(B, device=device).view(B, 1, 1)  # [B,1,1]
    idx_x0 = torch.arange(S, device=device).view(1, 1, S)  # [1,1,S]
    idx_xt = x_t.unsqueeze(-1)                               # [B,N,1]

    # q_t(a | x0) = qt0[b, x0, a]
    q_a_given_x0 = qt0[idx_b, idx_x0, idx_xt]               # [B, N, S_x0]
    q_a_given_x0 = torch.clamp(q_a_given_x0, min=eps)

    # Q_t(a -> b) / Q_t(b -> a)
    idx_b_row = torch.arange(B, device=device).view(B, 1)   # [B,1]
    forward_row = rate_matrix[idx_b_row, x_t]               # [B, N, S]  Q(a->b)
    rate_T      = rate_matrix.transpose(1, 2)               # [B, S, S]
    forward_col = rate_T[idx_b_row, x_t]                    # [B, N, S]  Q(b->a)

    # =====================================================
    # 1. Term1: ∑_b Ŝ(a -> b)
    # Ŝ(a -> b) = Q(b->a) * Σ_{x0} p(x0|a)/q(a|x0) * q(b|x0)
    # =====================================================
    ratio_1    = p0t / q_a_given_x0                         # [B, N, S_x0]
    inner_sum1 = torch.matmul(ratio_1, qt0)                 # [B, N, S]
    S_hat_out  = forward_col * inner_sum1                   # [B, N, S]
    S_hat_out  = torch.clamp(S_hat_out, min=0.0)
    S_hat_out.scatter_(2, x_t.unsqueeze(-1), 0.0)           # 去掉 a->a
    term1      = S_hat_out.sum(dim=2)                       # [B, N]

    # =====================================================
    # 2. Term2: Z_t(a) * E_{b~r_t} [ log Ŝ(b -> a) ]
    # Ŝ(b -> a) ≈ Q(a->b) * Σ_{x0} p(x0|a)*q(a|x0)/q(b|x0)
    # =====================================================
    num_2   = p0t * q_a_given_x0                            # [B, N, S_x0]
    inv_qt0 = 1.0 / torch.clamp(qt0, min=eps)               # [B, S_x0, S]
    inner2  = torch.matmul(num_2, inv_qt0)                  # [B, N, S]

    S_hat_in = forward_row * inner2                         # [B, N, S]
    S_hat_in = torch.clamp(S_hat_in, min=1e-8)              # 防止 log(0)

    # 前向分布 r_t(b|a)
    forward_row_no_diag = forward_row.clone()
    forward_row_no_diag.scatter_(2, x_t.unsqueeze(-1), 0.0)
    Z_t = forward_row_no_diag.sum(dim=2) + eps              # [B, N]

    r_probs = forward_row_no_diag / Z_t.unsqueeze(-1)       # [B, N, S]
    r_probs = torch.clamp(r_probs, min=0.0)
    r_probs = r_probs / (r_probs.sum(dim=-1, keepdim=True) + 1e-12)

    log_S_hat_in = torch.log(S_hat_in)                      # [B, N, S]
    term2 = Z_t * torch.sum(r_probs * log_S_hat_in, dim=-1) # [B, N]

    # -------- 3. 组合 & 归一化 --------
    raw_loss = term1 - term2                                # [B, N]

    if rate_normalize:
        per_pos_loss = raw_loss / (Z_t.detach() + eps)
    else:
        per_pos_loss = raw_loss

    # 只在 CDR 上做平均
    cdr_mask = (1.0 - fixed_mask) * mask                    # [B, N]
    loss = (per_pos_loss * cdr_mask).sum() / (cdr_mask.sum() + eps)

    return loss





def ce_loss(
    logits: torch.Tensor,        # 模型预测的logits，形状: [B, N, C]
    target: torch.Tensor,        # 目标序列，形状: [B, N]
    fixed_mask: torch.Tensor,      # CDR区域掩码，形状: [B, N]，1表示CDR区域，0表示非CDR区域
    mask: torch.Tensor,      # CDR区域掩码，形状: [B, N]，1表示CDR区域，0表示非CDR区域
    eps: float = 1e-8            # 数值稳定性常数
) -> torch.Tensor:
    """
    计算CDR区域的交叉熵损失。
    
    Args:
        logits: 模型输出的logits，形状为[批次大小, 序列长度, 类别数]
        target: 目标氨基酸序列索引，形状为[批次大小, 序列长度]
        cdr_mask: CDR区域掩码，形状为[批次大小, 序列长度]，1表示CDR区域，0表示非CDR区域
        weight: 损失权重系数
        eps: 防止除零的小常数
        
    Returns:
        torch.Tensor: CDR区域平均交叉熵损失
    """
    
    # 确保target是长整型

    target = target.long() * mask
    logits = logits * mask[..., None]
    
    # 获取形状信息
    batch_size, seq_len, num_classes = logits.shape
    
    # 确保cdi_mask的形状为 [B, N, 1]
    cdr_mask = 1 - fixed_mask
    
    # 重塑logits以适应F.cross_entropy
    flat_logits = logits.reshape(-1, num_classes)
    
    # 重塑target
    flat_target = target.reshape(-1)
    
    # 计算交叉熵损失
    per_position_loss = F.cross_entropy(flat_logits, flat_target, reduction='none')
    
    # 将损失重塑回原始序列形状 [B*N,] -> [B, N]
    per_position_loss = per_position_loss.reshape(batch_size, seq_len)
    
    # 应用 CDR 区域掩码
    masked_loss = per_position_loss * cdr_mask
    
    # 计算CDR区域的平均损失
    cdr_count = torch.sum(cdr_mask) + eps
    cdr_loss = torch.sum(masked_loss) / cdr_count

    return cdr_loss

def DSM_loss(
    pred_rot_score: torch.Tensor,        # 模型预测的logits，形状: [B, N, C]
    pred_trans_score: torch.Tensor,        # 目标序列，形状: [B, N]
    gt_rot_score: torch.Tensor,        # 目标序列，形状: [B, N]
    gt_trans_score: torch.Tensor,        # 目标序列，形状: [B, N]
    rigids_mask: torch.Tensor,        # 目标序列，形状: [B, N]
    rigids_t: torch.Tensor,        # 目标序列，形状: [B, N]
    t: torch.Tensor,        # 目标序列，形状: [B, N]
    rot_score_scaling: torch.Tensor,        # 目标序列，形状: [B, N]
    trans_score_scaling: torch.Tensor,        # 目标序列，形状: [B, N]
    eps: float = 1e-8,
    **kwargs,# 数值稳定性常数
) -> torch.Tensor:
                # ===== 1. 扩散损失 (L_Diff) =====
    # 1.1 L_DSM (旋转和平移损失)
    # pred_rot_score = outputs['heads']['folding']['rot_score']
    # pred_trans_score = outputs['heads']['folding']['trans_score']
    # gt_rot_score = batch['rot_score']
    # gt_trans_score = batch['trans_score']
    # rigids_mask = batch['fixed_mask']
    diffuse_mask = (1.0 - rigids_mask)
    
    # 旋转损失：分解为轴和角度
    gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
    gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-6)
    pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
    pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-6)
    
    # 轴损失
    rot_axis_loss = torch.sum(
        (gt_rot_axis - pred_rot_axis)**2 * diffuse_mask[..., None],
        dim=(-2, -1)
    ) / (diffuse_mask.sum(dim=-1) + 1e-6)
    
    # 角度损失
    rot_angle_loss = torch.sum(
        (gt_rot_angle - pred_rot_angle)**2 * diffuse_mask[..., None] / 
        rot_score_scaling[:, None, None]**2,
        dim=(-2, -1)
    ) / (diffuse_mask.sum(dim=-1) + 1e-6)
    rot_angle_loss *= t > kwargs["rot_angle_loss_t_filter"]
    # rot_angle_loss = rot_angle_loss * (t > 0.5).float()
    
    rot_loss = rot_axis_loss + rot_angle_loss
    
    # 平移损失
    trans_mse = (gt_trans_score - pred_trans_score) ** 2 * diffuse_mask[...,None]
    trans_loss = torch.sum(
        trans_mse / trans_score_scaling[:, None, None]**2,
        dim=(-2, -1)
    ) / (diffuse_mask.sum(dim=-1) + 1e-6)
    
    l_dsm = rot_loss + trans_loss
    
    l_dsm = torch.mean(l_dsm)
    return l_dsm


def dsm_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict
) -> torch.Tensor:
    """
    计算刚体（Rigids）的去噪得分匹配（DSM）损失。
    ✅ 该版本修正了广播操作中的维度不匹配问题。
    """
    # 1. 准备数据和掩码
    t = batch['t']
    diffuse_mask = (1.0 - batch["fixed_mask"]) * batch["mask"] # CDR区域掩码
    num_total_res = batch["mask"].sum(dim=-1).clamp(min=1.0)

    # 2. 平移损失 (混合策略)
    gt_trans_score = batch['trans_score']
    pred_trans_score = out['heads']['folding']['trans_score']
    trans_score_scaling = batch['trans_score_scaling']
    
    # ✅ 关键修正：使用 .unsqueeze(-1).unsqueeze(-1) 或更清晰的 [:, None, None] 语法
    # 将 [B] 的缩放因子变为 [B, 1, 1] 以正确广播
    trans_score_mse_sum = torch.sum(
        (gt_trans_score - pred_trans_score)**2 * diffuse_mask[..., None] / trans_score_scaling[:, None, None]**2,
        dim=(-1, -2)
    )
    trans_score_loss = trans_score_mse_sum / num_total_res

    gt_trans_x0 = batch['rigids_0'][..., 4:]
    pred_trans_x0 = out['heads']['folding']['rigids'][..., 4:]
    trans_x0_mse_sum = torch.sum((gt_trans_x0 - pred_trans_x0)**2 * diffuse_mask[..., None], dim=(-1, -2))
    trans_x0_loss = trans_x0_mse_sum / num_total_res
    
    trans_loss = torch.where(
        t > config.trans_x0_t_threshold, trans_score_loss, trans_x0_loss
    ) * config.trans_loss_weight

    # 3. 旋转损失 (分解策略)
    gt_rot_score = batch['rot_score']
    pred_rot_score = out['heads']['folding']['rot_score']
    rot_score_scaling = batch['rot_score_scaling']
    gt_rot_angle = torch.norm(gt_rot_score, dim=-1, keepdim=True)
    gt_rot_axis = gt_rot_score / (gt_rot_angle + 1e-8)
    pred_rot_angle = torch.norm(pred_rot_score, dim=-1, keepdim=True)
    pred_rot_axis = pred_rot_score / (pred_rot_angle + 1e-8)
    
    axis_loss_sum = torch.sum((gt_rot_axis - pred_rot_axis)**2 * diffuse_mask[..., None], dim=(-1, -2))
    axis_loss = axis_loss_sum / num_total_res

    # ✅ 关键修正：同样修正此处的维度
    angle_loss_unscaled_sum = torch.sum(
        (gt_rot_angle - pred_rot_angle)**2 * diffuse_mask[..., None] / rot_score_scaling[:, None, None]**2, 
        dim=(-1, -2)
    )
    angle_loss = angle_loss_unscaled_sum / num_total_res
    angle_loss *= (t > config.rot_loss_t_threshold)
    
    rot_loss = (axis_loss + angle_loss) * config.rot_loss_weight

    # 4. 最终聚合
    total_loss_per_sample = rot_loss + trans_loss
    return torch.mean(total_loss_per_sample)



def backbone_atom_loss(
    out: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    config: ml_collections.ConfigDict
) -> torch.Tensor:
    """
    计算骨架原子的物理约束损失，作为一种辅助正则化。
    ✅ 这是最终修正版，不再依赖复杂的原子构建函数，直接监督模型的核心坐标预测。
    """
    #import ipdb; ipdb.set_trace()
    t = batch['t']
    aatype = batch['seq']
    device = out['heads']['folding']['final_atom_positions'].device

    # --- 步骤 1: 直接从模型输出获取预测的x0原子坐标 ---
    # 我们使用 final_atom_positions，这是模型对结构的最终预测
    pred_atom37_pos = out['heads']['folding']['final_atom_positions']
    # 提取骨架部分 (N, CA, C, O, CB)
    pred_atom37_bb = pred_atom37_pos[..., :5, :]

    # --- 步骤 2: 从batch中获取真值坐标 ---
    gt_atom37_pos = batch['atom37_gt_positions']
    gt_atom37_bb = gt_atom37_pos[..., :5, :]

    # --- 步骤 3: 计算损失 (逻辑保持不变) ---
    diffuse_mask = (1.0 - batch["fixed_mask"]) * batch["mask"]
    
    # 从常量中获取骨架部分的原子是否存在掩码
    gt_mask_bb = torch.from_numpy(residue_constants.restype_atom37_mask[:, :5]).to(device)[aatype]
    
    loss_mask = diffuse_mask[..., None] * gt_mask_bb

    # 计算CDR区域内，所有有效骨架原子的L2损失总和
    bb_atom_l2_sum = torch.sum(
        (pred_atom37_bb - gt_atom37_bb)**2 * loss_mask[..., None],
        dim=(-1, -2, -3)
    )
    
    # 使用整条链上有效的骨架原子总数进行归一化，保证稳定性
    num_total_atoms = (batch["mask"][..., None] * gt_mask_bb).sum(dim=(-1, -2)).clamp(min=1.0)
    bb_atom_loss = bb_atom_l2_sum / num_total_atoms
    
    # 仅在t较小时激活此损失
    bb_atom_loss *= (t < config.get("bb_atom_loss_t_filter", 0.5))
    
    return torch.mean(bb_atom_loss)

def FPE_r3_loss(
    fpe_loss_r3: torch.Tensor,        # 模型预测的logits，形状: [B, N, C]
    global_step,# 目标序列，形状: [B, N]
    **kwargs,# 数值稳定性常数
) -> torch.Tensor:
    
    #损失值裁剪 (Loss Clipping)
    warmup    = 1
    gamma_fpe = min(1.0, global_step / warmup)
    fpe_loss = torch.clamp(gamma_fpe * fpe_loss_r3, max=20.0) # C 是一个超参数，例如1000，防止其过度主导总损失
    return fpe_loss

def FPE_so3_loss(
    fpe_loss_so3: torch.Tensor,        # 模型预测的logits，形状: [B, N, C]
    global_step,# 目标序列，形状: [B, N]
    **kwargs,# 数值稳定性常数
) -> torch.Tensor:
    
    #损失值裁剪 (Loss Clipping)
    warmup    = 1
    gamma_fpe = min(1.0, global_step / warmup)
    fpe_loss = torch.clamp(gamma_fpe * fpe_loss_so3, max=20.0) # C 是一个超参数，例如1000，防止其过度主导总损失
    return fpe_loss




def compute_renamed_ground_truth(
    batch: Dict[str, torch.Tensor],
    final_atom14_positions: torch.Tensor,
    eps=1e-10,
) -> Dict[str, torch.Tensor]:
    """
    Find optimal renaming of ground truth based on the predicted positions.

    Alg. 26 "renameSymmetricGroundTruthAtoms"

    This renamed ground truth is then used for all losses,
    such that each loss moves the atoms in the same direction.

    Args:
      batch: Dictionary containing:
        * atom14_gt_positions: Ground truth positions.
        * atom14_alt_gt_positions: Ground truth positions with renaming swaps.
        * atom14_atom_is_ambiguous: 1.0 for atoms that are affected by
            renaming swaps.
        * atom14_gt_exists: Mask for which atoms exist in ground truth.
        * atom14_alt_gt_exists: Mask for which atoms exist in ground truth
            after renaming.
        * atom14_atom_exists: Mask for whether each atom is part of the given
            amino acid type.
      final_atom14_positions: Array of atom positions in global frame with shape
    Returns:
      Dictionary containing:
        alt_naming_is_better: Array with 1.0 where alternative swap is better.
        renamed_atom14_gt_positions: Array of optimal ground truth positions
          after renaming swaps are performed.
        renamed_atom14_gt_exists: Mask after renaming swap is performed.
    """

    pred_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                final_atom14_positions[..., None, :, None, :]
                - final_atom14_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_gt_positions = batch["atom14_gt_positions"]
    gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_gt_positions[..., None, :, None, :]
                - atom14_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )

    atom14_alt_gt_positions = batch["atom14_alt_gt_positions"]
    alt_gt_dists = torch.sqrt(
        eps
        + torch.sum(
            (
                atom14_alt_gt_positions[..., None, :, None, :]
                - atom14_alt_gt_positions[..., None, :, None, :, :]
            )
            ** 2,
            dim=-1,
        )
    )
    lddt = torch.sqrt(eps + (pred_dists - gt_dists) ** 2)
    alt_lddt = torch.sqrt(eps + (pred_dists - alt_gt_dists) ** 2)

    atom14_gt_exists = batch["atom14_gt_exists"]
    atom14_atom_is_ambiguous = batch["atom14_atom_is_ambiguous"].float()
    mask = (
        atom14_gt_exists[..., None, :, None]
        * atom14_atom_is_ambiguous[..., None, :, None]
        * atom14_gt_exists[..., None, :, None, :]
        * (1.0 - atom14_atom_is_ambiguous[..., None, :, None, :])
    )

    per_res_lddt = torch.sum(mask * lddt, dim=(-1, -2, -3))
    alt_per_res_lddt = torch.sum(mask * alt_lddt, dim=(-1, -2, -3))
    diffuse_mask = 1 - batch["fixed_mask"]
    # if diffuse_mask is not None:
    #     per_res_lddt = per_res_lddt * diffuse_mask
    #     alt_per_res_lddt = alt_per_res_lddt * diffuse_mask
        

    fp_type = final_atom14_positions.dtype
    
    alt_naming_is_better = (alt_per_res_lddt < per_res_lddt).type(fp_type)
    
    renamed_atom14_gt_positions = (
                                      1.0 - alt_naming_is_better[..., None, None]
                                  ) * atom14_gt_positions + alt_naming_is_better[
                                      ..., None, None
                                  ] * atom14_alt_gt_positions

    renamed_atom14_gt_mask = (
                                 1.0 - alt_naming_is_better[..., None]
                             ) * atom14_gt_exists + alt_naming_is_better[..., None] * batch[
                                 "atom14_alt_gt_exists"
                             ]

    return {
        "alt_naming_is_better": alt_naming_is_better,
        "renamed_atom14_gt_positions": renamed_atom14_gt_positions,
        "renamed_atom14_gt_exists": renamed_atom14_gt_mask,
    }


def experimentally_resolved_loss(
    logits: torch.Tensor,
    atom37_atom_exists: torch.Tensor,
    atom37_gt_exists: torch.Tensor,
    resolution: torch.Tensor,
    min_resolution: float,
    max_resolution: float,
    eps: float = 1e-8,
    **kwargs,
) -> torch.Tensor:
    errors = sigmoid_cross_entropy(logits, atom37_gt_exists)
    loss = torch.sum(errors * atom37_atom_exists, dim=-1)
    loss = loss / (eps + torch.sum(atom37_atom_exists, dim=(-1, -2)).unsqueeze(-1))
    loss = torch.sum(loss, dim=-1)

    loss = loss * (
        (resolution >= min_resolution) & (resolution <= max_resolution)
    )

    loss = torch.mean(loss)

    return loss


def masked_msa_loss(logits, true_msa, bert_mask, num_classes, eps=1e-8, **kwargs):
    """
    Computes BERT-style masked MSA loss. Implements subsection 1.9.9.

    Args:
        logits: [*, N_seq, N_res, 23] predicted residue distribution
        true_msa: [*, N_seq, N_res] true MSA
        bert_mask: [*, N_seq, N_res] MSA mask
    Returns:
        Masked MSA loss
    """
    errors = softmax_cross_entropy(
        logits, torch.nn.functional.one_hot(true_msa, num_classes=num_classes)
    )

    # FP16-friendly averaging. Equivalent to:
    # loss = (
    #     torch.sum(errors * bert_mask, dim=(-1, -2)) /
    #     (eps + torch.sum(bert_mask, dim=(-1, -2)))
    # )
    loss = errors * bert_mask
    loss = torch.sum(loss, dim=-1)
    scale = 0.5
    denom = eps + torch.sum(scale * bert_mask, dim=(-1, -2))
    loss = loss / denom[..., None]
    loss = torch.sum(loss, dim=-1)
    loss = loss * scale

    loss = torch.mean(loss)

    return loss


def chain_center_of_mass_loss(
    all_atom_pred_pos: torch.Tensor,
    atom37_gt_positions: torch.Tensor,
    atom37_gt_exists: torch.Tensor,
    asym_id: torch.Tensor,
    clamp_distance: float = -4.0,
    weight: float = 0.05,
    eps: float = 1e-10, **kwargs
) -> torch.Tensor:
    """
    Computes chain centre-of-mass loss. Implements section 2.5, eqn 1 in the Multimer paper.

    Args:
        all_atom_pred_pos:
            [*, N_pts, 37, 3] All-atom predicted atom positions
        atom37_gt_positions:
            [*, N_pts, 37, 3] Ground truth all-atom positions
        atom37_gt_exists:
            [*, N_pts, 37] All-atom positions mask
        asym_id:
            [*, N_pts] Chain asym IDs
        clamp_distance:
            Cutoff above which distance errors are disregarded
        weight:
            Weight for loss
        eps:
            Small value used to regularize denominators
    Returns:
        [*] loss tensor
    """
    ca_pos = residue_constants.atom_order["CA"]
    all_atom_pred_pos = all_atom_pred_pos[..., ca_pos, :]
    atom37_gt_positions = atom37_gt_positions[..., ca_pos, :]
    atom37_gt_exists = atom37_gt_exists[..., ca_pos: (ca_pos + 1)]  # keep dim

    one_hot = torch.nn.functional.one_hot(asym_id.long()).to(dtype=atom37_gt_exists.dtype)
    one_hot = one_hot * atom37_gt_exists
    chain_pos_mask = one_hot.transpose(-2, -1)
    chain_exists = torch.any(chain_pos_mask, dim=-1).to(dtype=atom37_gt_positions.dtype)

    def get_chain_center_of_mass(pos):
        center_sum = (chain_pos_mask[..., None] * pos[..., None, :, :]).sum(dim=-2)
        centers = center_sum / (torch.sum(chain_pos_mask, dim=-1, keepdim=True) + eps)
        return Vec3Array.from_array(centers)

    pred_centers = get_chain_center_of_mass(all_atom_pred_pos)  # [B, NC, 3]
    true_centers = get_chain_center_of_mass(atom37_gt_positions)  # [B, NC, 3]

    pred_dists = euclidean_distance(pred_centers[..., None, :], pred_centers[..., :, None], epsilon=eps)
    true_dists = euclidean_distance(true_centers[..., None, :], true_centers[..., :, None], epsilon=eps)
    losses = torch.clamp((weight * (pred_dists - true_dists - clamp_distance)), max=0) ** 2
    loss_mask = chain_exists[..., :, None] * chain_exists[..., None, :]

    loss = masked_mean(loss_mask, losses, dim=(-1, -2))
    return loss




class AlphaFoldLoss(nn.Module):
    """
    Aggregation of the various losses, with a simplified curriculum learning strategy.
    """
    def __init__(self, config):
        super(AlphaFoldLoss, self).__init__()
        self.config = config

    def loss(self, out, batch, global_step, _return_breakdown=False):
        """
        Main loss aggregation function executing the simplified curriculum.
        """
        # --- 步骤 1: 预计算 (总是需要) ---
        if "violation" not in out.keys():
            out["violation"] = find_structural_violations(
                batch,
                out['heads']['folding']['final_atom14_positions'],
                self.config.folding.config.violation_tolerance_factor,
                self.config.folding.config.clash_overlap_tolerance,
            )
        if "renamed_atom14_gt_positions" not in out.keys():
            batch.update(compute_renamed_ground_truth(batch, out['heads']['folding']['final_atom14_positions']))

        # --- 步骤 2: 定义所有可能的损失函数 ---
          # --- AlphaFold Geometry Losses (全局计算，局部聚焦) ---
        # (这部分与您之前的代码一致)
        loss_fns = {
            "fape": lambda: fape_loss(
            out,
            batch,
            self.config.folding.config.fape,
        ),
            "plddt": lambda: lddt_loss(
            logits=out['heads']['predicted_lddt']["logits"],
            all_atom_pred_pos=out['heads']['folding']['final_atom_positions'],
            atom37_gt_positions=batch["atom37_gt_positions"],
            atom37_gt_exists=batch["atom37_gt_exists"],
            fixed_mask=batch["fixed_mask"],
            **self.config.plddt,
        ),
            "distogram": lambda: distogram_loss(
            logits=out['heads']['distogram']['logits'],
            pseudo_beta=batch['pseudo_beta'],
            pseudo_beta_mask=batch['pseudo_beta_mask'],
            **self.config.distogram,
        ),
            "violation": lambda: violation_loss(
            violations=out["violation"],
            atom14_atom_exists=batch["atom14_atom_exists"],
            fixed_mask=batch["fixed_mask"],
            **self.config.violation,
        ),
            "elbo": lambda: elbo_ctmc_loss(
                 logits=out['heads']['sequence_module']['logits'],
                 x_t=batch['seq_t'],
                 qt0=batch['q_t0'],
                 rate_matrix=batch['rate_t'],
                 mask=batch["mask"],
                 fixed_mask=batch["fixed_mask"]
            ),
                        
            "ce": lambda: ce_loss(
             logits=out['heads']['sequence_module']['logits'], 
             target=batch['seq'],
             fixed_mask=batch["fixed_mask"], 
             mask=batch["mask"]
        ),
            "DSM": lambda: dsm_loss(
             out=out, 
             batch=batch, 
             config =self.config.diffusion_rigids.config
        ),
        "bba": lambda: backbone_atom_loss(out, batch, self.config.backbone_atom),
        "energy": lambda: energy_head_joint_loss(self.training, batch, out['heads']['folding']['trans_score'], out['heads']['interface_energy'], batch['t'], batch['mask'], batch["fixed_mask"], self.config.energy),
        "FPE_r3": lambda: FPE_r3_loss(out['heads']['folding']['fpe_loss_r3'], global_step, **self.config.FPE_r3) if 'fpe_loss_r3' in out['heads']['folding'] else None,
        "FPE_so3": lambda: FPE_so3_loss(out['heads']['folding']['fpe_loss_so3'], global_step, **self.config.FPE_so3) if 'fpe_loss_so3' in out['heads']['folding'] else None
        }
        
        # --- 步骤 3: 根据 global_step 和您的简洁参数，动态计算权重 ---
        cfg = self.config
        phase1_end = cfg.get("curriculum_phase1_steps", 20000)
        phase2_end = cfg.get("curriculum_phase2_steps", 80000)
        ramp_len = cfg.get("curriculum_ramp_steps", 2000)
        def get_ramp(start_step, ramp_length):
            if ramp_length == 0:
                return 1.0
            # 计算从 0 到 1 的斜坡因子
            ramp_up = (global_step - start_step) / ramp_length
            return torch.clamp(torch.as_tensor(ramp_up), min=0.0, max=1.0)

        # 基础损失 (始终开启)
        weights = {
            "elbo": 0.0,
            "fape": 0.0,
            "plddt": 0.0,
            "energy": 0.0,
            "distogram": 0.0,
            "violation": 0.0,
            "FPE_r3": 0.0,
            "FPE_so3": 0.0,
        }
        # 阶段一 (0 - phase1_end): 基础去噪 + 序列学习
        # 理由: 任务最简单，只让模型学会最基本的“捏出形状”。
        weights["DSM"] = cfg.DSM.weight
        weights["ce"] = cfg.ce.weight
        #weights["elbo"] = cfg.elbo.weight
        stabilizer_weight_factor = cfg.get("stabilizer_weight_factor", 0.1)
        weights["plddt"] = cfg.plddt.weight * stabilizer_weight_factor
        
        # 阶段二损失 (distogram, plddt)
        if global_step >= phase1_end:
            ramp_factor = get_ramp(phase1_end, ramp_len)
            weights["fape"] = 1.0 * ramp_factor
            weights["energy"] = self.config.energy.weight * ramp_factor
            weights["plddt"] = cfg.plddt.weight * ramp_factor
            weights["elbo"] = cfg.elbo.weight * ramp_factor
            weights["distogram"] = cfg.distogram.weight * ramp_factor
            weights["FPE_r3"] = cfg.FPE_r3.weight * ramp_factor
            weights["FPE_so3"] = cfg.FPE_so3.weight * ramp_factor
            weights["bba"] = cfg.backbone_atom.weight * ramp_factor

        # 阶段三损失 (fape, violation)
        if global_step >= phase2_end:
            ramp_factor = get_ramp(phase2_end, ramp_len)
            weights["violation"] = cfg.violation.weight * ramp_factor
            weights["FPE_r3"] = cfg.FPE_r3.weight
            weights["FPE_so3"] = cfg.FPE_so3.weight

        # --- 步骤 4: 聚合损失 ---
        cum_loss = 0.
        losses = {}
        device = out['heads']['folding']['final_atom_positions'].device
        # 关键修改：我们遍历所有可能的损失，而不仅仅是权重非零的
        for loss_name, loss_fn in loss_fns.items():
            # 从课程学习的权重字典中获取当前步骤的权重
            weight = weights.get(loss_name, 0.0)
            
            if loss_name == "energy":
                # import ipdb; ipdb.set_trace()
                loss, eh_logs  = loss_fn()
                # 把 eh_logs 里的细项解包到 losses 里
                if eh_logs:
                    for k, v in eh_logs.items():
                        losses[k] = v.detach() # 这样 train_ema 就能记录 eh_loss_force 等了
            else:
                # 即使权重为0，我们仍然计算损失，以确保参数被使用
                loss = loss_fn()
            
            # 如果lambda返回了None (因为依赖的键不存在)，则优雅地跳过
            if loss is None:
                losses[loss_name] = torch.tensor(0.0) # 在breakdown中记录为0
                # logger.debug(f"[Step {global_step}] Skipping loss '{loss_name}' as its required inputs are not found in model output.")
                continue
            # # ===================== [DEBUG] 注入 NaN：专门测试 NaN 拦截是否生效 =====================
            # if (loss_name == "energy") and (global_step == 6003):
            #     sample_names = str(batch.get('name', 'unknown'))
            #     logging.warning(
            #         f"[DEBUG Inject] Step {global_step} | Type: {loss_name} | Name: {sample_names} | "
            #         f"Msg: Force energy loss to NaN for testing."
            #     )
            #     loss = loss * torch.tensor(float('nan'), device=loss.device, dtype=loss.dtype)
            # # ======================================================================================

            if torch.isnan(loss) or torch.isinf(loss):
                # [核心修复] 构造一个万能的 dummy_zero
                # 1. 连接主干 (Folding)
                zero_main = torch.nan_to_num(out['heads']['folding']['final_atom14_positions']).sum() * 0.0
                
                # 2. 连接能量头 (Energy) - 如果存在
                zero_energy = 0.0
                if 'interface_energy' in out['heads']:
                    ie_out = out['heads']['interface_energy']
                    # 遍历能量头的所有输出，凡是带梯度的都纳入 dummy 路径
                    # 这样能覆盖 E_pred (GNN参数) 和 iface_logit (Temperature参数)
                    # 遍历能量头返回的所有 Tensor (包括 pred, iface_logit, rmsd, forces 等)
                    # 只要它需要梯度，我们就把它加入 dummy 链路
                    for key, val in ie_out.items():
                        # 处理嵌套字典 (比如 forces)
                        if isinstance(val, dict):
                            for sub_val in val.values():
                                if isinstance(sub_val, torch.Tensor) and sub_val.requires_grad:
                                    clean_val = torch.nan_to_num(sub_val, nan=0.0, posinf=0.0, neginf=0.0)
                                    zero_energy = zero_energy + clean_val.sum() * 0.0
                        
                        # 处理直接 Tensor (比如 pred, iface_logit)
                        elif isinstance(val, torch.Tensor) and val.requires_grad:
                            # 必须先清洗，防止 NaN 传染
                            clean_val = torch.nan_to_num(val, nan=0.0, posinf=0.0, neginf=0.0)
                            # 累加到 zero_energy，建立计算图连接
                            zero_energy = zero_energy + clean_val.sum() * 0.0
                
                # 合并：这样 backward 时，梯度会同时流向主干和能量头
                dummy_zero = zero_main + zero_energy
                if loss_name == "energy":
                    # [策略 A] 能量头报错：物理奇异点，跳过\
                    sample_names = str(batch.get('name', 'unknown'))
                    logging.warning(
                        f"[Ignored NaN] Step {global_step} | Type: {loss_name} | Name: {sample_names} | "
                        f"Msg: Energy loss is NaN/Inf. Skipping."
                    )
                    loss = dummy_zero
                else:
                    logging.warning(f"[Step {global_step}] {loss_name} {batch['name']} loss is NaN or Inf! Skipping.")
                    import ipdb; ipdb.set_trace()
                    loss = torch.tensor(0.0, device=cum_loss.device, requires_grad=True)
                    loss = dummy_zero
            
            if loss_name == "DSM":
                loss = torch.clamp(loss, max=50.0) # 比如裁剪到50，防止其峰值过高
        
            losses[loss_name] = loss.detach().clone()
            
            # 乘以权重。如果 weight 为 0，这一项对总损失的贡献就是0
            cum_loss = cum_loss + weight * loss

        
        # losses["unscaled_loss"] = cum_loss.detach().clone()

        # # --- 步骤 5: 最终的全局缩放 ---
        # n_res = batch["seq"].shape[-1]
        # scaling_factor = torch.sqrt(torch.tensor(n_res, device=cum_loss.device, dtype=torch.float32))
        # cum_loss = cum_loss / scaling_factor
        losses["loss"] = cum_loss.detach().clone()

        if not _return_breakdown:
            return cum_loss

        return cum_loss, losses
    
    def forward(self, out, batch, global_step, _return_breakdown=False):
        if not _return_breakdown:
            return self.loss(out, batch, global_step, _return_breakdown=False)
        else:
            return self.loss(out, batch, global_step, _return_breakdown=True)