import numpy as np
import torch
from torch.nn import functional as F
from torch import Tensor
from einops import rearrange

from abx.model.utils import l2_normalize

# pylint: disable=bad-whitespace
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]  # rr
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]  # ii
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]  # jj
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]  # kk

QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # ij
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]  # ik
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]  # jk

QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]  # ir
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]  # jr
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]  # kr

QUAT_MULTIPLY = np.zeros((4, 4, 4), dtype=np.float32)
QUAT_MULTIPLY[:, :, 0] = [[ 1, 0, 0, 0],
                          [ 0,-1, 0, 0],
                          [ 0, 0,-1, 0],
                          [ 0, 0, 0,-1]]

QUAT_MULTIPLY[:, :, 1] = [[ 0, 1, 0, 0],
                          [ 1, 0, 0, 0],
                          [ 0, 0, 0, 1],
                          [ 0, 0,-1, 0]]

QUAT_MULTIPLY[:, :, 2] = [[ 0, 0, 1, 0],
                          [ 0, 0, 0,-1],
                          [ 1, 0, 0, 0],
                          [ 0, 1, 0, 0]]

QUAT_MULTIPLY[:, :, 3] = [[ 0, 0, 0, 1],
                          [ 0, 0, 1, 0],
                          [ 0,-1, 0, 0],
                          [ 1, 0, 0, 0]]

QUAT_MULTIPLY_BY_VEC = QUAT_MULTIPLY[:, 1:, :]
# pylint: enable=bad-whitespace

QUAT_TO_ROT_TORCH = torch.tensor(np.reshape(QUAT_TO_ROT, (4, 4, 9)))
QUAT_MULTIPLY_TORCH = torch.tensor(QUAT_MULTIPLY)
QUAT_MULTIPLY_BY_VEC_TORCH = torch.tensor(QUAT_MULTIPLY_BY_VEC)

def make_identity(out_shape, device):
    out_shape = (out_shape) + (3,)
    quaternions = F.pad(torch.zeros(out_shape, device=device), (1, 0), value=1.)
    translations = torch.zeros(out_shape, device = device)

    return quaternions, translations

def quat_to_rot(normalized_quat):
    rot_tensor = torch.sum(
            QUAT_TO_ROT_TORCH.to(normalized_quat.device) *
            normalized_quat[..., :, None, None] *
            normalized_quat[..., None, :, None],
            axis=(-3, -2))
    rot = rearrange(rot_tensor, '... (c d) -> ... c d', c=3, d=3)
    return rot

def quat_multiply_by_vec(quat, vec):
    return torch.sum(
            QUAT_MULTIPLY_BY_VEC_TORCH.to(quat.device) *
            quat[..., :, None, None] *
            vec[..., None, :, None],
            dim=(-3, -2))

def quat_multiply(quat1, quat2):
    assert quat1.shape == quat2.shape
    return torch.sum(
            QUAT_MULTIPLY_TORCH.to(quat1.device) *
            quat1[..., :, None, None] *
            quat2[..., None, :, None],
            dim=(-3, -2))

def quat_precompose_vec(quaternion, vector_quaternion_update):
    assert quaternion.shape[-1] == 4\
            and vector_quaternion_update.shape[-1] == 3\
            and quaternion.shape[:-1] == vector_quaternion_update.shape[:-1]
            
    new_quaternion = quaternion + quat_multiply_by_vec(quaternion, vector_quaternion_update)
    normalized_quaternion = l2_normalize(new_quaternion)

    return normalized_quaternion

# def quat_to_rotvec(quat, eps=1e-6):
#     # w > 0 to ensure 0 <= angle <= pi
#     flip = (quat[..., :1] < 0).float()
#     quat = (-1 * quat) * flip + (1 - flip) * quat

#     angle = 2 * torch.atan2(
#         torch.linalg.norm(quat[..., 1:], dim=-1),
#         quat[..., 0]
#     )

#     angle2 = angle * angle
#     small_angle_scales = 2 + angle2 / 12 + 7 * angle2 * angle2 / 2880
#     large_angle_scales = angle / torch.sin(angle / 2 + eps)

#     small_angles = (angle <= 1e-3).float()
#     rot_vec_scale = small_angle_scales * small_angles + (1 - small_angles) * large_angle_scales
#     rot_vec = rot_vec_scale[..., None] * quat[..., 1:]
#     return rot_vec

# def quat_to_rotvec(quaternions: torch.Tensor) -> torch.Tensor:
#     flip = torch.lt(quaternions.detach()[...,:1], 0.) * 1.0
#     quaternions = (-1. * quaternions) * flip + (1. - flip) * quaternions

#     norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
#     half_angles = torch.atan2(norms, quaternions[..., :1])
#     angles = 2 * half_angles
#     eps = 1e-6
#     small_angles = torch.lt(torch.abs(angles.detach()), eps)
#     sin_half_angles_over_angles = torch.empty_like(angles)
#     sin_half_angles_over_angles[~small_angles] = (
#         torch.sin(half_angles[~small_angles]) / angles[~small_angles]
#     )
#     # for x small, sin(x/2) is about x/2 - (x/2)^3/6
#     # so sin(x/2)/x is about 1/2 - (x*x)/48
#     sin_half_angles_over_angles[small_angles] = (
#         0.5 - (angles[small_angles] * angles[small_angles]) / 48
#     )
#     return quaternions[..., 1:] / sin_half_angles_over_angles


def quat_to_rotvec(q: Tensor,
                   eps: float = 1e-8,
                   taylor_th: float = 1e-4) -> Tensor:
    """
    Quaternion (w,x,y,z)  →  rotation-vector, 任意阶梯度均有限且无 NaN/Inf
    * 先规范化 & 强制 w≥0
    * 使用 atan2 形式避免 acos
    * 对 |v|→0 做 stop-grad 处理，使二阶导也不含 1/|v|
    """
    # --- 1. 归一化并统一符号 -----------------
    q = torch.nn.functional.normalize(q, dim=-1, eps=eps)
    w_neg = (q[..., 0:1] < 0).type_as(q)
    q = q * (1 - 2 * w_neg)             # 若 w<0 则整体取反，梯度安全

    qw   = q[..., 0:1]                  # (...,1)
    qv   = q[..., 1:]                   # (...,3)

    # --- 2. v_norm: 带 stop-grad 的安全范数 ----
    v_sq         = (qv * qv).sum(-1, keepdim=True)               # (...,1)
    v_safe       = torch.sqrt(v_sq + eps**2)                     # ≥ eps
    # stop-grad 当 |v| 很小时，把高阶导数截断
    v_norm       = v_safe.detach() + (v_safe - v_safe.detach())

    # --- 3. 角度 θ = 2·atan2(|v| , qw) -------------------------
    qw_safe      = torch.clamp(qw, min=eps)                      # 保证正
    theta        = 2.0 * torch.atan2(v_norm, qw_safe)            # (...,1)

    # --- 4. 比例系数 k = θ / |v| ，小角度用泰勒 ------------
    k            = theta / v_norm                                # 正常分支
    k_taylor     = 2.0 - (theta * theta) / 12.0                  # θ≈0 → k≈2 - θ²/12
    use_taylor   = (theta < taylor_th).expand_as(k)
    k            = torch.where(use_taylor, k_taylor, k)

    # --- 5. rotvec = k * v  ------------------------------------
    rotvec       = qv * k
    # 避免数值污染
    rotvec       = torch.nan_to_num(rotvec, nan=0.0, posinf=0.0, neginf=0.0)
    return rotvec

# def rotvec_to_quat(axis_angle: torch.Tensor) -> torch.Tensor:
#     import ipdb; ipdb.set_trace()
#     angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
#     half_angles = angles * 0.5
#     eps = 1e-6
#     small_angles = torch.lt(torch.abs(angles.detach()), eps)
#     sin_half_angles_over_angles = torch.empty_like(angles)
#     sin_half_angles_over_angles[~small_angles] = (
#         torch.sin(half_angles[~small_angles]) / angles[~small_angles]
#     )
#     # for x small, sin(x/2) is about x/2 - (x/2)^3/6
#     # so sin(x/2)/x is about 1/2 - (x*x)/48
#     sin_half_angles_over_angles[small_angles] = (
#         0.5 - (angles[small_angles] * angles[small_angles]) / 48
#     )
#     quaternions = torch.cat(
#         [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
#     )

#     return quaternions
def rotvec_to_quat(
    axis_angle: Tensor,
    eps: float = 1e-6,
    taylor_thresh: float = 1e-4,
) -> Tensor:
    """
    axis_angle: [..., 3]
    return:     [..., 4] (w, x, y, z), 保证对 axis_angle 处处可导，
              小角度用泰勒展开，且不做 identity 强制替换。
    """
    # 1. 角度 θ = ||ω||
    angle_sq = (axis_angle * axis_angle).sum(dim=-1, keepdim=True)   # [...,1]
    # clamp 一下避免 0 除，但不加额外的 stop-grad
    angles = torch.sqrt(torch.clamp(angle_sq, min=eps**2))           # [...,1]
    half_angles = 0.5 * angles                                       # [...,1]

    # 2. sin(θ/2)/θ 的稳定实现
    # mask 只影响分支选择，不进微分图
    small = (angles.detach() < taylor_thresh)

    sin_half = torch.sin(half_angles)                                # [...,1]
    ratio = torch.empty_like(angles)

    # 大角度：直接 sin/θ
    ratio[~small] = sin_half[~small] / angles[~small]

    # 小角度：泰勒展开 sin(θ/2)/θ ≈ 0.5 - θ²/48
    ratio[small] = 0.5 - angle_sq[small] / 48.0

    # 3. 组 quaternion
    qw = torch.cos(half_angles)                                      # [...,1]
    q_xyz = axis_angle * ratio                                       # [...,3]
    quat = torch.cat([qw, q_xyz], dim=-1)                            # [...,4]

    # 4. 可选：再单位化一下，防止积累误差
    quat = F.normalize(quat, dim=-1)

    return quat

# def rot_to_quat(rot: torch.Tensor,):
#     if(rot.shape[-2:] != (3, 3)):
#             raise ValueError("Input rotation is incorrectly shaped")
#     if rot.requires_grad:
#         raise ValueError("Input rotation is not differentiable")
#     rot = [[rot[..., i, j] for j in range(3)] for i in range(3)]
#     [[xx, xy, xz], [yx, yy, yz], [zx, zy, zz]] = rot 
    
#     k = [
#         [ xx + yy + zz,      zy - yz,      xz - zx,      yx - xy,],
#         [      zy - yz, xx - yy - zz,      xy + yx,      xz + zx,],
#         [      xz - zx,      xy + yx, yy - xx - zz,      yz + zy,],
#         [      yx - xy,      xz + zx,      yz + zy, zz - xx - yy,]
#     ]

#     k = (1./3.) * torch.stack([torch.stack(t, dim=-1) for t in k], dim=-2)
#     _, vectors = torch.linalg.eigh(k)
#     return vectors[..., -1]

def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def rot_to_quat(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
            #  `int`.
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :
    ].reshape(batch_dim + (4,))


def invert_quat(quat: torch.Tensor):
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sqrt(torch.sum(quat ** 2, dim=-1, keepdim=True))
    return inv



