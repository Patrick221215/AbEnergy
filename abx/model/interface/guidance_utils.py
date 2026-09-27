import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_cluster import radius_graph
from torch_geometric.nn import MessagePassing
from abx.model import atom, quat_affine
from abx.common import residue_constants
from abx.model.utils import batched_select

from typing import List, Optional, Dict, Tuple

class CosineCutoff(nn.Module):
    def __init__(self, cutoff: float, tau: float = 0.5, power: int = 4):
        super().__init__()
        self.cutoff = float(cutoff)
        self.tau = float(tau)
        self.power = int(power)

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # 单调、非负；dist=cutoff 时是 0.5**power（power=4 -> 0.0625）
        w = torch.sigmoid((self.cutoff - distances) / max(self.tau, 1e-4))
        if self.power > 1:
            w = w.pow(self.power)
        return w


class ExpNormalSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(ExpNormalSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        self.cutoff_fn = CosineCutoff(cutoff)
        self.alpha = 5.0 / cutoff

        means, betas = self._initial_params()
        if trainable:
            self.register_parameter("means", nn.Parameter(means))
            self.register_parameter("betas", nn.Parameter(betas))
        else:
            self.register_buffer("means", means)
            self.register_buffer("betas", betas)

    def _initial_params(self):
        start_value = torch.exp(torch.scalar_tensor(-self.cutoff))
        means = torch.linspace(start_value, 1, self.num_rbf)
        betas = torch.tensor([(2 / self.num_rbf * (1 - start_value)) ** -2] * self.num_rbf)
        return means, betas

    def reset_parameters(self):
        means, betas = self._initial_params()
        self.means.data.copy_(means)
        self.betas.data.copy_(betas)

    def forward(self, dist):
        dist = dist.unsqueeze(-1)
        return self.cutoff_fn(dist) * torch.exp(-self.betas * (torch.exp(self.alpha * (-dist)) - self.means) ** 2)


class GaussianSmearing(nn.Module):
    def __init__(self, cutoff=5.0, num_rbf=50, trainable=True):
        super(GaussianSmearing, self).__init__()
        self.cutoff = cutoff
        self.num_rbf = num_rbf
        self.trainable = trainable

        offset, coeff = self._initial_params()
        if trainable:
            self.register_parameter("coeff", nn.Parameter(coeff))
            self.register_parameter("offset", nn.Parameter(offset))
        else:
            self.register_buffer("coeff", coeff)
            self.register_buffer("offset", offset)

    def _initial_params(self):
        offset = torch.linspace(0, self.cutoff, self.num_rbf)
        coeff = -0.5 / (offset[1] - offset[0]) ** 2
        return offset, coeff

    def reset_parameters(self):
        offset, coeff = self._initial_params()
        self.offset.data.copy_(offset)
        self.coeff.data.copy_(coeff)

    def forward(self, dist):
        dist = dist.unsqueeze(-1) - self.offset
        return torch.exp(self.coeff * torch.pow(dist, 2))



class ShiftedSoftplus(nn.Module):
    def __init__(self):
        super(ShiftedSoftplus, self).__init__()
        self.shift = torch.log(torch.tensor(2.0)).item()

    def forward(self, x):
        return F.softplus(x) - self.shift


class Swish(nn.Module):
    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class Sphere(nn.Module):
    
    def __init__(self, l=2):
        super(Sphere, self).__init__()
        self.l = l
        
    def forward(self, edge_vec):
        edge_sh = self._spherical_harmonics(self.l, edge_vec[..., 0], edge_vec[..., 1], edge_vec[..., 2])
        return edge_sh
        
    @staticmethod
    def _spherical_harmonics(lmax: int, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> torch.Tensor:

        sh_1_0, sh_1_1, sh_1_2 = x, y, z
        
        if lmax == 1:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2], dim=-1)

        sh_2_0 = math.sqrt(3.0) * x * z
        sh_2_1 = math.sqrt(3.0) * x * y
        y2 = y.pow(2)
        x2z2 = x.pow(2) + z.pow(2)
        sh_2_2 = y2 - 0.5 * x2z2
        sh_2_3 = math.sqrt(3.0) * y * z
        sh_2_4 = math.sqrt(3.0) / 2.0 * (z.pow(2) - x.pow(2))

        if lmax == 2:
            return torch.stack([sh_1_0, sh_1_1, sh_1_2, sh_2_0, sh_2_1, sh_2_2, sh_2_3, sh_2_4], dim=-1)


class VecLayerNorm(nn.Module):
    def __init__(self, hidden_channels, trainable, norm_type="max_min"):
        super(VecLayerNorm, self).__init__()
        
        self.hidden_channels = hidden_channels
        self.eps = 1e-12
        
        weight = torch.ones(self.hidden_channels)
        if trainable:
            self.register_parameter("weight", nn.Parameter(weight))
        else:
            self.register_buffer("weight", weight)
        
        if norm_type == "rms":
            self.norm = self.rms_norm
        elif norm_type == "max_min":
            self.norm = self.max_min_norm
        else:
            self.norm = self.none_norm
        
        self.reset_parameters()

    def reset_parameters(self):
        # weight = torch.ones(self.hidden_channels)
        # self.weight.data.copy_(weight)
        self.weight.data.fill_(1.0)
    
    def none_norm(self, vec):
        return vec
        
    def rms_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        dist = torch.sqrt(torch.mean(dist ** 2, dim=-1))
        return vec / F.relu(dist).unsqueeze(-1).unsqueeze(-1)
    
    def max_min_norm(self, vec):
        # vec: (num_atoms, 3 or 5, hidden_channels)
        dist = torch.norm(vec, dim=1, keepdim=True)
        
        if (dist == 0).all():
            return torch.zeros_like(vec)
        
        dist = dist.clamp(min=self.eps)
        direct = vec / dist
        
        max_val, _ = torch.max(dist, dim=-1)
        min_val, _ = torch.min(dist, dim=-1)
        delta = (max_val - min_val).view(-1)
        delta = torch.where(delta == 0, torch.ones_like(delta), delta)
        dist = (dist - min_val.view(-1, 1, 1)) / delta.view(-1, 1, 1)
        
        return F.relu(dist) * direct

    def forward(self, vec):
        # vec: (num_atoms, 3 or 8, hidden_channels)
        # 关键：用 weight.clone() 参与计算，避免 autograd 图里保存的是那个会被
        # DDP/EMA/optimizer in-place 修改的同一个 tensor
        scale = self.weight.clone().view(1, 1, -1)   # (1, 1, hidden_channels)
        if vec.shape[1] == 3:
            vec = self.norm(vec)
            return vec * scale
        elif vec.shape[1] == 8:
            vec1, vec2 = torch.split(vec, [3, 5], dim=1)
            vec1 = self.norm(vec1)
            vec2 = self.norm(vec2)
            vec = torch.cat([vec1, vec2], dim=1)
            return vec * scale
        else:
            raise ValueError("VecLayerNorm only support 3 or 8 channels")


class Distance(nn.Module):
    def __init__(self, cutoff, max_num_neighbors=32, loop=True):
        super(Distance, self).__init__()
        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.loop = loop

    def forward(self, pos, batch):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch, loop=self.loop, max_num_neighbors=self.max_num_neighbors)
        edge_vec = pos[edge_index[0]] - pos[edge_index[1]]

        if self.loop:
            mask = edge_index[0] != edge_index[1]
            edge_weight = torch.zeros(edge_vec.size(0), device=edge_vec.device)
            edge_weight[mask] = torch.norm(edge_vec[mask], dim=-1)
        else:
            edge_weight = torch.norm(edge_vec, dim=-1)

        return edge_index, edge_weight, edge_vec


class NeighborEmbedding(MessagePassing):
    def __init__(self, hidden_channels, num_rbf, cutoff, max_z=100):
        super(NeighborEmbedding, self).__init__(aggr="add")
        self.embedding = nn.Embedding(max_z, hidden_channels)
        self.distance_proj = nn.Linear(num_rbf, hidden_channels)
        self.combine = nn.Linear(hidden_channels * 2, hidden_channels)
        self.cutoff = CosineCutoff(cutoff)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        self.embedding.reset_parameters()
        nn.init.xavier_uniform_(self.distance_proj.weight)
        nn.init.xavier_uniform_(self.combine.weight)
        self.distance_proj.bias.data.fill_(0)
        self.combine.bias.data.fill_(0)

    def forward(self, z, x, edge_index, edge_weight, edge_attr):
        # remove self loops
        mask = edge_index[0] != edge_index[1]
        if not mask.all():
            edge_index = edge_index[:, mask]
            edge_weight = edge_weight[mask]
            edge_attr = edge_attr[mask]

        C = self.cutoff(edge_weight)
        W = self.distance_proj(edge_attr) * C.view(-1, 1)

        x_neighbors = self.embedding(z)
        # propagate_type: (x: Tensor, W: Tensor)
        x_neighbors = self.propagate(edge_index, x=x_neighbors, W=W, size=None)
        x_neighbors = self.combine(torch.cat([x, x_neighbors], dim=1))
        return x_neighbors

    def message(self, x_j, W):
        return x_j * W

    
class EdgeEmbedding(MessagePassing):
    
    def __init__(self, num_rbf, hidden_channels):
        super(EdgeEmbedding, self).__init__(aggr=None)
        self.edge_proj = nn.Linear(num_rbf, hidden_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.edge_proj.weight)
        self.edge_proj.bias.data.fill_(0)
        
    def forward(self, edge_index, edge_attr, x):
        # propagate_type: (x: Tensor, edge_attr: Tensor)
        out = self.propagate(edge_index, x=x, edge_attr=edge_attr)
        return out
    
    def message(self, x_i, x_j, edge_attr):
        return (x_i + x_j) * self.edge_proj(edge_attr)
    
    def aggregate(self, features, index):
        # no aggregate
        return features
    
    
    

# -------------------- 数值与张量小工具 --------------------

def safe_norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """稳定范数 = sqrt(sum(x^2) + eps)"""
    return torch.sqrt(torch.clamp((x * x).sum(dim=dim), min=eps))


def batched_index_select(table: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """
    table: [K, ...], idx: [B, L] (每个元素 0..K-1)
    return: [B, L, ...]
    """
    b, l = idx.shape
    flat = table.index_select(0, idx.reshape(-1))
    return flat.view(b, l, *table.shape[1:])


def angle_feature(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    角度 a-b-c 的 (cos, sin) 特征（SE(3)-不变）。
    a,b,c: [..., 3]
    return: [..., 2]
    """
    v1 = a - b
    v2 = c - b
    v1n = v1 / (safe_norm(v1, dim=-1).unsqueeze(-1))
    v2n = v2 / (safe_norm(v2, dim=-1).unsqueeze(-1))
    cos = (v1n * v2n).sum(dim=-1, keepdim=True)
    sin = safe_norm(torch.cross(v1n, v2n, dim=-1), dim=-1).unsqueeze(-1)
    return torch.cat([cos, sin], dim=-1)


def build_atom14_from_rigids(
    rigids7: torch.Tensor,
    seq: torch.Tensor,
    torsion_sin_cos: torch.Tensor,
):
    """
    将 (rigids7, torsion_angles) → atom14。

    参数:
      rigids7:        [B, L, 7]   四元数 + 平移，(quat_xyzw, trans_xyz)
      seq:         [B, L] 
      torsion_sin_cos:[B, L, 7, 2]
                      主链 + 侧链 torsion 的 (sin, cos)，

    返回:
      atom14_pos:     [B, L, 14, 3]  —— 14 原子坐标
      atom14_exists:  [B, L, 14]     —— 是否存在的掩码 (0/1)
      torsion_sin_cos:[B, L, 7, 2]   —— 原样返回
    """
    device = rigids7.device

    # 1) rigids7 → (rot, trans)
    rot = quat_affine.quat_to_rot(rigids7[..., :4])   # [B,L,3,3]
    trans = rigids7[..., 4:]                 # [B,L,3]
    backb_to_global = (rot, trans)

    # 2) 利用真实的 torsion_angles_sin_cos，构造每个 rigid group 的局部框架
    frames = atom.torsion_angles_to_frames(
        seq,             # [B,L]
        backb_to_global,    # (rot, trans)
        torsion_sin_cos,    # [B,L,7,2]
    )

    # 3) literature 14-atom positions + 刚体 frame → atom14 坐标
    atom14_pos = atom.frames_and_literature_positions_to_atom14_pos(
        seq, frames
    )  # [B,L,14,3]

    atom14_exists = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=device), seq)

    return atom14_pos, atom14_exists



def knn_cross_edges(
    src_pos: torch.Tensor, dst_pos: torch.Tensor,
    dst_mask: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    从 src（AB）指向 dst（AG）的 KNN 边。
    src_pos: [B, N, 3]
    dst_pos: [B, M, 3]
    dst_mask: [B, M]
    return:
      knn_idx: [B, N, k]  (在 dst 的索引)
      knn_dist: [B, N, k]
    """
    if dst_pos.shape[1] == 0:
        # 空 dst
        B, N, _ = src_pos.shape
        device = src_pos.device
        return torch.zeros(B, N, 0, dtype=torch.long, device=device), torch.zeros(B, N, 0, device=device)

    d = torch.cdist(src_pos, dst_pos) + (~dst_mask[:, None, :]).float() * 1e6
    k_eff = min(k, dst_pos.shape[1])
    knn_dist, knn_idx = torch.topk(d, k=k_eff, dim=-1, largest=False)  # [B,N,k]
    return knn_idx, knn_dist

def build_ca_from_rigids7(rigids7: torch.Tensor, seq_idx: torch.Tensor) -> torch.Tensor:
    """
    rigids7: [B,L,7]  (quat_xyzw + trans_xyz)
    seq_idx: [B,L]    (0..19)
    return: Cα 坐标 [B,L,3]
    说明：可微链路（但这里通常只前向，用于构图）
    """
    B, L, _ = rigids7.shape
    rots = quat_affine.quat_to_rot(rigids7[..., :4])  # [B,L,3,3]
    trans = rigids7[..., 4:]                          # [B,L,3]
    backb_to_global = (rots, trans)

    tors = torch.zeros(B, L, 7, 2, device=rigids7.device, dtype=rigids7.dtype)
    tors[..., 1] = 1.0  # 占位
    frames = atom.torsion_angles_to_frames(seq_idx, backb_to_global, tors)
    atom14_pos = atom.frames_and_literature_positions_to_atom14_pos(seq_idx, frames)
    # 按 AlphaFold 的约定，atom14 的 CA 槽位为索引 1
    ca = atom14_pos[..., 1, :]  # [B,L,3]
    return ca



class LayerNorm(nn.Module):

    def __init__(self,
                 normal_shape,
                 gamma=True,
                 beta=True,
                 epsilon=1e-10):
        """Layer normalization layer
        See: [Layer Normalization](https://arxiv.org/pdf/1607.06450.pdf)
        :param normal_shape: The shape of the input tensor or the last dimension of the input tensor.
        :param gamma: Add a scale parameter if it is True.
        :param beta: Add an offset parameter if it is True.
        :param epsilon: Epsilon for calculating variance.
        """
        super().__init__()
        if isinstance(normal_shape, int):
            normal_shape = (normal_shape,)
        else:
            normal_shape = (normal_shape[-1],)
        self.normal_shape = torch.Size(normal_shape)
        self.epsilon = epsilon
        if gamma:
            self.gamma = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('gamma', None)
        if beta:
            self.beta = nn.Parameter(torch.Tensor(*normal_shape))
        else:
            self.register_parameter('beta', None)
        self.reset_parameters()

    def reset_parameters(self):
        if self.gamma is not None:
            self.gamma.data.fill_(1)
        if self.beta is not None:
            self.beta.data.zero_()

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.epsilon).sqrt()
        y = (x - mean) / std
        if self.gamma is not None:
            y *= self.gamma
        if self.beta is not None:
            y += self.beta
        return y

    def extra_repr(self):
        return 'normal_shape={}, gamma={}, beta={}, epsilon={}'.format(
            self.normal_shape, self.gamma is not None, self.beta is not None, self.epsilon,
        )



def calculate_cdr_rmsd(rigids_a: torch.Tensor, rigids_b: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
    """
    计算 interface 上的 RMSD。
    rigids_*: [B, L, 7]  (quat|trans). 我们关心的是坐标，所以需要把rigids7 -> CA坐标。
    我们只对界面残基计算，比如 CDR∪抗原。mask我们可以用:
       cdr_mask = (1 - batch['fixed_mask']).bool() & ab_mask
       ag_mask  = ag_mask
    然后 interface_mask = cdr_mask | ag_mask
    最后对这些残基的CA坐标做 RMSD。
    """
    with torch.no_grad():
        # 1. 构建 CA 坐标
        seq_t = batch['seq_t'].long()
        ca_a = build_ca_from_rigids7(rigids_a, seq_t)  # [B,L,3]
        ca_b = build_ca_from_rigids7(rigids_b, seq_t)  # [B,L,3]

        B, L, _ = ca_a.shape
        device = ca_a.device

        # 2. 构建 interface mask
        L_ab = batch['anchor_flag'].shape[1]
        ab_mask = torch.zeros(B, L, dtype=torch.bool, device=device); ab_mask[:, :L_ab] = True
        ag_mask = torch.zeros(B, L, dtype=torch.bool, device=device); ag_mask[:, L_ab:] = True
        cdr_mask = (1 - batch['fixed_mask']).bool() & ab_mask
        iface_mask = cdr_mask | ag_mask  # 只看界面CDR + 抗原

        # 3. 坐标差
        diff = ca_a - ca_b  # [B,L,3]
        diff2 = (diff ** 2).sum(dim=-1)  # [B,L]

        # 4. 只在 iface_mask 上平均
        diff2_iface = diff2 * iface_mask.float()
        # 避免除0
        denom = iface_mask.float().sum(dim=1).clamp_min(1.0)  # [B]
        mse_iface = diff2_iface.sum(dim=1) / denom  # [B]
        rmsd_iface = torch.sqrt(mse_iface + 1e-8)    # [B]
        return rmsd_iface  # [B]