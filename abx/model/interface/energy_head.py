from __future__ import annotations
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch import Tensor
import math
from torch_scatter import scatter, scatter_min
from torch_cluster import radius_graph
from einops import rearrange
from abx.model.quat_affine import quat_to_rotvec, rotvec_to_quat, quat_multiply, quat_to_rot

from .guidance_utils import (
    CosineCutoff,
    Distance, 
    EdgeEmbedding,
    NeighborEmbedding, 
    Sphere, 
    VecLayerNorm,
    ExpNormalSmearing,
    build_ca_from_rigids7,
    build_atom14_from_rigids,
    knn_cross_edges,
    
)


class GatedEquivariantBlock(nn.Module):
    """
    Gated Equivariant Block as defined in Schütt et al. (2021):
    Equivariant message passing for the prediction of tensorial properties and molecular spectra
    """
    def __init__(
        self,
        hidden_channels,
        out_channels,
        intermediate_channels=None,
        activation="tanh",
        scalar_activation=False,
    ):
        super(GatedEquivariantBlock, self).__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
        self.vec2_proj = nn.Linear(hidden_channels, out_channels, bias=False)

        self.update_net = nn.Sequential(
            nn.Linear(hidden_channels * 2, intermediate_channels),
            nn.Tanh(),
            nn.Linear(intermediate_channels, out_channels * 2),
        )

        self.act = nn.Tanh() if scalar_activation else None
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        nn.init.xavier_uniform_(self.update_net[0].weight)
        self.update_net[0].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.update_net[2].weight)
        self.update_net[2].bias.data.fill_(0)
    
    def forward(self, x, v):    
        vec1 = torch.norm(self.vec1_proj(v), dim=-2)
        vec2 = self.vec2_proj(v)

        x = torch.cat([x, vec1], dim=-1)
        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2
        # v = v.unsqueeze(1)
        # v = (v * vec2).contiguous()
    
        if self.act is not None:
            x = self.act(x)
        return x, v
    


class OutputModel(nn.Module):
    def __init__(self, allow_prior_model):
        super(OutputModel, self).__init__()
        self.allow_prior_model = allow_prior_model
        
    def reset_parameters(self):
        pass

    def pre_reduce(self, x, v, z, pos, batch):
        return
    
    def post_reduce(self, x):
        return x
    
class EquivariantScalar(OutputModel):
    def __init__(self, hidden_channels, activation="tanh", allow_prior_model=True):
        super(EquivariantScalar, self).__init__(allow_prior_model=allow_prior_model)
        self.output_network = nn.ModuleList([
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2, 
                    1, 
                    activation=activation,
                    scalar_activation=False,
                ),
        ])
        
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()
    
    def pre_reduce(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
      
        # # ✅ 正确：每个节点一个标量
        # v_norm = v.pow(2).sum(dim=(1, 2), keepdim=False)   # [N]
        # eps = 0.0
        # if getattr(self, "debug_v_injection", True):
        #     eps = 1e-6   # 只用来探测通路，别长期开
        # x = x + eps * v_norm.unsqueeze(-1)
        # return x.squeeze(-1)
        # # 显式利用 v 的模长，让任何影响 v 的几何都有机会影响 energy
        # v_norm = v.pow(2).sum(dim=(1, 2), keepdim=True)  # [N,1]
        # return x + 0.01 * v_norm   
        return x + v.sum() * 0
    
    
class ViS_MP(MessagePassing):
    def __init__(
        self,
        num_heads,
        hidden_channels,
        activation,
        attn_activation,
        cutoff,
        vecnorm_type,
        trainable_vecnorm,
        last_layer=False,
        w_min_guarantee: float = 5e-2,
    ):
        super(ViS_MP, self).__init__(aggr="add", node_dim=0)
        assert hidden_channels % num_heads == 0, (
            f"The number of hidden channels ({hidden_channels}) "
            f"must be evenly divisible by the number of "
            f"attention heads ({num_heads})"
        )

        self.num_heads = num_heads
        self.hidden_channels = hidden_channels
        self.head_dim = hidden_channels // num_heads
        self.last_layer = last_layer

        self.layernorm = nn.LayerNorm(hidden_channels)
        self.vec_layernorm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)


        def _act(name: str):
            name = (name or "tanh").lower()
            if name in ("silu", "swish"): return nn.SiLU()
            if name in ("relu",):          return nn.ReLU()
            if name in ("gelu",):          return nn.GELU()
            if name in ("tanh",):          return nn.Tanh()
            raise ValueError(f"Unsupported activation: {name}")
        # import ipdb; ipdb.set_trace()
        self.act = _act(activation)
        self.attn_activation = _act(attn_activation)

        # self.act = nn.SiLU(activation)
        # self.attn_activation = nn.SiLU(attn_activation)

        self.cutoff = CosineCutoff(cutoff)
        self.w_min_guarantee = float(w_min_guarantee)

        self.vec_proj = nn.Linear(hidden_channels, hidden_channels * 3, bias=False)
        
        self.q_proj = nn.Linear(hidden_channels, hidden_channels)
        self.k_proj = nn.Linear(hidden_channels, hidden_channels)
        self.v_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dk_proj = nn.Linear(hidden_channels, hidden_channels)
        self.dv_proj = nn.Linear(hidden_channels, hidden_channels)
        
        self.s_proj = nn.Linear(hidden_channels, hidden_channels * 2)
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels)
            self.w_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.w_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)

        self.o_proj = nn.Linear(hidden_channels, hidden_channels * 3)
        
        self.reset_parameters()
        
    @staticmethod
    def vector_rejection(vec, d_ij):
        vec_proj = (vec * d_ij.unsqueeze(2)).sum(dim=1, keepdim=True)
        return vec - vec_proj * d_ij.unsqueeze(2)

    def reset_parameters(self):
        self.layernorm.reset_parameters()
        self.vec_layernorm.reset_parameters()
        
        # 使用较小的 gain 来缩小初始化权重
        gain = 0.1
        nn.init.xavier_uniform_(self.q_proj.weight, gain)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight, gain)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight, gain)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight, gain)
        self.o_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.s_proj.weight, gain)
        self.s_proj.bias.data.fill_(0)
        
        if not self.last_layer:
            nn.init.xavier_uniform_(self.f_proj.weight, gain)
            self.f_proj.bias.data.fill_(0)
            nn.init.xavier_uniform_(self.w_src_proj.weight, gain)
            nn.init.xavier_uniform_(self.w_trg_proj.weight, gain)

        nn.init.xavier_uniform_(self.vec_proj.weight, gain)
        nn.init.xavier_uniform_(self.dk_proj.weight, gain)
        self.dk_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.dv_proj.weight, gain)
        self.dv_proj.bias.data.fill_(0)

        
    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij, edge_soft_weight: Optional[torch.Tensor] = None,edge_is_guarantee=None):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        if edge_soft_weight is None:
            edge_soft_weight = r_ij.new_ones(r_ij.size(0))
        else:
            edge_soft_weight = edge_soft_weight.to(device=r_ij.device, dtype=r_ij.dtype)
        self._edge_soft_weight = edge_soft_weight
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        if edge_is_guarantee is None:
            edge_is_guarantee = r_ij.new_zeros(r_ij.size(0), dtype=torch.bool)
        else:
            edge_is_guarantee = edge_is_guarantee.to(dtype=torch.bool, device=r_ij.device)

        self._edge_is_guarantee = edge_is_guarantee

        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        self._edge_is_guarantee = None 
        self._edge_soft_weight = None
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            return dx, dvec, df_ij
        else:
            return dx, dvec, None

    def message(self, q_i, k_j, v_j, vec_j, dk, dv, r_ij, d_ij):
        edge_is_guarantee = getattr(self, "_edge_is_guarantee", None)
        #attn = (q_i * k_j * dk).sum(dim=-1)
        
        attn = (q_i * k_j).sum(dim=-1) / math.sqrt(self.head_dim)
        attn = attn * dk.sum(dim=-1)  # 或 dk 也按 head_dim 缩放/限幅
        
        # attn = self.attn_activation(attn) * self.cutoff(r_ij).unsqueeze(1)
        w = self.cutoff(r_ij)  # [E]
        edge_soft = getattr(self, "_edge_soft_weight", None)
        if edge_soft is None:
            edge_soft = r_ij.new_ones(r_ij.size(0))
        w = w * edge_soft   

        # ====== ✅ guarantee 抬底：确保最少有 w_min_guarantee 的通路 ======
        edge_is_guarantee = getattr(self, "_edge_is_guarantee", None)
        if edge_is_guarantee is not None:
            w = torch.where(edge_is_guarantee, torch.clamp(w, min=self.w_min_guarantee), w)
        # ================================================================

        attn = self.attn_activation(attn) * w.unsqueeze(1)
        
        v_j = v_j * dv
        v_j = (v_j * attn.unsqueeze(2)).view(-1, self.hidden_channels)

        s1, s2 = torch.split(self.act(self.s_proj(v_j)), self.hidden_channels, dim=1)
        vec_j = vec_j * s1.unsqueeze(1) + s2.unsqueeze(1) * d_ij.unsqueeze(2)
    
        return v_j, vec_j
    
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        df_ij = self.act(self.f_proj(f_ij)) * w_dot
        return df_ij

    def aggregate(
        self,
        features: Tuple[torch.Tensor, torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        dim_size: Optional[int],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, vec = features
        x = scatter(x, index, dim=self.node_dim, dim_size=dim_size)
        vec = scatter(vec, index, dim=self.node_dim, dim_size=dim_size)
        return x, vec

    def update(self, inputs: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        return inputs
    
class ViS_MP_Vertex_Edge(ViS_MP):
    
    def __init__(
        self, 
        num_heads, 
        hidden_channels, 
        activation, 
        attn_activation, 
        cutoff, 
        vecnorm_type, 
        trainable_vecnorm, 
        last_layer=False
    ):
        super().__init__(num_heads, hidden_channels, activation, attn_activation, cutoff, vecnorm_type, trainable_vecnorm, last_layer)
        
        if not self.last_layer:
            self.f_proj = nn.Linear(hidden_channels, hidden_channels * 2)
            self.t_src_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            self.t_trg_proj = nn.Linear(hidden_channels, hidden_channels, bias=False)
            
    def edge_update(self, vec_i, vec_j, d_ij, f_ij):
        w1 = self.vector_rejection(self.w_trg_proj(vec_i), d_ij)
        w2 = self.vector_rejection(self.w_src_proj(vec_j), -d_ij)
        w_dot = (w1 * w2).sum(dim=1)
        
        t1 = self.vector_rejection(self.t_trg_proj(vec_i), d_ij)
        t2 = self.vector_rejection(self.t_src_proj(vec_i), -d_ij)
        t_dot = (t1 * t2).sum(dim=1)
        
        f1, f2 = torch.split(self.act(self.f_proj(f_ij)), self.hidden_channels, dim=-1)

        return f1 * w_dot + f2 * t_dot

    def forward(self, x, vec, edge_index, r_ij, f_ij, d_ij):
        x = self.layernorm(x)
        vec = self.vec_layernorm(vec)
        
        q = self.q_proj(x).reshape(-1, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(-1, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(-1, self.num_heads, self.head_dim)
        dk = self.act(self.dk_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        dv = self.act(self.dv_proj(f_ij)).reshape(-1, self.num_heads, self.head_dim)
        
        vec1, vec2, vec3 = torch.split(self.vec_proj(vec), self.hidden_channels, dim=-1)
        vec_dot = (vec1 * vec2).sum(dim=1)
        vec_dot = torch.clamp(vec_dot, -1000.0, 1000.0)
        
        # propagate_type: (q: Tensor, k: Tensor, v: Tensor, dk: Tensor, dv: Tensor, vec: Tensor, r_ij: Tensor, d_ij: Tensor)
        x, vec_out = self.propagate(
            edge_index,
            q=q,
            k=k,
            v=v,
            dk=dk,
            dv=dv,
            vec=vec,
            r_ij=r_ij,
            d_ij=d_ij,
            size=None,
        )
        
        o1, o2, o3 = torch.split(self.o_proj(x), self.hidden_channels, dim=1)
        dx = vec_dot * o2 + o3
        dvec = vec3 * o1.unsqueeze(1) + vec_out
        
        if not self.last_layer:
            # edge_updater_type: (vec: Tensor, d_ij: Tensor, f_ij: Tensor)
            df_ij = self.edge_updater(edge_index, vec=vec, d_ij=d_ij, f_ij=f_ij)
            
            return dx, dvec, df_ij
        else:
            return dx, dvec, None
    

class ViSNetBlockExternal(nn.Module):
    """
    纯外部特征驱动的SE(3)等变消息传递引擎。
    它不进行任何内部的图构建或初始特征化，只接收预先计算好的图组件。
    
    输入:
      x0:            [N, hidden]              - 外部计算的初始节点标量特征
      edge_index:    [2, E]                   - 外部构建的图拓扑
      edge_weight:   [E]                      - 边的标量距离 (可导)
      edge_dir:      [E, 3]                   - 边的单位方向向量 (可导)
      edge_attr:     [E, num_rbf]             - 外部融合的边标量特征
    返回:
      x_final:   [N, hidden]              - 经过多轮消息传递更新后的节点标量特征
      vec_final: [N, n_sh, hidden]          - 最终的节点矢量特征
    """

    def __init__(
        self,
        num_heads: int = 8,
        num_layers: int = 6,
        hidden_channels: int = 256,
        activation: str = "tanh",
        attn_activation: str = "tanh",
        cutoff: float = 12.0, # 用于ViS_MP内部的余弦截断
        vecnorm_type: str = "none",
        trainable_vecnorm: bool = False,
        w_min_guarantee: float = 5e-2,
    ):
        super().__init__()

        self.vis_mp_layers = nn.ModuleList()
        vis_mp_kwargs = dict(
            num_heads=num_heads,
            hidden_channels=hidden_channels,
            activation=activation,
            attn_activation=attn_activation,
            cutoff=cutoff,
            vecnorm_type=vecnorm_type,
            trainable_vecnorm=True,
            w_min_guarantee=w_min_guarantee,
        )
        for _ in range(num_layers - 1):
            self.vis_mp_layers.append(ViS_MP(last_layer=False, **vis_mp_kwargs))
        self.vis_mp_layers.append(ViS_MP(last_layer=True, **vis_mp_kwargs))

        # 3. 输出归一化层
        self.out_norm = nn.LayerNorm(hidden_channels)
        self.vec_out_norm = VecLayerNorm(hidden_channels, trainable=trainable_vecnorm, norm_type=vecnorm_type)

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.vis_mp_layers:
            layer.reset_parameters()
        self.out_norm.reset_parameters()
        self.vec_out_norm.reset_parameters()

    def forward(
        self,
        x: torch.Tensor,
        vec: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_vec: torch.Tensor,
        edge_soft_weight: Optional[torch.Tensor] = None,
        edge_is_guarantee: torch.Tensor=None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 运行多轮等变消息传递
        for layer in self.vis_mp_layers[:-1]:
            # 调用忠实复用的消息传递层
            dx, dvec, dedge_attr = layer(x, vec, edge_index, edge_weight, edge_attr, edge_vec, edge_soft_weight, edge_is_guarantee)
            # 残差更新
            x = x + dx
            vec = vec + dvec
            if dedge_attr is not None:
                edge_attr = edge_attr + dedge_attr

        dx, dvec, _ = self.vis_mp_layers[-1](x, vec, edge_index, edge_weight, edge_attr, edge_vec, edge_soft_weight, edge_is_guarantee)
        x = x + dx
        vec = vec + dvec
        # 4. 最终归一化并返回
        x_final = self.out_norm(x)
        vec_final = self.vec_out_norm(vec)
        return x_final, vec_final


    
class InterfaceIndexEmbedder(nn.Module):
    """
    生成与时间嵌入等宽的 index 特征：pos_emb(residx) + chain_emb(chain_id)
    返回：
      seq_feats: [B, L, index_embed_size]
      pair_feats: [B, L, L, 2*index_embed_size]  # 通过 cross concat
    """
    def __init__(self, index_embed_size: int, max_res_idx: int = 4096, num_chains: int = 4):
        super().__init__()
        self.pos_emb = nn.Embedding(max_res_idx + 2, index_embed_size)
        self.chain_emb = nn.Embedding(num_chains, index_embed_size)

    def _cross_concat(self, feats_1d: torch.Tensor) -> torch.Tensor:
        # feats_1d: [B,L,C]
        B, L, C = feats_1d.shape
        f_row = feats_1d.unsqueeze(2).expand(B, L, L, C)
        f_col = feats_1d.unsqueeze(1).expand(B, L, L, C)
        return torch.cat([f_row, f_col], dim=-1)  # [B,L,L,2C]

    def forward(self, residx: torch.Tensor, chain_id: torch.Tensor):
        # residx:   [B,L] 残基绝对编号（或相对编号，保持单调即可）
        # chain_id: [B,L] 链标记（例如 0=抗体, 1=抗原；多链可拓展）
        p = self.pos_emb(residx.clamp_min(0))             # [B,L,C]
        c = self.chain_emb(chain_id.clamp_min(0))         # [B,L,C]
        node_idx_feat = p + c                              # [B,L,C]
        pair_idx_feat = self._cross_concat(node_idx_feat)  # [B,L,L,2C]
        return node_idx_feat, pair_idx_feat
    

def pair_concat(pair_1, pair_2):
    assert pair_1.shape[0] == pair_2.shape[0] and pair_1.shape[-1] == pair_2.shape[-1]
    assert pair_1.device == pair_2.device
    device = pair_1.device
    batch_size = pair_1.shape[0]
    channel = pair_1.shape[-1]

    length_1 = pair_1.shape[1]
    length_2 = pair_2.shape[1]
    concat_dim1 = torch.cat(
        (
        pair_1, 
        torch.zeros((batch_size, length_2, length_1, channel), device=device)
        ), dim=1)
    
    concat_dim2 = torch.cat(
        (
        torch.zeros((batch_size, length_1, length_2, channel), device=device), 
        pair_2
        ), dim=1)
    pair_all = torch.cat([concat_dim1, concat_dim2], dim=2)
    return pair_all

  
class InterfaceRepresentation(nn.Module):
    """
    ie_seq/ie_pair（干净） → InterfaceIndexEmbedder（替代时间嵌入，注入绝对位置+AB链别）→ IE 专用 Seqformer → Interface KNN（抗体→抗原） → RBF(distance) ⊕ Linear(pair_act)（边融合）→ ViSNetBlock → x_ctx, v_ctx → EquivariantScalar → energy
    """
    def __init__(
        self,
        d_seq: int,                # ie_seq 的通道数（=主干 seq_channel）
        d_pair: int,               # ie_pair 的通道数（=主干 pair_channel）
        node_dim: int = 256,       #  输出的节点维度
        hidden_dim: int = 256,     # ViSNetBlock 的 hidden_channels
        num_layers: int = 6,       # ViSNet 层数（可比主干浅）
        num_heads: int = 8,
        lmax: int = 2,             # 球谐阶
        num_rbf: int = 32,
        iface_cutoff: float = 8.0,       # 几何截断（Å），界面 KNN 用
        iface_soft_tau: float = 1.0,          # sigmoid 温度，Å
        iface_soft_center: Optional[float] = None,  # 默认用 iface_cutoff
        cdr_intra_cutoff: float = 12.0,          # 每个Cα原子为中心的球形邻域。只有当另一个Cα原子落入这个球内时，两者之间才会建立一条边
        min_edges_per_src: int = 4,
        max_num_neighbors: int = 32,
        k_iface: int = 24,         # 每个抗体残基连 k 个最近抗原残基
        w_min_guarantee: float = 5e-2,
        pair2rbf_mode: str = "sum",# "sum" 或 "concat"
        backbone=None,               # 主干网络实例（用于权重共享）
        **kwargs # 接收其他未使用参数
    ):
        super().__init__()
        self.d_seq  = d_seq
        self.d_pair = d_pair
        self.node_dim = node_dim
        self.num_rbf = num_rbf
        self.max_num_neighbors = max_num_neighbors
        self.cdr_intra_cutoff = cdr_intra_cutoff
        self.min_edges_per_src = min_edges_per_src
        self.iface_cutoff = iface_cutoff
        self.k_iface = k_iface
        self.pair2rbf_mode = pair2rbf_mode
        self.lmax = lmax
        self.hidden_dim = hidden_dim
        
        self.iface_soft_tau = float(iface_soft_tau)
        self.iface_soft_center = float(iface_cutoff if iface_soft_center is None else iface_soft_center)

        # ori_feat: [u_i_local(3), u_j_local(3), rel6(6)] => 12 dims
        self.edge_ori_proj = nn.Linear(12, num_rbf)

        # 让旋转特征不被 edge_attr_geom / pair_rbf 淹没
        self.log_ori_scale = nn.Parameter(torch.tensor(math.log(10.0)))  # 初值=10倍

        self.log_geom_scale = nn.Parameter(torch.tensor(0.0))  # 初始=1
        self.log_pair_scale = nn.Parameter(torch.tensor(0.0))  # 初始=1
        self.edge_attr_norm = nn.LayerNorm(num_rbf)


        # 输入维度现在是Seqformer的输出维度
        self.node_proj = nn.Linear(d_seq, hidden_dim)
        
        # 3) pair 边特征 -> RBF 维度 的投影
        # 输入维度现在也是Seqformer的输出维度
        self.pair_to_rbf = nn.Linear(d_pair, num_rbf)

        
        # 4) 几何展开器（RBF+球谐），与 ViSNet 的设置对齐
        self.rbf = ExpNormalSmearing(cutoff=iface_cutoff, num_rbf=num_rbf, trainable=False)
        self.sphere = Sphere(l=lmax)
        self.edge_embedding = EdgeEmbedding(num_rbf, hidden_dim)

        # 5) ViSNetBlock 主体
        self.visnet = ViSNetBlockExternal(
            num_heads=num_heads,
            num_layers=num_layers,
            hidden_channels=hidden_dim,
            activation="tanh",
            attn_activation="tanh",
            cutoff=iface_cutoff,
            w_min_guarantee=w_min_guarantee,
        )

        # 6) 边融合后再对齐为 ViSNet 的 edge_attr 维度
        # ViSNetBlock 里 edge_embedding 接受的 edge_attr 是 num_rbf 维；我们融合后仍是 num_rbf
        # 若你想 concat，则需要一个线性层把 2*num_rbf -> num_rbf
        if pair2rbf_mode == "concat":
            self.edge_cat_proj = nn.Linear(2 * num_rbf, num_rbf)
        else:
            self.edge_cat_proj = None
    def reset_parameters(self):
        self.visnet.reset_parameters()
        self.rbf.reset_parameters()
        self.edge_embedding.reset_parameters()
        # 初始化 edge_ori_proj
        nn.init.xavier_uniform_(self.edge_ori_proj.weight, gain=0.1)
        if self.edge_ori_proj.bias is not None:
            self.edge_ori_proj.bias.data.fill_(0.0)

        
    @torch.no_grad()
    def _build_iface_graph_topology(
        self,
        ca: torch.Tensor,         # [B,L,3] 所有残基 CA
        ab_mask: torch.Tensor,    # [B,L] 抗体 mask
        ag_mask: torch.Tensor,    # [B,L] 抗原 mask
        cdr_mask: torch.Tensor,   # [B,L] CDR mask
    ) -> Tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
        """
        构建一个包含CDR内部边和“软保底”跨界面边的混合图。
        返回:
          edge_index: [2, E_total] or None
          is_iface_node_mask: [B,L] (标记哪些CDR节点是界面节点)
        """
        B, L, _ = ca.shape
        device = ca.device
        
        edge_index_list = []
        edge_guarantee_list = []   #对齐 edge_index_list 的 bool 标记
        final_iface_ab_cdr_mask = torch.zeros(B, L, device=device, dtype=torch.bool)
        
        for b in range(B):
            # --- 1. 确定所有相关的节点集 ---
            all_cdr_ids = torch.where(ab_mask[b] & cdr_mask[b])[0]
            all_ag_ids = torch.where(ag_mask[b])[0]
            
            if all_cdr_ids.numel() == 0 or all_ag_ids.numel() == 0:
                continue

            # --- 2. [第一部分] 构建CDR内部的边 (使用独立的cdr_intra_cutoff) ---
            ca_cdr_b = ca[b, all_cdr_ids]
            cdr_edge_index_local = radius_graph(
                ca_cdr_b, 
                r=self.cdr_intra_cutoff, # 使用专门的内部cutoff
                max_num_neighbors=self.max_num_neighbors
            )
            cdr_edge_index_global = all_cdr_ids[cdr_edge_index_local] + b * L

            cdr_is_guarantee = torch.zeros(
                cdr_edge_index_global.shape[1],
                device=device,
                dtype=torch.bool
            )
            
            # --- 3. [第二部分] 构建“软保底”的跨界面kNN边 ---
            ca_ag_b = ca[b, all_ag_ids]
            ag_mask_b = ag_mask[b, all_ag_ids]

            # a. 为所有CDR节点构建kNN边
            nn_idx, nn_dist = knn_cross_edges(
                src_pos=ca_cdr_b.unsqueeze(0),
                dst_pos=ca_ag_b.unsqueeze(0),
                dst_mask=ag_mask_b.unsqueeze(0),
                k=self.k_iface
            )
            nn_idx, nn_dist = nn_idx.squeeze(0), nn_dist.squeeze(0) # [N_cdr, k]

            # b. 先排序，再保底，再过滤
            if nn_dist.numel() > 0:
                # 排序 (topk不保证升序)
                knn_dist_sorted, order = torch.sort(nn_dist, dim=-1)
                knn_idx_sorted = torch.gather(nn_idx, -1, order)

                # 永远保留前 k（k_iface），但如果实际 dst 少于 k，会自动变短
                k_eff = knn_idx_sorted.shape[1]
                if k_eff == 0:
                    edge_index_list.append(cdr_edge_index_global)
                    edge_guarantee_list.append(cdr_is_guarantee)
                else:
                    src_local_idx = torch.arange(knn_idx_sorted.shape[0], device=device).unsqueeze(1).expand(-1, k_eff).reshape(-1)
                    trg_local_ids = knn_idx_sorted.reshape(-1)

                    src_global_ids = all_cdr_ids[src_local_idx] + b * L
                    trg_global_ids = all_ag_ids[trg_local_ids] + b * L

                    cross_edge_index_forward = torch.stack([src_global_ids, trg_global_ids], dim=0)
                    cross_edge_index_backward = torch.stack([trg_global_ids, src_global_ids], dim=0)

                    # 保留 guarantee 标记但不再用于抬底（可用于 debug）
                    # 例如：把每个 src 的前 m 条标为 guarantee
                    m = min(int(self.min_edges_per_src), k_eff)
                    guarantee_mask = torch.zeros((knn_idx_sorted.shape[0], k_eff), device=device, dtype=torch.bool)
                    if m > 0:
                        guarantee_mask[:, :m] = True
                    cross_is_guarantee_forward = guarantee_mask.reshape(-1)
                    cross_is_guarantee_backward = cross_is_guarantee_forward.clone()

                    edge_index_b = torch.cat(
                        [cdr_edge_index_global, cross_edge_index_forward, cross_edge_index_backward],
                        dim=1
                    )
                    edge_guarantee_b = torch.cat(
                        [cdr_is_guarantee, cross_is_guarantee_forward, cross_is_guarantee_backward],
                        dim=0
                    )
                    edge_index_list.append(edge_index_b)
                    edge_guarantee_list.append(edge_guarantee_b)

                    # 物理界面节点：仍用真实距离<iface_cutoff 来标
                    cutoff_mask = (knn_dist_sorted < self.iface_cutoff)
                    iface_src_local = torch.where(cutoff_mask.any(dim=-1))[0]
                    if iface_src_local.numel() > 0:
                        final_iface_ab_cdr_mask[b, all_cdr_ids[iface_src_local]] = True
            else:
                # 如果kNN本身就是空的，也只保留CDR内部边
                edge_index_list.append(cdr_edge_index_global)
                edge_guarantee_list.append(cdr_is_guarantee)

        if not edge_index_list:
            return None, None, final_iface_ab_cdr_mask

        edge_index_total = torch.cat(edge_index_list, dim=1)
        edge_is_guarantee_total = torch.cat(edge_guarantee_list, dim=0)  # [E_total]
        
        return edge_index_total, edge_is_guarantee_total, final_iface_ab_cdr_mask
    

    def forward(
        self,
        batch: Dict,
        seq: torch.Tensor,
        rigids7: torch.Tensor,    # [B,L,7]       —— 当前构象（用于 CA&构图）
        ie_seq: torch.Tensor,
        ie_pair: torch.Tensor,
        atom14_positions: torch.Tensor,    # [B,L,7]
        angles_sin_cos: torch.Tensor,    # [B,L,7] 
        atom14_exists: torch.Tensor,    # [B,L,7] 
        return_pos_for_grad: bool = False,   # 只有在要力时才需要 True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回：
          x_ctx, v_ctx —— ViSNetBlock 的上下文化输出（节点标量/向量），给能量头读出。
        """
        
        B, L = seq.shape
        device = seq.device
        
        # --- 0. 从 rigids7 中取出旋转矩阵 R，用于构造“相对朝向特征” ---
        quat = rigids7[..., :4]                  # [B,L,4]
        rot  = quat_to_rot(quat)    
        rot_flat = rot.reshape(B * L, 3, 3).contiguous()
        

        # --- 1. 单次几何构建与“锚点-变量”分离 ---
        # # atom14：用于 encode_residue_emb / encode_pair_emb
        # atom14_positions, atom14_exists = build_atom14_from_rigids(
        #     rigids7=rigids7,
        #     seq=seq.long(),
        #     torsion_sin_cos=angles_sin_cos,
        # )   # [B,L,14,3], [B,L,14]

        ca_var = build_ca_from_rigids7(rigids7, seq)  
        ca_anchor = ca_var.detach()

        residx = batch.get("residx")           # [B,L]
        # 抗体/抗原链标（最简单：前 L_ab 为 0，后段为 1）
        L_ab = batch['anchor_flag'].shape[1]
        chain_id = torch.zeros_like(residx)
        chain_id[:, L_ab:] = 1
        

        # --- 3. 构建稀疏界面图 (使用锚点坐标) ---
        ab_mask = torch.zeros(B, L, device=device, dtype=torch.bool); ab_mask[:, :L_ab] = True
        ag_mask = torch.zeros(B, L, device=device, dtype=torch.bool); ag_mask[:, L_ab:] = True
        # 使用'fixed_mask'的反转作为CDR区域的掩码
        cdr_mask = (1 - batch['fixed_mask']).bool()
        
        #拓扑本身不需要对 rot 有梯度。旋转力要从 feature 通路走，不是从“图结构”通路走。
        # edge_index, edge_dist_anchor, _, batch_idx_nodes, iface_ab_cdr_mask = self._build_iface_edges(ca_anchor, ab_mask, ag_mask, cdr_mask)
        edge_index, edge_is_guarantee, is_iface_node_mask = self._build_iface_graph_topology(
            ca_anchor, ab_mask, ag_mask, cdr_mask
        )


        if edge_index is None:
            # 如果没有边，返回初始特征和占位符
            n_sh = ((self.lmax + 1) ** 2) - 1
            v_placeholder = torch.zeros(B, L, n_sh, self.hidden_dim, device=device)
            # 节点特征我们返回经过Seqformer处理后的结果
            x_out = self.node_proj(ie_seq)
            iface_weight = torch.zeros(B, L, device=device, dtype=ca_var.dtype)
            if return_pos_for_grad:
                return x_out, v_placeholder, ca_var, iface_weight
            return x_out, v_placeholder, iface_weight
    
        # --- 5. 为图中的所有边计算可导特征并融合 ---
        src, dst = edge_index[0], edge_index[1]
        
        pos_flat = ca_var.view(-1, 3)
        p_i, p_j = pos_flat[src], pos_flat[dst]
        
        edge_vec = p_j - p_i
        edge_weight = edge_vec.norm(dim=-1)
        
        # =========================
        # 5.0) 计算每个 CDR 节点的 soft iface 权重（基于当前几何距离）
        # =========================
        valid_mask = batch["mask"].bool()                 # [B,L]
        cdr_node_mask = (ab_mask & cdr_mask & valid_mask) # [B,L]
        cdr_flat = cdr_node_mask.view(-1)                 # [B*L]
        ag_flat  = ag_mask.view(-1)                       # [B*L]

        src_is_cdr = cdr_flat[src]
        dst_is_cdr = cdr_flat[dst]
        src_is_ag  = ag_flat[src]
        dst_is_ag  = ag_flat[dst]

        # cross edge: CDR <-> antigen（双向边你都加了，这里两种方向都收）
        cross_mask = (src_is_cdr & dst_is_ag) | (src_is_ag & dst_is_cdr)
        if cross_mask.any():
            # 对每条 cross edge，取 CDR 端点作为聚合 index
            cdr_end = torch.where(src_is_cdr & dst_is_ag, src, dst)  # [E_cross]
            d_cross = edge_weight[cross_mask]                         # [E_cross]
            idx_cdr = cdr_end[cross_mask]                             # [E_cross]

            # idx_cdr 必须是 Long
            idx_cdr = idx_cdr.long()

            # ✅ 正确的 scatter_min：用 fill_value（更稳定、语义清晰）
            try:
                min_dist_flat, _ = scatter_min(
                    d_cross,
                    idx_cdr,
                    dim=0,
                    dim_size=B * L,
                    fill_value=1e8,
                )
            except TypeError:
                # ✅ 兼容老版本：没有 fill_value 时，用 dim_size 再手动补空位
                min_dist_flat, _ = scatter_min(
                    d_cross,
                    idx_cdr,
                    dim=0,
                    dim_size=B * L,
                )
                # 老版本对空 bucket 可能给 inf 或 0，这里统一成 1e8
                bad = ~torch.isfinite(min_dist_flat)
                if bad.any():
                    min_dist_flat = min_dist_flat.clone()
                    min_dist_flat[bad] = 1e8

            min_dist = min_dist_flat.view(B, L)

        else:
            min_dist = edge_weight.new_full((B, L), 1e8)

        # soft 权重：w_i = sigmoid((center - d_i)/tau)
        center = self.iface_soft_center
        tau = max(self.iface_soft_tau, 1e-3)
        iface_weight = torch.sigmoid((center - min_dist) / tau)       # [B,L]
        iface_weight = iface_weight * cdr_node_mask.to(iface_weight.dtype)
 
        # ============ edge_soft_weight：只软化 cross edges，拓扑固定但影响连续 ============
        edge_soft_weight = edge_weight.new_ones(edge_weight.size(0))  # [E]
        if cross_mask.any():
            center = float(self.iface_soft_center)
            tau = max(float(self.iface_soft_tau), 1e-3)
            edge_soft_weight[cross_mask] = torch.sigmoid((center - edge_weight[cross_mask]) / tau)
        # 可选：防止极端饱和
        edge_soft_weight = edge_soft_weight.clamp(0.0, 1.0)


        edge_attr_geom = self.rbf(edge_weight) # 几何特征
        
        # 归一化方向向量  
        norm = torch.norm(edge_vec, dim=1).unsqueeze(1)
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)  # 将零范数替换为1，以避免除以零
        unit_vec = edge_vec / norm
        
        # 再做球谐展开
        edge_vec_sh = self.sphere(unit_vec) 


        # --- 5.1 基于刚体旋转 R_i, R_j 和边方向 u_ij 构造“相对朝向特征” ---
        # 这里要的性质是：
        #   - 对整体刚体旋转不变（只看局部 frame 和边方向的相对关系）
        #   - 对单个残基的局部旋转敏感（rotvec 变，R 变，特征变）
        #
        # 做法：对每条边 (i->j)，计算：
        #   u_ij_global = unit_vec
        #   u_ij_in_frame_i = R_i^T @ u_ij_global
        #   u_ji_in_frame_j = R_j^T @ (-u_ij_global)
        # 然后把这两个 3 维向量 concat 成 [6] 维特征。
        
        # 取出每条边对应的 R_i, R_j
        R_i = rot_flat[src]                       # [E,3,3]
        R_j = rot_flat[dst]                       # [E,3,3]

        u_ij = unit_vec                           # [E,3]
        u_ij_neg = -u_ij                          # [E,3]

        # 转到各自局部 frame：u_local = R^T @ u_global
        u_i_local = torch.matmul(
            R_i.transpose(-1, -2),                # [E,3,3]
            u_ij.unsqueeze(-1)                    # [E,3,1]
        ).squeeze(-1)                             # [E,3]

        u_j_local = torch.matmul(
            R_j.transpose(-1, -2),
            u_ij_neg.unsqueeze(-1)
        ).squeeze(-1)                             # [E,3]

        # -------- 相对旋转：R_rel = R_i^T R_j --------
        R_rel = torch.matmul(R_i.transpose(-1, -2), R_j)  # [E,3,3]

        # 6D rotation representation：取前两列 (连续、无四元数翻转问题)
        rel6 = R_rel[..., :, :2].reshape(-1, 6)           # [E,6]
        # 可选：限幅，避免数值爆炸（正常应该在 [-1,1]）
        rel6 = rel6.clamp(-1.0, 1.0)

        # 拼成 [E,12]
        ori_feat = torch.cat([u_i_local, u_j_local, rel6], dim=-1)  # [E,12]
        ori_feat = ori_feat.clamp(-1.0, 1.0)

        geom_scale = torch.exp(self.log_geom_scale).clamp(0.01, 100.0)
        pair_scale = torch.exp(self.log_pair_scale).clamp(0.01, 100.0)
        ori_scale  = torch.exp(self.log_ori_scale).clamp(0.1, 100.0)

        edge_attr_ori = self.edge_ori_proj(ori_feat) * ori_scale


        # 上下文特征
        batch_idx_nodes = torch.arange(B, device=device).repeat_interleave(L)
        b_idx, i_loc, j_loc = batch_idx_nodes[src], src % L, dst % L
        pair_ij = ie_pair[b_idx, i_loc, j_loc] 
        pair_rbf = self.pair_to_rbf(pair_ij)
        
        # 将距离 RBF、pair RBF 和“相对朝向 RBF”一起融合给 ViSNet 使用
        if self.pair2rbf_mode == "sum":
            edge_attr_fused = geom_scale * edge_attr_geom + pair_scale * pair_rbf + edge_attr_ori
            edge_attr_fused = self.edge_attr_norm(edge_attr_fused)  # ✅ 关键
        else:
            #edge_attr_fused = self.edge_cat_proj(torch.cat([edge_attr_geom, pair_rbf], dim=-1))
            # 这里保持原先 concat，然后线性压回 num_rbf 的逻辑
            edge_attr_fused = torch.cat(
                [edge_attr_geom + edge_attr_ori, pair_rbf], dim=-1
            )  # [E, 2*num_rbf]
            edge_attr_fused = self.edge_cat_proj(edge_attr_fused)  # [E, num_rbf]

        # --- 6. 准备ViSNetBlock的输入并调用 ---
        # 节点特征: 来自Seqformer处理后的ie_seq
        x_init_flat = self.node_proj(ie_seq).view(-1, self.hidden_dim)
        vec = torch.zeros(x_init_flat.size(0), ((self.lmax + 1) ** 2) - 1, x_init_flat.size(1), device=x_init_flat.device)

        edge_attr_mp = self.edge_embedding(edge_index, edge_attr_fused, x_init_flat)


        # 调用我们重构后的、接口清晰的ViSNetBlock
        x_ctx_flat, v_ctx_flat = self.visnet(
            x=x_init_flat,
            vec=vec,
            edge_index=edge_index,
            edge_weight=edge_weight,
            edge_attr=edge_attr_mp,
            edge_vec=edge_vec_sh,
            edge_soft_weight=edge_soft_weight,   # <<< 新增：软权重进入 attention 截断
            edge_is_guarantee=edge_is_guarantee,   # <<< 新增
        )
    
        # --- 7. 恢复批处理形状并返回 ---
        x_ctx = x_ctx_flat.view(B, L, -1)
        v_ctx = v_ctx_flat.view(B, L, v_ctx_flat.shape[1], -1)
        
        if return_pos_for_grad:
            return x_ctx, v_ctx, ca_var, iface_weight
        return x_ctx, v_ctx, iface_weight


    
class InterfaceEnergy(nn.Module):
    def __init__(
        self,
        repr_cfg: Dict,          # InterfaceRepresentation 的构造参数
        readout: Dict,       # {"reduce_op": "add"/"mean", "mean": float, "std": float}
        derivative: bool = False, # 默认是否返回力
        lambda_iface: float = 1.0,   # 界面权重正则化系数（暂未使用）
        lambda_noniface: float = 0.2, # 非界面权重正则化系数（暂未使用）
        backbone=None,               # 主干网络实例（用于权重共享）
    ):
        
        super(InterfaceEnergy, self).__init__()
        # 1) 表征模块（完全使用你已有的 InterfaceRepresentation）
        self.repr = InterfaceRepresentation(**repr_cfg, backbone=backbone)

        # 2) 读出层
        self.reduce_op = readout.get("reduce_op", "add")
        hidden_channels = repr_cfg.get("hidden_dim", 256)  # 与 InterfaceRepresentation 里 ViSNet hidden 对齐
        self.readout = EquivariantScalar(hidden_channels, activation=readout.activation)

        # 3) 归一化/聚合配置
        mean_val = readout.get("mean", None)
        std_val = readout.get("std", None)
        
        mean = torch.scalar_tensor(0.0) if mean_val is None else torch.scalar_tensor(float(mean_val))
        self.register_buffer("mean", mean)
        std = torch.scalar_tensor(1.0) if std_val is None else torch.scalar_tensor(float(std_val))
        self.register_buffer("std", std)
        
        self.derivative = derivative
        self.lambda_iface = float(readout.get("lambda_iface", lambda_iface))
        self.lambda_noniface = float(readout.get("lambda_noniface", lambda_noniface))
        self.reset_parameters()

    def reset_parameters(self):
        self.repr.reset_parameters()
        self.readout.reset_parameters()
        # if self.prior_model is not None:
        #     self.prior_model.reset_parameters()

    def forward(
            self,
            seq: torch.Tensor,      # [B,L,d_seq]
            rigids7: torch.Tensor,    # [B,L,7]
            atom14_positions: torch.Tensor,    # [B,L,7]
            angles_sin_cos: torch.Tensor,    # [B,L,7] 
            atom14_exists: torch.Tensor,    # [B,L,7] 
            ie_seq: torch.Tensor,    
            ie_pair: torch.Tensor,   
            batch: Dict,
            return_force: Optional[bool] = None
        ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
            """
            返回:
            energy: [B]
            forces: [B,L,3] or None   （对 Cα 的力）
            """
            return_force = self.derivative if return_force is None else return_force

            B, L = seq.shape
            device = rigids7.device
            L_ab = batch['anchor_flag'].shape[1]

            idx = torch.arange(L, device=device)[None, :]
            ab_mask     = (idx < L_ab).expand(B, L)
            cdr_mask    = (batch['fixed_mask'] == 0)
            valid_mask  = batch['mask'].bool()
            node_mask   = (ab_mask & cdr_mask & valid_mask)  # [B,L]

            # --- repr ---
            dbg = None
            if return_force:
                x_ctx, v_ctx, ca_var_probe, iface_weight = self.repr(
                    batch=batch,
                    seq=seq,
                    rigids7=rigids7,
                    ie_seq=ie_seq,
                    ie_pair=ie_pair,
                    atom14_positions=atom14_positions,
                    angles_sin_cos=angles_sin_cos,
                    atom14_exists=atom14_exists,
                    return_pos_for_grad=True
                )
            else:
                x_ctx, v_ctx, iface_weight = self.repr(
                    batch=batch,
                    seq=seq,
                    rigids7=rigids7,
                    ie_seq=ie_seq,
                    ie_pair=ie_pair,
                    atom14_positions=atom14_positions,
                    angles_sin_cos=angles_sin_cos,
                    atom14_exists=atom14_exists,
                )

            # --- 能量读出 ---
            x_flat = x_ctx.view(B * L, -1)
            v_flat = v_ctx.view(B * L, v_ctx.shape[-2], v_ctx.shape[-1])

            node_scalar = self.readout.pre_reduce(x_flat, v_flat)
            node_scalar = node_scalar.squeeze()

            if node_scalar.dim() != 1:
                raise RuntimeError(f"node_scalar must be 1D [N], got shape={tuple(node_scalar.shape)}")

            node_scalar_norm = node_scalar * self.std + self.mean
            node_scalar_for_grad = node_scalar

            # --- 后面 alpha / scatter 聚合逻辑保持不变 ---
            w = iface_weight.to(node_scalar_norm.dtype)                 # [B,L]
            alpha = (self.lambda_noniface + (self.lambda_iface - self.lambda_noniface) * w)
            alpha = alpha * node_mask.to(alpha.dtype)                   # [B,L]
            alpha_flat = alpha.reshape(-1)
            node_scalar_weighted_for_grad = node_scalar_for_grad * alpha_flat
            node_scalar_weighted = node_scalar_norm * alpha_flat
  
            # ---- 2) masked 聚合：mean 只除以 mask 个数 ----
            batch_idx = torch.arange(B, device=device).repeat_interleave(L)
            

            # ---- 聚合：先统一算 energy_sum / denom / energy_mean ----
            energy_sum = scatter(node_scalar_weighted, batch_idx, dim=0, reduce="sum")  # [B]
            denom = scatter(alpha_flat, batch_idx, dim=0, reduce="sum").clamp_min(1e-6) # [B]
            if energy_sum.dim() != 1 or denom.dim() != 1:
                raise RuntimeError(f"energy_sum/denom must be [B], got energy_sum={tuple(energy_sum.shape)}, denom={tuple(denom.shape)}")


            energy_mean = energy_sum / denom                                            # [B]

              # ---- 对外输出（不改变你原有语义）----
            if self.reduce_op in ("add", "sum"):
                energy_out = energy_sum
            elif self.reduce_op == "mean":
                energy_out = energy_mean
            else:
                raise ValueError(f"Unsupported reduce_op: {self.reduce_op}")

            # ---- 不求力：直接返回 ----
            if not return_force:
                return energy_out, None

            # ---- 求力：用“更有信号且不被分母抵消”的标量来反传 ----
            # 核心：mean 模式用 denom.detach()，避免分母反传削弱梯度
            if self.reduce_op == "mean":
                # ✅ force 用 raw 版本聚合（并且 mean 的分母 detach）
                energy_sum_fg = scatter(node_scalar_weighted_for_grad, batch_idx, dim=0, reduce="sum")
                denom_fg = denom.detach()
                energy_for_grad = (energy_sum_fg / denom_fg).sum()
            else:
                 energy_for_grad = scatter(node_scalar_weighted_for_grad, batch_idx, dim=0, reduce="sum").sum()


            # ================= [验证/推理保护] =================
            if (not torch.is_grad_enabled()) or (not rigids7.requires_grad):
                forces = {
                    "trans": torch.zeros_like(rigids7[..., 4:]),
                    "rot":   torch.zeros_like(rigids7[..., :4]),
                }
                return energy_out, forces

            # ---- autograd ----
            (grad_rigids7,) = torch.autograd.grad(
                outputs=energy_for_grad,
                inputs=rigids7,
                retain_graph=True,
                create_graph=False,
            )
            
            forces_rot   = -grad_rigids7[..., :4]
            forces_trans = -grad_rigids7[..., 4:]
            forces = {"trans": forces_trans, "rot": forces_rot}

            return energy_out, forces

        


