import sys
import functools

from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from einops import rearrange

from abx.common import residue_constants

from abx.model import atom, quat_affine
from abx.utils import *
from abx.model.common_modules import(
        Linear,
        LayerNorm)
from abx.model.utils import squared_difference, plddt, batched_select
from abx.model.score_network import IpaScore
from abx.model.interface.energy_head import InterfaceEnergy
from abx.model.interface.guidance_utils import calculate_cdr_rmsd

class DistogramHead(nn.Module):
    """Head to predict a distogram.
    """
    def __init__(self, config, num_in_channel):
        super().__init__()

        c = config

        self.breaks = torch.linspace(c.first_break, c.last_break, steps=c.num_bins-1)
        self.proj = Linear(num_in_channel+2*c.index_embed_size, c.num_bins, init='final')

        self.config = config

    def forward(self, headers, representations, batch):

        x = representations['pair']
        x = self.proj(x)
        logits = (x + rearrange(x, 'b i j c -> b j i c')) * 0.5
        breaks = self.breaks.to(logits.device)  
        return dict(logits=logits, breaks=breaks)

    
class DiffusionHead(nn.Module):
    """Head to Diffusion 3d struct.
    """
    def __init__(self, config, config_seqformer, diffuser):
        super().__init__()
        self.ScoreNetwork = IpaScore(config, config_seqformer, diffuser)


    def forward(self, headers, batch, global_step):

        return self.ScoreNetwork(batch, global_step)


class MetricDict(dict):
    def __add__(self, o):
        n = MetricDict(**self)
        for k in o:
            if k in n:
                n[k] = n[k] + o[k]
            else:
                n[k] = o[k]
        return n

    def __mul__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] * o
        return n

    def __truediv__(self, o):
        n = MetricDict(**self)
        for k in n:
            n[k] = n[k] / o
        return n

class MetricDictHead(nn.Module):
    """Head to calculate metrics
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, headers, representations, batch):

        metrics = MetricDict()
        if 'distogram' in headers:
            assert 'logits' in headers['distogram'] and 'breaks' in headers['distogram']
            logits, breaks = headers['distogram']['logits'], headers['distogram']['breaks']
            positions = batch['pseudo_beta']
            # positions = torch.cat((positions, batch['antigen_pseudo_beta']), dim=1)
            mask = batch['pseudo_beta_mask']
            # mask = torch.cat((mask, batch['antigen_pseudo_beta_mask']), dim=-1)
            cutoff = self.config.get('contact_cutoff', 8.0)
            t =  torch.sum(breaks <= cutoff)
            pred = F.softmax(logits, dim=-1)
            pred = torch.sum(pred[...,:t+1], dim=-1)
            #truth = torch.cdist(positions, positions, p=2)
            truth = torch.sqrt(torch.sum(squared_difference(positions[:,:,None], positions[:,None]), dim=-1))
            precision_list = contact_precision(
                    pred, truth, mask=mask,
                    ratios=self.config.get('contact_ratios'),
                    ranges=self.config.get('contact_ranges'),
                    cutoff=cutoff)
            metrics['contact'] = MetricDict()
            for (i, j), ratio, precision in precision_list:
                i, j = default(i, 0), default(j, 'inf')
                metrics['contact'][f'[{i},{j})_{ratio}'] = precision
        return dict(loss=metrics) if metrics else None

class TMscoreHead(nn.Module):
    """Head to predict TM-score.
    """
    def __init__(self, config):
        super().__init__()

        self.config = config

    def forward(self, headers, representations, batch):

        c = self.config
        # only for CA atom
        if 'atom14_gt_positions' in batch and 'atom14_gt_exists' in batch:
            preds, labels = headers['folding']['final_atom_positions'][...,1,:].detach(), batch['atom14_gt_positions'][...,1,:].detach()
            gt_mask = batch['atom14_gt_exists'][...,1]

            #import ipdb; ipdb.set_trace()
            tmscore = 0.
            for b in range(preds.shape[0]):
                mask = gt_mask[b]
                pred_aligned, label_aligned = Kabsch(
                        rearrange(preds[b][mask], 'c d -> d c'),
                        rearrange(labels[b][mask], 'c d -> d c'))

                tmscore += TMscore(pred_aligned[None,:,:], label_aligned[None,:,:], L=torch.sum(mask).item())

            return dict(loss = tmscore / preds.shape[0])
        return None
    
class SequenceHead(nn.Module):
    """Head to Diffusion 3d struct.
    """
    def __init__(self, config, num_res=20):
        super().__init__()
        c = config
        dim = c.num_channel
        hidden_dim = c.num_hidden_channel
        self.net = nn.Sequential(
            LayerNorm(dim),
            Linear(dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, num_res, init='relu', bias=True),

        )
        self.config = config

    def forward(self, headers, representations, batch):
        """
        SequenceHead's forward pass. Implements CFG logic for logits mixing.
        """
        # 1. 断言检查，确保上游模块已运行
        assert 'folding' in headers, "'folding' head (IpaScore) must be run before SequenceHead."
        
        reps_dict = headers['folding']['representations']
        
        # 2. 安全地获取CFG配置
        #    getattr(self.config, 'CFG', {}) 是一种更安全的写法，以防'CFG'键不存在
        cfg_config = self.config.get('cfg', {})
        is_inference = not self.training
        cfg_enabled = cfg_config.get('enable', False)

        # 3. 核心CFG逻辑判断
        is_cfg_inference_path = is_inference and cfg_enabled

        if is_cfg_inference_path:
            # ============ CFG推理模式：双前向 + logits混合 ============

            act_c = reps_dict.get('structure_module_cond')
            act_u = reps_dict.get('structure_module_uncond')

            # 健壮性检查：如果上游没有提供双通道激活，则安全退化
            if act_c is None or act_u is None:
                logger.warning("CFG is enabled for SequenceHead, but conditional/unconditional activations are missing. Falling back to single path.")
                act = reps_dict.get('structure_module')
                final_logits = self.net(act)
            else:
                # 4. 分别通过MLP网络，得到cond和uncond的logits
                logits_c = self.net(act_c)
                logits_u = self.net(act_u)
                
                # 5. 从配置中读取gamma_seq的值，并进行混合
                gamma_seq = float(cfg_config.get('gamma_seq', 1.0))
                final_logits = logits_u + gamma_seq * (logits_c - logits_u)
                # logger.debug(f"SequenceHead: CFG applied with gamma_seq={gamma_seq}")

        else:
            # ============ 训练模式 或 非CFG推理模式：单前向 ============
            
            # 行为与原始代码完全一致
            # 使用.get()确保即使在CFG推理模式下，如果只提供了主激活，也能正常工作
            act = reps_dict.get('structure_module', reps_dict.get('structure_module_cond'))
            if act is None:
                raise ValueError("Could not find 'structure_module' or 'structure_module_cond' in representations.")
            final_logits = self.net(act)
                
        p_0t = F.softmax(final_logits, dim=-1)
        seq_0 = torch.max(p_0t, dim=-1)[1]
        fixed_mask = batch['fixed_mask']
        seq_0 = seq_0 * (1 - fixed_mask) + batch['seq_t'] * fixed_mask
        
        # --- (全原子重建部分，代码保持不变) ---
        angles = headers['folding']['sidechains']['angles_sin_cos']
        rigids = headers['folding']['rigids']
        rots = quat_affine.quat_to_rot(rigids[...,:4])
        trans = rigids[..., 4:]
        backb_to_global = (rots, trans)

        all_frames_to_global = atom.torsion_angles_to_frames(seq_0, backb_to_global, angles)
        pred_positions = atom.frames_and_literature_positions_to_atom14_pos(seq_0, all_frames_to_global)
        final_atom_positions = batched_select(pred_positions, batch['residx_atom37_to_atom14'], batch_dims=2)
        
        atom14_atom_exists = batched_select(torch.tensor(residue_constants.restype_atom14_mask, device=seq_0.device), seq_0)
        atom37_atom_exists = batched_select(torch.tensor(residue_constants.restype_atom37_mask, device=seq_0.device), seq_0)

        headers['folding'].update(
            final_atom14_positions = pred_positions,
            final_atom_positions = final_atom_positions,
            atom14_atom_exists = atom14_atom_exists,
            atom37_atom_exists = atom37_atom_exists,
        )

        headers['folding']['sidechains'].update(
            atom_pos = pred_positions,
            frames = all_frames_to_global
        )

        # 最终返回混合后的logits和基于它生成的seq_0
        return dict(logits=final_logits, seq_0=seq_0)
    

class PredictedLDDTHead(nn.Module):
    def __init__(self, config, bins=50):
        super().__init__()
        c = config
        dim = c.num_channel
        hidden_dim = c.num_hidden_channel
        self.net = nn.Sequential(
            LayerNorm(dim),
            Linear(dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, hidden_dim, init='relu', bias=True),
            nn.ReLU(),
            Linear(hidden_dim, bins, init='relu', bias=True),

        )
        self.config = config

    def forward(self, headers, representations, batch):

        assert 'folding' in headers
        act = headers['folding']['representations']['structure_module']
        logits = self.net(act)
        pLDDT = plddt(logits)
        return dict(logits=logits, pLDDT=pLDDT)
    

class PredictedEnergyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.energy_config = config
        if self.energy_config.enable:
            self.energy_head = InterfaceEnergy(self.energy_config['repr_cfg'],
                                                self.energy_config['readout'],
                                                self.energy_config['derivative']
                                                )
        # 不再直接用 float，而是可学习 log_temp
        self.log_bce_temp = nn.Parameter(torch.log(torch.tensor(float(self.energy_config.bce_temp))))
    def forward(self, headers, representations, batch):
        #计算物理能量对
        aux_energy_pair = {}
        if self.energy_config.enable:    
            # 1. 准备输入数据
            # 建议：为了口径一致，Pred 和 GT 都尽量只基于原子坐标 (angles_sin_cos=None)
            # 或者 Pred 用 Pred 角度，GT 用 GT 角度
            pred_rigids = headers['folding']['rigids']
            pred_atom14 = headers['folding']['final_atom14_positions']
            pred_atom14_exists = headers['folding']['atom14_atom_exists']
            angles_pred = headers['folding']['sidechains']['angles_sin_cos']
            
            # seq_t = ['heads']['sequence_module']['seq_0']

            # 2. 计算 Pred 能量 (Student Path, 带梯度)
            # 梯度流向：EnergyHead(Frozen) -> pred_atom14 -> Backbone
            E_pred, forces = self.energy_head(
                seq=batch['seq_t'], 
                rigids7=pred_rigids, 
                atom14_positions=pred_atom14,
                angles_sin_cos=angles_pred, 
                atom14_exists=pred_atom14_exists, 
                ie_seq =headers['folding']['ie_seq'],
                ie_pair =headers['folding']['ie_pair'],
                batch=batch,
                return_force=True,
            )


            # 1) GT x0 能量
            E0, _ = self.energy_head(
                seq=batch['seq'],                    # clean seq 更合理
                rigids7=batch['rigids_0'],
                atom14_positions=batch['atom14_gt_positions'],
                angles_sin_cos=batch['torsion_angles_sin_cos'],
                atom14_exists=batch['atom14_gt_exists'],
                ie_seq=headers['folding']['ie_seq_clean'], 
                ie_pair=headers['folding']['ie_pair_clean'],
                batch=batch,
                return_force=False,
            )

            # 学习到的温度（限制在 [1, 50] 防止爆炸）
            bce_temp = self.log_bce_temp.exp().clamp(1.0, 50.0)
            
            # BCE (基于 CDR RMSD)
            rmsd_pred  = calculate_cdr_rmsd(pred_rigids,  batch['rigids_0'], batch)
        
            logit_pred    = -E_pred  / bce_temp
            
            aux_energy_pair = {
                'E0_gt': E0,
                'pred': E_pred,
                'iface_logit': logit_pred,
                'rmsd': rmsd_pred,
                'forces': forces
            }
        return aux_energy_pair
    

class HeaderBuilder:
    @staticmethod
    def build(config, config_seqformer, parent, diffuser=None, cfg_config={}):
        head_factory = OrderedDict(
                diffusion_module = functools.partial(DiffusionHead, 
                config_seqformer=config_seqformer,diffuser=diffuser),
                sequence_module = functools.partial(SequenceHead) ,
                distogram = functools.partial(DistogramHead, num_in_channel=config_seqformer.pair_channel),
                metric = functools.partial(MetricDictHead),
                tmscore = functools.partial(TMscoreHead),
                predicted_lddt = functools.partial(PredictedLDDTHead),
                interface_energy = functools.partial(PredictedEnergyHead))

        def gen():
            for head_name, h in head_factory.items():
                if head_name not in config:
                    continue
                head_config = config[head_name]
                
                # # ---> [核心修改 6] <---
                # # 将顶层CFG配置，注入到每个Head自己的配置中
                # # 这样每个Head都能通过self.config.get('cfg', {})来访问它
                # if cfg_config:
                #     head_config['cfg'] = cfg_config
                    
                head = h(config=head_config)

                if isinstance(parent, nn.Module):
                    parent.add_module(head_name, head)
                
                if head_name == 'diffusion_module':
                    head_name = 'folding'
                yield head_name, head, head_config

        return list(gen())
