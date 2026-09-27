import torch
import math 
from torch import nn
from torch.nn import functional as F

from abx.model.folding import InvariantPointAttention as IPA 
from abx.model.seqformer import EmbeddingAndSeqformer 

from abx.model.common_modules import (
        pseudo_beta_fn_v2,
        dgram_from_positions) 
from abx.model.sidechain import MultiRigidSidechain 
from abx.model.utils import batched_select 


from abx.model import quat_affine, r3, atom 
from abx.model.common_modules import(
        Linear,
        LayerNorm)


try:
    from . import fpe_operators 
except ImportError:
    import fpe_operators 

import logging
logger = logging.getLogger(__name__)

class IpaScore(nn.Module):
    def __init__(self, config_ipa, config_seqformer, diffuser):
        super(IpaScore, self).__init__()
        self.score_network_config = config_ipa 
        self.ipa_config = config_ipa.IPA 
        self.embed_config = config_ipa.embed 
        self.seqformer_config = config_seqformer
        self.diffuser = diffuser 
        

        self.embedding_and_seqformer_module = EmbeddingAndSeqformer(self.seqformer_config)
        
        num_in_seq_channel =config_seqformer.seq_channel
        num_in_pair_channel = config_seqformer.pair_channel
        effective_num_in_seq_channel = self.embed_config.index_embed_size + num_in_seq_channel
        effective_num_in_pair_channel = (2 * self.embed_config.index_embed_size) + num_in_pair_channel

        self.proj_init_seq_act = Linear(effective_num_in_seq_channel, self.ipa_config.num_channel, init='linear')
        self.proj_init_pair_act = Linear(effective_num_in_pair_channel, num_in_pair_channel, init='linear')
        self.init_seq_layer_norm = LayerNorm(self.ipa_config.num_channel)
        self.init_pair_layer_norm = LayerNorm(num_in_pair_channel)
        self.proj_seq = Linear(self.ipa_config.num_channel, self.ipa_config.num_channel, init='linear')
        self.attention_module = IPA(self.ipa_config, num_in_pair_channel)
        self.attention_layer_norm = LayerNorm(self.ipa_config.num_channel)
        transition_module_layers = []
        for k in range(self.ipa_config.num_layer_in_transition):
            is_last = (k == self.ipa_config.num_layer_in_transition - 1)
            transition_module_layers.append(
                    Linear(self.ipa_config.num_channel, self.ipa_config.num_channel, init='linear' if is_last else 'final'))
            if not is_last:
                transition_module_layers.append(nn.ReLU())
        self.transition_module = nn.Sequential(*transition_module_layers)
        self.transition_layer_norm = LayerNorm(self.ipa_config.num_channel)
        self.affine_update = Linear(self.ipa_config.num_channel, 6, init='final')
        
        self.sidechain_module = MultiRigidSidechain(
            self.ipa_config, 
            self.ipa_config.num_channel 
        )

        self.r3_fpe_operator = None
        self.so3_fpe_operator = None
        self.fpe_config = None
        if self.score_network_config.IPA.fpe.enable:
            self.fpe_config = self.score_network_config.IPA.fpe
            logger.info("FPE regularization enabled in IpaScore.")
            if self.fpe_config.vector_residual.enable:
                self.r3_fpe_operator = fpe_operators.R3ScoreFPEOperator(
                    self.diffuser._r3_diffuser,
                    self.fpe_config
                )
                logger.info("R3 FPE Operator Initialized.")
                
                self.so3_fpe_operator = fpe_operators.SO3ScoreFPEOperator(
                    self.diffuser._so3_diffuser,
                    self.fpe_config
                )
                logger.info("SO3 FPE Operator Initialized.")
            if self.fpe_config.scalar_residual.enable:
                pass

            
    def _apply_mask(self, tensor_to_mask, fixed_value_at_mask, one_minus_fixed_mask):
        mask_expanded = one_minus_fixed_mask
        while mask_expanded.ndim < tensor_to_mask.ndim:
            mask_expanded = mask_expanded.unsqueeze(-1)
        return mask_expanded * tensor_to_mask + (1.0 - mask_expanded) * fixed_value_at_mask

    def _get_time_conditioneds(self, batch, t):
        """
        Internal: Generates time-conditioned seq_act and pair_act for IPA using a specific time.
        """
        temp_batch_for_seqformer = batch.copy() 
        temp_batch_for_seqformer['t'] = t
        seq_act, pair_act, ie_seq_act, ie_pair_act = self.embedding_and_seqformer_module(temp_batch_for_seqformer) 
        
        return {'seq': seq_act, 'pair': pair_act, 'ie_seq': ie_seq_act, 'ie_pair': ie_pair_act}
    
    def _predict_z0(self, 
                                  representations, # {'seq': seq_act(t'), 'pair': pair_act(t')}
                                  current_rigids_t_unscaled, # [B,N,7] (quat, trans_unscaled)
                                  node_mask,    # [B,N]
                                  fixed_mask,   # [B,N]
                                  seq,     # [B,N] (e.g., batch['seq_t'])
                                  batch, # Original batch for residx etc.
                                  compute_sidechains_for_z0=False):
        """
        Core IPA trunk to predict z0. Uses pre-computed time-conditioned representations.
        """
        ipa_c = self.score_network_config.IPA 
        
        one_minus_fixed_mask = 1.0 - fixed_mask.type(torch.float32)
        one_minus_fixed_mask_exp = one_minus_fixed_mask.unsqueeze(-1)

        initial_quats_t = current_rigids_t_unscaled[..., :4].clone()
        initial_trans_t_unscaled = current_rigids_t_unscaled[..., 4:].clone()


        curr_quats= initial_quats_t.clone()
        curr_trans_scaled = (initial_trans_t_unscaled / ipa_c.position_scale).clone()
        
        seq_act_input_proj = self.proj_init_seq_act(representations['seq'])
        pair_act_input_proj = self.proj_init_pair_act(representations['pair'])
        seq_act_norm = self.init_seq_layer_norm(seq_act_input_proj)
        pair_act_norm = self.init_pair_layer_norm(pair_act_input_proj)
        
        seq_act = self.proj_seq(seq_act_norm)
        initial_seq_act_proj_for_sidechain = seq_act_norm.clone() 

        traj_for_z0_pred = [] 

        for fold_iter_idx in range(ipa_c.num_layer):
            curr_rots = quat_affine.quat_to_rot(curr_quats) 

            seq_act_update = self.attention_module(
                inputs_1d=seq_act, 
                inputs_2d=pair_act_norm, 
                mask=node_mask, 
                in_rigids=(curr_rots, curr_trans_scaled)
            )
            seq_act = seq_act + seq_act_update
            seq_act = F.dropout(seq_act, p=ipa_c.dropout, training=self.training) 
            seq_act = self.attention_layer_norm(seq_act)
            
            seq_act_transition_update = self.transition_module(seq_act)
            seq_act = seq_act + seq_act_transition_update
            seq_act = F.dropout(seq_act, p=ipa_c.dropout, training=self.training)
            seq_act = self.transition_layer_norm(seq_act)

            affine_updates_quat, affine_updates_trans = self.affine_update(seq_act).chunk(2, dim=-1)
            
            curr_quats = quat_affine.quat_precompose_vec(curr_quats, affine_updates_quat)
            curr_trans_scaled = r3.rigids_mul_vecs((curr_rots, curr_trans_scaled), affine_updates_trans)
            
            curr_quats = self._apply_mask(curr_quats, initial_quats_t, one_minus_fixed_mask_exp)
            curr_trans_scaled = self._apply_mask(curr_trans_scaled, initial_trans_t_unscaled / ipa_c.position_scale, one_minus_fixed_mask_exp)
            
            traj_for_z0_pred.append(
                torch.cat([
                    curr_quats.clone(), 
                    (curr_trans_scaled.clone() * ipa_c.position_scale)
                ], dim=-1)
            )
       
        pred_R0_quat = curr_quats 
        pred_R0_rot= quat_affine.quat_to_rot(pred_R0_quat)
        pred_R0_rotvecs= quat_affine.quat_to_rotvec(pred_R0_quat)
        pred_X0_T_unscaled = curr_trans_scaled * ipa_c.position_scale
       
        sidechain_for_z0 = None
        if compute_sidechains_for_z0:
            # import ipdb; ipdb.set_trace()
            final_pred_rots_for_sidechain = pred_R0_rot
            # print("final_pred_rots_for_sidechain shape:", final_pred_rots_for_sidechain.shape)
            # print("pred_X0_T_unscaled shape:", pred_X0_T_unscaled.shape)

            sidechain_for_z0 = self.sidechain_module(
                seq, 
                (final_pred_rots_for_sidechain, pred_X0_T_unscaled),
                [seq_act, initial_seq_act_proj_for_sidechain], 
                batch, 
                compute_atom_pos=True 
            )
            
        return {
            'x0_trans_unscaled': pred_X0_T_unscaled, 
            'x0_rot_rotvecs': pred_R0_rotvecs,
            'x0_rot_rot': pred_R0_rot,
            'x0_rot_quat': pred_R0_quat,
            'sidechain': sidechain_for_z0, 
            'final_trajectory': traj_for_z0_pred, 
            'final_seq_act': seq_act,
            # Return the time-conditioned representations that went INTO IPA's first projection
            # if they are needed by other heads (e.g., Distogram) or for recycling.
            'representations': representations 
        }
    
    def predict_z0_at_specific_time(self, 
                                    batch, 
                                    t):
        """
        High-level predictor called by FPE operator's _calculate_dt_s_theta.
        """
        # 1. Get time-conditioned IPA inputs using `time_for_prediction_t_prime`
        representations_at_t_prime = self._get_time_conditioneds(batch,t)

        # 2. Call core IPA trunk with these new representations
        #    Spatial state for IPA init comes from original_batch_dict['rigids_t']
        z0_predictions = self._predict_z0(
            representations_at_t_prime,
            batch['rigids_t'],
            batch['mask'],
            batch['fixed_mask'],
            batch['seq_t'],
            batch,
            compute_sidechains_for_z0=False # No sidechains for FPE intermediate calls
        )
        # Return only components needed for score calculation by diffuser
        return {
            'x0_trans_unscaled': z0_predictions['x0_trans_unscaled'],
            'x0_rot_rotvecs': z0_predictions['x0_rot_rotvecs'],
            'x0_rot_quat': z0_predictions['x0_rot_quat'] 
        }
        
    def _forward_single(self, batch, global_step): 
        outputs = {}
        current_t = batch['t']

        # 1. Get time-conditioned IPA inputs for the current batch['t']
        representations = self._get_time_conditioneds(batch, current_t)
        with torch.no_grad():
            batch_clean = dict(batch)
            batch_clean['seq_t'] = batch['seq']                 # clean seq
            batch_clean['t'] = torch.zeros_like(batch['t'])     # t=0
            reps_clean = self._get_time_conditioneds(batch_clean, batch_clean['t'])
        # 把 clean 的 ie_* 透传给 head
        outputs['ie_seq_clean']  = reps_clean['ie_seq']
        outputs['ie_pair_clean'] = reps_clean['ie_pair']

        outputs['ie_seq'] = representations['ie_seq']
        outputs['ie_pair'] = representations['ie_pair']
        
        # 2. Predict z0_theta using these current time representations
        z0_predictions = self._predict_z0(
            representations,
            batch['rigids_t'],
            batch['mask'],
            batch['fixed_mask'],
            batch['seq_t'],
            batch,
            True 
        )
        
        
            
            
        pred_X0_T_unscaled = z0_predictions['x0_trans_unscaled']
        #pred_R0_rotvecs = z0_predictions['x0_rot_rotvecs']
        pred_R0_quat = z0_predictions['x0_rot_quat']

        # Store main prediction outputs
        outputs['rigids'] = torch.cat([pred_R0_quat, pred_X0_T_unscaled], dim=-1)
        if z0_predictions['sidechain'] is not None:
            outputs['sidechains'] = z0_predictions['sidechain']


        outputs['traj'] = z0_predictions['final_trajectory']
        # Store representations that other heads might need (e.g., Distogram)
        # These should be the ones that went INTO the IPA projections (after EmbeddingAndSeqformer)
        outputs['representations'] = {
            'structure_module': z0_predictions['final_seq_act'], # Output of IPA
            'representations_for_heads': z0_predictions['representations'] # Input to IPA
        }


        # 3. Indirect scores for DSM Loss
        current_X_t_unscaled = batch['rigids_t'][..., 4:] 
        current_R_t_quat = batch['rigids_t'][..., :4]  
        
        # For s_theta_X, use unscaled X_t and unscaled predicted X0_T
        # The r3_diffuser.score method with scale_input_coords=True will handle internal scaling.
        s_theta_X_unscaled = self.diffuser.calc_trans_score(
            current_X_t_unscaled, 
            pred_X0_T_unscaled, 
            current_t, 
            True 
        )
        outputs['trans_score'] = s_theta_X_unscaled 

        
        s_theta_R_rotvec = self.diffuser.calc_quat_score(
            current_R_t_quat,
            pred_R0_quat,
            current_t
        ) # Output is an so(3) element (rotation vector)
        outputs['rot_score'] = s_theta_R_rotvec
        

        # ==================== MODIFICATION START ====================
        fpe_loss_r3 = torch.tensor(0.0, device=batch['t'].device)
        fpe_loss_so3 = torch.tensor(0.0, device=batch['t'].device)

        # 检查是否满足计算FPE的条件
        should_compute_fpe = False
        if self.fpe_config and self.fpe_config.enable:
            fpe_start_step = self.fpe_config.get('curriculum_start_step', 0)
            fpe_prob = self.fpe_config.get('compute_probability', 1.0)
            
            if global_step >= fpe_start_step and torch.rand(1).item() < fpe_prob:
                should_compute_fpe = True
            #should_compute_fpe = True
        
        # 4. FPE Calculation (if enabled and training) 物理一致性的正则化项
        #if self.fpe_config and self.fpe_config.enable:
        if should_compute_fpe:
            # 在进入FPE计算之前，缓存必要的张量
            # 使用一个专用的key，避免污染batch
            batch['_fpe_cache'] = {
                'z0_x': pred_X0_T_unscaled.detach(),
                'z0_r': pred_R0_quat.detach(),
                's_x': s_theta_X_unscaled.detach(),
                's_r': s_theta_R_rotvec.detach()
            }
            # 传递predictor函数，以备用（例如，如果配置为'central_diff'）
            batch['main_predictor_fn'] = self.predict_z0_at_specific_time


            # R3 FPE Loss Contribution
            if self.fpe_config.vector_residual.get('alpha_X', 0.0) > 0:
                r3_diffuser_obj = self.diffuser._r3_diffuser
                D_X_flat = current_X_t_unscaled.reshape(current_X_t_unscaled.shape[0], -1).shape[1]
                
                def score_fn_r3_for_G_spatial(x_t_scaled_flat, t_fixed, local_batch):
                    
                    r3_diffuser_obj = self.diffuser._r3_diffuser
                    x_t_scaled_struct = x_t_scaled_flat.reshape(local_batch['rigids_t'][..., 4:].shape)
                    
                    x0_pred_r3_unscaled_dynamic = local_batch['_fpe_cache']['z0_unscaled']
                    x0_pred_r3_scaled_dynamic = r3_diffuser_obj._scale(x0_pred_r3_unscaled_dynamic)
                    
                    s_val_scaled = r3_diffuser_obj.score(
                        x_t_scaled_struct, x0_pred_r3_scaled_dynamic, t_fixed, False)
                    return s_val_scaled.reshape(x_t_scaled_flat.shape[0], -1)
        


                loss_r3_per_sample = self.r3_fpe_operator.calculate_fpe_loss_contribution(
                    batch, # Pass original batch for predict_z0_at_specific_time
                    r3_diffuser_obj,
                    score_fn_r3_for_G_spatial,
                )
                # fpe_loss_r3 = torch.mean(loss_r3_per_sample) * \
                #                                  self.fpe_config.vector_residual.alpha_X
                fpe_loss_r3 = torch.mean(loss_r3_per_sample)    
                
    
            if self.so3_fpe_operator and self.fpe_config.vector_residual.get('alpha_R', 0.0) > 0:


                def score_fn_so3_for_G_spatial(R_t_flat_rotvecs, t_fixed, local_batch):
                    so3_diffuser_obj = self.diffuser._so3_diffuser
                    temp_quat = local_batch['rigids_t'][..., :4]
                    rotvec_shape_template = quat_affine.quat_to_rotvec(temp_quat).shape

                    Rt_struct_rotvecs = R_t_flat_rotvecs.reshape(rotvec_shape_template)

                    R0_theta_quat_dynamic = local_batch['_fpe_cache']['z0_unscaled']
                    R_t_quat = quat_affine.rotvec_to_quat(Rt_struct_rotvecs)

                    rel_rot_quat_spatial = quat_affine.quat_multiply(quat_affine.invert_quat(R0_theta_quat_dynamic), R_t_quat)
                    rel_rot_rotvec_spatial = quat_affine.quat_to_rotvec(rel_rot_quat_spatial)
                    
                    s_val_rotvec = so3_diffuser_obj.score(rel_rot_rotvec_spatial, t_fixed)
                    return s_val_rotvec.reshape(R_t_flat_rotvecs.shape[0], -1)

                #with torch.autograd.set_detect_anomaly(True):
                loss_so3_per_sample = self.so3_fpe_operator.calculate_fpe_loss_contribution(
                        batch, 
                        self.diffuser,
                        score_fn_so3_for_G_spatial,
                    )
                # fpe_loss_so3 = torch.mean(loss_so3_per_sample) * \
                #                                   self.fpe_config.vector_residual.alpha_R
                fpe_loss_so3 = torch.mean(loss_so3_per_sample)
                #import ipdb; ipdb.set_trace()
    
        outputs['fpe_loss_r3'] = fpe_loss_r3
        outputs['fpe_loss_so3'] = fpe_loss_so3
        #logger.info(f"FPE Loss R3: {fpe_loss_r3.item():.4f}, SO3: {fpe_loss_so3.item():.4f}")
    
        return outputs  
    
    def _make_cfg_overrides(self, batch, cond: bool):
        """
        基于 batch['diffused_mask'] 生成按位 α-map：
        - cond   : CDR/非CDR 全 1（保持完全条件）
        - uncond : CDR 位 0，非CDR 位 1（只在设计位拔条件）
        要求：
        - batch['diffused_mask'] : (B, L) 0/1 张量，1=需要设计/扩散的 CDR 位
        """
        device = batch['mask'].device
        dtype  = batch['mask'].dtype
        diffused_mask = 1 - batch['fixed_mask']  # (B, L), 0/1

        if cond:
            alpha_seq = torch.ones_like(diffused_mask, device=device, dtype=dtype)
            alpha_str = torch.ones_like(diffused_mask, device=device, dtype=dtype)
        else:
            # 只在 CDR 位拔条件：CDR=0，非CDR=1
            # 等价：(1 - diffused_mask)
            alpha_seq = (1.0 - diffused_mask).to(dtype=dtype)
            alpha_str = alpha_seq.clone()

        return {
            "alpha_seq_cond_map":   alpha_seq,  # (B, L), float
            "alpha_struct_cond_map": alpha_str,  # (B, L), float
        }
    
    @torch.no_grad()
    def _predict_z0_nograd(self, reps, batch):
        """no_grad 帮助函数：签名与 _predict_z0 保持一致，减少冗余代码 & 显存占用。"""
        return self._predict_z0(
            reps,
            batch['rigids_t'],
            batch['mask'],
            batch.get('fixed_mask', None),
            batch['seq_t'],
            batch,
            compute_sidechains_for_z0=False
        )


    def _forward_cfg(self, batch, cfg_config):
        """
        CFG 推理（效率优先 + 理论稳健）：
        1) 用 cond/uncond α-map 得到 reps_c / reps_u；
        2) reps_mixed = reps_u + γ * (reps_c - reps_u)；
        3) 仅对 reps_mixed 跑一次 IPA（_predict_z0）得到主输出（rigids/sidechains/traj & final_seq_act）；
        4) 用 no_grad 分别跑 reps_c / reps_u 得到 z0_c / z0_u -> 计算 score 并做 CFG 混合；
        5) outputs['representations'] 同时提供 mixed/cond/uncond，兼容后续 γ_seq 的需要。
        """
        outputs   = {}
        current_t = batch['t']

        # 1) 构造 cond / uncond 覆盖（只覆盖 α-map，不动其它条件）
        batch_cond   = dict(batch)
        batch_uncond = dict(batch)
        batch_cond.update(self._make_cfg_overrides(batch, cond=True))
        batch_uncond.update(self._make_cfg_overrides(batch, cond=False))

        # 2) 只跑一次 “表征构建”（计算量较低）
        reps_c = self._get_time_conditioneds(batch_cond,   current_t)   # {'seq':..., 'pair':...}
        reps_u = self._get_time_conditioneds(batch_uncond, current_t)

        # 3) 混合表征（γ 控结构探索半径；建议支持命令行/配置热调）
        gamma_struct = float(cfg_config.get("gamma_struct", 1.0))
        reps_mixed = {
            "seq":  reps_u["seq"]  + gamma_struct * (reps_c["seq"]  - reps_u["seq"]),
            "pair": reps_u["pair"] + gamma_struct * (reps_c["pair"] - reps_u["pair"]),
        }

        # 4) 只跑一次 IPA：主输出走混合路径（信息流自洽）
        z0_mixed = self._predict_z0(
            reps_mixed,
            batch['rigids_t'],
            batch['mask'],
            batch.get('fixed_mask', None),
            batch['seq_t'],
            batch,
            True
        )
    
        # 5) 用 no_grad 分别算 cond / uncond 的 z0 -> score -> CFG 混合（理论更稳）
        z0_c = self._predict_z0_nograd(reps_c, batch)
        z0_u = self._predict_z0_nograd(reps_u, batch)

        sX_c = self.diffuser.calc_trans_score(
            batch['rigids_t'][..., 4:],    # trans part of rigids_t
            z0_c['x0_trans_unscaled'],
            current_t,
            True
        )
        sR_c = self.diffuser.calc_quat_score(
            batch['rigids_t'][..., :4],    # quat part of rigids_t
            z0_c['x0_rot_quat'],
            current_t
        )
        sX_u = self.diffuser.calc_trans_score(
            batch['rigids_t'][..., 4:],
            z0_u['x0_trans_unscaled'],
            current_t,
            True
        )
        sR_u = self.diffuser.calc_quat_score(
            batch['rigids_t'][..., :4],
            z0_u['x0_rot_quat'],
            current_t
        )

        outputs['trans_score'] = sX_u + gamma_struct * (sX_c - sX_u)
        outputs['rot_score']   = sR_u + gamma_struct * (sR_c - sR_u)

        # 6) 主输出（来自混合路径）：rigids/sidechains/traj
        # 注意：你的实现里这些 key 的名字可能略有不同，按需对齐
        outputs['rigids']     = torch.cat([z0_mixed['x0_rot_quat'], z0_mixed['x0_trans_unscaled']], dim=-1)
        outputs['sidechains'] = z0_mixed.get('sidechain', None)
        outputs['traj']       = z0_mixed.get('final_trajectory', None)

        # 7) representations：提供 mixed/cond/uncond 的激活，方便 SequenceHead 也做 γ_seq
        outputs['representations'] = {
            'structure_module'        : z0_mixed['final_seq_act'],  # 主激活（mixed）
            'structure_module_cond'   : z0_c['final_seq_act'],
            'structure_module_uncond' : z0_u['final_seq_act'],
            # 如果你的 heads 还需要 pair/中间表征，可在 _predict_z0 返回中取并补充：
            # 'representations_for_heads_cond'  : z0_c['representations'],
            # 'representations_for_heads_uncond': z0_u['representations'],
            # 为 Recycle (get_prev) 提供它所需要的输入表征。
            # 我们选择提供 cond 分支的表征，因为它信息最完整，与非CFG模式下的行为最一致。
            'representations_for_heads' : z0_c['representations'],
        }
        
        return outputs
    
    def forward(self, batch, global_step):
        
        is_inference = not self.training
        cfg_config = getattr(self.score_network_config, 'cfg', {})
        cfg_enabled = cfg_config.get('enable', False)
        
        if is_inference and cfg_enabled:
            # ============ 推理阶段：双前向CFG逻辑 ============
            return self._forward_cfg(batch, cfg_config)
        else:
            # ============ 训练阶段或非CFG推理：单前向逻辑 ============
            return self._forward_single(batch, global_step)
    