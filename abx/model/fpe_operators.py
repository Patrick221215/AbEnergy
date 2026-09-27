import torch
from abx.model import quat_affine
import logging
try:
    from . import diff_utils 
except ImportError:
    import diff_utils 

logger = logging.getLogger(__name__)



# fpe_operators.py (FINAL and CORRECTED VERSION)

def _subsample_fpe_inputs(batch, sample_rate, is_coord_flag):
    """
    Subsamples FPE inputs AND/OR maps specific cache keys to generic keys.
    This function is now robust for both sampling and non-sampling cases.
    """
    num_res = batch['rigids_t'].shape[1]
    
    # --- 决定是否进行采样 ---
    perform_sampling = (
        batch.get('is_training', False) and 
        sample_rate < 1.0 and 
        num_res > 1
    )

    if perform_sampling:
        num_sample = max(1, int(num_res * sample_rate))
        valid_indices = torch.where(batch['mask'][0] == 1)[0]
        
        if len(valid_indices) <= num_sample:
            # 有效残基数不足，退回到不采样
            perform_sampling = False
        else:
            perm = torch.randperm(len(valid_indices), device=batch['rigids_t'].device)
            sampled_indices = valid_indices[perm[:num_sample]]
            D_total_flat_dim = num_sample * 3

    # --- 准备一个新的、干净的batch字典 ---
    # 无论是否采样，我们都创建一个新字典，以避免副作用
    batch_for_fpe = {k: v for k, v in batch.items() if not isinstance(v, torch.Tensor) or k == 't'}
    
    # --- 对Tensor进行处理 (采样或直接拷贝) ---
    keys_to_sample = ['rigids_t', 'mask', 'seq_t', 'fixed_mask']
    for key, tensor in batch.items():
        if isinstance(tensor, torch.Tensor):
            if perform_sampling and key in keys_to_sample and tensor.ndim > 1 and tensor.shape[1] == num_res:
                batch_for_fpe[key] = tensor[:, sampled_indices]
            elif key not in batch_for_fpe:
                batch_for_fpe[key] = tensor
    
    # --- 【核心修正】处理FPE缓存的映射和采样 ---
    _fpe_cache_subset = {}
    
    # 1. 确定具体的源key
    cache_key_s_specific = 's_x' if is_coord_flag else 's_r'
    cache_key_z0_specific = 'z0_x' if is_coord_flag else 'z0_r'
    
    # 2. 从原始缓存中获取tensor
    s_tensor = batch['_fpe_cache'][cache_key_s_specific]
    z0_tensor = batch['_fpe_cache'][cache_key_z0_specific]

    # 3. 如果需要，进行采样
    if perform_sampling:
        s_tensor = s_tensor[:, sampled_indices]
        z0_tensor = z0_tensor[:, sampled_indices]

    # 4. 用【抽象的】key名存入新的缓存
    _fpe_cache_subset['s_unscaled'] = s_tensor
    _fpe_cache_subset['z0_unscaled'] = z0_tensor

    # 5. 拷贝其他可能存在的缓存项
    for key, value in batch['_fpe_cache'].items():
        if key not in [cache_key_s_specific, cache_key_z0_specific]:
            _fpe_cache_subset[key] = value

    batch_for_fpe['_fpe_cache'] = _fpe_cache_subset
    
    # 如果没有采样，D_total_flat_dim返回None
    final_D_dim = D_total_flat_dim if perform_sampling else None
    
    return batch_for_fpe, final_D_dim

class BaseScoreFPEOperator:
    def __init__(self, config_fpe):
        self.fpe_config = config_fpe
        self.m_power = float(config_fpe.get('m_power', 2.0))
        self.normalize_residual = config_fpe.get('normalize_by_dim', True)
        self.fpe_weight_type = config_fpe.get('fpe_weight_type', 'constant')

    def _calculate_dt_s_theta(self, diffuser_obj_for_score, batch, t_current, is_coord):
        main_predictor_fn = batch['main_predictor_fn']
        return diff_utils.t_finite_diff_indirect_score(
            main_predictor_fn, diffuser_obj_for_score, batch, t_current, is_coord,
            delta_t_val=self.fpe_config.get('delta_t_finite_diff', 1e-3)
        )

    # 【重构】这个方法现在只负责计算最终的loss，不再处理采样和dt/ds
    def _calculate_final_loss(self, residual_flat, t_current, D_total_flat_dim, diffuser_obj_ref, is_coord_flag):
        # 5. Calculate powered residual norm per sample
        residual_norm_sq_per_sample = torch.sum(residual_flat**2, dim=1)
        
        if self.m_power == 1.0:
            powered_residual_norm = torch.sqrt(torch.clamp(residual_norm_sq_per_sample, min=1e-12))
        elif self.m_power == 2.0:
            powered_residual_norm = residual_norm_sq_per_sample
        else:
            powered_residual_norm = torch.pow(
                torch.sqrt(torch.clamp(residual_norm_sq_per_sample, min=1e-12)), 
                self.m_power
            )
        
        # 6. Apply time weighting factor
        weight_factor_lambda_fp = torch.ones_like(t_current)
        
        # 你的 g(t) 权重计算逻辑...
        if is_coord_flag:
            g_val = diffuser_obj_ref.diffusion_coef(t_current)
        else:
            # 对于SO3，需要访问_so3_diffuser
            g_val = diffuser_obj_ref._so3_diffuser.diffusion_coef(t_current) if hasattr(diffuser_obj_ref, '_so3_diffuser') else diffuser_obj_ref.diffusion_coef(t_current)

        current_g_sq_val = g_val**2
        
        if self.fpe_weight_type == 'g_sq':
            weight_factor_lambda_fp = current_g_sq_val.squeeze()
        elif self.fpe_weight_type == 'g':
            g_val_sq = torch.sqrt(current_g_sq_val).squeeze()
            g_batch_rms = torch.sqrt(torch.mean(g_val_sq ** 2) + 1e-9)
            weight_factor_lambda_fp = g_val_sq / g_batch_rms
        
        final_loss_term_per_sample = weight_factor_lambda_fp * powered_residual_norm

        # 7. Normalize
        if self.normalize_residual and D_total_flat_dim > 0:
            final_loss_term_per_sample /= (D_total_flat_dim ** (self.m_power / 2.0))
        
        return final_loss_term_per_sample
    
    # 子类需要实现这个方法
    def _calculate_G_specific(self, s_theta_flat, x_flat, t, score_fn, diffuser, D_flat):
        raise NotImplementedError

    # 【核心】这是新的统一入口，但具体实现由子类完成
    def calculate_fpe_loss_contribution(self, batch, diffuser_obj_ref, score_fn_for_spatial_derivs, is_coord_flag):
        raise NotImplementedError
    
    
    # fpe_operators.py

class R3ScoreFPEOperator(BaseScoreFPEOperator):
    def __init__(self, r3_diffuser_ref, config_fpe):
        super().__init__(config_fpe)
        self.r3_diffuser = r3_diffuser_ref

    def _calculate_G_specific(self, s_theta_x_t_flat_scaled, x_t_perturbed_flat_scaled, t_current,
                              score_fn_callable_for_spatial_derivs, diffuser_obj_is_self_r3, D_X_from_input):
        # 您的 get_L_X_operator_terms_vp 和 gradient_autograd 逻辑保持不变...
        with torch.enable_grad():
            L_X_s_theta_scalar_field = self.r3_diffuser.get_L_X_operator_terms_vp(
                s_theta_x_t_flat_scaled, x_t_perturbed_flat_scaled, t_current,
                score_fn_callable_for_spatial_derivs, D_X_from_input
            )
            rhs_G_X_flat_scaled = diff_utils.gradient_autograd(
                y_scalar_batch_sum=L_X_s_theta_scalar_field.sum(), 
                x_input=x_t_perturbed_flat_scaled,
                retain_graph=True, create_graph=False
            )
        return rhs_G_X_flat_scaled

    # 【核心】R3的FPE计算总入口
    def calculate_fpe_loss_contribution(self, batch, diffuser_obj_ref, score_fn_for_spatial_derivs):
        is_coord_flag = True
        t_current = batch['t']
        # 1. Subsampling (Corrected Implementation)
        sample_rate = self.fpe_config.get('subsample_rate', 1.0)
        
        # 1. Subsampling (调用辅助函数)
        batch_for_fpe, D_total_flat_dim_sub = _subsample_fpe_inputs(
            batch, sample_rate, is_coord_flag
        )
        s_theta_unscaled = batch_for_fpe['_fpe_cache']['s_unscaled']
        x0_pred_unscaled = batch_for_fpe['_fpe_cache']['z0_unscaled']
    
        if D_total_flat_dim_sub is None: # 如果没有进行采样
            D_total_flat_dim = s_theta_unscaled.reshape(s_theta_unscaled.shape[0], -1).shape[1]
        else:
            D_total_flat_dim = D_total_flat_dim_sub
        
       

        # 2. Calculate d(s)/dt
        time_est_method = self.fpe_config.get('time_est_method', 'frozen_z0')
        if time_est_method == 'frozen_z0':
            dt_s_theta_scaled_flat = diff_utils.analytical_dt_s_theta(
                diffuser_obj_ref, s_theta_unscaled, x0_pred_unscaled, t_current, is_coord_flag)
        else:
            # 有限差分法应该作用在采样后的batch上
            dt_s_theta_scaled_flat = self._calculate_dt_s_theta(
                diffuser_obj_ref, batch_for_fpe, t_current, is_coord_flag)
        
        # 3. Prepare x_t for G[s]
        x_t_spatial_unscaled = batch_for_fpe['rigids_t'][..., 4:]
        x_t_spatial_scaled = diffuser_obj_ref._scale(x_t_spatial_unscaled.detach().clone())
        x_t_spatial_scaled_flat = x_t_spatial_scaled.reshape(x_t_spatial_scaled.shape[0], -1)
        x_t_spatial_scaled_flat.requires_grad_(True)
        
        # 4. Calculate G[s]
        hutchinson_callable = lambda x, t: score_fn_for_spatial_derivs(x, t, batch_for_fpe)
    
        s_theta_for_G_scaled_flat = score_fn_for_spatial_derivs(x_t_spatial_scaled_flat, t_current, batch_for_fpe)
        rhs_G_scaled_flat = self._calculate_G_specific(
            s_theta_for_G_scaled_flat, x_t_spatial_scaled_flat, t_current,
            hutchinson_callable, diffuser_obj_ref, D_total_flat_dim)
        
        # 5. Calculate Residual
        residual_flat = dt_s_theta_scaled_flat - rhs_G_scaled_flat
        
        # 6. Calculate Final Loss using Base class method
        return self._calculate_final_loss(residual_flat, t_current, D_total_flat_dim, diffuser_obj_ref, is_coord_flag)



class SO3ScoreFPEOperator(BaseScoreFPEOperator):
    """
    FPE operator for SO(3) space, which uses a VE-SDE.
    This class handles its own logic for subsampling and calculating
    the time derivative and spatial operator G[s] for rotations.
    """
    def __init__(self, so3_diffuser_ref, config_fpe):
        """
        Args:
            so3_diffuser_ref: An instance of the SO3Diffuser.
            config_fpe: The FPE configuration dictionary.
        """
        super().__init__(config_fpe)
        self.so3_diffuser = so3_diffuser_ref
        self.config_fpe = config_fpe

    def _calculate_G_specific(
        self, 
        s_theta_R_t_rotvecs_flat, 
        R_t_rotvecs_flat, 
        t_current,
        score_fn_callable_for_spatial_derivs_so3, 
        diffuser_obj_is_self_so3,  # This should be the FullDiffuser instance
        D_R_flat_from_input
    ):
        """
        Calculates the G_R operator for SO(3).
        This involves computing L_R and then its Euclidean gradient on the rotvec space.
        """
        with torch.enable_grad():
            use_exact_divergence_flag = self.config_fpe.get('so3_divergence_exact', False)
            
            # get_L_R_operator_terms is a method of SO3Diffuser
            L_R_s_R_scalar_field = self.so3_diffuser.get_L_R_operator_terms(
                s_theta_R_t_rotvecs_flat,
                R_t_rotvecs_flat,
                t_current,
                score_fn_callable_for_spatial_derivs_so3,
                D_R_flat_from_input,
                use_exact_divergence_flag
            )

            # G_R = nabla_R(L_R), approximated by Euclidean gradient on rotvecs
            rhs_G_flat = diff_utils.gradient_autograd(
                y_scalar_batch_sum=L_R_s_R_scalar_field.sum(),
                x_input=R_t_rotvecs_flat,
                retain_graph=True,
                create_graph=False
            )
        return rhs_G_flat

    def calculate_fpe_loss_contribution(self, batch, diffuser_obj_ref, score_fn_for_spatial_derivs):
        """
        Main entry point for calculating the SO(3) FPE loss contribution.
        This method orchestrates subsampling, and calculation of d(s)/dt and G[s].

        Args:
            batch (dict): The full data batch, containing '_fpe_cache', 'rigids_t', etc.
            diffuser_obj_ref (FullDiffuser): The main diffuser instance.
            score_fn_for_spatial_derivs (callable): Lambda function for G[s] calculation.
        
        Returns:
            torch.Tensor: A tensor of shape [B] with the FPE loss per sample.
        """
        is_coord_flag = False
        t_current = batch['t']

        # 1. Subsampling Logic (Corrected and self-contained)
        sample_rate = self.fpe_config.get('subsample_rate', 1.0)
        batch_for_fpe, D_total_flat_dim_sub = _subsample_fpe_inputs(
        batch, sample_rate, is_coord_flag
    )
        s_theta_unscaled = batch_for_fpe['_fpe_cache']['s_unscaled']
        x0_pred_unscaled = batch_for_fpe['_fpe_cache']['z0_unscaled']

        if D_total_flat_dim_sub is None: # 如果没有进行采样
            D_total_flat_dim = s_theta_unscaled.reshape(s_theta_unscaled.shape[0], -1).shape[1]
        else:
            D_total_flat_dim = D_total_flat_dim_sub
        
        

        # 2. Calculate d(s)/dt
        time_est_method = self.fpe_config.get('time_est_method', 'frozen_z0')
        if time_est_method == 'frozen_z0':
            dt_s_theta_flat = diff_utils.analytical_dt_s_theta(
                self.so3_diffuser,  # Pass the correct diffuser instance
                s_theta_unscaled,
                x0_pred_unscaled,
                t_current,
                is_coord_flag
            )
        else:
            # Fallback to finite difference on the (potentially subsampled) batch
            dt_s_theta_flat = self._calculate_dt_s_theta(
                diffuser_obj_ref, batch_for_fpe, t_current, is_coord_flag
            )

        # 3. Prepare x_t for G[s] calculation
        # SO(3) operates on unscaled rotation vectors
        x_t_spatial_unscaled = quat_affine.quat_to_rotvec(batch_for_fpe['rigids_t'][..., :4])
        
        x_t_spatial_flat = x_t_spatial_unscaled.detach().clone().reshape(x_t_spatial_unscaled.shape[0], -1)
        x_t_spatial_flat.requires_grad_(True)

        # 4. Calculate G[s]       
    
        hutchinson_callable = lambda x, t: score_fn_for_spatial_derivs(x, t, batch_for_fpe)

        
        s_theta_for_G_flat = score_fn_for_spatial_derivs(x_t_spatial_flat, t_current, batch_for_fpe)
        rhs_G_flat = self._calculate_G_specific(
            s_theta_for_G_flat,
            x_t_spatial_flat,
            t_current,
            hutchinson_callable,
            diffuser_obj_ref, # Pass the FullDiffuser
            D_total_flat_dim
        )
        
        # Clean up gradient on the temporary tensor
        if x_t_spatial_flat.grad is not None:
            x_t_spatial_flat.grad.zero_()
        x_t_spatial_flat.requires_grad_(False)

        # 5. Calculate Residual
        # Both dt_s_theta_flat and rhs_G_flat are in the unscaled rotvec space
        residual_flat = dt_s_theta_flat - rhs_G_flat
        
        # 6. Calculate Final Loss using Base class method
        return self._calculate_final_loss(
            residual_flat, t_current, D_total_flat_dim, diffuser_obj_ref, is_coord_flag
        )