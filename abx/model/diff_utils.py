import torch
from abx.model import quat_affine
import math
import logging

logger = logging.getLogger(__name__)

def gradient_autograd(y_scalar_batch_sum, x_input, retain_graph=True, create_graph=True):
    """
    Computes d(y_scalar_batch_sum)/dx_input where y_scalar_batch_sum is typically y.sum().
    Args:
        y_scalar_batch_sum: A scalar tensor (e.g., sum of a batch of scalar values).
        x_input: The tensor with respect to which the gradient is computed. Must have requires_grad=True.
        retain_graph: Passed to torch.autograd.grad.
        create_graph: Passed to torch.autograd.grad.
    Returns:
        Gradient tensor of the same shape as x_input.
    """
    
    if not x_input.requires_grad:
        # For safety, ensure requires_grad is set. This might be redundant if caller handles it.
        # However, if x_input is a non-leaf tensor that had its grad status turned off,
        # this might not re-enable it correctly for the graph.
        # It's best if the caller ensures x_input is a leaf or has .retain_grad() called
        # and .requires_grad is True *before* this function is called if it's part of a graph.
        logger.warning("gradient_autograd: x_input did not require grad. Enabling. Ensure this is intended.")
        x_input.requires_grad_(True) 
            
    if x_input.grad is not None:
        x_input.grad.zero_() # Zero out any existing gradients on x_input
        
    grads = torch.autograd.grad(
        outputs=y_scalar_batch_sum,
        inputs=x_input,
        retain_graph=retain_graph,
        create_graph=create_graph, 
        allow_unused=False # Ensure x_input is part of the graph leading to y_scalar_batch_sum
    )[0]

    return grads



def hutchinson_divergence_autograd(score_fn_of_x_only, x_input_flat, t_current, num_samples=1):
    """
    Computes div_X s(X) using Hutchinson's trace estimator: E_v[v^T grad_x (s(x)^T v)].
    This is the formulation Tr(J) = E[v^T J v] which is E[v^T d(s)/dx v] if J is symmetric,
    or more generally E_v [ v^T grad_x ( sum_i s_i(x) v_i ) ] for Tr(J_s) = sum_i ds_i/dx_i.
    The latter matches FPDiffusion's diff.py implementation.

    Args:
        score_fn_of_x_only: A function that takes x_input_flat [B, D_X] (requires_grad=True) 
                            and returns scores [B, D_X]. Time 't' is considered fixed.
        x_input_flat: Input tensor [B, D_X]. Must have requires_grad=True when passed.
        num_samples: Number of random vectors v for Hutchinson's estimator.
    Returns:
        div_s_estimate: Estimated divergence [B].
    """
    if not x_input_flat.requires_grad:
        raise ValueError("x_input_flat must have requires_grad=True for hutchinson_divergence_autograd.")

    B, D_X = x_input_flat.shape
    div_s_estimate_sum = torch.zeros(B, device=x_input_flat.device)

    #import ipdb; ipdb.set_trace()
    with torch.enable_grad():  
        for _ in range(num_samples):
            v_noise = torch.randn_like(x_input_flat) # Standard Gaussian vector

            # Calculate s(x) * v (dot product for each batch element)
            
            s_output = score_fn_of_x_only(x_input_flat, t_current) # [B, D_X]
            s_output_dot_v_noise = torch.sum(s_output * v_noise, dim=1) # [B]

        
            grad_s_output_dot_v_noise_wrt_x = torch.autograd.grad(
                outputs=s_output_dot_v_noise.sum(), # Sum over batch to make it scalar
                inputs=x_input_flat,
                retain_graph=True, # Keep graph if num_samples > 1 or for higher-order FPE terms
                create_graph=True # Typically False for FPE loss calculation
            )[0] # [B, D_X]
            
            # Calculate v^T [grad_x (s(x)^T v)]
            div_s_estimate_sum += torch.sum(grad_s_output_dot_v_noise_wrt_x * v_noise, dim=1) # [B]
    N = D_X // 3         
    div_s_estimate_sum = div_s_estimate_sum / N
    return div_s_estimate_sum / num_samples

def hutchinson_divergence_autogradR(score_fn_of_x_only, x_input_flat, t_current, num_samples=1):
    """
    Computes div_X s(X) using Hutchinson's trace estimator: E_v[v^T grad_x (s(x)^T v)].
    This is the formulation Tr(J) = E[v^T J v] which is E[v^T d(s)/dx v] if J is symmetric,
    or more generally E_v [ v^T grad_x ( sum_i s_i(x) v_i ) ] for Tr(J_s) = sum_i ds_i/dx_i.
    The latter matches FPDiffusion's diff.py implementation.

    Args:
        score_fn_of_x_only: A function that takes x_input_flat [B, D_X] (requires_grad=True) 
                            and returns scores [B, D_X]. Time 't' is considered fixed.
        x_input_flat: Input tensor [B, D_X]. Must have requires_grad=True when passed.
        num_samples: Number of random vectors v for Hutchinson's estimator.
    Returns:
        div_s_estimate: Estimated divergence [B].
    """
    if not x_input_flat.requires_grad:
        raise ValueError("x_input_flat must have requires_grad=True for hutchinson_divergence_autograd.")

    B, D_X = x_input_flat.shape
    div_s_estimate_sum = torch.zeros(B, device=x_input_flat.device)

    #import ipdb; ipdb.set_trace()
    with torch.enable_grad():  
        for _ in range(num_samples):
            v_noise = torch.randn_like(x_input_flat) # Standard Gaussian vector

            # Calculate s(x) * v (dot product for each batch element)
            
            s_output = score_fn_of_x_only(x_input_flat, t_current) # [B, D_X]
            s_output_dot_v_noise = torch.sum(s_output * v_noise, dim=1) # [B]


            # Calculate grad_x ( s(x)^T v )
            # We need to sum s_output_dot_v_noise over the batch to pass a scalar to autograd
            grad_s_output_dot_v_noise_wrt_x = torch.autograd.grad(
                outputs=s_output_dot_v_noise.sum(), # Sum over batch to make it scalar
                inputs=x_input_flat,
                retain_graph=True, # Keep graph if num_samples > 1 or for higher-order FPE terms
                create_graph=True # Typically False for FPE loss calculation
            )[0] # [B, D_X]
            
            # Calculate v^T [grad_x (s(x)^T v)]
            div_s_estimate_sum += torch.sum(grad_s_output_dot_v_noise_wrt_x * v_noise, dim=1) # [B]
    #import ipdb; ipdb.set_trace()
    return div_s_estimate_sum / num_samples



def exact_div_so3( # Renamed for clarity
    so3_diffuser_obj, 
    rotvec: torch.Tensor, 
    t: torch.Tensor,
    eps_theta_norm: float  = 1e-8,     # 计算 θ=‖ω‖ 时的最小值，避免 0      
    eps_theta_denom: float = 1e-3, # Clamp theta in 2k/theta denominator (was _CLAMP_TH)
    theta0_taylor: float = 1e-2, # Threshold for using 3*kappa'(0) approx
) -> torch.Tensor:

    """
    **解析计算** IGSO(3) Score `s_R` 的散度，公式（Methods, Eq. 14）：
        div s = κ'(θ,t) + 2 κ(θ,t) / θ

    泰勒极限校正：
        当 θ < θ0 (默认 0.01 rad) 时，
        div ≈ 3 κ'(θ0 , t)     —— 见正文“Numerical Stability”推导。

    结果对每个样本按原子 **平均**，形状 [B]；
    若需要总和，可自行乘以 N_atoms。
    ------------------------------------------------------------------
    参数
    ----
    so3_diffuser_obj : SO3Diffuser
        必须实现 _interp_kappa_and_grad(theta,t) → (κ, κ′)
    rotvec : torch.Tensor
        平铺的旋转向量，形状可以是
        • [B, 3N]  —— 本函数会 reshape(-1,3)
        • [B, N, 3]
    t : torch.Tensor
        time；shape [B]
    eps_theta_norm
        计算 θ 时的最小值，避免 nan/0
    eps_theta_denom
        分母 θ→0 时的 clamp，用于 2κ/θ
    theta0_taylor
        θ0，小于它用泰勒极限 (3 κ′(θ0))
    ------------------------------------------------------------------
    返回
    ----
    div_mean_over_atoms : torch.Tensor  [B]
        每个 batch 样本的散度均值
    """
    
    if not hasattr(so3_diffuser_obj, '_interp_kappa_and_grad'):
        raise AttributeError("so3_diffuser_obj must have _interp_kappa_and_grad method.")

    # 1. Reshape rotvec and calculate theta = ||omega||
    original_ndim = rotvec.ndim
    if original_ndim == 2: 
        B, D_flat = rotvec.shape # Capture B from here
        if D_flat % 3 != 0: raise ValueError("Flat rotvec not divisible by 3.")
        num_atoms = D_flat // 3
        omega = rotvec.view(B, num_atoms, 3)
    elif original_ndim == 3 and rotvec.shape[-1] == 3:
        omega = rotvec
        B, num_atoms, _ = omega.shape
    else:
        raise ValueError(f"Unsupported rotvec shape: {rotvec.shape}")

    # Calculate original theta for mask, clamp for interpolation
    theta_orig = omega.norm(dim=-1).clamp(min=eps_theta_norm)   # [B,N]  # Shape [B, N_atoms]
    # theta_for_interp should be clamped to the range of omega_table if necessary,
    # or _interp_kappa_and_grad handles boundary conditions robustly.
    # Let's use a small clamp for safety before interpolation.

    # 2. Interpolate KAPPA and DKAPPA/DTHETA
    # These are now assumed to be reasonably bounded due to clipping in _interp_kappa_and_grad
    # ------------------------------------------------------------
    # 2. 插值 κ(θ,t) 与 κ'(θ,t)
    # ------------------------------------------------------------
    kappa, dkappa = so3_diffuser_obj._interp_kappa_and_grad(theta_orig, t) # Each is [B, N_atoms]
    
    # 3. Calculate the term 2 * kappa / theta
    #    theta_for_denom is crucial. It must not be too small.
    #    Using a potentially larger clamp here than for interpolation.
    # ------------------------------------------------------------
    # 3. 计算散度，两种分支
    #    (i) θ >= θ0 : 公式 κ' + 2κ/θ
    #    (ii) θ <  θ0 : 泰勒极限 3 κ'(θ0)
    # ------------------------------------------------------------
    
    theta_clamped_for_division = theta_orig.clamp(min=eps_theta_denom)
    term_2k_over_theta = 2.0 * kappa / theta_clamped_for_division
    
    # Optional: Clip this term directly if it's still the source of explosion
    # term_2k_over_theta = term_2k_over_theta.clamp(-clip_2k_over_theta_term, clip_2k_over_theta_term)

    # 4. Calculate divergence using formula: kappa' + 2*kappa/theta
    div_formula_val = dkappa + term_2k_over_theta

    # 5. For very small original theta, use the Taylor limit: div(s_R) --> 3 * kappa'(0,t)
    #    kappa_prime_at_zero_approx will use dkappa which is kappa'(clamped_theta_for_interp)
    #    This is an approximation of kappa'(0).
        # —— 泰勒分支 —— #
    small_mask = theta_orig < theta0_taylor                 # [B,N] bool
    if small_mask.any():
        div_formula_val[small_mask] = 3.0 * dkappa[small_mask]
    
    # ------------------------------------------------------------
    # 4. 按原子 **平均**；若想求和把 mean → sum
    # ------------------------------------------------------------
    
    div_mean_over_atoms = div_formula_val.mean(dim=-1)    # [B]
   

    if torch.isnan(div_mean_over_atoms).any() or torch.isinf(div_mean_over_atoms).any():
        logger.error("NaN/Inf in final div_mean_over_atoms!")
        # Fallback or error
        div_mean_over_atoms = torch.nan_to_num(div_mean_over_atoms, nan=0.0, posinf=1e3, neginf=-1e3)


    return div_mean_over_atoms


def t_finite_diff_indirect_score(
    main_abx_F_theta_predictor_fn, # e.g., IpaScore.predict_z0_at_specific_time
    diffuser_obj,                  # R3Diffuser or SO3Diffuser instance
    original_batch_for_z0_pred,    # Original batch dict passed to predictor_fn
    t_current_center,              # Current time for F_theta input [B]
    is_coord_score_flag,           # Boolean: True for R3, False for SO3
    delta_t_val=1e-3
):
    """
    Approximates d s_theta(z_t, t) / d t for AbX's indirect score using central finite differences.
    s_theta(z_t, t) = KernelScore(z_t, z0_theta(z_t, t_Ftheta), t_kernel)
    Here, t_Ftheta and t_kernel are varied together as t_plus/t_minus.
    z_t (spatial part) is fixed from original_batch_for_z0_pred['rigids_t'].
    """
    if not callable(main_abx_F_theta_predictor_fn):
        raise TypeError("main_abx_F_theta_predictor_fn must be a callable function/method.")

    _t = t_current_center
    # Ensure t_current_center is a tensor and correctly shaped for broadcasting if needed
    if not isinstance(_t, torch.Tensor):
        # Assuming x_t_spatial_arg_unscaled is within original_batch_for_z0_pred['rigids_t']
        ref_device = original_batch_for_z0_pred['rigids_t'].device
        ref_dtype = original_batch_for_z0_pred['rigids_t'].dtype
        _t = torch.tensor(_t, device=ref_device, dtype=ref_dtype)
    
    if _t.ndim == 0 and original_batch_for_z0_pred['rigids_t'].shape[0] > 1:
        _t = _t.repeat(original_batch_for_z0_pred['rigids_t'].shape[0])
    elif _t.ndim == 1 and original_batch_for_z0_pred['rigids_t'].ndim > 1 and \
         _t.shape[0] != original_batch_for_z0_pred['rigids_t'].shape[0]:
        _t = _t.expand(original_batch_for_z0_pred['rigids_t'].shape[0])


    EPS_T = 1e-4   # 可配置
    t_plus  = torch.clamp(_t + delta_t_val, min=EPS_T,     max=1.0-EPS_T)
    t_minus = torch.clamp(_t - delta_t_val, min=EPS_T,     max=1.0-EPS_T)

    # t_plus = torch.clamp(_t + delta_t_val, min=0.0, max=1.0) # Clamp to SDE interval [0,1]
    # t_minus = torch.clamp(_t - delta_t_val, min=0.0, max=1.0)
    
    # Avoid issues if t_plus and t_minus become too close due to clamping at boundaries
    actual_delta_t = t_plus - t_minus 
    # Set a minimum delta_t to prevent division by zero if t_plus ~= t_minus
    # This can happen if t_current_center is very close to 0 or 1 and delta_t_val is too large.
    actual_delta_t = torch.where(actual_delta_t < 1e-6, torch.full_like(actual_delta_t, 1e-6), actual_delta_t)


    # Get z0_theta predictions at t_plus and t_minus
    # predict_z0_at_specific_time expects (original_batch_dict, time_for_prediction)
    # It will internally use the time_for_prediction to condition F_theta (IpaScore core)
    with torch.no_grad(): # z0 predictions for finite difference should not track gradients back to F_theta here
        z0_predictions_at_t_plus = main_abx_F_theta_predictor_fn(
            original_batch_for_z0_pred, t_plus
        )
        z0_predictions_at_t_minus = main_abx_F_theta_predictor_fn(
            original_batch_for_z0_pred, t_minus
        )
       
    # Extract relevant z0 component (unscaled)
    if is_coord_score_flag:
        z0_at_plus_unscaled = z0_predictions_at_t_plus['x0_trans_unscaled']
        z0_at_minus_unscaled = z0_predictions_at_t_minus['x0_trans_unscaled']
        x_t_spatial_unscaled_for_score = original_batch_for_z0_pred['rigids_t'][..., 4:]
    else: # is_rot_score_flag
        z0_at_plus_unscaled = z0_predictions_at_t_plus['x0_rot_quat']
        z0_at_minus_unscaled = z0_predictions_at_t_minus['x0_rot_quat']
        x_t_spatial_unscaled_for_score = original_batch_for_z0_pred['rigids_t'][..., :4]
   
    # Scale inputs for diffuser.score method
    # diffuser_obj's score method with scale_input_coords=True will handle this.
    # So we pass unscaled x_t and unscaled z0 to it.

    # Compute scores using the diffuser's score method
    # The diffuser's score method takes (z_t_unscaled, z_0_unscaled, time_for_kernel)
    # and handles scaling internally if scale_input_coords=True.
    if is_coord_score_flag:
        s_t_plus_dt_unscaled = diffuser_obj.score( # Returns unscaled score
            x_t_spatial_unscaled_for_score, 
            z0_at_plus_unscaled, 
            t_plus, 
            True 
        )
        s_t_minus_dt_unscaled = diffuser_obj.score(
            x_t_spatial_unscaled_for_score, 
            z0_at_minus_unscaled, 
            t_minus, 
            True
        )
    else:
        #import ipdb; ipdb.set_trace()
        s_t_plus_dt_unscaled = diffuser_obj.calc_quat_score( # Returns unscaled score
            x_t_spatial_unscaled_for_score, 
            z0_at_plus_unscaled, 
            t_plus
        )
        s_t_minus_dt_unscaled = diffuser_obj.calc_quat_score(
            x_t_spatial_unscaled_for_score, 
            z0_at_minus_unscaled, 
            t_minus
        )
    if is_coord_score_flag:
        # Scale the scores for FPE residual calculation consistency if G is also in scaled space
        s_t_plus_dt_scaled = diffuser_obj._scale(s_t_plus_dt_unscaled)
        s_t_minus_dt_scaled = diffuser_obj._scale(s_t_minus_dt_unscaled)
    else:
        s_t_plus_dt_scaled = s_t_plus_dt_unscaled
        s_t_minus_dt_scaled = s_t_minus_dt_unscaled

    # Reshape actual_delta_t if necessary for broadcasting
    if actual_delta_t.ndim < s_t_plus_dt_scaled.ndim :
        actual_delta_t_br = actual_delta_t.view(-1, *([1]*(s_t_plus_dt_scaled.ndim-1)))
    else:
        actual_delta_t_br = actual_delta_t

    dt_s_approx_scaled = (s_t_plus_dt_scaled - s_t_minus_dt_scaled) / (actual_delta_t_br)
    
    # Ensure flattened output [B, D_flat]
    soft_C = 300.0
    dt_s_approx_scaled = soft_C * torch.tanh(dt_s_approx_scaled / soft_C)
    return dt_s_approx_scaled.reshape(dt_s_approx_scaled.shape[0], -1)



def bochner_so3_exact(score_fn_fixed_t, rotvecs_scaled_B3):
    """
    Computes the exact Bochner Laplacian Delta_B sigma for a batch of score fields
    sigma(rotvecs_scaled_B3) on SO(3), parameterized by scaled rotation vectors.

    Args:
        score_fn_fixed_t: Callable that takes scaled rotation vectors [B, 3]
                          and returns scaled score vectors [B, 3]. Time 't' is fixed.
        rotvecs_scaled_B3: Input batch of scaled rotation vectors [B, 3].
                           Must have requires_grad=True if not already set by caller.

    Returns:
        lap_b_sigma: Bochner Laplacian of sigma, [B, 3], scaled.
    """
    B, D = rotvecs_scaled_B3.shape
    assert D == 3, "Bochner Laplacian here is defined for SO(3) elements (rotvec dim 3)"

    # Ensure input requires grad for Jacobian/Hessian computation
    original_requires_grad = rotvecs_scaled_B3.requires_grad
    if not original_requires_grad:
        rotvecs_scaled_B3.requires_grad_(True)

    # --- Term 1: Euclidean Laplacian of each component sigma_i ---
    # laplacian_eucl_sigma[b, i] = sum_k (d^2 sigma_i / d omega_k^2) for batch item b
    laplacian_eucl_sigma = torch.zeros_like(rotvecs_scaled_B3) # [B, 3]

    # Loop over batch elements to compute Hessians, as functional.hessian is not batched for outputs
    for b_idx in range(B):
        rotvec_sample = rotvecs_scaled_B3[b_idx:b_idx+1] # Keep batch dim: [1, 3]
        
        # For sigma_0 component
        hess_s0 = torch.autograd.functional.hessian(
            lambda x: score_fn_fixed_t(x)[:, 0].sum(), # sum is for scalar output if x is [1,3]
            rotvec_sample, create_graph=True, strict=True
        ) # hess_s0 will be [1,3,1,3] or similar, need to extract [3,3]
        laplacian_eucl_sigma[b_idx, 0] = torch.trace(hess_s0.squeeze()) # Squeeze to [3,3]

        # For sigma_1 component
        hess_s1 = torch.autograd.functional.hessian(
            lambda x: score_fn_fixed_t(x)[:, 1].sum(),
            rotvec_sample, create_graph=True, strict=True
        )
        laplacian_eucl_sigma[b_idx, 1] = torch.trace(hess_s1.squeeze())

        # For sigma_2 component
        hess_s2 = torch.autograd.functional.hessian(
            lambda x: score_fn_fixed_t(x)[:, 2].sum(),
            rotvec_sample, create_graph=True, strict=True
        )
        laplacian_eucl_sigma[b_idx, 2] = torch.trace(hess_s2.squeeze())
    
    # --- Term 2: Lie "cross" term: -curl(sigma) equivalent ---
    # We need the Jacobian J_jk = d sigma_j / d omega_k
    # functional.jacobian can compute this.
    # For batched input [B,3] and batched output [B,3], jacobian will be [B,3,B,3]
    # We need J[b, j_out, k_in] for each b.
    
    # To get J[b, j_out, k_in]:
    # Option 1: Loop over batch for Jacobian (safer for autograd)
    J_batch = [] # List of [3,3] Jacobians
    for b_idx in range(B):
        rotvec_sample = rotvecs_scaled_B3[b_idx:b_idx+1] # [1,3]
        # We want Jacobian of score_fn_fixed_t(rotvec_sample) [1,3] w.r.t rotvec_sample [1,3]
        # The lambda should return the [1,3] vector.
        # functional.jacobian expects a function R^N -> R^M, returns M x N
        # Here, N=3 (input rotvec), M=3 (output score)
        jac_b = torch.autograd.functional.jacobian(
            lambda x: score_fn_fixed_t(x), # score_fn_fixed_t already handles [1,3]->[1,3]
            rotvec_sample, create_graph=True, strict=True
        ) # jac_b will be [1,3,1,3] if inputs/outputs are kept as [1,3]
        J_batch.append(jac_b.squeeze()) # Squeeze to [3,3]
    J_tensor = torch.stack(J_batch, dim=0) # [B, 3, 3], J_tensor[b, j, k] = d sigma_j / d omega_k

    cross_term = torch.zeros_like(rotvecs_scaled_B3) # [B, 3]
    cross_term[:, 0] = -(J_tensor[:, 1, 2] - J_tensor[:, 2, 1]) # -(d(sigma_1)/d(omega_2) - d(sigma_2)/d(omega_1))
    cross_term[:, 1] = -(J_tensor[:, 2, 0] - J_tensor[:, 0, 2]) # -(d(sigma_2)/d(omega_0) - d(sigma_0)/d(omega_2))
    cross_term[:, 2] = -(J_tensor[:, 0, 1] - J_tensor[:, 1, 0]) # -(d(sigma_0)/d(omega_1) - d(sigma_1)/d(omega_0))

    # --- Term 3: Ricci curvature term -2*sigma ---
    # sigma_output is needed here again. If score_fn_fixed_t is expensive, compute once.
    # If create_graph=True for J_tensor, sigma_output from Term 1 might have stale graph.
    # Recompute sigma_output if there's any doubt, or ensure graph is passed.
    # Assuming sigma_output used for laplacian_eucl_sigma is still valid (e.g. if called after Term 1 setup)
    # For safety, re-evaluate if score_fn_fixed_t is not just a simple lookup:
    sigma_output_for_ricci = score_fn_fixed_t(rotvecs_scaled_B3)
    ricci_term = -2.0 * sigma_output_for_ricci

    # Combine all terms
    lap_b_sigma = laplacian_eucl_sigma + cross_term + ricci_term

    # Reset requires_grad if it was changed by this function
    if not original_requires_grad:
        rotvecs_scaled_B3.requires_grad_(False)
    if rotvecs_scaled_B3.grad is not None: # Clean up any grads on input
        rotvecs_scaled_B3.grad.zero_()
        
    return lap_b_sigma


# diff_utils.py

def analytical_dt_s_theta(diffuser_obj, s_theta_unscaled, x0_pred_unscaled, t_current, is_coord_flag):
    """
    Calculates d(s)/dt analytically assuming z0 is constant w.r.t. t.
    Correctly handles scaling for R3 and no-scaling for SO3.
    """
    # ==================== R³ (VP-SDE) Case with Scaling ====================
    if is_coord_flag:
        alpha_t, sigma_t_sq, alpha_prime_t, sigma_sq_prime_t = diffuser_obj.get_coeffs_and_derivs(t_current)
        
        # Reshape for broadcasting
        alpha_prime_t_b = alpha_prime_t.view(-1, *([1]*(x0_pred_unscaled.ndim-1)))
        sigma_t_sq_b = sigma_t_sq.view(-1, *([1]*(x0_pred_unscaled.ndim-1)))
        sigma_sq_prime_t_b = sigma_sq_prime_t.view(-1, *([1]*(x0_pred_unscaled.ndim-1)))
        
        # Formula: d(s_X)/dt = (α'_t / σ_t^2) * x_0 - s_X * ( d(σ_t^2)/dt / σ_t^2 )
        term1 = (alpha_prime_t_b / (sigma_t_sq_b + 1e-9)) * x0_pred_unscaled
        term2 = s_theta_unscaled * (sigma_sq_prime_t_b / (sigma_t_sq_b + 1e-9))
        dt_s_unscaled = term1 - term2
        
        dt_s_scaled = diffuser_obj._scale(dt_s_unscaled)
        return dt_s_scaled.reshape(dt_s_scaled.shape[0], -1)

    # ==================== SO(3) (VE-SDE) Case without Scaling ====================
    else:
        sigma_t, sigma_prime_t = diffuser_obj.get_sigma_and_deriv(t_current)
        
        # Reshape for broadcasting
        sigma_t_b = sigma_t.view(-1, *([1]*(s_theta_unscaled.ndim-1)))
        sigma_prime_t_b = sigma_prime_t.view(-1, *([1]*(s_theta_unscaled.ndim-1)))
        
        # Formula for VE SDE: d(s_R)/dt ≈ -(σ'/σ)·s_R
        dt_s_unscaled = -(sigma_prime_t_b / (sigma_t_b + 1e-9)) * s_theta_unscaled
        
        return dt_s_unscaled.reshape(dt_s_unscaled.shape[0], -1)
    