"""R^3 diffusion methods."""
import numpy as np
from scipy.special import gamma
import torch
import pdb
import logging
import math
from abx.model import diff_utils

logger = logging.getLogger(__name__)


class R3Diffuser:
    """VP-SDE diffuser class for translations."""

    def __init__(self, r3_conf):
        """
        Args:
            min_b: starting value in variance schedule.
            max_b: ending value in variance schedule.
        """
        self._r3_conf = r3_conf
        self.min_b = r3_conf['min_b']
        self.max_b = r3_conf['max_b']

    def _scale(self, x):
        return x * torch.tensor(self._r3_conf['coordinate_scaling'],device=x.device)

    def _unscale(self, x):
        return x / torch.tensor(self._r3_conf['coordinate_scaling'], device=x.device)

    def b_t(self, t):
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        return torch.tensor(self.min_b, device=t.device) + t*torch.tensor((self.max_b - self.min_b),device=t.device)

    def diffusion_coef(self, t):
        """Time-dependent diffusion coefficient."""
        return torch.sqrt(self.b_t(t))[:, None, None]

    def drift_coef(self, x, t):
        """Time-dependent drift coefficient."""
        return -1/2 * self.b_t(t)[:, None, None] * x

    def sample_ref(self, n_samples: np.array, device='cpu'):
        return torch.randn(size=(*n_samples,3),device=device)

    def marginal_b_t(self, t):
        return t*torch.tensor(self.min_b, device=t.device) + (1/2)*(t**2)*(torch.tensor(self.max_b-self.min_b, device=t.device))

    def calc_trans_0(self, score_t, x_t, t):
        beta_t = self.marginal_b_t(t)
        beta_t = beta_t[..., None, None]
        exp_fn = torch.exp
        cond_var = 1 - exp_fn(-beta_t)
        return (score_t * cond_var + x_t) / exp_fn(-1/2*beta_t)

    def forward(self, x_t_1: torch.tensor, t: torch.tensor, num_t: int):
        """Samples marginal p(x(t) | x(t-1)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_t_1 = self._scale(x_t_1)
        b_t = torch.tensor(self.marginal_b_t(t) / num_t, device=x_t_1.device)
        z_t_1 = torch.randn(size=x_t_1.shape, device=x_t_1.device)
        x_t = torch.sqrt(1 - b_t) * x_t_1 + torch.sqrt(b_t) * z_t_1
        return x_t

    def distribution(self, x_t, score_t, t, mask, dt):
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        std = g_t * torch.sqrt(dt)
        mu = x_t - (f_t - g_t**2 * score_t) * dt
        if mask is not None:
            mu *= mask[..., None]
        return mu, std

    def forward_marginal(self, x_0: torch.tensor, t: torch.tensor):
        """Samples marginal p(x(t) | x(0)).

        Args:
            x_0: [..., n, 3] initial positions in Angstroms.
            t: continuous time in [0, 1].

        Returns:
            x_t: [..., n, 3] positions at time t in Angstroms.
            score_t: [..., n, 3] score at time t in scaled Angstroms.
        """
        x_0 = self._scale(x_0)

        log_mean_coeff = -0.5 * self.marginal_b_t(t)
        cast_shape = [log_mean_coeff.shape[0]] + [1] * (len(x_0.shape) - 1)
        log_mean_coeff = log_mean_coeff.view(*cast_shape)

        mean = torch.exp(log_mean_coeff) * x_0
        std = torch.sqrt(1.0 - torch.exp(2.0 * log_mean_coeff))

        x_t = torch.normal(mean=mean, std=std).to(device=x_0.device)
        score_t = self.score(x_t, x_0, t)
        x_t = self._unscale(x_t)
        return x_t, score_t

    def score_scaling(self, t: torch.tensor):
        return 1 / torch.sqrt(self.conditional_var(t))

    def reverse(
            self,
            *,
            x_t: torch.tensor,
            score_t: torch.tensor,
            t: torch.tensor,
            dt: torch.tensor,
            mask: torch.tensor=None,
            center: bool=True,
            noise_scale: float=1.0,
        ):
        """Simulates the reverse SDE for 1 step

        Args:
            x_t: [..., 3] current positions at time t in angstroms.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] positions at next step t-1.
        """
        x_t = self._scale(x_t)
        g_t = self.diffusion_coef(t)
        f_t = self.drift_coef(x_t, t)
        z = noise_scale * torch.randn(size=score_t.shape, device=score_t.device)
        perturb = (f_t - g_t**2 * score_t) * dt + g_t * dt * z

        if mask is not None:
            perturb *= mask[..., None]
        else:
            mask = torch.ones(x_t.shape[:-1], device=x_t.device)
        x_t_1 = x_t - perturb
        if center:
            com = torch.sum(x_t_1, dim=-2) / torch.sum(mask, dim=-1, keepdims=True)
            x_t_1 -= com[..., None, :]
        x_t_1 = self._unscale(x_t_1)
        return x_t_1

    def conditional_var(self, t):
        """Conditional variance of p(xt|x0).

        Var[x_t|x_0] = conditional_var(t)*I

        """
        return 1 - torch.exp(-self.marginal_b_t(t))

    def score(self, x_t, x_0, t, scale=False):
        if scale:
            x_t = self._scale(x_t)
            x_0 = self._scale(x_0)

        t = t[:, None, None]
        return -(x_t - torch.exp(-1/2*self.marginal_b_t(t)) * x_0) / self.conditional_var(t)
    
    #D_X 对应N_CDR, X_dot_s_theta_scaled对应X*Sx, div_s_theta_scaled (通过 Hutchinson 估计) 对应div_x(Sx)
    #norm_sq_s_theta_scaled对应||Sx||**2
    def get_L_X_operator_terms_vp(self, 
                                s_theta_x_t_flat_scaled,      # Score s(x,t), scaled, flat [B, D_X], Graph w.r.t. x_t expected
                                x_t_perturbed_flat_scaled,  # x_t, scaled, flat [B, D_X], Must have requires_grad=True
                                t_current,                    # [B]
                                # score_fn_callable_for_div: fn(x_flat_scaled_arg) -> s_flat_scaled (t_current fixed inside)
                                # This callable is needed by hutchinson_divergence_autograd.
                                # It should be constructed by the caller (IpaScore) to ensure t_current is correctly bound.
                                hutchinson_callable_score_at_t_current, 
                                D_X):                         # Scalar, total dimensionality
        """
        Calculates the scalar field L_X[s_theta](x_t, t) for VP-SDE (Eq. 19 in Methods).
        L_X[s] = 0.5 * beta_X(t) * [ D_X + X_scaled . s_scaled + div_X_scaled(s_scaled) + ||s_scaled||^2 ]
        All terms are w.r.t. SCALED coordinates and SCALED score.
        """
        beta_X_t_val = self.diffusion_coef(t_current)**2 # [B], instantaneous rate beta_X(t)
        N = D_X // 3
        # div_X s_theta (divergence of scaled score w.r.t. scaled coordinates)
        # x_t_perturbed_flat_scaled must have requires_grad=True for hutchinson_divergence_autograd
        # The hutchinson_callable_score_at_t_current is:
        # lambda x_arg_scaled: actual_abx_score_fn(x_arg_scaled, t_current, F_theta_fixed, ...)
        div_s_theta_scaled = diff_utils.hutchinson_divergence_autograd(
            hutchinson_callable_score_at_t_current, 
            x_t_perturbed_flat_scaled, # This x needs requires_grad=True
            t_current,
            num_samples=1 
        ) # [B]
        
        # X_scaled . s_theta_scaled
        # s_theta_x_t_flat_scaled is assumed to be s_theta(x_t_perturbed_flat_scaled, t_current)
        # and has a graph w.r.t. x_t_perturbed_flat_scaled.
        X_dot_s_theta_scaled = torch.mean(x_t_perturbed_flat_scaled * s_theta_x_t_flat_scaled, dim=1)* 3  # [B]

        # ||s_theta_scaled||^2
        norm_sq_s_theta_scaled = torch.mean(s_theta_x_t_flat_scaled**2, dim=1)* 3  # [B]
        
        # L_X[s_theta] (scalar field for each batch item, has graph w.r.t. x_t_perturbed_flat_scaled)
        # beta_X_t_val is [B], result L_X_s_theta is [B]

        # logger.info(f"---------------------------------D_X {D_X}")
        # L_X_s_theta_scalar_field = 0.5 * beta_X_t_val * \
        #                         (D_X + X_dot_s_theta_scaled + div_s_theta_scaled + norm_sq_s_theta_scaled)
        SOFT_C = 300.0
        # --- soft-clip 三项 ---
        X_dot_s_theta_scaled  = SOFT_C * torch.tanh(X_dot_s_theta_scaled  / SOFT_C)
        div_s_theta_scaled = SOFT_C * torch.tanh(div_s_theta_scaled / SOFT_C)
        norm_sq_s_theta_scaled  = SOFT_C * torch.tanh(norm_sq_s_theta_scaled  / SOFT_C)
        
        L_X_s_theta_scalar_field = 0.5 * beta_X_t_val * \
                                (3 + X_dot_s_theta_scaled + div_s_theta_scaled + norm_sq_s_theta_scaled)
        
        return L_X_s_theta_scalar_field

    
    def get_coeffs_and_derivs(self, t):
        """
        Computes VP-SDE coefficients and their time derivatives.
        alpha_t = exp(-0.5 * integral(beta_s ds))
        sigma_t^2 = 1 - alpha_t^2
        """
        # 从已有函数中获取 beta_t 和 marginal_b_t
        beta_t = self.b_t(t)
        marginal_beta_t = self.marginal_b_t(t)

        # alpha_t 和 sigma_t^2
        log_alpha_t = -0.5 * marginal_beta_t
        alpha_t = torch.exp(log_alpha_t)
        sigma_t_sq = 1. - torch.exp(2. * log_alpha_t)

        # d(alpha_t)/dt
        alpha_prime_t = -0.5 * beta_t * alpha_t
        
        # d(sigma_t^2)/dt
        sigma_sq_prime_t = -2. * alpha_t * alpha_prime_t
        
        return alpha_t, sigma_t_sq, alpha_prime_t, sigma_sq_prime_t