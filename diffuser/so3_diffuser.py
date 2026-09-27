"""SO(3) diffusion methods."""
import numpy as np
import os
# from data import utils as du
import logging
import torch
from abx.utils import torch_interp
from abx.model.r3 import compose_rotvec
from abx.model.quat_affine import rotvec_to_quat, quat_multiply, quat_to_rotvec
from scipy.signal import savgol_filter
from abx.model import diff_utils

logger = logging.getLogger(__name__)


def igso3_expansion(omega, eps, L=1000):
    """Truncated sum of IGSO(3) distribution.

    This function approximates the power series in equation 5 of
    "DENOISING DIFFUSION PROBABILISTIC MODELS ON SO(3) FOR ROTATIONAL
    ALIGNMENT"
    Leach et al. 2022

    This expression diverges from the expression in Leach in that here, eps =
    sqrt(2) * eps_leach, if eps_leach were the scale parameter of the IGSO(3).

    With this reparameterization, IGSO(3) agrees with the Brownian motion on
    SO(3) with t=eps^2.

    Args:
        omega: rotation of Euler vector (i.e. the angle of rotation)
        eps: std of IGSO(3).
        L: Truncation level
    """

    ls = torch.arange(L)    
    ls = ls.to(omega.device)
    if len(omega.shape) == 2:
        # Used during predicted score calculation.
        ls = ls[None, None]  # [1, 1, L]
        omega = omega[..., None]  # [num_batch, num_res, 1]
        eps = eps[..., None]
    elif len(omega.shape) == 1:
        # Used during cache computation.
        ls = ls[None]  # [1, L]
        omega = omega[..., None]  # [num_batch, 1]
    else:
        raise ValueError("Omega must be 1D or 2D.")
    p = (2*ls + 1) * torch.exp(-ls*(ls+1)*eps**2/2) * torch.sin(omega*(ls+1/2)) / torch.sin(omega/2)
    return p.sum(dim=-1)


def density(expansion, omega, marginal=True):
    """IGSO(3) density.

    Args:
        expansion: truncated approximation of the power series in the IGSO(3)
        density.
        omega: length of an Euler vector (i.e. angle of rotation)
        marginal: set true to give marginal density over the angle of rotation,
            otherwise include normalization to give density on SO(3) or a
            rotation with angle omega.
    """
    if marginal:
        # if marginal, density over [0, pi], else over SO(3)
        return expansion * (1-torch.cos(omega))/torch.tensor(np.pi)
        # return expansion * (1-np.cos(omega))/ np.pi
    else:
        # the constant factor doesn't affect any actual calculations though
        return expansion / 8 / torch.tensor(np.pi)**2


def score(exp, omega, eps, L=1000):  # score of density over SO(3)
    """score uses the quotient rule to compute the scaling factor for the score
    of the IGSO(3) density.

    This function is used within the Diffuser class to when computing the score
    as an element of the tangent space of SO(3).

    This uses the quotient rule of calculus, and take the derivative of the
    log:
        d hi(x)/lo(x) = (lo(x) d hi(x)/dx - hi(x) d lo(x)/dx) / lo(x)^2
    and
        d log expansion(x) / dx = (d expansion(x)/ dx) / expansion(x)

    Args:
        exp: truncated expansion of the power series in the IGSO(3) density
        omega: length of an Euler vector (i.e. angle of rotation)
        eps: scale parameter for IGSO(3) -- as in expansion() this scaling
            differ from that in Leach by a factor of sqrt(2).
        L: truncation level

    Returns:
        The d/d omega log IGSO3(omega; eps)/(1-cos(omega))

    """

    # lib = torch
    ls = torch.arange(L, device=omega.device)
    ls = ls[None]
    if len(omega.shape) == 2:
        ls = ls[None]
    elif len(omega.shape) > 2:
        raise ValueError("Omega must be 1D or 2D.")
    omega = omega[..., None]
    eps = eps[..., None]
    hi = torch.sin(omega * (ls + 1 / 2))
    dhi = (ls + 1 / 2) * torch.cos(omega * (ls + 1 / 2))
    lo = torch.sin(omega / 2)
    dlo = 1 / 2 * torch.cos(omega / 2)
    dSigma = (2 * ls + 1) * torch.exp(-ls * (ls + 1) * eps**2/2) * (lo * dhi - hi * dlo) / lo ** 2
    dSigma = dSigma.sum(dim=-1)
    return dSigma / (exp + 1e-4)


class SO3Diffuser:

    def __init__(self, so3_conf):
        self.schedule = so3_conf['schedule']

        self.min_sigma = so3_conf['min_sigma']
        self.max_sigma = so3_conf['max_sigma']

        self.num_sigma = so3_conf['num_sigma']
        self.use_cached_score = so3_conf['use_cached_score']
        self._log = logging.getLogger(__name__)

        # Discretize omegas for calculating CDFs. Skip omega=0.
        self.discrete_omega = torch.linspace(0, np.pi, so3_conf['num_omega']+1)[1:]

        # Precompute IGSO3 values.
        replace_period = lambda x: str(x).replace('.', '_')
        cache_dir = os.path.join(
            so3_conf['cache_dir'],
            f'eps_{so3_conf["num_sigma"]}_omega_{so3_conf["num_omega"]}_min_sigma_{replace_period(so3_conf["min_sigma"])}_max_sigma_{replace_period(so3_conf["max_sigma"])}_schedule_{so3_conf["schedule"]}'
        )

        # If cache directory doesn't exist, create it
        if not os.path.isdir(cache_dir):
            os.makedirs(cache_dir)
        pdf_cache = os.path.join(cache_dir, 'pdf_vals.npy')
        cdf_cache = os.path.join(cache_dir, 'cdf_vals.npy')
        score_norms_cache = os.path.join(cache_dir, 'score_norms.npy')

        if os.path.exists(pdf_cache) and os.path.exists(cdf_cache) and os.path.exists(score_norms_cache):
            self._log.info(f'Using cached IGSO3 in {cache_dir}')
            self._pdf = torch.from_numpy(np.load(pdf_cache))
            self._cdf = torch.from_numpy(np.load(cdf_cache))
            self._score_norms = torch.from_numpy(np.load(score_norms_cache))
            # # ADD THIS LINE TO LOAD THE PRECOMPUTED DERIVATIVE IF IT EXISTS
            # dkappa_cache = os.path.join(cache_dir, 'dkappa_vals.npy')
            # if os.path.exists(dkappa_cache):
            #     self._log.info(f'Using cached dkappa/dtheta from {dkappa_cache}')
            #     self._dkappa_dtheta = torch.from_numpy(np.load(dkappa_cache))
            #     self._log.info(f"SO3Diffuser: _dkappa_dtheta table min={self._dkappa_dtheta.min().item()}, max={self._dkappa_dtheta.max().item()}, mean={self._dkappa_dtheta.mean().item()}")
            # else: # If dkappa cache doesn't exist, compute and save it
            #     # d_theta_spacing = (self.discrete_omega[1] - self.discrete_omega[0]).item() # <<< MODIFICATION
            #     # # Use torch.gradient to numerically compute the derivative of _score_norms (kappa) w.r.t. omega
            #     # self._dkappa_dtheta = torch.gradient(self._score_norms, spacing=d_theta_spacing, dim=-1)[0] # spacing now a float
            #     # self._log.info(f"SO3Diffuser: _dkappa_dtheta table min={self._dkappa_dtheta.min().item()}, max={self._dkappa_dtheta.max().item()}, mean={self._dkappa_dtheta.mean().item()}")
                                
            #     # np.save(dkappa_cache, self._dkappa_dtheta.cpu().numpy())
            #     self._log.info('Computing and caching dkappa/dtheta.')
            #     S_np = self._score_norms.cpu().numpy().astype(np.float64)
            #     WINDOW_LENGTH = 11  # odd number, typical values 9~15
            #     POLYORDER = 3       # 3rd order polynomial fit

            #     for i in range(S_np.shape[0]):
            #         S_np[i] = savgol_filter(S_np[i], window_length=WINDOW_LENGTH, polyorder=POLYORDER, mode="interp")

            #     # Save smoothed _score_norms back
            #     self._score_norms = torch.from_numpy(S_np).to(self._score_norms.dtype).to(self._score_norms.device)

            #     # # Now compute dkappa_dtheta with central differences
            #     # ω = self.discrete_omega.cpu().numpy()
            #     # h = (ω[2:] - ω[:-2]) / 2.0

            #     # # Central difference interior
            #     # dS = (S_np[:, 2:] - S_np[:, :-2]) / h

            #     # # One-sided difference boundaries
            #     # left  = (S_np[:, 1] - S_np[:, 0]) / (ω[1] - ω[0])
            #     # right = (S_np[:, -1] - S_np[:, -2]) / (ω[-1] - ω[-2])

            #     # # Concatenate full derivative
            #     # dS = np.concatenate([left[:, None], dS, right[:, None]], axis=-1)

            #     # # Clip derivative to avoid numerical explosions
            #     # CLIP_DKAPPA = 1000
            #     # dS = np.clip(dS, -CLIP_DKAPPA, CLIP_DKAPPA)

            #     # # Save back to tensor
            #     # self._dkappa_dtheta = torch.from_numpy(dS).to(self._score_norms.dtype).to(self._score_norms.device)

            #     # # Logging
            #     # self._log.info(f"SO3Diffuser: _dkappa_dtheta clipped min={self._dkappa_dtheta.min().item():.1f}, "
            #     #             f"max={self._dkappa_dtheta.max().item():.1f}, "
            #     #             f"mean={self._dkappa_dtheta.mean().item():.5f}")

            #     # # Save to cache
            #     # dkappa_cache = os.path.join(cache_dir, 'dkappa_vals.npy')
            #     # np.save(dkappa_cache, self._dkappa_dtheta.cpu().numpy())
                

        else:
            self._log.info(f'Computing IGSO3. Saving in {cache_dir}')
            # compute the expansion of the power series
            exp_vals = torch.stack(
                [igso3_expansion(self.discrete_omega, sigma) for sigma in self.discrete_sigma])
            
            # Compute the pdf and cdf values for the marginal distribution of the angle
            # of rotation (which is needed for sampling)
            self._pdf  = torch.stack(
                [density(x, self.discrete_omega, marginal=True) for x in exp_vals])
            self._cdf = torch.stack(
                [torch.cumsum(pdf, dim=0) / so3_conf['num_omega'] * torch.tensor(np.pi) for pdf in self._pdf])
            
            # Compute the norms of the scores.  This are used to scale the rotation axis when
            # computing the score as a vector.
            self._score_norms = torch.stack(
                [score(exp_vals[i], self.discrete_omega, x) for i, x in enumerate(self.discrete_sigma)])
            
            # Cache the precomputed values
            pdf = self._pdf.cpu().numpy()
            cdf = self._cdf.cpu().numpy()
            score_norm = self._score_norms.cpu().numpy()
            np.save(pdf_cache, pdf)
            np.save(cdf_cache, cdf)
            np.save(score_norms_cache, score_norm)
            
            #self._log.info('Computing and caching dkappa/dtheta.')
            # d_theta_spacing = (self.discrete_omega[1] - self.discrete_omega[0]).item()
            # # Use torch.gradient to numerically compute the derivative of _score_norms (kappa) w.r.t. omega
            # self._dkappa_dtheta = torch.gradient(self._score_norms, spacing=d_theta_spacing, dim=-1)[0] # spacing now a float

            # self._log.info(f"SO3Diffuser: _dkappa_dtheta table min={self._dkappa_dtheta.min().item()}, max={self._dkappa_dtheta.max().item()}, mean={self._dkappa_dtheta.mean().item()}")
                            
            # # Cache the precomputed values
            # dkappa_cache = os.path.join(cache_dir, 'dkappa_vals.npy')
            # np.save(dkappa_cache, self._dkappa_dtheta.cpu().numpy())
            S_np = self._score_norms.cpu().numpy().astype(np.float64)

            WINDOW_LENGTH = 11  # odd number, typical values 9~15
            POLYORDER = 3       # 3rd order polynomial fit

            for i in range(S_np.shape[0]):
                S_np[i] = savgol_filter(S_np[i], window_length=WINDOW_LENGTH, polyorder=POLYORDER, mode="interp")

            # Save smoothed _score_norms back
            self._score_norms = torch.from_numpy(S_np).to(self._score_norms.dtype).to(self._score_norms.device)

            # # Now compute dkappa_dtheta with central differences
            # ω = self.discrete_omega.cpu().numpy()
            # h = (ω[2:] - ω[:-2]) / 2.0

            # # Central difference interior
            # dS = (S_np[:, 2:] - S_np[:, :-2]) / h

            # # One-sided difference boundaries
            # left  = (S_np[:, 1] - S_np[:, 0]) / (ω[1] - ω[0])
            # right = (S_np[:, -1] - S_np[:, -2]) / (ω[-1] - ω[-2])

            # # Concatenate full derivative
            # dS = np.concatenate([left[:, None], dS, right[:, None]], axis=-1)

            # # Clip derivative to avoid numerical explosions
            # CLIP_DKAPPA = 1000
            # import ipdb; ipdb.set_trace()
            # dS = np.clip(dS, -CLIP_DKAPPA, CLIP_DKAPPA)

            # # Save back to tensor
            # self._dkappa_dtheta = torch.from_numpy(dS).to(self._score_norms.dtype).to(self._score_norms.device)

            # # Logging
            # self._log.info(f"SO3Diffuser: _dkappa_dtheta clipped min={self._dkappa_dtheta.min().item():.1f}, "
            #             f"max={self._dkappa_dtheta.max().item():.1f}, "
            #             f"mean={self._dkappa_dtheta.mean().item():.5f}")

            # # Save to cache
            # dkappa_cache = os.path.join(cache_dir, 'dkappa_vals.npy')
            # np.save(dkappa_cache, self._dkappa_dtheta.cpu().numpy())


            


        self._score_scaling = torch.sqrt(torch.abs(
            torch.sum(
                self._score_norms**2 * self._pdf, axis=-1) / torch.sum(
                    self._pdf, axis=-1)
        )) / torch.tensor(np.sqrt(3))
        

    @property
    def discrete_sigma(self):
        return self.sigma(
            torch.linspace(0.0, 1.0, self.num_sigma)
        )


    def _interp_kappa_and_grad(self, theta: torch.Tensor, t: torch.Tensor):
        """
        线性插值 κ(θ,t) 并同时返回 κ'(θ,t)（用线段斜率）。
        不再依赖 _dkappa_dtheta 表，彻底避免差分噪声。

        参数
        ----
        theta : [B, N]   每个样本 N 个角度值 (||ω||)  
        t     : [B]      连续时间

        返回
        ----
        kappa_interp   : [B, N]
        kappa_prime    : [B, N]   （已裁剪在 ± CLIP_KAPPA_P）
        """
        device = theta.device
        B, N_atoms = theta.shape # N_atoms is N from your comment

        omega_tbl = self.discrete_omega.to(device, dtype=theta.dtype)     # [M_omega]
        kappa_tbl = self._score_norms.to(device, dtype=theta.dtype)       # [S_sigma, M_omega]

        # 1. Get kappa_row [B, M_omega] for current time t
        sigma_idx = torch.tensor(self.t_to_idx(t), device=device, dtype=torch.long)
        kappa_row = kappa_tbl[sigma_idx]               

        # 2. Find theta's interval indices [idx_lo, idx_hi] in omega_tbl
        idx_hi = torch.bucketize(theta, omega_tbl, right=True).clamp(max=omega_tbl.numel()-1)
        idx_lo = (idx_hi - 1).clamp(min=0)

        # Get omega values for the interval bounds
        # Need to use advanced indexing for [B,N] output from [M] and [B,N] indices
        # This requires omega_tbl to be 1D.
        # A simpler way if idx_lo/hi are already [B,N]:
        omega_lo = torch.gather(omega_tbl.expand(B, -1), 1, idx_lo) if N_atoms > 0 else torch.empty_like(theta) # Handles N=0
        omega_hi = torch.gather(omega_tbl.expand(B, -1), 1, idx_hi) if N_atoms > 0 else torch.empty_like(theta)

        # Calculate interpolation weights
        interval_width = omega_hi - omega_lo 
        
        w_hi = torch.zeros_like(theta)
        valid_interval_for_weights_mask = interval_width.abs() > 1e-9 # Where ω_hi != ω_lo
        
        # Calculate w_hi only for valid intervals
        w_hi[valid_interval_for_weights_mask] = (
            (theta[valid_interval_for_weights_mask] - omega_lo[valid_interval_for_weights_mask]) /
             interval_width[valid_interval_for_weights_mask]
        )
        w_hi = w_hi.clamp(0.0, 1.0) # Clamp weights due to theta potentially being outside omega_tbl range
        w_lo = 1.0 - w_hi

        # 3. Gather kappa_lo / kappa_hi
        # kappa_row is [B, M_omega]. idx_lo/idx_hi are [B, N_atoms].
        # We need, for each b in B, to gather from kappa_row[b, :] using indices idx_lo[b, :].
        batch_indices_expanded = torch.arange(B, device=device)[:, None].expand_as(theta) # [B, N_atoms]
        
        kappa_at_lo = kappa_row[batch_indices_expanded, idx_lo] if N_atoms > 0 else torch.empty_like(theta)
        kappa_at_hi = kappa_row[batch_indices_expanded, idx_hi] if N_atoms > 0 else torch.empty_like(theta)

        # 4. Linear interpolation for kappa & calculate slope for kappa_prime
        kappa_interp  = w_lo * kappa_at_lo + w_hi * kappa_at_hi
        
        kappa_prime_interp = torch.zeros_like(kappa_interp) # Initialize
        # Calculate slope only for valid intervals where interval_width > 0
        kappa_prime_interp[valid_interval_for_weights_mask] = (
            (kappa_at_hi[valid_interval_for_weights_mask] - kappa_at_lo[valid_interval_for_weights_mask]) /
             (interval_width[valid_interval_for_weights_mask] + 1e-9 )
        )

        # 5. Handle boundary conditions for kappa_prime_interp more robustly
        #    These masks identify [B,N] elements where theta was at/beyond table ends.
        #    idx_lo == idx_hi happens when theta is clamped to a boundary omega_tbl point
        #    or falls outside the range and bucketize maps it to the first/last bucket repeatedly.
        
        # Mask for elements where theta might be less than or equal to the first omega_tbl point
        # These would have idx_lo = 0 and idx_hi = 0 (or 1 if it landed right on omega_tbl[0])
        # A simpler way: if idx_lo == 0 and idx_hi == 0 (or idx_hi points to first valid interval)
        at_left_boundary_mask = (idx_lo == 0) & (idx_hi <= 1) # Check if in first segment or before
                                                            # or more simply if interval_width is zero due to left boundary
        
        # Mask for elements where theta might be greater than or equal to the last omega_tbl point
        at_right_boundary_mask = (idx_hi == omega_tbl.numel() - 1) & (idx_lo >= omega_tbl.numel() - 2)
                                                                # or more simply if interval_width is zero due to right boundary
        
        # Override slope for these boundary cases if interval_width was effectively zero there
        # This needs to be done carefully if valid_interval_for_weights_mask already handled some.
        # Consider elements where interval_width was too small AND they are at a boundary.
        
        zero_interval_at_left = (~valid_interval_for_weights_mask) & (idx_lo == 0)
        if zero_interval_at_left.any() and omega_tbl.numel() > 1:
            slope0 = (kappa_row[:, 1] - kappa_row[:, 0]) / (omega_tbl[1] - omega_tbl[0] + 1e-9) # [B]
            # Apply this slope0 to all N_atoms for batch items where the first atom is at left boundary
            # This needs to map [B] slope to [B,N] elements that satisfy the mask.
            # slope0_expanded = slope0.unsqueeze(1).expand(-1, N_atoms)
            # kappa_prime_interp[zero_interval_at_left] = slope0_expanded[zero_interval_at_left]
            # More robust:
            for b in range(B):
                if zero_interval_at_left[b].any(): # If any atom in this batch item is at left zero interval
                    kappa_prime_interp[b, zero_interval_at_left[b]] = slope0[b]


        zero_interval_at_right = (~valid_interval_for_weights_mask) & (idx_hi == omega_tbl.numel() - 1)
        if zero_interval_at_right.any() and omega_tbl.numel() > 1:
            slopeL = (kappa_row[:, -1] - kappa_row[:, -2]) / (omega_tbl[-1] - omega_tbl[-2] + 1e-9)  # [B]
            # slopeL_expanded = slopeL.unsqueeze(1).expand(-1, N_atoms)
            # kappa_prime_interp[zero_interval_at_right] = slopeL_expanded[zero_interval_at_right]
            for b in range(B):
                if zero_interval_at_right[b].any():
                     kappa_prime_interp[b, zero_interval_at_right[b]] = slopeL[b]


        # 6. Final clip
        #import ipdb; ipdb.set_trace()
        CLIP_KAPPA_P = 33.0
        kappa_prime_interp = kappa_prime_interp.clamp(-CLIP_KAPPA_P, CLIP_KAPPA_P)
        # logger.info("-----------------kappa_prime_interp---------------------")
        # logger.info(f"min= {kappa_prime_interp.min()} max= {kappa_prime_interp.max()} mean= {kappa_prime_interp.mean()}")

        return kappa_interp, kappa_prime_interp


        
    def sigma_idx(self, sigma: torch.tensor):
        """Calculates the index for discretized sigma during IGSO(3) initialization."""
        device = sigma.device
        discrete_sigma = self.discrete_sigma
        discrete_sigma = discrete_sigma.to(device=device)
        
        # TODO: Need to check
        return torch.sum(discrete_sigma[None,...] <= sigma[...,None]+1e-5, -1) - 1

    

    def sigma(self, t: torch.tensor):
        """Extract \sigma(t) corresponding to chosen sigma schedule."""
        if torch.any(t < 0) or torch.any(t > 1):
            raise ValueError(f'Invalid t={t}')
        if self.schedule == 'logarithmic':
            return torch.log(t * torch.exp(torch.tensor(self.max_sigma)) + (1 - t) * torch.exp(torch.tensor(self.min_sigma)))
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')

    def diffusion_coef(self, t):
        """Compute diffusion coefficient (g_t)."""
        if self.schedule == 'logarithmic':
            sigma_t = self.sigma(t)
            g_t = torch.sqrt(
                2 * (torch.exp(torch.tensor(self.max_sigma, device=t.device)) - torch.exp(torch.tensor(self.min_sigma, device=t.device))) * sigma_t / torch.exp(sigma_t)
            ).to(device=t.device)
        else:
            raise ValueError(f'Unrecognize schedule {self.schedule}')
        return g_t

    def t_to_idx(self, t: np.ndarray):
        """Helper function to go from time t to corresponding sigma_idx."""
        return self.sigma_idx(self.sigma(t)).tolist()

    def sample_igso3(
            self,
            t: torch.tensor,
            n_samples: np.array):
        """Uses the inverse cdf to sample an angle of rotation from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_samples: number of samples to draw.

        Returns:
            [n_samples] angles of rotation.
        """
        device = t.device
        x = torch.rand(n_samples, device=device)
        batch_size = t.shape[0]
        discrete_omega = self.discrete_omega[None,...].to(device=device).expand(batch_size, -1)
        return torch_interp(x, self._cdf[self.t_to_idx(t)].to(device=device), discrete_omega)

    def sample(
            self,
            t: torch.tensor,
            n_samples: np.array):
        """Generates rotation vector(s) from IGSO(3).

        Args:
            t: continuous time in [0, 1].
            n_sample: number of samples to generate.

        Returns:
            [n_samples, 3] axis-angle rotation vectors sampled from IGSO(3).
        """
        device = t.device
        x = torch.randn((*n_samples, 3),device=device)
        x /= torch.linalg.norm(x, dim=-1, keepdims=True)

        return x * self.sample_igso3(t, n_samples=n_samples)[..., None]

    def sample_ref(self, n_samples: np.array, device='cpu'):
        t = torch.ones(n_samples[0],device=device)
        return self.sample(t, n_samples=n_samples)

    def score(
            self,
            vec: torch.tensor,
            t: torch.tensor,
            eps: float=1e-6,
        ):
        """Computes the score of IGSO(3) density as a rotation vector.

        Same as score function but uses pytorch and performs a look-up.

        Args:
            vec: [..., 3] array of axis-angle rotation vectors.
            t: continuous time in [0, 1].

        Returns:
            [..., 3] score vector in the direction of the sampled vector with
            magnitude given by _score_norms.
        """
        #omega = torch.linalg.norm(vec, dim=-1) + eps
        small_th=1e-3
        omega = torch.linalg.norm(vec, dim=-1, keepdim=True).clamp(min=small_th)
        omega = omega.squeeze(-1)
        if self.use_cached_score:
            score_norms_t = self._score_norms[self.t_to_idx(t)]
            score_norms_t = score_norms_t.to(vec.device)
            omega_idx = torch.bucketize(
                omega, self.discrete_omega[:-1].to(device=vec.device))
            omega_scores_t = torch.gather(
                score_norms_t, 1, omega_idx)

        else:
            sigma = self.discrete_sigma[self.t_to_idx(t)]
            sigma = sigma.to(vec.device)
            omega_vals = igso3_expansion(omega, sigma[:, None])
            omega_scores_t = score(omega_vals, omega, sigma[:, None])

        return omega_scores_t[..., None] * vec / (omega[..., None] + eps)

    def score_scaling(self, t: torch.tensor):
        """Calculates scaling used for scores during trianing."""
        return self._score_scaling[self.t_to_idx(t)].to(device=t.device)

    def forward_marginal(self, rot_0: torch.tensor, t: torch.tensor):
        """Samples from the forward diffusion process at time index t.

        Args:
            rot_0: [..., 3] initial rotations.
            t: continuous time in [0, 1].

        Returns:
            rot_t: [..., 3] noised rotation vectors.
            rot_score: [..., 3] score of rot_t as a rotation vector.
        """
        n_samples = rot_0.shape[:-1]
        sampled_rots = self.sample(t, n_samples=n_samples)
        rot_score = self.score(sampled_rots, t).reshape(rot_0.shape)
        # Right multiply.
        # rot_0 = rot_0.reshape(-1,3)
        # sampled_rots = sampled_rots.reshape(-1,3)
        quat_0 = rotvec_to_quat(rot_0)
        sample_quats = rotvec_to_quat(sampled_rots)

        quat_t = quat_multiply(quat_0, sample_quats)
        rot_t = quat_to_rotvec(quat_t)
        # rot_t = compose_rotvec(rot_0, sampled_rots).reshape(rot_score.shape)
        return rot_t, rot_score

    def reverse(
            self,
            rot_t: torch.tensor,
            score_t: torch.tensor,
            t: torch.tensor,
            dt: torch.tensor,
            mask: torch.tensor=None,
            noise_scale: float=1.0,
            ):
        """Simulates the reverse SDE for 1 step using the Geodesic random walk.

        Args:
            rot_t: [..., 3] current rotations at time t.
            score_t: [..., 3] rotation score at time t.
            t: continuous time in [0, 1].
            dt: continuous step size in [0, 1].
            add_noise: set False to set diffusion coefficent to 0.
            mask: True indicates which residues to diffuse.

        Returns:
            [..., 3] rotation vector at next step.
        """
        # if not np.isscalar(t): raise ValueError(f'{t} must be a scalar.')
        g_t = self.diffusion_coef(t)[:, None, None]
        z = noise_scale * torch.randn(size=score_t.shape, device=t.device)
        perturb = (g_t ** 2) * score_t * dt + g_t * torch.sqrt(dt) * z

        if mask is not None: 
            perturb *= mask[..., None]
        perturb_quat = rotvec_to_quat(perturb)
        quat_t = rotvec_to_quat(rot_t)
        quat_t_1 = quat_multiply(quat_t, perturb_quat)
        rot_t_1 = quat_to_rotvec(quat_t_1)
        return rot_t_1
    
    def get_L_R_operator_terms(self, 
                                     s_theta, # [B, D_R_flat], score, scaled, WITH graph w.r.t. R_t_flat
                                     R_t_flat,         # [B, D_R_flat], state, scaled, WITH requires_grad=True
                                     t_current,                  
                                     score_fn, # fn(R_flat_scaled) -> s_R_flat_scaled
                                     D_R_flat,
                                     use_exact_so3_divergence: bool = False # New flag): # Not directly used here if calculations are per-atom then summed
                                    ):
        """
        Calculates L_{R,VE}[s_R](R_t,t) for VE-SDE on SO(3) using exact divergence.
        L_{R,VE}[s_R](R,t) = 0.5 * beta_R(t) * (div_SO3(s_R) + ||s_R||^2_R)
        """
        #import ipdb; ipdb.set_trace()
        beta_R_t_val = self.diffusion_coef(t_current)**2 # [B], which is g_R(t)^2
        
        # 1. ||s_R||^2_R (Euclidean norm squared of scaled so(3) elements)
        # s_theta is [B, N_atoms*3]
        # Reshape to [B*N_atoms, 3] for per-rotation norm, then sum, then mean over atoms
        num_batch = s_theta.shape[0]
        num_atoms = s_theta.shape[1] // 3
        
        s_theta_per_atom = s_theta.view(num_batch, num_atoms, 3)
        norm_sq_s_R_per_atom = torch.sum(s_theta_per_atom**2, dim=-1) # [B, N_atoms]
        norm_sq_s_R_summed_over_atoms = norm_sq_s_R_per_atom.mean(dim=-1) # [B], if L_R is for the whole system

        # 2. div_SO3(s_R) - Using the exact version
        # exact_div_so3 expects rotvecs to have requires_grad=True.
        # score_fn should be:
        # lambda r_flat: score_network_output_for_R_flat(F_theta, r_flat_to_struct(r_flat), fixed_t_current)
        # The callable needs to match the input/output of exact_div_so3's score_fn arg.
        # R_t_flat already has requires_grad = True from the caller (SO3ScoreFPEOperator)
        
        # The score_fn must be adaptable:
        # exact_div_so3 internally reshapes its 'rotvec' arg to [B, N, 3] if needed.
        # So, score_fn should accept [B, N*3] and handle it.
         # 2. div_SO3(s_R)
        #use_exact_so3_divergence = False
        if use_exact_so3_divergence:
            #logger.info(f"Using exact SO(3) divergence for L_R_calculation.")
            # This is only used for L_R loss value, not for calculating G.
            div_s_R_summed = diff_utils.exact_div_so3(self, R_t_flat, t_current) # Returns [B] (summed over atoms)
        else:
            #logger.warning("Using approximate Euclidean divergence for SO(3) L_R calculation.")
            # R_t_flat_scaled must have requires_grad=True for hutchinson_divergence_autograd
            div_s_R_summed = diff_utils.hutchinson_divergence_autogradR(
                score_fn, # fn(R_flat_scaled_arg) -> s_R_flat_scaled
                R_t_flat,
                t_current,
                num_samples=1
            ) # Returns [B] (summed over flat dimensions)
        #import ipdb; ipdb.set_trace()
        # 3. Combine to get L_R (scalar per batch item)
        # beta_R_t_val is [B], div_exact_summed is [B], norm_sq_s_R_summed_over_atoms is [B]
        SOFT_C = 300.0
        div_s_R_summed = SOFT_C * torch.tanh(div_s_R_summed / SOFT_C)
        norm_sq_s_R_summed_over_atoms = SOFT_C * torch.tanh(norm_sq_s_R_summed_over_atoms / SOFT_C)        # div_S_R 已 mean

        L_R_scalar_field = 0.5 * beta_R_t_val * (div_s_R_summed + norm_sq_s_R_summed_over_atoms)
        #import ipdb; ipdb.set_trace()
        return L_R_scalar_field
    
    def get_sigma_and_deriv(self, t):
        """
        Computes sigma(t) and its time derivative sigma'(t) for VE-SDE.
        sigma(t) = log(t * exp(max_sigma) + (1-t) * exp(min_sigma))
        """
        if self.schedule == 'logarithmic':
            # 内部计算
            exp_max_sigma = torch.exp(torch.tensor(self.max_sigma, device=t.device))
            exp_min_sigma = torch.exp(torch.tensor(self.min_sigma, device=t.device))
            
            # sigma(t)
            sigma_t = self.sigma(t)

            # d(sigma_t)/dt
            # d/dt log(f(t)) = f'(t)/f(t)
            # f(t) = t * exp_max + (1-t) * exp_min
            # f'(t) = exp_max - exp_min
            # d(sigma_t)/dt = (exp_max - exp_min) / (t * exp_max + (1-t) * exp_min)
            #             = (exp_max - exp_min) / exp(sigma(t))
            sigma_prime_t = (exp_max_sigma - exp_min_sigma) / torch.exp(sigma_t)
            
            return sigma_t, sigma_prime_t
        else:
            raise ValueError(f'Unrecognized schedule {self.schedule}')
    