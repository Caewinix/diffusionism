import torch
from torch import Tensor
import numpy as np
from ..functions import extract
from .diffusion_buffer import DiffusionBuffer


class GaussianDiffusionBuffer(DiffusionBuffer):
    def __init__(
        self,
        betas: torch.Tensor,
        v_posterior = 0., # weight for choosing posterior variance as sigma = (1-v) * beta_tilde + v * beta
        logvar_init = 0.,
        original_elbo_weight=0.,
        l_simple_weight=1.
    ):
        self.v_posterior = v_posterior
        self.original_elbo_weight = original_elbo_weight
        self.l_simple_weight = l_simple_weight
        
        self.betas = betas
        super().__init__(betas.size(0))
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat((self.alphas_cumprod.new_tensor([1.]), self.alphas_cumprod[:-1]), dim=0)
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variances = ((1 - self.v_posterior) * self.betas * (1. - self.alphas_cumprod_prev) /
                                   (1. - self.alphas_cumprod) + self.v_posterior * self.betas)
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        
        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.posterior_log_variance_clipped = torch.log(np.maximum(self.posterior_variances, 1e-20))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        self.logvar = torch.full(fill_value=logvar_init, size=(self.num_timesteps,))
    
    def degradation_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # beta_t
        return extract(self.betas, timestep, input.shape)

    def retention_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # alpha_t = 1 - beta_t
        return extract(self.alphas, timestep, input.shape)
    
    def degradation_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # beta_t
        return torch.sqrt(extract(self.betas, timestep, input.shape))

    def retention_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # alpha_t = 1 - beta_t
        return torch.sqrt(extract(self.alphas, timestep, input.shape))
    
    def degradation_total_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \hat{beta_t}
        return torch.square(extract(self.sqrt_one_minus_alphas_cumprod, timestep, input.shape))

    def retention_total_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \hat{alpha_t}
        return extract(self.alphas_cumprod, timestep, input.shape)
    
    def degradation_total_std(self, input: Tensor, timestep: Tensor, *args, **kwarg) -> Tensor:
        # \sqrt{\hat{beta_t}}
        return extract(self.sqrt_one_minus_alphas_cumprod, timestep, input.shape)

    def retention_total_std(self, input: Tensor, timestep: Tensor, *args, **kwarg) -> Tensor:
        # \sqrt{\hat{alpha_t}}
        return extract(self.sqrt_alphas_cumprod, timestep, input.shape)

    def reciprocal_retention_total_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \sqrt{1 / \hat{alpha_t}}
        return extract(self.sqrt_recip_alphas_cumprod, timestep, input.shape)
    
    def complementary_reciprocal_retention_total_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \sqrt{1 / \hat{alpha_t} - 1}
        return extract(self.sqrt_recipm1_alphas_cumprod, timestep, input.shape)

    def posterior_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return extract(self.posterior_variances, timestep, input.shape)
    
    def posterior_log_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return extract(self.posterior_log_variance_clipped, timestep, input.shape)
    
    def posterior_mean_start_coefficient(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # (\sqrt{\bar{\alpha}_{t-1}} \beta_t) / (1 - \bar{\alpha}_t)
        return extract(self.posterior_mean_coef1, timestep, input.shape)
    
    def posterior_mean_current_coefficient(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # (\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})) / (1 - \bar{\alpha}_t)
        return extract(self.posterior_mean_coef2, timestep, input.shape)