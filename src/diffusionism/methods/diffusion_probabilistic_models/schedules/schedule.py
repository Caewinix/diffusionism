from typing import Optional
import torch
from torch import Tensor
from ....diffusion.utils.functions import extract
from ....diffusion.schedules.linear_schedule import AncestralDiffusionSchedule
from ....diffusion.hyperparameters import Hyperparameters


class DiffusionProbabilisticModelsDiscreteSchedule(AncestralDiffusionSchedule):
    def __init__(
        self,
        betas: torch.Tensor,
        logvar_init = 0.,
        v_posterior = 0.,
        hyperparameters: Optional[Hyperparameters] = None
    ):
        super().__init__(torch.long, hyperparameters)
        
        self.betas = betas
        self.v_posterior = v_posterior
        num_total_timesteps = betas.size(0)
        
        self.alphas = 1 - self.betas
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
        self.posterior_log_variance_clipped = torch.log(torch.maximum(self.posterior_variances, torch.tensor(1e-20)))
        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1. - self.alphas_cumprod)
        
        self.logvar = torch.full(fill_value=logvar_init, size=(num_total_timesteps,))
    
    def step_noise_scale(self, input, timestep, *args, differential_timestep = 1, **kwargs):
        return extract(self.betas, timestep, input.shape).sqrt_()
    
    def step_noise_variance(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # beta_t
        return extract(self.betas, timestep, input.shape)
    
    def step_input_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep = 1, **kwargs) -> Tensor:
        return extract(self.alphas, timestep, input.shape).sqrt_()

    def step_input_scale_square(self, input: Tensor, timestep: Tensor, *args, differential_timestep = 1, **kwargs) -> Tensor:
        # alpha_t = 1 - beta_t
        return extract(self.alphas, timestep, input.shape)
    
    def noise_variance(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \hat{beta_t}
        return torch.square(extract(self.sqrt_one_minus_alphas_cumprod, timestep, input.shape))

    def input_scale_square(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \hat{alpha_t}
        return extract(self.alphas_cumprod, timestep, input.shape)
    
    def noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwarg) -> Tensor:
        # \sqrt{\hat{beta_t}}
        return extract(self.sqrt_one_minus_alphas_cumprod, timestep, input.shape)

    def input_scale(self, input: Tensor, timestep: Tensor, *args, **kwarg) -> Tensor:
        # \sqrt{\hat{alpha_t}}
        return extract(self.sqrt_alphas_cumprod, timestep, input.shape)

    def reciprocal_cumulative_mean(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the reciprocal of the mean product of the diffusion,
        from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The reciprocal of the mean product of the diffusion, from the start
                to the current timestep.
        
        """
        # \sqrt{1 / \hat{alpha_t}}
        return extract(self.sqrt_recip_alphas_cumprod, timestep, input.shape)
    
    def complementary_reciprocal_cumulative_mean(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        # \sqrt{1 / \hat{alpha_t} - 1}
        """Indicates the complementary reciprocal of the mean product of the diffusion,
        from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The complementary reciprocal of the mean product of the diffusion, from the start
                to the current timestep.
        
        """
        return extract(self.sqrt_recipm1_alphas_cumprod, timestep, input.shape)

    def posterior_variance(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the variance of the posterior.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variance of the posterior.
        
        """
        return extract(self.posterior_variances, timestep, input.shape)
    
    def posterior_logarithm_variance(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the logarithm variance of the posterior.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The logarithm variance of the posterior.
        
        """
        return extract(self.posterior_log_variance_clipped, timestep, input.shape)
    
    def posterior_mean_start_coefficient(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the 'start' part coefficient of the posterior mean.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The 'start' part coefficient of the posterior mean.
        
        """
        # (\sqrt{\bar{\alpha}_{t-1}} \beta_t) / (1 - \bar{\alpha}_t)
        return extract(self.posterior_mean_coef1, timestep, input.shape)
    
    def posterior_mean_current_coefficient(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the 'current timestep' part coefficient of the posterior mean.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The 'current timestep' part coefficient of the posterior mean.
        
        """
        # (\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})) / (1 - \bar{\alpha}_t)
        return extract(self.posterior_mean_coef2, timestep, input.shape)
    
    # def on_scaled(self, old_resolution_scale_square: Optional[float]):
    #     self.__init_schedule()
    
    # def shift_schedule(self, variance: Tensor) -> Tuple[Tensor, Tensor]:
    #     if self.resolution_scale_square is not None:
    #         variance_mul_scale = variance * self.resolution_scale_square
    #         variance = variance_mul_scale / (variance_mul_scale - variance + 1)
    #     return 1. - variance, variance