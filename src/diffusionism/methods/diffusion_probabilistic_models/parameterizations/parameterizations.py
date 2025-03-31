from typing import Union
import torch
from torch import Tensor
from ....diffusion.parameterizations.linear.conventional import (
    NoiseParameterization,
    InputStartParameterization,
    VelocityParameterization
)
from ..schedules.schedule import DiffusionProbabilisticModelsDiscreteSchedule


class NoiseParameterization(NoiseParameterization):
    def __init__(self, simple_weight: float = 1., elbo_weight: float = 0.):
        """
        Args:
            simple_weight (float): The weight of the simple part loss.
            elbo_weight (float): The weight of the ELBO part loss.
        
        """
        super().__init__()
        self.simple_weight = simple_weight
        self.elbo_weight = elbo_weight
    
    @property
    def is_loss_complex(self) -> bool:
        """
        Indicates whether it should calculate the loss complexly.
        """
        return self.simple_weight != 1. or self.elbo_weight != 0.
    
    def adjust_loss(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        loss: Tensor,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        if self.is_loss_complex:
            loss_simple = loss * self.simple_weight
            loss_vlb = self.constant_weight(
                self,
                diffusion_schedule,
                input,
                timestep,
                differential_timestep,
                *args,
                diffusion_schedule.v_posterior,
                **kwargs
            ) * loss
            loss = loss_simple + self.elbo_weight * loss_vlb
        return loss
    
    def constant_weight(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        betas = diffusion_schedule.step_noise_variance(input, timestep, *args, **kwargs)
        alphas_cumprod_prev = diffusion_schedule.input_scale_square(input, timestep - differential_timestep, *args, **kwargs)
        alphas_cumprod = diffusion_schedule.input_scale_square(input, timestep, *args, **kwargs)
        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            (1 - posterior_weight) * betas * (1. - alphas_cumprod_prev) /
            (1. - alphas_cumprod) + posterior_weight * betas
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        timestep = timestep.clone()
        timestep[timestep == 0] = 1
        return (
            torch.square(betas) / 
            (
                2 * posterior_variance *
                diffusion_schedule.step_input_scale_square(input, timestep, *args, **kwargs) *
                (1 - alphas_cumprod)
            )
        )
    
    def input_start(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        mean_square_product = diffusion_schedule.input_scale_square(input, timestep, *args, **kwargs)
        
        pred_x0 = (input - torch.sqrt(1. - mean_square_product) * target) / torch.sqrt(mean_square_product)
        return pred_x0
    
    def reverse_ordinary_step(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int] = 1,
        *args,
        **kwargs
    ) -> Tensor:
        mean = diffusion_schedule.reciprocal_cumulative_mean(input, timestep, *args, **kwargs) * input
        std = diffusion_schedule.complementary_reciprocal_cumulative_mean(input, timestep, *args, **kwargs)
        return mean - std * target


class InputParameterization(InputStartParameterization):
    def __init__(self, simple_weight: float = 1., elbo_weight: float = 0., posterior_weight = 0.):
        """
        Args:
            simple_weight (float): The weight of the simple part loss.
            elbo_weight (float): The weight of the ELBO part loss.
        
        """
        super().__init__()
        self.simple_weight = simple_weight
        self.elbo_weight = elbo_weight
        self.posterior_weight = posterior_weight
    
    @property
    def is_loss_complex(self) -> bool:
        """
        Indicates whether it should calculate the loss complexly.
        """
        return self.simple_weight != 1. or self.elbo_weight != 0.
    
    def adjust_loss(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        loss: Tensor,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        if self.is_loss_complex:
            loss_simple = loss * self.simple_weight
            loss_vlb = self.constant_weight(
                self,
                diffusion_schedule,
                input,
                timestep,
                differential_timestep,
                *args,
                self.posterior_weight,
                **kwargs
            ) * loss
            loss = loss_simple + self.elbo_weight * loss_vlb
        return loss
    
    def constant_weight(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int] = 1,
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        timestep = timestep.clone()
        timestep[timestep == 0] = 1
        mean_square_product = diffusion_schedule.input_scale_square(input, timestep, *args, **kwargs)
        return 0.5 * torch.sqrt(mean_square_product) / (2. * 1 - mean_square_product)
    
    def reverse_ordinary_step(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int] = 1,
        *args,
        **kwargs
    ) -> Tensor:
        return target


class VelocityParameterization(VelocityParameterization):
    def __init__(self, simple_weight: float = 1., elbo_weight: float = 0., posterior_weight = 0.):
        """
        Args:
            simple_weight (float): The weight of the simple part loss.
            elbo_weight (float): The weight of the ELBO part loss.
        
        """
        super().__init__()
        self.simple_weight = simple_weight
        self.elbo_weight = elbo_weight
        self.posterior_weight = posterior_weight
    
    @property
    def is_loss_complex(self) -> bool:
        """
        Indicates whether it should calculate the loss complexly.
        """
        return self.simple_weight != 1. or self.elbo_weight != 0.
    
    def adjust_loss(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        loss: Tensor,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        if self.is_loss_complex:
            loss_simple = loss * self.simple_weight
            loss_vlb = self.constant_weight(
                self,
                diffusion_schedule,
                input,
                timestep,
                differential_timestep,
                *args,
                self.posterior_weight,
                **kwargs
            ) * loss
            loss = loss_simple + self.elbo_weight * loss_vlb
        return loss
    
    def truth_target(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return (
            diffusion_schedule.input_scale(input, timestep, *args, **kwargs) * noise -
            diffusion_schedule.noise_scale(input, timestep, *args, **kwargs) * input_start
        )
    
    def input_start(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        mean_product = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        standard_deviation_product = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        
        pred_x0 = mean_product * input - standard_deviation_product * target
        return pred_x0
    
    def noise(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        mean_product = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        standard_deviation_product = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        
        return mean_product * target + standard_deviation_product * input
    
    def constant_weight(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        return torch.ones_like(timestep)
    
    def reverse_ordinary_step(
        self,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int] = 1,
        *args,
        **kwargs
    ) -> Tensor:
        raise TypeError(f"{type(self).__name__} is only suitbale for DDIM samplers, and the current sampling process might not be DDIM")