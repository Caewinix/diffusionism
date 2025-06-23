from typing import Union
import torch
from torch import Tensor
from ...parameterizations import (
    NoiseParameterization,
    InputStartParameterization,
    VelocityParameterization,
    ScoreParameterization
)
from ....schedules import LinearDiffusionSchedule


class NoiseParameterization(NoiseParameterization):
    @classmethod
    def constant_weight(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_delta_t = diffusion_schedule.scaling(input, timestep + differential_timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_delta_t = diffusion_schedule.noise_level(input, timestep + differential_timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        
        original_weight = sigma_derivative / sigma * differential_timestep
        return (1 - posterior_weight) * s * sigma / (s_delta_t * sigma_delta_t) * original_weight + posterior_weight * original_weight
    
    @classmethod
    def input_start(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        
        return input / s - c - sigma * target
    
    @classmethod
    def velocity(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return s_derivative / s * input + s * c_derivative + s * sigma_derivative * target
    
    @classmethod
    def score(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        
        return -target / (s * sigma)

    @classmethod
    def reverse_ordinary_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        
        return input * (1 + s_derivative / s * differential_timestep) + s * sigma_derivative * differential_timestep * target
    
    @classmethod
    def reverse_stochastic_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return (1 + s_derivative / s * differential_timestep) * input + s * differential_timestep * (
            c_derivative + 2 * sigma_derivative * target
        ) + s * torch.sqrt_(2 * sigma_derivative * sigma * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)


class InputStartParameterization(InputStartParameterization):
    @classmethod
    def constant_weight(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_delta_t = diffusion_schedule.scaling(input, timestep + differential_timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_delta_t = diffusion_schedule.noise_level(input, timestep + differential_timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        
        original_weight = sigma_derivative / sigma.square() * differential_timestep
        return (1 - posterior_weight) * s * sigma / (s_delta_t * sigma_delta_t) * original_weight + posterior_weight * original_weight
    
    @classmethod
    def noise(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        
        return (input / s - target - c) / sigma
    
    @classmethod
    def velocity(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return (s_derivative / s + sigma_derivative / sigma) * input + s * (c_derivative - sigma_derivative / sigma * (target + c))
    
    @classmethod
    def score(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        
        return ((target + c) / s - input) / sigma.square()

    @classmethod
    def reverse_ordinary_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        
        return input * (
            1 + differential_timestep * (s_derivative / s + sigma_derivative / sigma)
        ) - s * sigma_derivative / sigma * differential_timestep * (target + c)
    
    @classmethod
    def reverse_stochastic_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return input * (
            1 + differential_timestep * (s_derivative / s + 2 * sigma_derivative / sigma)
        ) + s * differential_timestep * (
            c_derivative - 2 * sigma_derivative / sigma * (target + c)
        ) + s * torch.sqrt_(2 * sigma_derivative * sigma * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)


class VelocityParameterization(VelocityParameterization):
    @classmethod
    def truth_target(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return s_derivative * (input_start + c) + s * c_derivative + noise * (s_derivative * sigma + s * sigma_derivative)
    
    @classmethod
    def constant_weight(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_delta_t = diffusion_schedule.scaling(input, timestep + differential_timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_delta_t = diffusion_schedule.noise_level(input, timestep + differential_timestep, *args, **kwargs)
        
        original_weight = differential_timestep / (s * sigma)
        return (1 - posterior_weight) * s * sigma / (s_delta_t * sigma_delta_t) * original_weight + posterior_weight * original_weight
    
    @classmethod
    def input_start(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        sigma_div_sigma_derivative = sigma / sigma_derivative
        
        return input / s * (s_derivative / s * sigma_div_sigma_derivative + 1) + c + sigma_div_sigma_derivative * (
            c_derivative - target / s
        )
    
    @classmethod
    def score(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return ((s_derivative / s * input - target) / s + c_derivative) / (s * sigma_derivative * sigma)

    @classmethod
    def reverse_ordinary_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return input + differential_timestep * (target - s * c_derivative)
    
    @classmethod
    def reverse_stochastic_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return (1 - s_derivative / s * differential_timestep) * input + differential_timestep * (
            2 * target - s * c_derivative
        ) + s * torch.sqrt_(2 * sigma_derivative * sigma * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)


class ScoreParameterization(ScoreParameterization):
    @classmethod
    def truth_target(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        
        return -noise / (s * sigma)
    
    @classmethod
    def constant_weight(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_delta_t = diffusion_schedule.scaling(input, timestep + differential_timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_delta_t = diffusion_schedule.noise_level(input, timestep + differential_timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        
        original_weight = sigma_derivative * s * differential_timestep
        return (1 - posterior_weight) * s * sigma / (s_delta_t * sigma_delta_t) * original_weight + posterior_weight * original_weight
    
    @classmethod
    def input_start(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        
        return input / s - c + sigma.square() * target * s
    
    @classmethod
    def noise(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        
        return -target * s * sigma
    
    @classmethod
    def velocity(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return s_derivative / s * input + s * c_derivative - s.square() * sigma * sigma_derivative * target

    @classmethod
    def reverse_ordinary_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        
        return input * (1 + s_derivative / s * differential_timestep) - s.square() * sigma * sigma_derivative * differential_timestep * target
    
    @classmethod
    def reverse_stochastic_step(
        cls,
        diffusion_schedule: LinearDiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        s = diffusion_schedule.scaling(input, timestep, *args, **kwargs)
        s_derivative = diffusion_schedule.scaling_derivative(input, timestep, *args, scaling=s, **kwargs)
        sigma = diffusion_schedule.noise_level(input, timestep, *args, **kwargs)
        sigma_derivative = diffusion_schedule.noise_level_derivative(input, timestep, *args, noise_level=sigma, **kwargs)
        c = diffusion_schedule.shifting(input, timestep, *args, **kwargs)
        c_derivative = diffusion_schedule.shifting_derivative(input, timestep, *args, shifting=c, **kwargs)
        
        return (1 + s_derivative / s * differential_timestep) * input + s * differential_timestep * (
            c_derivative - 2 * sigma_derivative * target * s * sigma
        ) + s * torch.sqrt_(2 * sigma_derivative * sigma * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)