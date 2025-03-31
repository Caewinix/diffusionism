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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_delta_t = diffusion_schedule.noise_scale(input, timestep + differential_timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        
        original_weight = (beta_derivative / beta - alpha_derivative / alpha) * differential_timestep
        return (1 - posterior_weight) * beta / beta_delta_t * original_weight + posterior_weight * original_weight

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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        
        return (input - gamma - beta * target) / alpha
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return alpha_derivative / alpha * (input - gamma) + gamma_derivative + (beta_derivative - beta * alpha_derivative / alpha) * target
    
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
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        
        return -target / beta

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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        
        return input * (1 + alpha_derivative / alpha * differential_timestep) + (beta_derivative - beta * alpha_derivative / alpha) * differential_timestep * target
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return (1 + alpha_derivative / alpha * differential_timestep) * input + differential_timestep * (
            gamma_derivative - alpha_derivative / alpha * gamma + 2 * (beta_derivative - beta * alpha_derivative / alpha) * target
        ) + torch.sqrt_(2 * beta_derivative * (beta_derivative - beta * alpha_derivative / alpha) * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)


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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_delta_t = diffusion_schedule.noise_scale(input, timestep + differential_timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        
        original_weight = (alpha_derivative / beta - alpha * beta_derivative / beta.square()) * differential_timestep
        return (1 - posterior_weight) * beta / beta_delta_t * original_weight + posterior_weight * original_weight
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        
        return (input - alpha * target - gamma) / beta
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        alpha_derivative_div_alpha = alpha_derivative / alpha
        
        return beta_derivative / beta * input + gamma_derivative - alpha_derivative_div_alpha * gamma + (
            alpha_derivative_div_alpha - beta_derivative / beta
        ) * (alpha * target + gamma)
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        
        return (alpha * target + gamma - input) / beta.square()

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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        
        alpha_derivative_div_alpha = alpha_derivative / alpha
        beta_derivative_div_beta = beta_derivative / beta
        
        return input * (1 + differential_timestep * beta_derivative_div_beta) + (
            alpha_derivative_div_alpha - beta_derivative_div_beta
        ) * differential_timestep * (alpha * target + gamma)
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        alpha_derivative_div_alpha = alpha_derivative / alpha
        beta_derivative_div_beta = beta_derivative / beta
        
        return input * (1 + differential_timestep * (2 * alpha_derivative_div_alpha - beta_derivative_div_beta)) + differential_timestep * (
            gamma_derivative - alpha_derivative_div_alpha * gamma + 2 * ((alpha_derivative - alpha * beta_derivative / beta) * target + (alpha_derivative_div_alpha - beta_derivative_div_beta) * gamma)
        ) + torch.sqrt_(2 * beta_derivative * (beta_derivative - beta * alpha_derivative / alpha) * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)


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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return alpha_derivative * input_start + gamma_derivative + noise * beta_derivative
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_delta_t = diffusion_schedule.noise_scale(input, timestep + differential_timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        
        original_weight = (beta_derivative / beta - alpha_derivative / alpha) * differential_timestep / beta
        return (1 - posterior_weight) * beta / beta_delta_t * original_weight + posterior_weight * original_weight
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return (beta_derivative * input - beta * target - gamma * beta_derivative + beta * gamma_derivative) / (alpha_derivative * beta - alpha * beta_derivative)
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return (alpha_derivative * (input - gamma) + alpha * (gamma_derivative - target)) / (beta * (alpha * beta_derivative - alpha_derivative * beta))

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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return input + differential_timestep * (target - gamma_derivative + alpha_derivative / alpha * gamma)
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        alpha_derivative_div_alpha = alpha_derivative / alpha
        
        return (1 - alpha_derivative_div_alpha * differential_timestep) * input + differential_timestep * (
            2 * target - gamma_derivative + alpha_derivative_div_alpha * gamma
        ) + torch.sqrt_(2 * beta_derivative * (beta_derivative - beta * alpha_derivative / alpha) * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)


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
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        
        return -noise / beta
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_delta_t = diffusion_schedule.noise_scale(input, timestep + differential_timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        
        original_weight = (beta_derivative / beta - alpha_derivative / alpha) * differential_timestep * beta
        return (1 - posterior_weight) * beta / beta_delta_t * original_weight + posterior_weight * original_weight
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        
        return (input - gamma + beta.square() * target) / alpha
    
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
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        
        return -target * beta
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return alpha_derivative / alpha * (input - gamma) + gamma_derivative - (beta_derivative - beta * alpha_derivative / alpha) * target * beta

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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        
        return input * (1 + alpha_derivative / alpha * differential_timestep) + (beta * alpha_derivative / alpha - beta_derivative) * differential_timestep * target * beta
    
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
        alpha = diffusion_schedule.input_scale(input, timestep, *args, **kwargs)
        alpha_derivative = diffusion_schedule.input_scale_derivative(input, timestep, *args, input_scale=alpha, **kwargs)
        beta = diffusion_schedule.noise_scale(input, timestep, *args, **kwargs)
        beta_derivative = diffusion_schedule.noise_scale_derivative(input, timestep, *args, noise_scale=beta, **kwargs)
        gamma = diffusion_schedule.bias(input, timestep, *args, **kwargs)
        gamma_derivative = diffusion_schedule.bias_derivative(input, timestep, *args, bias=gamma, **kwargs)
        
        return (1 + alpha_derivative / alpha * differential_timestep) * input + differential_timestep * (
            gamma_derivative - alpha_derivative / alpha * gamma - 2 * (beta_derivative - beta * alpha_derivative / alpha) * target * beta
        ) + torch.sqrt_(2 * beta_derivative * (beta_derivative - beta * alpha_derivative / alpha) * torch.abs_(torch.as_tensor(differential_timestep, device=timestep.device))) * torch.randn_like(target)