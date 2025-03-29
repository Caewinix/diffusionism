from typing import overload, Callable, Optional, Tuple
from torch import nn
from torch import Tensor
from ....diffusion.diffusers.diffuser import Diffuser
from ..parameterizations import InputStartParameterization
from ..schedules.schedules import ElucidatedDiffusionModelsSchedule
from ....diffusion.parameterizations import Parameterization
from ....diffusion import losses
from ....diffusion.schedules.context import distributions


class ElucidatedDiffusionModelsDiffuser(Diffuser, schedule=ElucidatedDiffusionModelsSchedule):
    diffusion_schedule: ElucidatedDiffusionModelsSchedule
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        *,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization = InputStartParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss
    ):
        ...
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization = InputStartParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        ...
    
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization = InputStartParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        super().__init__(
            backbone,
            *args,
            timesteps_distribution=timesteps_distribution,
            parameterization=parameterization,
            loss_function=loss_function,
            **kwargs
        )
    
    @classmethod
    def diffuse(
        cls,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input_start: Tensor,
        timestep: Tensor,
        *diffusion_args,
        noise: Optional[Tensor] = None,
        **diffusion_kwargs
    ) -> Tensor:
        if noise is None:
            noise = cls.degrade(diffusion_schedule, input_start, timestep, *diffusion_args, **diffusion_kwargs)
        sigma = diffusion_schedule.noise_level(input_start, timestep, *diffusion_args, **diffusion_kwargs)
        return input_start + noise * sigma
    
    @classmethod
    def predict(
        cls,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tensor]:
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        backbone_args, backbone_kwargs = diffusion_schedule.get_backbone_arguments(*args, **kwargs)
        c_in, c_out, c_skip, c_noise, loss_weighting = diffusion_schedule.preconditioning_with_weighting(input, timestep, *diffusion_args, **diffusion_kwargs)
        c_noise = c_noise.view(c_noise.size(0))
        model_output = backbone((c_in * input).to(input.dtype), c_noise.to(input.dtype), *backbone_args, **backbone_kwargs)
        denoised: Tensor = c_out * model_output + c_skip * input
        return denoised.to(input.dtype), loss_weighting.to(input.dtype)