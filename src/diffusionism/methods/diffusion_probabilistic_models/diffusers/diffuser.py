from typing import Callable, Optional
from torch import nn
from torch import Tensor
from ....diffusion.diffusers.diffuser import Diffuser
from ....diffusion.parameterizations import Parameterization
from ..parameterizations import NoiseParameterization
from ..schedules.schedule import DiffusionProbabilisticModelsDiscreteSchedule
from ....diffusion import losses
from ....diffusion.schedules.context import distributions


class DiffusionProbabilisticModelsDiffuser(Diffuser, schedule=DiffusionProbabilisticModelsDiscreteSchedule):
    diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule
    
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        timesteps_distribution: distributions.Context = distributions.uniform_int.init(1000),
        parameterization: Parameterization = NoiseParameterization(),
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
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input_start: Tensor,
        timestep: Tensor,
        *diffusion_args,
        noise: Optional[Tensor] = None,
        **diffusion_kwargs
    ) -> Tensor:
        # q_sample
        if noise is None:
            noise = cls.degrade(diffusion_schedule, input_start, timestep, *diffusion_args, **diffusion_kwargs)
        mean = diffusion_schedule.input_scale(input_start, timestep, *diffusion_args, **diffusion_kwargs) * input_start
        std = diffusion_schedule.noise_scale(input_start, timestep, *diffusion_args, **diffusion_kwargs)
        x_t = mean + std * noise
        return x_t