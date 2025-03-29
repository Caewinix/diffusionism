from typing import Optional, Sequence, Tuple, Union, Any, Dict, overload
from numpy import sqrt
import torch
from torch import nn
from torch import Tensor
from ....diffusion.samplers.sampler import Sampler
from ....diffusion.parameterizations import Parameterization
from ....diffusion.samplers import OutputHandler
from ....diffusion.utils.range_clipper import RangeClipper
from ....diffusion.schedules.context import sequencings
from ..diffusers.diffuser import ElucidatedDiffusionModelsDiffuser
from ..parameterizations import InputStartParameterization
from ..schedules.schedules import ElucidatedDiffusionModelsSchedule
from ..hyperparameters.churn_stochastic_hyperparameters import ChurnStochasticHyperparameters


class ElucidatedDiffusionModelsEulerSampler(Sampler, schedule=ElucidatedDiffusionModelsSchedule, diffuser=ElucidatedDiffusionModelsDiffuser):
    diffusion_schedule: ElucidatedDiffusionModelsSchedule
    diffuser: ElucidatedDiffusionModelsDiffuser
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        *,
        parameterization: Parameterization = InputStartParameterization(),
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)]
    ):
        ...
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        *args,
        parameterization: Parameterization = InputStartParameterization(),
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)],
        **kwargs
    ):
        ...
    
    def __init__(
        self,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        *args,
        parameterization: Parameterization = InputStartParameterization(),
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)],
        **kwargs
    ):
        super().__init__(
            backbone,
            timesteps,
            *args,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        )
    
    @classmethod
    def sequence_timesteps(
        cls,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: ElucidatedDiffusionModelsSchedule
    ) -> Tensor:
        timesteps = super().sequence_timesteps(timesteps, diffusion_schedule)
        return diffusion_schedule.inverse_noise_level(timesteps)
    
    @classmethod
    def process_ending(
        cls,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input_end: Tensor,
        timestep: Tensor,
        *diffusion_args,
        **diffusion_kwargs
    ) -> Tensor:
        dtype = input_end.dtype
        return (
            input_end.to(torch.float64) * (
                diffusion_schedule.noise_level(input_end, timestep, *diffusion_args, **diffusion_kwargs)
                * diffusion_schedule.scaling(input_end, timestep, *diffusion_args, **diffusion_kwargs)
            ).to(torch.float64)
        ).to(dtype)
    
    @classmethod
    def presample_state(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input_middle: Tensor,
        timestep: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        
        t_cur = timestep
        x_next = input_middle
        
        x_cur = x_next
        
        sigma_current = diffusion_schedule.noise_level(input_middle, t_cur, *diffusion_args, **diffusion_kwargs)

        hyperparameters: ChurnStochasticHyperparameters = diffusion_schedule.hyperparameters
        S_churn, S_min, S_max, S_noise = (
            hyperparameters.stochastic_churn,
            hyperparameters.stochastic_tmin,
            hyperparameters.stochastic_tmax,
            hyperparameters.stochastic_noise
        )
        
        # Increase noise temporarily.
        gamma = torch.as_tensor(min(S_churn / num_steps, sqrt(2) - 1) if S_min <= sigma_current <= S_max else 0, dtype=timestep.dtype, device=timestep.device)
        t_hat = diffusion_schedule.inverse_noise_level(sigma_current + gamma * sigma_current)
        sigma_hat = diffusion_schedule.noise_level(input_middle, t_hat, *diffusion_args, **diffusion_kwargs)
        s_hat = diffusion_schedule.scaling(input_middle, t_hat, *diffusion_args, **diffusion_kwargs)
        s_cur = diffusion_schedule.scaling(input_middle, t_cur, *diffusion_args, **diffusion_kwargs)
        c_hat = diffusion_schedule.shifting(input_middle, t_hat, *diffusion_args, **diffusion_kwargs)
        c_cur = diffusion_schedule.shifting(input_middle, t_cur, *diffusion_args, **diffusion_kwargs)
        
        noise = cls.diffuser.degrade(diffusion_schedule, x_cur, t_cur, *diffusion_args, **diffusion_kwargs)
        x_hat = s_hat * (x_cur / s_cur + (sigma_hat ** 2 - sigma_current ** 2).clip_(min=0).sqrt_() * S_noise * noise + c_hat - c_cur)
        return super().presample_state(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            x_hat,
            timestep,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        ), t_hat
    
    @classmethod
    def predict(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input: Tensor,
        timestep: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        s = diffusion_schedule.scaling(input, timestep, *diffusion_args, **diffusion_kwargs)
        c = diffusion_schedule.shifting(input, timestep, *diffusion_args, **diffusion_kwargs)
        
        return super().predict(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            input / s - c,
            timestep,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        )
    
    @classmethod
    @torch.inference_mode()
    def sample_step(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input_middle: Tensor,
        presampled_middle: Union[Tensor, Tuple[Tensor, ...]],
        timesteps: Sequence[Tensor],
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tensor:
        x_hat, t_hat = presampled_middle
        
        denoised = cls.predict(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            x_hat,
            t_hat,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        )
        refined = cls.order_based_sample_step(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            denoised,
            presampled_middle,
            timesteps,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        )
        if isinstance(refined, Tuple):
            result = refined[0]
        else:
            result = refined
        
        return result
    
    @classmethod
    def order_based_sample_step(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input: Tensor,
        presampled_middle: Union[Tensor, Tuple[Tensor, ...]],
        timesteps: Sequence[Tensor],
        *args,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        **kwargs
    ) -> Tensor:
        t_cur, t_next = timesteps
        x_hat, t_hat = presampled_middle
        denoised = input
        
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        
        h = t_next - t_hat
        return parameterization.reverse_ordinary_step(
            diffusion_schedule,
            x_hat,
            t_hat,
            denoised,
            h,
            *diffusion_args,
            **diffusion_kwargs
        ), h


class ElucidatedDiffusionModelsHeunSampler(ElucidatedDiffusionModelsEulerSampler):
    @classmethod
    def order_based_sample_step(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: ElucidatedDiffusionModelsSchedule,
        input: Tensor,
        presampled_middle: Union[Tensor, Tuple[Tensor, ...]],
        timesteps: Sequence[Tensor],
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tensor:
        x_hat, t_hat = presampled_middle
        denoised = input
        x_next, h = super().order_based_sample_step(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            input,
            presampled_middle,
            timesteps,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        )
        if step_index != num_steps - 1:
            hyperparameters: ChurnStochasticHyperparameters = diffusion_schedule.hyperparameters
            alpha = hyperparameters.alpha
            
            diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
            d_cur = (x_next - x_hat) / h
            
            x_prime = x_hat + alpha * h * d_cur
            t_prime = t_hat + alpha * h
            
            denoised = cls.predict(
                step_index,
                num_steps,
                backbone,
                diffusion_schedule,
                x_prime,
                t_prime,
                *args,
                inpaint_reference=inpaint_reference,
                mask=mask,
                variables=variables,
                parameterization=parameterization,
                output_handlers=output_handlers,
                **kwargs
            )
            
            new_x_prime = parameterization.reverse_ordinary_step(
                diffusion_schedule,
                x_prime,
                t_prime,
                denoised,
                h,
                *diffusion_args,
                **diffusion_kwargs
            )
            x_next = (1 - 1 / alpha) * x_next + (x_hat + new_x_prime) / (2 * alpha)
        return x_next