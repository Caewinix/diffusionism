from typing import Type, overload, Union, Optional, Tuple, Sequence, Callable
import re
import inspect
import torch
from torch import Tensor
from torch import nn
from torchflint.nn import Module
from ..schedules.diffusion_schedule import DiffusionSchedule
from ..parameterizations import Parameterization
from ..schedules.context import distributions, sequencings
from ..samplers.sampling_progress import OutputHandler
from .. import losses
from ..utils.range_clipper import RangeClipper
from ..diffusers.diffuser import Diffuser
from ..samplers.sampler import Sampler


def _validate_keyword_arguments(func, *args, **kwargs):
    while True:
        try:
            return func(*args, **kwargs)
        except TypeError as e:
            match = re.match(r"(.*?)\(\) got an unexpected keyword argument '(.*?)'", e.args[0])
            if match is None:
                raise
            keyword = match.group(2)
            kwargs.pop(keyword)


class DiffusionModel(Module):
    """The diffusion model that manages both forward and reverse process.
    """
    diffuser: Diffuser
    sampler: Sampler
    
    @overload
    def __init__(self, diffuser: Diffuser, sampler: Sampler):
        """
        Args:
            diffuser (Diffuser): The diffuser instance.
            sampler (Sampler): The sampler instance.

        """
        ...
    
    @overload
    def __init__(
        self,
        diffuser: Type[Diffuser],
        sampler: Type[Sampler],
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: DiffusionSchedule,
        *,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)]
    ):
        """
        Args:
            diffuser (Type[Diffuser]): The diffuser type that is expected to use.
            sampler (Type[Sampler]): The sampler type that is expected to use.
            backbone (nn.Module): The backbone model of the diffusion model.
            timesteps (Union[sequencings.Context, Tensor]): The timesteps sequenceing method that
                returns a sequence of timesteps in the descending order, or the descending order
                timesteps sequence itself.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
        
        """
        ...
    
    @overload
    def __init__(
        self,
        diffuser: Type[Diffuser],
        sampler: Type[Sampler],
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        *args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)],
        **kwargs
    ):
        """
        Args:
            diffuser (Type[Diffuser]): The diffuser type that is expected to use.
            sampler (Type[Sampler]): The sampler type that is expected to use.
            backbone (nn.Module): The backbone model of the diffusion model.
            timesteps (Union[sequencings.Context, Tensor]): The timesteps sequenceing method that
                returns a sequence of timesteps in the descending order, or the descending order
                timesteps sequence itself.
            *args: The arguments that are used to construct the instance of
                :class:`DiffusionSchedule`.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that are used to construct the instance of
                :class:`DiffusionSchedule`.
        
        """
        ...
    
    def __init__(self, diffuser: Union[Type[Diffuser], Diffuser], sampler: Union[Type[Sampler], Sampler], *args, **kwargs):
        if isinstance(diffuser, Diffuser):
            self.diffuser = diffuser
            if isinstance(sampler, Sampler):
                self.sampler = sampler
            elif issubclass(sampler, Sampler):
                try:
                    self.sampler = sampler(
                        self.diffuser.backbone,
                        self.diffuser.diffusion_schedule,
                        parameterization=self.diffuser.parameterization
                    )
                except:
                    self.sampler = sampler(*args, **kwargs)
            else:
                raise TypeError(f"'sampler' should be a subclass or an instance of '{Sampler.__name__}', but got a counterpart of '{sampler.__name__ if isinstance(sampler, type) else type(sampler).__name__}'")
        elif issubclass(diffuser, Diffuser):
            if isinstance(sampler, Sampler):
                self.sampler = sampler
                try:
                    self.diffuser = diffuser(
                        self.sampler.backbone,
                        self.sampler.diffusion_schedule,
                        parameterization=self.sampler.parameterization,
                        **kwargs
                    )
                except:
                    self.diffuser = diffuser(*args, **kwargs)
            elif issubclass(sampler, Sampler):
                diffuser_init_parameter_keys = set(inspect.signature(diffuser).parameters.keys())
                sampler_init_parameter_keys = set(inspect.signature(sampler).parameters.keys())
                intersection_keys = diffuser_init_parameter_keys.intersection(sampler_init_parameter_keys)
                kwarg_keys = set(kwargs.keys())
                sampler_kwarg_keys = kwarg_keys.difference(diffuser_init_parameter_keys.difference(intersection_keys))
                diffuser_kwarg_keys = kwarg_keys.difference(sampler_init_parameter_keys.difference(intersection_keys))
                try:
                    self.diffuser = diffuser(*args, **{key : kwargs[key] for key in diffuser_kwarg_keys})
                except:
                    try:
                        self.diffuser = diffuser(*args, **kwargs)
                    except:
                        self.diffuser = _validate_keyword_arguments(diffuser, *args, **kwargs)
                try:
                    self.sampler = sampler(*args, **{key : kwargs[key] for key in sampler_kwarg_keys})
                except:
                    try:
                        self.sampler = sampler(*args, **kwargs)
                    except:
                        self.sampler = _validate_keyword_arguments(sampler, *args, **kwargs)
            else:
                raise TypeError(f"'sampler' should be a subclass or an instance of '{Sampler.__name__}', but got a counterpart of '{sampler.__name__ if isinstance(sampler, type) else type(sampler).__name__}'")
        else:
            raise TypeError(f"'diffuser' should be a subclass or an instance of '{Diffuser.__name__}', but got a counterpart of '{diffuser.__name__ if isinstance(diffuser, type) else type(diffuser).__name__}'")
    
    def _apply(self, fn, recurse = True) -> nn.Module:
        result = super()._apply(fn, recurse)
        result.sampler.backbone = result.diffuser.backbone
        result.sampler.diffusion_schedule = result.diffuser.diffusion_schedule
        return result
    
    @property
    def backbone(self) -> nn.Module:
        assert self.diffuser.backbone is self.sampler.backbone
        return self.diffuser.backbone
    
    @backbone.setter
    def backbone(self, value: nn.Module):
        self.diffuser.backbone = value
        self.sampler.backbone = value
    
    @property
    def diffusion_schedule(self) -> DiffusionSchedule:
        assert self.diffuser.diffusion_schedule is self.sampler.diffusion_schedule
        return self.diffuser.diffusion_schedule
    
    @diffusion_schedule.setter
    def diffusion_schedule(self, value: DiffusionSchedule):
        self.diffuser.diffusion_schedule = value
        self.sampler.diffusion_schedule = value
    
    @property
    def parameterization(self) -> Parameterization:
        assert self.diffuser.parameterization is self.sampler.parameterization
        return self.diffuser.parameterization
    
    @parameterization.setter
    def parameterization(self, value: Parameterization):
        self.diffuser.parameterization = value
        self.sampler.parameterization = value
    
    def diffuse(
        self,
        input_start: Tensor,
        timestep: Tensor,
        *args,
        noise: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Diffuses the clean input to a degraded one.

        Args:
            input_start (Tensor): The clean input which means it is at the first step.
            timestep (Tensor): The required diffusion timestep, making the clean
                input degrade.
            *args: The arguments that drive the diffusion process.
            noise (Optional[Tensor]): A noise, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this noise, or not require a
                noise input.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A diffused result at the given timestep.
        
        """
        return self.diffuser.diffuse(
            self.diffuser.diffusion_schedule,
            input_start,
            timestep,
            *args,
            noise=noise,
            **kwargs
        )
    
    def predict(
        self,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Union[Tensor, float, int]]]:
        """Denoise the degraded input at the given timestep into the clean one, and the weighting
        for losses if it is necessary.
        
        Args:
            input (Tensor): The input, but not requiring at a concrete timestep.
            timestep (Tensor): The diffusion timestep according to the input.
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.
        
        Returns:
            Tensor: A predicted clean state, and the weighting fr the losses if needed.
        
        """
        return self.diffuser.predict(
            self.diffuser.backbone,
            self.diffuser.diffusion_schedule,
            input,
            timestep,
            *args,
            **kwargs
        )
    
    def degrade(
        self,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Makes a noise.

        Args:
            input (Tensor): The input, but not requiring at a concrete timestep.
            timestep (Tensor): The diffusion timestep according to the input.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A noise.
        
        """
        return self.diffuser.degrade(
            self.diffuser.diffusion_schedule,
            input,
            timestep,
            *args,
            **kwargs
        )
    
    def get_ending(
        self,
        input_end: Union[Tensor, Sequence[int]],
        timestep: Tensor,
        *args,
        backbone: nn.Module,
        **kwargs
    ) -> Tensor:
        """Returns the end step state.

        Args:
            input_end (Tensor): The input at the final timestep.
            timestep (Tensor): The final timestep according to the input.
            *args: The arguments that drive the diffusion process.
            backbone (nn.Module): The backbone model which predicts the result.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: An end step state.
        
        """
        return self.sampler.get_ending(
            self.sampler.diffusion_schedule,
            input_end,
            timestep,
            *args,
            backbone=backbone,
            **kwargs
        )
    
    @torch.no_grad()
    def sample(
        self,
        input_end: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 1.0,
        **kwargs
    ) -> Tensor:
        """Samples the degraded input to the predicted input.

        Args:
            input_end (Tensor): The input at the final (or the initial state) step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            initial_state (Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result at the given timestep.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        return self.sampler(
            input_end,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            initial_state=initial_state,
            strength=strength
            **kwargs
        )
    
    def step_loss(
        self,
        input_start: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the calculated loss.

        Args:
            input_start (Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: The loss results.
        
        """
        return self.diffuser(input_start, *args, **kwargs)

    @overload
    def forward(
        self,
        input_start: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the calculated loss.

        Args:
            input_start (Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: The loss results.
        
        """
        ...
    
    @overload
    @torch.no_grad()
    def forward(
        self,
        input_end: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Tensor:
        """Samples the degraded input to the predicted input.

        Args:
            input_end (Tensor): The input at the final (or the initial state) step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            initial_state (Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result at the given timestep.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        ...

    def forward(self, *args, **kwargs) -> Tensor:
        assert self.diffuser.backbone is self.sampler.backbone
        assert self.diffuser.diffusion_schedule is self.sampler.diffusion_schedule
        if self.training:
            return self.diffuser(*args, **kwargs)
        else:
            return self.sampler(*args, **kwargs)