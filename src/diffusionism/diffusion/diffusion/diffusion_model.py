from typing import Type, overload, Union, Callable, Optional, Iterable
import torch
from torch import Tensor
from torch import nn
from ..buffers.diffusion_buffer import DiffusionBuffer
from ..parameterizations import Parameterization, NoiseParameterization
from .. import losses
from .diffusion import Diffusion


class DiffusionModel(Diffusion):
    """The diffusion model that manages both forward and reverse process.
    """
    diffuser: nn.Module
    sampler: nn.Module
    
    def __init__(self, diffuser: Union[Type[nn.Module], nn.Module], sampler: Union[Type[nn.Module], nn.Module], *args, **kwargs):
        super().__init__()
        from ..diffusers.diffuser import Diffuser
        from ..samplers.sampler import Sampler
        if isinstance(diffuser, Diffuser):
            self.diffuser = diffuser
            if isinstance(sampler, Sampler):
                self.sampler = sampler
            elif issubclass(sampler, Sampler):
                try:
                    self.sampler = sampler(
                        self.diffuser.backbone,
                        self.diffuser.diffusion_buffer,
                        parameterization=self.diffuser.parameterization
                    )
                except:
                    self.sampler = sampler(*args, **kwargs)
            else:
                raise TypeError(f"'sampler' should be a subclass or an instance of '{Sampler.__name__}', but got a counterpart of '{sampler.__name__ if isinstance(sampler, type) else type(sampler).__name__}'.")
        elif issubclass(diffuser, Diffuser):
            if isinstance(sampler, Sampler):
                self.sampler = sampler
                try:
                    self.diffuser = diffuser(
                        self.sampler.backbone,
                        self.sampler.diffusion_buffer,
                        parameterization=self.sampler.parameterization,
                        **kwargs
                    )
                except:
                    self.diffuser = diffuser(*args, **kwargs)
            elif issubclass(sampler, Sampler):
                from inspect import signature
                diffuser_init_parameter_keys = set(signature(diffuser).parameters.keys())
                sampler_init_parameter_keys = set(signature(sampler).parameters.keys())
                intersection_keys = diffuser_init_parameter_keys.intersection(sampler_init_parameter_keys)
                kwarg_keys = set(kwargs.keys())
                sampler_kwarg_keys = kwarg_keys.difference(diffuser_init_parameter_keys.difference(intersection_keys))
                diffuser_kwarg_keys = kwarg_keys.difference(sampler_init_parameter_keys.difference(intersection_keys))
                
                self.diffuser = diffuser(*args, **{key : kwargs[key] for key in diffuser_kwarg_keys})
                self.sampler = sampler(*args, **{key : kwargs[key] for key in sampler_kwarg_keys})
            else:
                raise TypeError(f"'sampler' should be a subclass or an instance of '{Sampler.__name__}', but got a counterpart of '{sampler.__name__ if isinstance(sampler, type) else type(sampler).__name__}'.")
        else:
            raise TypeError(f"'diffuser' should be a subclass or an instance of '{Diffuser.__name__}', but got a counterpart of '{diffuser.__name__ if isinstance(diffuser, type) else type(diffuser).__name__}'.")
    
    def diffuse(
        self,
        input_start: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Diffuses the clean input to a degraded one.

        Args:
            input_start (Tensor): The clean input which means it is at the first step.
            timestep (Tensor): The required diffusion timestep, making the clean
                input degrade.
            *args: The arguments that drive the diffusion process.
            degradation (Optional[Tensor]): A degradation, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this degradation, or not require a
                degradation input.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A diffused result at the given timestep.
        
        """
        return self.diffuser.diffuse(
            self.diffuser.diffusion_buffer,
            input_start,
            timestep,
            *args,
            degradation=degradation,
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
        return self.sampler.sample(
            self.sampler.backbone,
            self.sampler.diffusion_buffer,
            self.sampler.parameterization,
            input_end,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            initial_state=initial_state,
            strength=strength,
            **kwargs
        )
    
    def get_losses(
        self,
        input_start: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Calculates the losses regarding to any timesteps.

        Args:
            input_start (Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: The loss results.
        
        """
        return self.diffuser.get_losses(input_start, *args, **kwargs)

    @overload
    @torch.no_grad()
    def forward(
        self,
        input_start: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Calculates the losses regarding to any timesteps.

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
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
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
        if self.training:
            return self.diffuser(*args, **kwargs)
        else:
            return self.sampler(*args, **kwargs)