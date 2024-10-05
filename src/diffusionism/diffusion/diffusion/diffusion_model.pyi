from typing import Type, overload, Union, Callable, Optional, Iterable
import torch
from torch import Tensor
from torch import nn
from ..buffers.diffusion_buffer import DiffusionBuffer
from ..parameterizations import Parameterization, NoiseParameterization
from .. import losses
from .diffusion import Diffusion
from ..diffusers.diffuser import Diffuser
from ..samplers.sampler import Sampler
from ..range_clipper import RangeClipper


class DiffusionModel(Diffusion):
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
        diffusion_buffer: DiffusionBuffer,
        *,
        parameterization: Parameterization = NoiseParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        range_clipper: RangeClipper = RangeClipper(-1, 1)
    ):
        """
        Args:
            diffuser (Type[Diffuser]): The diffuser type that is expected to use.
            sampler (Type[Sampler]): The sampler type that is expected to use.
            backbone (nn.Module): The backbone model of the diffusion model.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
        
        """
        ...
    
    @overload
    def __init__(
        self,
        diffuser: Type[Diffuser],
        sampler: Type[Sampler],
        backbone: nn.Module,
        *args,
        parameterization: Parameterization = NoiseParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        range_clipper: RangeClipper = RangeClipper(-1, 1),
        **kwargs
    ):
        """
        Args:
            diffuser (Type[Diffuser]): The diffuser type that is expected to use.
            sampler (Type[Sampler]): The sampler type that is expected to use.
            backbone (nn.Module): The backbone model of the diffusion model.
            *args: The arguments that are used to construct the instance of
                :class:`DiffusionBuffer`.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
            **kwargs: The keyword arguments that are used to construct the instance of
                :class:`DiffusionBuffer`.
        
        """
        ...
    
    def diffuse(
        self,
        input_start: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor: ...
    
    @torch.no_grad()
    def sample(
        self,
        input_end: Tensor,
        *args,
        additional_residuals: Optional[Iterable[Tensor]] = None,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Tensor: ...
    
    def get_losses(
        self,
        input_start: Tensor,
        *args,
        additional_residuals: Optional[Iterable[Tensor]] = None,
        **kwargs
    ) -> Tensor: ...

    @overload
    @torch.no_grad()
    def forward(
        self,
        input_start: Tensor,
        *args,
        additional_residuals: Optional[Iterable[Tensor]] = None,
        **kwargs
    ) -> Tensor: ...
    
    @overload
    @torch.no_grad()
    def forward(
        self,
        input_end: Tensor,
        *args,
        additional_residuals: Optional[Iterable[Tensor]] = None,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 0.8,
        **kwargs
    ) -> Tensor: ...