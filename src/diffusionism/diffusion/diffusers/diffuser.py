from typing import overload, Callable, Optional, Iterable, Tuple
import torch
from torch import Tensor
from torch import nn
from ..buffers.diffusion_buffer import DiffusionBuffer
from ..diffusion import MetaDiffusion, Diffusion
from ..parameterizations import Parameterization, NoiseParameterization
from .. import losses


class Diffuser(Diffusion, buffer=DiffusionBuffer):
    """The diffuser is an implementation that contains the forward process part of
    the diffusion model.
    """
    diffusion_buffer: DiffusionBuffer
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        diffusion_buffer: DiffusionBuffer,
        *,
        parameterization: Parameterization = NoiseParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
        
        """
        ...
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        parameterization: Parameterization = NoiseParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            *args: The arguments that are used to construct the instance of
                :class:`DiffusionBuffer`.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            **kwargs: The keyword arguments that are used to construct the instance of
                :class:`DiffusionBuffer`.
        
        """
        ...
    
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        parameterization: Parameterization = NoiseParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        super(Diffusion, self).__init__()
        self.backbone = backbone
        self.parameterization = parameterization
        self.loss_function = loss_function
        self.diffusion_buffer = MetaDiffusion.get_diffusion_buffer(
            type(self).buffer, *args, error_head=f"{type(self).__name__}.{type(self).__init__.__code__.co_name}()", **kwargs
        )
    
    @classmethod
    def diffuse(
        cls,
        diffusion_buffer: DiffusionBuffer,
        input_start: Tensor,
        timestep: Tensor,
        *diffusion_args,
        degradation: Optional[Tensor] = None,
        **diffusion_kwargs
    ) -> Tensor:
        """Diffuses the clean input to a degraded one.

        Args:
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input_start (Tensor): The clean input which means it is at the first step.
            timestep (Tensor): The required diffusion timestep, making the clean
                input degrade.
            *diffusion_args: The arguments that drive the diffusion process.
            degradation (Optional[Tensor]): A degradation, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this degradation, or not require a
                degradation input.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A diffused result at the given timestep.
        
        """
        # q_sample
        pass
    
    @classmethod
    def degrade(
        cls,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *diffusion_args,
        **diffusion_kwargs
    ) -> Tensor:
        """Makes a degradation.

        Args:
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input (Tensor): The input, but not requiring at a concrete timestep.
            timestep (Tensor): The diffusion timestep according to the input.
            *diffusion_args: The arguments that drive the diffusion process.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A degradation.
        
        """
        pass
    
    def parameters(self, recurse = True):
        return self.backbone.parameters(recurse)
    
    def construct_optimization(
        self,
        input_start: Tensor,
        timestep: Tensor,
        *diffusion_args,
        degradation: Optional[Tensor] = None,
        **diffusion_kwargs
    ) -> Tensor:
        """Constructs the optimization target.

        Args:
            input_start (Tensor): The clean input at the first timestep, usually used as shape tips.
            timestep (Tensor): The diffusion timestep that is used to extract the optimization target.
            *diffusion_args: The arguments that drive the diffusion process.
            degradation (Optional[Tensor]): A degradation, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this degradation, or not require a
                degradation input.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: An optimization target.
        
        """
        return self.parameterization.optimization_target(self.diffusion_buffer, input_start, timestep, *diffusion_args, degradation=degradation, **diffusion_kwargs)

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
        # p_losses "Algorithm 1"
        diffusion_args, diffusion_kwargs = self.diffusion_buffer.get_diffusion_arguments(*args, **kwargs)
        backbone_args, backbone_kwargs = self.diffusion_buffer.get_backbone_arguments(*args, **kwargs)
        
        timestep = torch.randint(self.diffusion_buffer.num_timesteps, size=input_start.shape[:1], device=input_start.device)
        degradation = self.degrade(self.diffusion_buffer, input_start, timestep, *diffusion_args, **diffusion_kwargs)
        x_t = self.diffuse(self.diffusion_buffer, input_start, timestep, *diffusion_args, degradation=degradation, **diffusion_kwargs)
        
        prediction = self.backbone(x_t, timestep, *backbone_args, **backbone_kwargs)
        optimization_target = self.construct_optimization(input_start, timestep, *diffusion_args, degradation=degradation, **diffusion_kwargs)
        loss = self.loss_function(prediction, optimization_target)
        
        if self.parameterization.is_loss_complex:
            loss_simple = loss * self.parameterization.simple_weight
            loss_vlb = self.parameterization.variational_weights(self.diffusion_buffer, x_t, timestep, *diffusion_args, **diffusion_kwargs) * loss
            loss = loss_simple + self.parameterization.elbo_weight * loss_vlb
        
        return loss
    
    @overload
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
    
    def forward(self, *args, **kwargs) -> Tensor:
        return self.get_losses(*args, **kwargs)