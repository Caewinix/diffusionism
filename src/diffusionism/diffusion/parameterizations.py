from typing import Optional
from abc import ABC, abstractmethod
import torch
from torch import Tensor
from .buffers.diffusion_buffer import DiffusionBuffer


class Parameterization(ABC):
    """The parameterization situation of the diffusion model.
    """
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
    
    @abstractmethod
    def optimization_target(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Returns the optimization target.

        Args:
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input (Tensor): The input which is usually used as shape tips.
            timestep (Tensor): The diffusion timestep that is used to extract the optimization target.
            *args: The arguments that drive the diffusion process.
            degradation (Optional[Tensor]): A degradation, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this degradation, or not require a
                degradation input.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: An optimization target.
        
        """
        pass
    
    @abstractmethod
    def variational_weights(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the variational lower bound loss weights.

        Args:
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            *args: The arguments that drive the diffusion process.
            degradation (Optional[Tensor]): A degradation, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this degradation, or not require a
                degradation input.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variational lower bound loss weights.
        
        """
        pass
    
    @abstractmethod
    def predict_current_and_start(
        self,
        backbone_prediction: Tensor,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        retention_total_var: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Predicts the state at the current timestep and the beginning timestep, respectively.

        Args:
            backbone_prediction (Tensor): The raw prediction that is onlynoutput from the
                backbone model.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            *args: The arguments that drive the diffusion process.
            retention_total_var (Optional[Tensor]): The variance of retention degree, from
                the start to the current timestep. If ``None``, the function should calculate
                that through the :param:`diffusion_buffer`.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The state at the current timestep, and the state at the beginning timestep.
        
        """
        pass
    
    @abstractmethod
    def reconstruct_step(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        """Reconstructs the state to the next timestep.

        Args:
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            *args: The arguments that drive the diffusion process.
            degradation (Optional[Tensor]): A degradation, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this degradation, or not require a
                degradation input.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The reconstructed state to the next timestep.
        
        """
        pass


class NoiseParameterization(Parameterization):
    def optimization_target(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        return degradation
    
    def variational_weights(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        timestep = timestep.clone()
        timestep[timestep == 0] = 1
        return (
            torch.square(diffusion_buffer.degradation_var(input, timestep, *args, **kwargs)) / 
            (
                2 * diffusion_buffer.posterior_var(input, timestep, *args, **kwargs) *
                diffusion_buffer.retention_var(input, timestep, *args, **kwargs) *
                (1 - diffusion_buffer.retention_total_var(input, timestep, *args, **kwargs))
            )
        )
    
    def predict_current_and_start(
        self,
        backbone_prediction: Tensor,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        retention_total_var: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        if retention_total_var is None:
            retention_total_var = diffusion_buffer.retention_total_var(input, timestep, *args, **kwargs)
        
        e_t = backbone_prediction
        pred_x0 = (input - torch.sqrt(1. - retention_total_var) * e_t) / torch.sqrt(retention_total_var)
        return e_t, pred_x0
    
    def reconstruct_step(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        mean = diffusion_buffer.reciprocal_retention_total_std(input, timestep, *args, **kwargs) * input
        std = diffusion_buffer.complementary_reciprocal_retention_total_std(input, timestep, *args, **kwargs)
        return mean - std * degradation


class InputParameterization(NoiseParameterization):
    def optimization_target(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        return input

    def variational_weights(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        timestep = timestep.clone()
        timestep[timestep == 0] = 1
        return 0.5 * torch.sqrt(diffusion_buffer.retention_total_var(input, timestep, *args, **kwargs)) / (2. * 1 - diffusion_buffer.retention_total_var(input, timestep, *args, **kwargs))
    
    def reconstruct_step(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Tensor,
        **kwargs
    ) -> Tensor:
        return degradation

class VParameterization(Parameterization):
    def optimization_target(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Tensor,
        **kwargs
    ) -> Tensor:
        return (
            diffusion_buffer.retention_total_std(input, timestep, *args, **kwargs) * degradation -
            diffusion_buffer.degradation_total_std(input, timestep, *args, **kwargs) * input
        )
    
    def predict_current_and_start(
        self,
        backbone_prediction: Tensor,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        retention_total_var: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        retention_total_std = diffusion_buffer.retention_total_std(input, timestep, *args, **kwargs)
        degradation_total_std = diffusion_buffer.degradation_total_std(input, timestep, *args, **kwargs)
        
        e_t = retention_total_std * backbone_prediction + degradation_total_std * input
        pred_x0 = retention_total_std * input - degradation_total_std * backbone_prediction
        return e_t, pred_x0
    
    def variational_weights(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        timestep = timestep.clone()
        timestep[timestep == 0] = 1
        return torch.ones_like(
            torch.square(diffusion_buffer.degradation_var(input, timestep, *args, **kwargs)) / 
            (
                2 * diffusion_buffer.posterior_var(input, timestep, *args, **kwargs) *
                diffusion_buffer.retention_var(input, timestep, *args, **kwargs) *
                (1 - diffusion_buffer.retention_total_var(input, timestep, *args, **kwargs))
            )
        )
    
    def reconstruct_step(
        self,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        raise TypeError(f"{type(self).__name__} is only suitbale for DDIM samplers, and the current sampling process might not be DDIM.")