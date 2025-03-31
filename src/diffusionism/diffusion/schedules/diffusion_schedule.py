from typing import Optional, Tuple, Union
import numpy as np
import torch
from torch import Tensor
import torchflint as te
from ..hyperparameters import Hyperparameters


class DiffusionSchedule(te.nn.BufferObject):
    """The buffer which contains all necessary mathematical variables regarding to the diffusion model.
    """
    def __init__(self, default_dtype: Optional[torch.dtype] = None, hyperparameters: Optional[Hyperparameters] = None):
        self.persistent = False
        self.default_dtype = default_dtype
        self.hyperparameters = hyperparameters
    
    def get_diffusion_arguments(self, *args, **kwargs) -> Tuple[tuple, dict]:
        """Extracts the input arguments into the arguments that drive the diffusion process.

        Args:
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tuple: The arguments and keyword arguments that drive the diffusion process.
        
        """
        return tuple(), {}
    
    def get_backbone_arguments(self, *args, **kwargs) -> Tuple[tuple, dict]:
        """Extracts the input arguments into the arguments that drive the backbone model.

        Args:
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tuple: The arguments and keyword arguments that drive the backbone model.
        
        """
        return args, kwargs


class StochasticDifferentialEquationsSchedule(DiffusionSchedule):
    def drift_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        pass
    
    def diffusion_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        pass
    
    def drift(self, input: Tensor, timestep: Tensor, differential_timestep: Union[Tensor, float, int] = 1.e-3, *args, **kwargs) -> Tensor:
        return self.drift_term(input, timestep, *args, **kwargs) * differential_timestep
    
    def diffusion(self, input: Tensor, timestep: Tensor, differential_timestep: Union[Tensor, float, int] = 1.e-3, *args, **kwargs) -> Tensor:
        return self.diffusion_term(input, timestep, *args, **kwargs) * np.sqrt(differential_timestep)
    
    def reverse_drift_term(self, input: Tensor, timestep: Tensor, score: Tensor, *args, **kwargs) -> Tensor:
        return (
            self.drift_term(input, timestep, *args, **kwargs)
            - self.diffusion_term(input, timestep, *args, **kwargs).square_() * score
        )
    
    def reverse_diffusion_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.diffusion_term(input, timestep, *args, **kwargs)
    
    def reverse_drift(self, differential_timestep: float, input: Tensor, timestep: Tensor, score: Tensor, *args, **kwargs) -> Tensor:
        return self.reverse_drift_term(input, timestep, score, *args, **kwargs) * differential_timestep
    
    def reverse_diffusion(self, differential_timestep: float, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.reverse_diffusion_term(input, timestep, *args, **kwargs) * np.sqrt(differential_timestep)
    
    def ordinary_drift_term(self, input: Tensor, timestep: Tensor, score: Tensor, *args, **kwargs) -> Tensor:
        # for ODE
        return (
            self.drift_term(input, timestep, *args, **kwargs)
            - self.diffusion_term(input, timestep, *args, **kwargs).square_() * score / 2.
        )
    
    def ordinary_drift(self, differential_timestep: float, input: Tensor, timestep: Tensor, score: Tensor, *args, **kwargs) -> Tensor:
        return self.ordinary_drift_term(input, timestep, score, *args, **kwargs) * differential_timestep
    
    # def on_scaled(self, old_resolution_scale_square: Optional[float]):
    #     pass
    
    # def shift_schedule(self, variance: Tensor) -> Tuple[Tensor, Tensor]:
    #     """Shifts the diffusion variance into a target resolution level, since the SNR is resolution-sensitive.
        
    #     Args:
    #         variance (Tensor): The noise variance.
        
    #     Returns:
    #         Tuple: The target mean square and variance.
        
    #     """
    #     pass
    
    # def __setattr__(self, name, value):
    #     if name == 'resolution_scale_square':
    #         try:
    #             old_resolution_scale_square = self.resolution_scale_square
    #             super().__setattr__(name, value)
    #             if value != old_resolution_scale_square:
    #                 self.on_scaled(old_resolution_scale_square)
    #         except AttributeError:
    #             super().__setattr__(name, value)
    #             self.on_scaled(None)
    #     else:
    #         return super().__setattr__(name, value)