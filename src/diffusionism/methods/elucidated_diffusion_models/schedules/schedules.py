from typing import Tuple, Optional
import torch
from torch import Tensor
from ....diffusion.schedules.linear_schedule import EssentialDiffusionSchedule
from ....diffusion.utils.functions import append_dims
from ..hyperparameters import ChurnStochasticHyperparameters


class ElucidatedDiffusionModelsSchedule(EssentialDiffusionSchedule):
    def __init__(
        self,
        data_std: float = 0.5,
        default_dtype: Optional[torch.dtype] = None,
        hyperparameters: ChurnStochasticHyperparameters = ChurnStochasticHyperparameters()
    ):
        super().__init__(default_dtype, hyperparameters)
        self.data_std = data_std
    
    def noise_level(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return append_dims(timestep, input.ndim)
    
    def noise_level_derivative(self, input: Tensor, timestep: Tensor, *args, noise_level: Optional[Tensor] = None, **kwargs) -> Tensor:
        return append_dims(torch.ones_like(timestep), input.ndim)
    
    def inverse_noise_level(self, noise_level: Tensor) -> Tensor:
        """Indicates the inverse noise schedule of the diffusion.

        Args:
            noise_level (Tensor): The noise schedule of the diffusion.

        Returns:
            Tensor: The inverse noise schedule of the diffusion.
        
        """
        sigma = noise_level
        return sigma
    
    def scaling(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return append_dims(torch.ones_like(timestep), input.ndim)
    
    def scaling_derivative(self, input: Tensor, timestep: Tensor, *args, scaling: Optional[Tensor] = None, **kwargs) -> Tensor:
        return append_dims(torch.zeros_like(timestep), input.ndim)
    
    def preconditioning_with_weighting(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Indicates the preconditioning and the loss weighting of the diffusion.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tuple: The the preconditioned "in", "out", "skip" and "noise" coefficients,
                as well as the loss weighting of the diffusion.
        
        """
        sigma = self.noise_level(input, timestep, *args, **kwargs)
        sigma_data = torch.as_tensor(self.data_std, dtype=input.dtype, device=input.device)
        sigma_data_square = sigma_data.square()
        sigma_square_plus_sigma_data_square = sigma.square() + sigma_data_square
        sqrt_sigma_square_plus_sigma_data_square = sigma_square_plus_sigma_data_square.sqrt()
        sigma_multiply_sigma_data = sigma * sigma_data
        c_in = 1. / sqrt_sigma_square_plus_sigma_data_square
        c_out = sigma_multiply_sigma_data / sqrt_sigma_square_plus_sigma_data_square
        c_skip = sigma_data_square / sigma_square_plus_sigma_data_square
        c_noise = 0.25 * torch.log(sigma)
        loss_weighting = sigma_square_plus_sigma_data_square / sigma_multiply_sigma_data.square()
        return c_in, c_out, c_skip, c_noise, loss_weighting


class VariancePreservingSchedule(ElucidatedDiffusionModelsSchedule):
    def __init__(
        self,
        min_variance: float = 0.1,
        delta_variance: float = 19.9,
        time_scale: float = 1000,
        default_dtype: Optional[torch.dtype] = None,
        hyperparameters: ChurnStochasticHyperparameters = ChurnStochasticHyperparameters()
    ):
        super(ElucidatedDiffusionModelsSchedule, self).__init__(default_dtype, hyperparameters)
        self.min_variance = min_variance
        self.delta_variance = delta_variance
        self.time_scale = time_scale
    
    def noise_level(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        beta_min = self.min_variance
        beta_d = self.delta_variance
        return append_dims(torch.sqrt(torch.expm1(0.5 * beta_d * timestep.square() + beta_min * timestep)), input.ndim)
    
    def noise_level_derivative(self, input: Tensor, timestep: Tensor, *args, noise_level: Optional[Tensor] = None, **kwargs) -> Tensor:
        sigma = noise_level
        beta_min = self.min_variance
        beta_d = self.delta_variance
        return 0.5 * (beta_min + beta_d * timestep) * (sigma + 1 / sigma)
    
    def inverse_noise_level(self, noise_level: Tensor) -> Tensor:
        sigma = noise_level
        beta_min = self.min_variance
        beta_d = self.delta_variance
        return ((beta_min ** 2 + 2 * beta_d * (1 + sigma ** 2).log_()).sqrt_() - beta_min) / beta_d
    
    def scaling(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        beta_min = self.min_variance
        beta_d = self.delta_variance
        return append_dims(1. / torch.sqrt(torch.exp(0.5 * beta_d * timestep.square() + beta_min * timestep)), input.ndim)
    
    def scaling_derivative(self, input: Tensor, timestep: Tensor, *args, scaling: Optional[Tensor] = None, **kwargs) -> Tensor:
        beta_min = self.min_variance
        beta_d = self.delta_variance
        timestep = append_dims(timestep, input.ndim)
        return -((beta_d * timestep + beta_min) * torch.exp_(-(((beta_d * timestep.square()) / 2 + beta_min * timestep) / 2))) / 2
    
    def preconditioning_with_weighting(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sigma = self.noise_level(input, timestep, *args, **kwargs)
        sigma_square = sigma.square()
        c_in = 1. / torch.sqrt_(sigma_square + 1)
        c_out = -sigma
        c_skip = append_dims(torch.ones_like(timestep), input.ndim)
        c_noise = (self.time_scale - 1) * self.inverse_noise_level(sigma)
        loss_weighting = 1. / sigma_square
        return c_in, c_out, c_skip, c_noise, loss_weighting


class VarianceExplodingSchedule(ElucidatedDiffusionModelsSchedule):
    def __init__(self, default_dtype: Optional[torch.dtype] = None, hyperparameters: ChurnStochasticHyperparameters = ChurnStochasticHyperparameters()):
        super(ElucidatedDiffusionModelsSchedule, self).__init__(default_dtype, hyperparameters)
    
    def noise_level(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return append_dims(torch.sqrt(timestep), input.ndim)
    
    def noise_level_derivative(self, input: Tensor, timestep: Tensor, *args, noise_level: Optional[Tensor] = None, **kwargs) -> Tensor:
        return append_dims(0.5 / timestep.sqrt(), input.ndim)
    
    def inverse_noise_level(self, noise_level: Tensor) -> Tensor:
        sigma = noise_level
        return sigma.square()
    
    def scaling(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return append_dims(torch.ones_like(timestep), input.ndim)
    
    def scaling_derivative(self, input: Tensor, timestep: Tensor, *args, scaling: Optional[Tensor] = None, **kwargs) -> Tensor:
        return append_dims(torch.zeros_like(timestep), input.ndim)
    
    def preconditioning_with_weighting(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        sigma = self.noise_level(input, timestep, *args, **kwargs)
        c_in = append_dims(torch.ones_like(timestep), input.ndim)
        c_out = sigma
        c_skip = c_in
        c_noise = torch.log_(0.5 * sigma)
        loss_weighting = 1. / sigma.square()
        return c_in, c_out, c_skip, c_noise, loss_weighting