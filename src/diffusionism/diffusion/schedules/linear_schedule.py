from typing import Optional, Union
import torch
from torch import Tensor
from .diffusion_schedule import StochasticDifferentialEquationsSchedule


class LinearDiffusionSchedule(StochasticDifferentialEquationsSchedule):
    """The linear SDE schedule, which means the drift term has a linear relationship with input, and due to the diffusion
    term followed by DMs is input independent, the linear relationship here only applies to the drift term. However, in order
    to make the format consistent with the SDE parent class and to obtain other attributes of the input, including its shape,
    each function here requires an input.
    
    """
    def drift_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        input_scale = self.input_scale(input, timestep, *args, **kwargs)
        input_scale_derivative = self.input_scale_derivative(input, timestep, *args, input_scale=input_scale, **kwargs)
        bias = self.bias(input, timestep, *args, **kwargs)
        
        input_scale_derivative_div_input_scale = input_scale_derivative / input_scale
        
        return input_scale_derivative_div_input_scale * input + self.bias_derivative(input, timestep, *args, bias=bias, **kwargs) - input_scale_derivative_div_input_scale * bias
    
    def diffusion_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.diffusion_noise_scale(input, timestep, *args, **kwargs)
    
    def input_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        pass
    
    def input_scale_derivative(self, input: Tensor, timestep: Tensor, *args, input_scale: Optional[Tensor] = None, **kwargs) -> Tensor:
        pass
    
    def noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        pass
    
    def noise_scale_derivative(self, input: Tensor, timestep: Tensor, *args, noise_scale: Optional[Tensor] = None, **kwargs) -> Tensor:
        pass
    
    def bias(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return 0
    
    def bias_derivative(self, input: Tensor, timestep: Tensor, *args, bias: Optional[Tensor] = None, **kwargs) -> Tensor:
        return 0
    
    def step_input_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.drift_input_scale(input, timestep, *args, **kwargs) * differential_timestep + 1
    
    def drift_input_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        input_scale = self.input_scale(input, timestep, *args, **kwargs)
        return self.input_scale_derivative(input, timestep, *args, input_scale=input_scale, **kwargs) / input_scale
    
    def step_noise_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.diffusion_noise_scale(input, timestep, *args, **kwargs) * torch.sqrt_(torch.as_tensor(differential_timestep, device=timestep.device).abs_())
    
    def diffusion_noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        noise_scale = self.noise_scale(input, timestep, *args, **kwargs)
        input_scale = self.input_scale(input, timestep, *args, **kwargs)
        return torch.sqrt_(2 * noise_scale * (
            self.noise_scale_derivative(input, timestep, *args, noise_scale=noise_scale, **kwargs)
            - self.input_scale_derivative(input, timestep, *args, input_scale=input_scale, **kwargs) / input_scale * noise_scale
        ))
    
    def step_bias(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.drift_bias(input, timestep, *args, **kwargs) * differential_timestep
    
    def drift_bias(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        input_scale = self.input_scale(input, timestep, *args, **kwargs)
        bias = self.bias(input, timestep, *args, **kwargs)
        return self.bias_derivative(input, timestep, *args, bias=bias, **kwargs) - self.input_scale_derivative(input, timestep, *args, input_scale=input_scale, **kwargs) / input_scale * bias
    
    def input_scale_square(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.input_scale(input, timestep, *args, **kwargs).square_()

    def step_input_scale_square(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.step_input_scale(input, timestep, *args, differential_timestep=differential_timestep, **kwargs).square_()
    
    def noise_variance(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.noise_scale(input, timestep, *args, **kwargs).square_()
    
    def step_noise_variance(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.step_noise_scale(input, timestep, *args, differential_timestep=differential_timestep, **kwargs).square_()
    
    def scaling(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.input_scale(input, timestep, *args, **kwargs)
    
    def scaling_derivative(self, input: Tensor, timestep: Tensor, *args, scaling: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.input_scale_derivative(input, timestep, *args, input_scale=scaling, **kwargs)
    
    def noise_level(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.noise_scale(input, timestep, *args, **kwargs) / self.input_scale(input, timestep, *args, **kwargs)
    
    def noise_level_derivative(self, input: Tensor, timestep: Tensor, *args, noise_level: Optional[Tensor] = None, **kwargs) -> Tensor:
        input_scale = self.input_scale(input, timestep, *args, **kwargs)
        noise_scale = self.noise_scale(input, timestep, *args, **kwargs)
        return (
            self.noise_scale_derivative(input, timestep, *args, noise_scale=noise_scale, **kwargs) / input_scale
            - self.input_scale_derivative(input, timestep, *args, input_scale=input_scale, **kwargs) / input_scale.square() * noise_scale
        )
    
    def shifting(self, input: Tensor, timestep: Tensor, *args, **kwargs):
        return self.bias(input, timestep, *args, **kwargs) / self.input_scale(input, timestep, *args, **kwargs)

    def shifting_derivative(self, input: Tensor, timestep: Tensor, *args, shifting: Optional[Tensor] = None, **kwargs) -> Tensor:
        input_scale = self.input_scale(input, timestep, *args, **kwargs)
        bias = self.bias(input, timestep, *args, **kwargs)
        return (
            self.bias_derivative(input, timestep, *args, bias=bias, **kwargs) / input_scale
            - self.input_scale_derivative(input, timestep, *args, input_scale=input_scale, **kwargs) / input_scale.square() * bias
        )


class AncestralDiffusionSchedule(LinearDiffusionSchedule):
    def drift_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.drift_input_scale(input, timestep, *args, **kwargs) * input + self.drift_bias(input, timestep, *args, **kwargs)
    
    def diffusion_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.diffusion_noise_scale(input, timestep, *args, **kwargs)
    
    def input_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        pass
    
    def step_input_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        pass
    
    def noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        pass
    
    def step_noise_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        pass
    
    def bias(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return 0
    
    def step_bias(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return 0
    
    def input_scale_derivative(self, input: Tensor, timestep: Tensor, *args, input_scale: Tensor, **kwargs) -> Tensor:
        return self.drift_input_scale(input, timestep, *args, **kwargs) * input_scale
    
    def drift_input_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.step_input_scale(input, timestep, *args, differential_timestep=1, **kwargs) - 1
    
    def noise_scale_derivative(self, input: Tensor, timestep: Tensor, *args, noise_scale: Tensor, **kwargs) -> Tensor:
        return self.diffusion_noise_scale(input, timestep, *args, **kwargs).square_() / noise_scale / 2 + self.drift_input_scale(input, timestep, *args, **kwargs) * noise_scale
    
    def diffusion_noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.step_noise_scale(input, timestep, *args, differential_timestep=1, **kwargs)
    
    def bias_derivative(self, input: Tensor, timestep: Tensor, *args, bias: Tensor, **kwargs) -> Tensor:
        return self.drift_bias(input, timestep, *args, **kwargs) + self.drift_input_scale(input, timestep, *args, **kwargs) * bias
    
    def drift_bias(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.step_bias(input, timestep, *args, differential_timestep=1, **kwargs)


class EssentialDiffusionSchedule(LinearDiffusionSchedule):
    def drift_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        scaling = self.scaling(input, timestep, *args, **kwargs)
        shifting = self.shifting(input, timestep, *args, **kwargs)
        scaling_derivative = self.scaling_derivative(input, timestep, *args, scaling=scaling, **kwargs)
        return scaling_derivative / scaling * input + scaling * self.shifting_derivative(input, timestep, *args, shifting=shifting, **kwargs)
    
    def diffusion_term(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.diffusion_noise_scale(input, timestep, *args, **kwargs)
    
    def scaling(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the scaling of the diffusion.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The scaling of the diffusion.
        
        """
        pass
    
    def scaling_derivative(self, input: Tensor, timestep: Tensor, *args, scaling: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Indicates the derivative of the scaling with respect to the :params:`timestep` of the diffusion.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            scaling (Tensor): The scaling of the diffusion.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The derivative of the scaling with respect to the :params:`timestep` of the diffusion.
        
        """
        pass
    
    def noise_level(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the noise level of the diffusion.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The noise level of the diffusion.
        
        """
        pass
    
    def noise_level_derivative(self, input: Tensor, timestep: Tensor, *args, noise_level: Optional[Tensor] = None, **kwargs) -> Tensor:
        """Indicates the derivative of the noise level with respect to the :params:`timestep` of the diffusion.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            noise_level (Tensor): The noise level of the diffusion.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The derivative of the scaling with respect to the :params:`timestep` of the diffusion.
        
        """
        pass
    
    def shifting(self, input: Tensor, timestep: Tensor, *args, **kwargs):
        return 0

    def shifting_derivative(self, input: Tensor, timestep: Tensor, *args, shifting: Optional[Tensor] = None, **kwargs) -> Tensor:
        return 0
    
    def input_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.scaling(input, timestep, *args, **kwargs)
    
    def input_scale_derivative(self, input: Tensor, timestep: Tensor, *args, input_scale: Optional[Tensor] = None, **kwargs) -> Tensor:
        return self.scaling_derivative(input, timestep, *args, scaling=input_scale, **kwargs)
    
    def step_input_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.drift_input_scale(input, timestep, *args, **kwargs) * differential_timestep + 1
    
    def drift_input_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        scaling = self.scaling(input, timestep, *args, **kwargs)
        return self.scaling_derivative(input, timestep, *args, scaling=scaling, **kwargs) / scaling
    
    def noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.scaling(input, timestep, *args, **kwargs) * self.noise_level(input, timestep, *args, **kwargs)
    
    def noise_scale_derivative(self, input: Tensor, timestep: Tensor, *args, noise_scale: Optional[Tensor] = None, **kwargs) -> Tensor:
        scaling = self.scaling(input, timestep, *args, **kwargs)
        noise_level = self.noise_level(input, timestep, *args, **kwargs)
        return self.scaling_derivative(input, timestep, *args, scaling=scaling, **kwargs) * noise_level + scaling * self.noise_level_derivative(input, timestep, *args, noise_level=noise_level, **kwargs)
    
    def step_noise_scale(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.diffusion_noise_scale(input, timestep, *args, **kwargs) * torch.sqrt_(torch.as_tensor(differential_timestep, device=timestep.device).abs_())
    
    def diffusion_noise_scale(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        noise_level = self.noise_level(input, timestep, *args, **kwargs)
        return self.scaling(input, timestep, *args, **kwargs) * torch.sqrt_(2 * noise_level * self.noise_level_derivative(input, timestep, *args, noise_level=noise_level, **kwargs))
    
    def bias(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        return self.scaling(input, timestep, *args, **kwargs) * self.shifting(input, timestep, *args, **kwargs)
    
    def bias_derivative(self, input: Tensor, timestep: Tensor, *args, bias: Optional[Tensor] = None, **kwargs) -> Tensor:
        scaling = self.scaling(input, timestep, *args, **kwargs)
        shifting = self.shifting(input, timestep, *args, **kwargs)
        return self.scaling_derivative(input, timestep, *args, scaling=scaling, **kwargs) * shifting + scaling * self.shifting_derivative(input, timestep, *args, shifting=shifting, **kwargs)
    
    def step_bias(self, input: Tensor, timestep: Tensor, *args, differential_timestep: Union[Tensor, float, int] = 1.e-3, **kwargs) -> Tensor:
        return self.drift_bias(input, timestep, *args, **kwargs) * differential_timestep
    
    def drift_bias(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        scaling = self.scaling(input, timestep, *args, **kwargs)
        shifting = self.shifting(input, timestep, *args, **kwargs)
        return scaling * self.shifting_derivative(input, timestep, *args, shifting=shifting, **kwargs)