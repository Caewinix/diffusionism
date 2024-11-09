from typing import Tuple
import numpy as np
from torch import Tensor
import torchflint as te


class DiffusionBuffer(te.nn.BufferObject):
    """The buffer which contains all necessary mathematical variables regarding to the diffusion model.
    """
    def __init__(self, num_timesteps: int = 1000):
        """
        Args:
            num_timesteps (int): The number of timesteps of the diffusion model.
        
        """
        super().__init__()
        self.num_timesteps = num_timesteps
        self.timesteps = np.arange(self.num_timesteps)
        self.prev_timesteps = np.concatenate([[0], self.timesteps[:-1]])
        
        self.eta = 0.
        self.persistent = False

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
    
    def degradation_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the variance of degradation degree.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variance of degradation degree.
        
        """
        # beta_t
        pass

    def retention_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the variance of retention degree.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variance of retention degree.
        
        """
        # alpha_t = 1 - beta_t
        pass
    
    def degradation_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the standard deviation of degradation degree.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The standard deviation of degradation degree.
        
        """
        # \sqrt{beta_t}
        pass

    def retention_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the standard deviation of retention degree.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The standard deviation of retention degree.
        
        """
        # \sqrt{alpha_t}
        pass
    
    def degradation_total_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the variance of degradation degree, from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variance of degradation degree from the start to the current timestep.
        
        """
        # \hat{beta_t}
        pass

    def retention_total_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the variance of retention degree, from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variance of retention degree from the start to the current timestep.
        
        """
        # \hat{alpha_t}
        pass
    
    def degradation_total_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the standard deviation of degradation degree, from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The standard deviation of degradation degree from the start to the current timestep.
        
        """
        # \sqrt{\hat{beta_t}}
        pass

    def retention_total_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the standard deviation of retention degree, from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The standard deviation of retention degree from the start to the current timestep.
        
        """
        # \sqrt{\hat{alpha_t}}
        pass

    def reciprocal_retention_total_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the reciprocal of the standard deviation of retention degree,
        from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The reciprocal of the standard deviation of retention degree from the start
                to the current timestep.
        
        """
        # \sqrt{1 / \hat{alpha_t}}
        pass
    
    def complementary_reciprocal_retention_total_std(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the complementary reciprocal of the standard deviation of retention degree,
        from the start to the current timestep.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The complementary reciprocal of the standard deviation of retention degree from the start
                to the current timestep.
        
        """
        # \sqrt{1 / \hat{alpha_t} - 1}
        pass

    def posterior_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the variance of the posterior.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The variance of the posterior.
        
        """
        pass
    
    def posterior_log_var(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the logarithm variance of the posterior.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The logarithm variance of the posterior.
        
        """
        pass
    
    def posterior_mean_start_coefficient(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the 'start' part coefficient of the posterior mean.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The 'start' part coefficient of the posterior mean.
        
        """
        # (\sqrt{\bar{\alpha}_{t-1}} \beta_t) / (1 - \bar{\alpha}_t)
        pass
    
    def posterior_mean_current_coefficient(self, input: Tensor, timestep: Tensor, *args, **kwargs) -> Tensor:
        """Indicates the 'current timestep' part coefficient of the posterior mean.

        Args:
            input (Tensor): The input which typically contributes to the shape extraction,
                sometimes can be introduced to the result calculation.
            timestep (Tensor): The timestep for the diffusion model that is used to extract the
                result at the given timestep.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The 'current timestep' part coefficient of the posterior mean.
        
        """
        # (\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})) / (1 - \bar{\alpha}_t)
        pass