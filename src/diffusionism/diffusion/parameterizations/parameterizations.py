from typing import Union, Type
from torch import Tensor
from ..schedules.diffusion_schedule import DiffusionSchedule


class Parameterization:
    """The parameterization situation of the diffusion model.
    """
    @classmethod
    def truth_target(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the ground truth of the target of this parameterization.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            loss (Tensor): The original loss.
            input_start (Tensor): The input at the start timestep.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            noise (Tensor): A noise, such as the noise tensor. If ``None``,
                that means this diffusion model will generate this noise, or not require a
                noise input.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The ground truth of the target of this parameterization.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def constant_weight(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        posterior_weight: float = 0.,
        **kwargs
    ) -> Union[Tensor, float, int]:
        """Returns the constant weight of the loss that can be ignored.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            differential_timestep (Union[float, int]): The differential timestep at this process.
            *args: The arguments that drive the diffusion process.
            posterior_weight (float): The weight of the posterior part.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The constant weight of the loss that can be ignored.
        
        """
        return 1
    
    @classmethod
    def input_start(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the input start.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The input start.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def noise(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the noise.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The noise.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def velocity(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the velocity.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The velocity.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def score(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the score.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The score.
        
        """
        raise NotImplementedError()

    @classmethod
    def reverse_ordinary_step(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int], # less than 0
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the step of the reverse Ordinary Differential Equation.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            differential_timestep (Union[float, int]): The differential timestep at this process.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The step of the reverse Ordinary Differential Equation.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def reverse_stochastic_step(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int], # less than 0
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the step of the reverse Stochastic Differential Equation.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            differential_timestep (Union[float, int]): The differential timestep at this process.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The step of the reverse Stochastic Differential Equation.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def skipped_reverse_ordinary_step(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep_start: Tensor,
        timestep_end: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the step of the reverse Ordinary Differential Equation after skipping many steps.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            differential_timestep (Union[float, int]): The differential timestep at this process.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The step of the reverse Ordinary Differential Equation after skipping many steps.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def skipped_reverse_stochastic_step(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep_start: Tensor,
        timestep_end: Tensor,
        target: Tensor,
        differential_timestep: Union[float, int], # less than 0
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the step of the reverse Stochastic Differential Equation after skipping many steps.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            target (Tensor): The parameterization target.
            differential_timestep (Union[float, int]): The differential timestep at this process.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The step of the reverse Stochastic Differential Equation after skipping many steps.
        
        """
        raise NotImplementedError()
    
    @classmethod
    def adjust_loss(
        cls,
        diffusion_schedule: DiffusionSchedule,
        loss: Tensor,
        input: Tensor,
        timestep: Tensor,
        differential_timestep: Union[float, int],
        *args,
        **kwargs
    ) -> Tensor:
        """Returns the adjusted loss.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            loss (Tensor): The original loss.
            input (Tensor): The input at any timestep.
            timestep (Tensor): The diffusion timestep corresponding to the input.
            differential_timestep (Union[float, int]): The differential timestep at this process.
            *args: The arguments that drive the diffusion process.
            **kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The adjusted loss.
        
        """
        return loss


class NoiseParameterization(Parameterization):
    @classmethod
    def truth_target(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return noise
    
    @classmethod
    def noise(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return target


class InputStartParameterization(Parameterization):
    @classmethod
    def truth_target(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return input_start
    
    @classmethod
    def input_start(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return target


class VelocityParameterization(Parameterization):
    @classmethod
    def velocity(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return target


class ScoreParameterization(Parameterization):
    @classmethod
    def score(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        target: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return target