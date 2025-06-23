from typing import overload, Callable, Sequence, Optional, Union, Tuple, Any, Dict
import torch
from torch import Tensor
from torch import nn
from ..schedules.diffusion_schedule import DiffusionSchedule
from ..unification import Diffusion
from ..parameterizations import Parameterization
from .. import losses
from ..schedules.context import Context
from ..schedules.context import distributions
from ..utils import data_type


class Diffuser(Diffusion, schedule=DiffusionSchedule):
    """The diffuser is an implementation that contains the forward process part of
    the diffusion model.
    """
    diffusion_schedule: DiffusionSchedule
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        *,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
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
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            *args: The arguments that are used to construct the instance of
                :class:`DiffusionSchedule`.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            **kwargs: The keyword arguments that are used to construct the instance of
                :class:`DiffusionSchedule`.
        
        """
        ...
    
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        self.backbone = backbone
        self.timesteps_distribution = timesteps_distribution
        self.parameterization = parameterization
        self.loss_function = loss_function
        
        super().__init__(*args, **kwargs)
    
    @classmethod
    def distribute_timesteps(
        cls,
        diffusion_schedule: DiffusionSchedule,
        timesteps_distribution: distributions.Context,
        shape: Sequence[int],
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None
    ) -> Tensor:
        """Distributes the random timesteps from the given distribution.
        
        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            shape (Sequence[int]): The shape that is needed to be sampled.
            device (Optional[torch.device]): The device of the timesteps.
            dtype (Optional[torch.dtype]): The data type of the timesteps. If ``None``, it will be
                the :attr:`diffusion_schedule`.:attr:`default_dtype`. In addition, the final data
                type of the timesteps will be determined by both this data type and the
                diffusion schedule data type.
        
        Returns:
            Tensor: The randomly sampled timesteps from the given distribution.
        
        """
        dtype = data_type.determine_timesteps_dtype(diffusion_schedule.default_dtype, dtype)
        if isinstance(timesteps_distribution, distributions.ShapeContext):
            return timesteps_distribution(shape, dtype=dtype, device=device)
        elif isinstance(timesteps_distribution, distributions.ScheduledShapeContext):
            return timesteps_distribution(diffusion_schedule, shape, dtype=dtype, device=device)
        elif isinstance(timesteps_distribution, distributions.ScheduledContext):
            return timesteps_distribution(diffusion_schedule, dtype=dtype, device=device)
        elif isinstance(timesteps_distribution, distributions.Context):
            return timesteps_distribution(dtype=dtype, device=device)
        elif isinstance(timesteps_distribution, Context):
            return timesteps_distribution()
        else:
            return timesteps_distribution
    
    @classmethod
    def diffuse(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        timestep: Tensor,
        *diffusion_args,
        noise: Optional[Tensor] = None,
        **diffusion_kwargs
    ) -> Tensor:
        """Diffuses the clean input to a degraded one.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_start (Tensor): The clean input which means it is at the first step.
            timestep (Tensor): The required diffusion timestep, making the clean
                input degrade.
            *diffusion_args: The arguments that drive the diffusion process.
            noise (Optional[Tensor]): A noise tensor. If ``None``, that means this diffusion
                model will generate this noise, or not require a noise input.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A diffused result at the given timestep.
        
        """
        # q_sample
        pass
    
    @classmethod
    def predict(
        cls,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, Union[Tensor, float, int]]]:
        """Predicts the degraded input at the given timestep into the clean one, and the weighting
        for losses if it is necessary.
        
        Args:
            backbone (nn.Module): The backbone model which predicts the noise.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input, but not requiring at a concrete timestep.
            timestep (Tensor): The diffusion timestep according to the input.
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.
        
        Returns:
            Tensor: A predicted clean state, and the weighting fr the losses if needed.
        
        """
        backbone_args, backbone_kwargs = diffusion_schedule.get_backbone_arguments(*args, **kwargs)
        return backbone(input, timestep, *backbone_args, **backbone_kwargs)
    
    @classmethod
    def degrade(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input: Union[Tensor, Sequence[int]],
        timestep: Tensor,
        *diffusion_args,
        **diffusion_kwargs
    ) -> Tensor:
        """Makes a noise.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input, but not requiring at a concrete timestep. If the input is
                not a Tensor, it should be a shape.
            timestep (Tensor): The diffusion timestep according to the input.
            *diffusion_args: The arguments that drive the diffusion process.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A noise.
        
        """
        if isinstance(input, Tensor):
            return torch.randn_like(input)
        else:
            return torch.randn(input, device=timestep.device)
    
    def parameters(self, recurse = True):
        return self.backbone.parameters(recurse)
    
    @classmethod
    def construct_optimization(
        cls,
        diffusion_schedule: DiffusionSchedule,
        prediction: Tensor,
        input_start: Tensor,
        input: Tensor,
        timestep: Tensor,
        noise: Tensor,
        *diffusion_args,
        parameterization: Parameterization,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        **diffusion_kwargs
    ) -> Tuple[Tensor, Tensor]:
        """Constructs the optimization arguments for the loss function, usually the prediction and the target.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            prediction (Tensor): The denoised prediction.
            input_start (Tensor): The clean input at the first timestep.
            input (Tensor): The input at the given timestep.
            timestep (Tensor): The diffusion timestep that is used to extract the optimization target.
            noise (Optional[Tensor]): A noise tensor. If ``None``, that means this diffusion model
                will generate this noise, or not require a noise input.
            *diffusion_args: The arguments that drive the diffusion process.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`step_loss` to keep the context.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The optimization arguments for the loss function.
        
        """
        target = parameterization.truth_target(
            diffusion_schedule,
            input_start,
            input,
            timestep,
            noise,
            *diffusion_args,
            **diffusion_kwargs
        )
        return prediction, target

    @classmethod
    def make_step(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        *diffusion_args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        **diffusion_kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Returns the timestep, noise and the step itself.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_start (Tensor): The clean input which means it is at the first step.
            *diffusion_args: The arguments that drive the diffusion process.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`step_loss` to keep the context.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensors: The timestep, noise and the step itself.
        
        """
        timestep = cls.distribute_timesteps(diffusion_schedule, timesteps_distribution, input_start.shape[:1], input_start.device, input_start.dtype)
        noise = cls.degrade(diffusion_schedule, input_start, timestep, *diffusion_args, **diffusion_kwargs)
        x_t = cls.diffuse(diffusion_schedule, input_start, timestep, *diffusion_args, noise=noise, **diffusion_kwargs)
        return timestep, noise, x_t
    
    @classmethod
    def weighting_prediction(
        cls,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input_middle: Tensor,
        timestep: Tensor,
        *args,
        parameterization: Parameterization,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        **kwargs
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Returns the prediction and the weighting for the loss.

        Args:
            backbone (nn.Module): The backbone model which predicts the noise.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The input at the given timestep.
            timestep (Tensor): The diffusion timestep that is used to extract the optimization target.
            *args: The arguments that drive both the diffusion process and the backbone model.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`step_loss` to keep the context.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: The prediction and the weighting for the loss.
        
        """
        prediction = cls.predict(backbone, diffusion_schedule, input_middle, timestep, *args, **kwargs)
        if isinstance(prediction, tuple):
            prediction, weighting = prediction
        else:
            weighting = None
        return prediction, weighting

    @classmethod
    def make_loss(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_middle: Tensor,
        timestep: Tensor,
        optimization_arguments: Tuple[Tensor, Tensor],
        weighting: Optional[Tensor],
        *diffusion_args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        **diffusion_kwargs
    ) -> Tensor:
        """Returns the loss.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_middle (Tensor): The input at the given timestep.
            timestep (Tensor): The diffusion timestep that is used to extract the optimization target.
            optimization_arguments (Tuple[Tensor, Tensor]): The optimization arguments that are used to feed to
                the loss function.
            weighting (Optional[Tensor]): The weighting for the loss.
            *diffusion_args: The arguments that drive the diffusion process.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`step_loss` to keep the context.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: The loss.
        
        """
        loss = loss_function(*optimization_arguments)
        if weighting is not None:
            loss = loss * weighting
        
        return parameterization.adjust_loss(
            diffusion_schedule,
            loss,
            input_middle,
            timestep,
            timesteps_distribution.differential_timestep,
            *diffusion_args,
            **diffusion_kwargs
        )

    @classmethod
    def calculate_loss(
        cls,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        *args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        **kwargs
    ) -> Tensor:
        """Calculates the loss regarding to any timesteps.

        Args:
            backbone (nn.Module): The backbone model which predicts the noise.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_start (Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`step_loss` to keep the context.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: The loss results.
        
        """
        # p_losses "Algorithm 1"
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        timestep, noise, x_t = cls.make_step(
            diffusion_schedule,
            input_start,
            *diffusion_args,
            timesteps_distribution=timesteps_distribution,
            parameterization=parameterization,
            variables=variables,
            **diffusion_kwargs
        )
        
        prediction, weighting = cls.weighting_prediction(
            backbone,
            diffusion_schedule,
            x_t,
            timestep,
            *args,
            parameterization=parameterization,
            variables=variables,
            **kwargs
        )
        
        optimization_arguments = cls.construct_optimization(
            diffusion_schedule,
            prediction,
            input_start,
            x_t,
            timestep,
            noise,
            *diffusion_args,
            parameterization=parameterization,
            variables=variables,
            **diffusion_kwargs
        )
        
        return cls.make_loss(
            diffusion_schedule,
            x_t,
            timestep,
            optimization_arguments,
            weighting,
            *diffusion_args,
            timesteps_distribution=timesteps_distribution,
            parameterization=parameterization,
            loss_function=loss_function,
            variables=variables,
            **diffusion_kwargs
        )
    
    @classmethod
    def initialize_variables(cls) -> Union[Sequence[Any], Dict[str, Any], None]:
        """Initializes variables for the context of training.
        
        Returns:
            Any: The variables container.
        
        """
        return None

    @classmethod
    def step_loss(
        cls,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input_start: Tensor,
        *args,
        timesteps_distribution: distributions.Context,
        parameterization: Parameterization,
        loss_function: Callable[[Tensor, Tensor], Tensor],
        **kwargs
    ) -> Tensor:
        """Returns the calculated loss.

        Args:
            backbone (nn.Module): The backbone model which predicts the noise.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_start (Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            timesteps_distribution (distributions.Context): The context that generates timesteps
                from this given distribution.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            loss_function (Callable[[Tensor, Tensor], Tensor]): The loss function that
                is used to optimize.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: The loss results.
        
        """
        return cls.calculate_loss(
            backbone,
            diffusion_schedule,
            input_start,
            *args,
            timesteps_distribution=timesteps_distribution,
            parameterization=parameterization,
            loss_function=loss_function,
            variables=cls.initialize_variables(),
            **kwargs
        )

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
    
    def forward(self, input_start: Tensor, *args, **kwargs) -> Tensor:
        return self.step_loss(
            self.backbone,
            self.diffusion_schedule,
            input_start,
            *args,
            timesteps_distribution=self.timesteps_distribution,
            parameterization=self.parameterization,
            loss_function=self.loss_function,
            **kwargs
        )