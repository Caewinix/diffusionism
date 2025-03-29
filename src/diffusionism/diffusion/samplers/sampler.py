from typing import overload, Optional, Iterable, Sequence, Tuple, Union, Any, Dict
import torch
from torch import Tensor
from torch import nn
from ..schedules.diffusion_schedule import DiffusionSchedule
from ..unification import Diffusion
from ..parameterizations import Parameterization
from ..diffusers.diffuser import Diffuser
from .sampling_progress import SamplingProgress, OutputHandler
from ..utils.range_clipper import RangeClipper
from ..schedules.context import Context
from ..schedules.context import sequencings
from ..utils import data_type


def _timestep_dtype(timestep):
    return torch.as_tensor(timestep).dtype


class Sampler(Diffusion, schedule=DiffusionSchedule, diffuser=Diffuser):
    """The sampler is an implementation that contains the reverse process part of
    the diffusion model.
    """
    diffusion_schedule: DiffusionSchedule
    diffuser: Diffuser
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: DiffusionSchedule,
        *,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)]
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            timesteps (Union[sequencings.Context, Tensor]): The timesteps sequenceing method that
                returns a sequence of timesteps in the descending order, or the descending order
                timesteps sequence itself.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
        
        """
        ...
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        *args,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)],
        **kwargs
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            timesteps (Union[sequencings.Context, Tensor]): The timesteps sequenceing method that
                returns a sequence of timesteps in the descending order, or the descending order
                timesteps sequence itself.
            *args: The arguments that are used to construct the instance of
                :class:`DiffusionSchedule`.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that are used to construct the instance of
                :class:`DiffusionSchedule`.
        
        """
        ...
    
    def __init__(
        self,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        *args,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)],
        **kwargs
    ):
        self.progress = SamplingProgress()
        
        self.backbone = backbone
        self.timesteps = timesteps
        self.parameterization = parameterization
        
        for handler in output_handlers:
            handler.attach(self.progress)
        self.output_handlers = output_handlers
        
        super().__init__(*args, **kwargs)
    
    def __del__(self):
        for handler in self.output_handlers:
            handler.detach()
    
    @classmethod
    def sequence_timesteps(
        cls,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: DiffusionSchedule
    ) -> Tensor:
        """Sequences the timesteps from the given sequencing method, in the descending order.
        
        Args:
            timesteps (Union[sequencings.Context, Tensor]): The timesteps sequenceing method that
                returns a sequence of timesteps in the descending order, or the descending order
                timesteps sequence itself.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
        
        Returns:
            Tensor: The sequence of timesteps in the descending order.
        
        """
        dtype = diffusion_schedule.default_dtype
        if isinstance(timesteps, Tensor):
            return timesteps.to(dtype=dtype)
        elif isinstance(timesteps, sequencings.ScheduledContext):
            return timesteps(diffusion_schedule, dtype=dtype)
        elif isinstance(timesteps, sequencings.Context):
            return timesteps(dtype=dtype)
        elif isinstance(timesteps, Context):
            return timesteps()
        else:
            return timesteps
    
    @classmethod
    def get_ending(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_end: Union[Tensor, Sequence[int]],
        timestep: Tensor,
        *diffusion_args,
        backbone: nn.Module,
        **diffusion_kwargs
    ) -> Tensor:
        """Returns the end step state.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_end (Tensor): The input at the final timestep.
            timestep (Tensor): The final timestep according to the input.
            *diffusion_args: The arguments that drive the diffusion process.
            backbone (nn.Module): The backbone model which predicts the result.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: An end step state.
        
        """
        if isinstance(input_end, Tensor):
            if timestep.dim() == 0:
                timestep = torch.tensor([timestep], dtype=timestep.dtype, device=input_end.device)
        else:
            if timestep.dim() == 0:
                timestep = torch.tensor([timestep], dtype=timestep.dtype, device=next(backbone.parameters()).device)
            input_end = cls.diffuser.degrade(diffusion_schedule, input_end, timestep, *diffusion_args, **diffusion_kwargs)
        input_end = cls.process_ending(
            diffusion_schedule,
            input_end,
            timestep,
            *diffusion_args,
            **diffusion_kwargs
        )
        return input_end
    
    @classmethod
    def process_ending(
        cls,
        diffusion_schedule: DiffusionSchedule,
        input_end: Tensor,
        timestep: Tensor,
        *diffusion_args,
        **diffusion_kwargs
    ) -> Tensor:
        """process the end step state.

        Args:
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_end (Tensor): The input at the final timestep.
            timestep (Tensor): The final timestep according to the input.
            *diffusion_args: The arguments that drive the diffusion process.
            **diffusion_kwargs: The keyword arguments that drive the diffusion process.

        Returns:
            Tensor: A processed end step state.
        
        """
        return input_end
    
    @classmethod
    def foresee_number_timesteps(cls) -> int:
        """Returns the number of more than current timesteps in each iteration.

        Returns:
            Tensor: The number of more than current timesteps in each iteration.
        
        """
        return 1
    
    @classmethod
    def foresee_timesteps(cls, current_timesteps: Tensor, number: int = 0) -> Tuple[Tensor, ...]:
        """Returns more than current timesteps.

        Args:
            current_timesteps (Tensor): The unforseen timesteps sequence without any displacement.
            number (int): The number of more than current timesteps in each iteration

        Returns:
            Tensor: More than current timesteps.
        
        """
        return tuple(torch.cat((current_timesteps[i:], current_timesteps.new_zeros((i,)))) for i in range(1, number + 1))
    
    @classmethod
    def initialize_sampling(
        cls,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: DiffusionSchedule,
        input_end: Tensor,
        initial_state: Optional[Tensor] = None,
        strength: float = 1.,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        backbone: nn.Module,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tuple[Tensor, Tuple[Iterable[int]], int]:
        """Initializes the initial state of the sampling process, which skips particular timesteps.

        Args:
            timesteps (Union[sequencings.Context, Tensor]): The timesteps sequenceing method that
                returns a sequence of timesteps in the descending order, or the descending order
                timesteps sequence itself.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_end (Tensor): The state at the final timestep.
            initial_state (Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`sample` to keep the context.
            backbone (nn.Module): The backbone model which predicts the result.
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            sampling_progress (SamplingProgress): The process that indicates the process of current sampling.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tuple: An initialized state at the particular timestep, given by :param:`strength`, 
                a sequence of timesteps that can be iterated during sampling, and the length of
                the timesteps iteration.
        
        """
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        current_timesteps = cls.sequence_timesteps(timesteps, diffusion_schedule)
        foreseen_timesteps = cls.foresee_timesteps(current_timesteps, cls.foresee_number_timesteps())
        if initial_state is None:
            input_end = cls.get_ending(diffusion_schedule, input_end, current_timesteps[0], backbone=backbone)
            x_t = input_end
        else:
            if strength > 0 and strength < 1.:
                num_timesteps = len(current_timesteps)
                end_index = num_timesteps - round(num_timesteps * strength)
                current_timesteps = current_timesteps[end_index:]
                foreseen_timesteps = tuple(timesteps[end_index:] for timesteps in foreseen_timesteps)
                strengthened_t = torch.full((input_end.shape[0],), current_timesteps[-1], dtype=current_timesteps.dtype, device=input_end.device)
                x_t = cls.diffuser.diffuse(
                    diffusion_schedule,
                    initial_state,
                    strengthened_t,
                    *diffusion_args,
                    **diffusion_kwargs
                )
            else:
                current_timesteps = current_timesteps[:0]
                foreseen_timesteps = tuple(timesteps[:0] for timesteps in foreseen_timesteps)
                x_t = initial_state
        return x_t, (current_timesteps, *foreseen_timesteps), len(current_timesteps)
    
    @classmethod
    def presample_state(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input_middle: Tensor,
        timestep: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Presample the current state and other corresponding things before the formal sampling.

        Args:
            step_index (int): The current step index of the sampling.
            num_steps (int): The total steps that the sampling needs.
            backbone (nn.Module): The backbone model which predicts the result.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input_middle (Tensor): The state at the given timestep.
            timestep (Tensor): The timestep corresponding to the :param:`input_middle`.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`sample` to keep the context.
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor...: The presampled result and other corresponding things. 
        
        """
        if mask is None:
            if inpaint_reference is None:
                return input_middle
            else:
                raise ValueError(f"when 'mask' is `None`, 'inpaint_reference' should be `None` at the same time")
        else:
            if inpaint_reference is None:
                return input_middle
            else:
                diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
                diffused_inpainting_x = cls.diffuser.diffuse(
                    diffusion_schedule,
                    inpaint_reference,
                    timestep,
                    *diffusion_args,
                    **diffusion_kwargs
                ) # TODO: deterministic forward pass?
                return diffused_inpainting_x * mask + (1. - mask) * input_middle
    
    @classmethod
    def predict(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input: Tensor,
        timestep: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Predicts the state and other corresponding things at the current timestep.

        Args:
            step_index (int): The current step index of the sampling.
            num_steps (int): The total steps that the sampling needs.
            backbone (nn.Module): The backbone model which predicts the result.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The state at the given timestep.
            timestep (Tensor): The timestep corresponding to the :param:`input_middle`.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`sample` to keep the context.
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor...: The predicted result and other corresponding things. 
        
        """
        prediction = cls.diffuser.predict(
            backbone,
            diffusion_schedule,
            input,
            timestep,
            *args,
            **kwargs
        )
        if isinstance(prediction, Tuple):
            prediction = prediction[0]
        else:
            prediction = prediction
        prediction = cls.refine_prediction(
            step_index,
            num_steps,
            diffusion_schedule,
            prediction,
            input,
            timestep,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            **kwargs
        )
        if prediction is not None:
            if isinstance(prediction, Tuple):
                value = prediction[0]
                for handler in output_handlers:
                    value = _apply_handler(handler, value)
                prediction = (value, *prediction[1:])
            else:
                for handler in output_handlers:
                    prediction = _apply_handler(handler, prediction)
        return prediction

    @classmethod
    def refine_prediction(
        cls,
        step_index: int,
        num_steps: int,
        diffusion_schedule: DiffusionSchedule,
        prediction: Tensor,
        input: Tensor,
        timestep: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        """Refines the predicted state from the predicting method of the diffuser.

        Args:
            step_index (int): The current step index of the sampling.
            num_steps (int): The total steps that the sampling needs.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            prediction (Tensor): The predicted state from the diffuser.
            input (Tensor): The state at the given timestep.
            timestep (Tensor): The timestep corresponding to the :param:`input_middle`.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`sample` to keep the context.
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor...: The predicted result and other corresponding things. 
        
        """
        return prediction
    
    @classmethod
    @torch.inference_mode()
    def sample_step(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: DiffusionSchedule,
        input_middle: Tensor,
        presampled_middle: Union[Tensor, Tuple[Tensor, ...]],
        timesteps: Sequence[Tensor],
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tensor:
        """Samples the input to the given timestep.

        Args:
            step_index (int): The current step index of the sampling.
            num_steps (int): The total steps that the sampling needs.
            backbone (nn.Module): The backbone model which predicts the result.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            input (Tensor): The state at the given timestep.
            timestep (Tensor): The timestep corresponding to the :param:`input_middle`.
            *args: The arguments that drive both the diffusion process and the backbone model.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`sample` to keep the context.
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result at the given timestep.
        
        """
        # p_sample
        pass
    
    @classmethod
    def initialize_variables(cls) -> Union[Sequence[Any], Dict[str, Any], None]:
        """Initializes variables for the context of sampling.
        
        Returns:
            Any: The variables container.
        
        """
        return None

    @classmethod
    @torch.inference_mode()
    def sample_loop(
        cls,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: DiffusionSchedule,
        input_end: Union[Tensor, Sequence[int]],
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 1.,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        sampling_progress: SamplingProgress,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tensor:
        """Implements the sampling loop to sample the target which is at the start timestep.

        Args:
            backbone (nn.Module): The backbone model which predicts the noise.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
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
            variables (Union[Sequence[Any], Dict[str, Any], None]): The variables stored in the
                :attr:`sample` to keep the context.
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            sampling_progress (SamplingProgress): The process that indicates the process of current sampling.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        # p_sample_loop "Algorithm 2."
        x_t, timestep_iters, num_steps = cls.initialize_sampling(
            timesteps,
            diffusion_schedule,
            input_end,
            initial_state,
            strength,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            backbone=backbone,
            output_handlers=output_handlers,
            parameterization=parameterization,
            **kwargs
        )
        if not isinstance(timestep_iters, tuple):
            timestep_iters = (timestep_iters,)
        
        for i, timesteps in sampling_progress(num_steps, *timestep_iters):
            timesteps = tuple(torch.tensor([timestep], device=x_t.device, dtype=data_type.determine_timesteps_dtype(_timestep_dtype(timestep), x_t.dtype)) for timestep in timesteps)
            presampled_t = cls.presample_state(
                i,
                num_steps,
                backbone,
                diffusion_schedule,
                x_t,
                timesteps[0],
                *args,
                inpaint_reference=inpaint_reference,
                mask=mask,
                variables=variables,
                parameterization=parameterization,
                output_handlers=output_handlers,
                **kwargs
            )
            x_t = cls.sample_step(
                i,
                num_steps,
                backbone,
                diffusion_schedule,
                x_t,
                presampled_t,
                timesteps,
                *args,
                inpaint_reference=inpaint_reference,
                mask=mask,
                variables=variables,
                parameterization=parameterization,
                output_handlers=output_handlers,
                **kwargs
            )
        
        for handler in output_handlers:
            x_t = handler(x_t)
        sampling_progress.reset()
        return x_t
    
    @classmethod
    @torch.inference_mode()
    def sample(
        cls,
        backbone: nn.Module,
        timesteps: Union[sequencings.Context, Tensor],
        diffusion_schedule: DiffusionSchedule,
        input_end: Union[Tensor, Sequence[int]],
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 1.,
        parameterization: Parameterization,
        sampling_progress: SamplingProgress,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tensor:
        """Samples the target.

        Args:
            backbone (nn.Module): The backbone model which predicts the noise.
            diffusion_schedule (DiffusionSchedule): The schedule of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
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
            parameterization (Parameterization): The parameterization that informs which prediction
                mode is used.
            sampling_progress (SamplingProgress): The process that indicates the process of current sampling.
            output_handlers (Sequence[SamplingOutputHandler]): The handler that handles the output at each
                timestep.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result after given timesteps.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        return cls.sample_loop(
            backbone,
            timesteps,
            diffusion_schedule,
            input_end,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            initial_state=initial_state,
            strength=strength,
            variables=cls.initialize_variables(),
            parameterization=parameterization,
            sampling_progress=sampling_progress,
            output_handlers=output_handlers,
            **kwargs
        )
    
    @torch.inference_mode()
    def forward(
        self,
        input_end: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 1.,
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
        return self.sample(
            self.backbone,
            self.timesteps,
            self.diffusion_schedule,
            input_end,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            initial_state=initial_state,
            strength=strength,
            parameterization=self.parameterization,
            sampling_progress=self.progress,
            output_handlers=self.output_handlers,
            **kwargs
        )
    
    # @classmethod
    # def extract_current_timestep(cls, timesteps: Iterable[Tensor]) -> Tensor:
    #     """Extracts the current timestep which is relevant to the :param:`timesteps`
    #     in the method :attr:`sample_step`.

    #     Args:
    #         timesteps (Iterable[Tensor]): The required sampling timestep(s).

    #     Returns:
    #         Tensor: The main timestep from the given timesteps sequence.

    #     """
    #     return timesteps[0]


def _apply_handler(handler, variable):
    if isinstance(variable, Sequence):
        return tuple(var if not isinstance(var, Tensor) else handler(var) for var in variable)
    else:
        if isinstance(variable, Tensor):
            return handler(variable)
        else:
            return variable