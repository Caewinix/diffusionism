from typing import overload, Callable, Optional, Tuple, Union, Iterable
import torch
from torch import nn
from monai.metrics import SSIMMetric
from ..generative import GenerativeRunner


class DiffusionRunner(GenerativeRunner):
    """The runner that drives the diffusion model, containing the training, validation and test parts.
    """
    def __init__(
        self,
        diffusion_model,
        target_data_getter: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
        source_data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        initial_state_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        initial_noise_strength: float = 1.0,
        *,
        metrics: Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]] = SSIMMetric(spatial_dims=2, data_range=1.0, win_size=7)
    ):
        """
        Args:
            diffusion_model (DiffusionModel): The diffusion model instance.
            target_data_getter (Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]):
                The target input data for the diffusion.
            source_data_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]):
                The source data that may be used to translate the source data into the target data.
                If ``None``, no translation task will be runned.
            data_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]): The
                data that is not regarding to the direct diffusion, but may be for guidance. If ``None``,
                no other data will be used.
            initial_state_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]):
                The initial state getter which is used to start the sampling process from this gotten state.
                If ``None``, :param:`initial_noise_strength` should be ``1.0``, and that means the initial
                state will be the pure random.
            initial_noise_strength (float): The initial noise strength value that will be applied to the
                :param:`initial_state_getter`.
            metrics (Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]]):
                The metrics list that contain all metrics calculation function.
        
        """
        super().__init__(target_data_getter, source_data_getter, data_getter, metrics=metrics)
        self.diffusion_model = diffusion_model
        self.initial_noise_strength = initial_noise_strength
        
        if self.initial_noise_strength != 1.:
            if initial_state_getter is None:
                if source_data_getter is not None:
                    self.initial_state_getter = self.source_data_getter
                else:
                    raise ValueError(f"'initial_state_getter' and 'source_data_getter' should not be `None` at the same time.")
            else:
                self.initial_state_getter = initial_state_getter
        else:
            self.initial_state_getter = lambda _: None
    
    @overload
    @torch.no_grad()
    def forward(
        self,
        input_start: torch.Tensor,
        *args,
        additional_residuals: Optional[Iterable[torch.Tensor]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Calculates the losses regarding to any timesteps.

        Args:
            input_start (torch.Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            additional_residuals (Optional[Iterable[torch.Tensor]]): The additional parts that
                need to be added into the backbone model, in a residual form. If ``None``,
                no any residuals will be added into the backbone.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            torch.Tensor: The loss results.
        
        """
        ...
    
    @overload
    @torch.no_grad()
    def forward(
        self,
        input_end: torch.Tensor,
        *args,
        additional_residuals: Optional[Iterable[torch.Tensor]] = None,
        inpaint_reference: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        strength: float = 0.8,
        **kwargs
    ) -> torch.Tensor:
        """Samples the degraded input to the predicted input.

        Args:
            input_end (torch.Tensor): The input at the final (or the initial state) step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            additional_residuals (Optional[Iterable[torch.Tensor]]): The additional parts that
                need to be added into the backbone model, in a residual form. If ``None``,
                no any residuals will be added into the backbone.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
            inpaint_reference (Optional[torch.Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[torch.Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            initial_state (torch.Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            torch.Tensor: A sampled result at the given timestep.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        ...
    
    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)
    
    def get_additional_residuals(self, batch, batch_idx) -> Iterable[torch.Tensor]:
        """Returns the additional residuals that will be added into the backbone model.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.

        Returns:
            Iterable[torch.Tensor]: A sequence of additional residuals.
        
        """
        return None
    
    def select_main_module(self) -> nn.Module:
        return self.diffusion_model.diffuser.backbone
    
    def train_at_step(self, batch, batch_idx) -> torch.Tensor:
        target_input = self.get_target_data(batch)
        losses: torch.Tensor = self(
            target_input,
            *self.get_source_data(batch),
            *self.get_additional_data(batch),
            additional_residuals=self.get_additional_residuals(batch, batch_idx)
        )
        return losses
    
    def generate(self, batch, batch_idx, source_input: Tuple[torch.Tensor, ...], data: Tuple[torch.Tensor, ...], target_input: torch.Tensor) -> torch.Tensor:
        initial_state = self.initial_state_getter(batch)
        degradation = self.diffusion_model.degrade(target_input, torch.zeros((target_input.size(0), 1)))
        return self(
            degradation,
            *source_input,
            *data,
            additional_residuals=self.get_additional_residuals(batch, batch_idx),
            initial_state=initial_state,
            strength=self.initial_noise_strength
        )