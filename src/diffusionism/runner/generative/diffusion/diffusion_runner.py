from typing import overload, Callable, Optional, Sequence, Tuple, Union
import torch
from torch import nn
from ..generative_runner import GenerativeRunner
from ....diffusion import DiffusionModel


def _check_func_none(func, batch):
    if func is None:
        return None
    return func(batch)


def _combine_residuals(plugins, batch):
    if plugins is not None:
        result = plugins[0](batch)
        if not isinstance(result, Sequence):
            result = [result]
        if len(plugins) > 1:
            for plugin in plugins[1:]:
                current_result = plugin(batch)
                if not isinstance(current_result, Sequence):
                    current_result = [current_result]
                for i, layer_result in enumerate(current_result):
                    if i < len(result):
                        result[i] = result[i] + layer_result
                    else:
                        result.append(layer_result)
        return result


class DiffusionRunner(GenerativeRunner):
    """The runner that drives the diffusion model, containing the training, validation and test parts.
    """
    diffusion_model: DiffusionModel
    initial_state_getter: Optional[Callable[..., Tuple[torch.Tensor, ...]]]
    initial_noise_strength: float
    
    def __init__(
        self,
        diffusion_model,
        target_data_getter: Callable[[torch.Tensor], torch.Tensor],
        source_data_getter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        data_getter: Optional[Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]] = None,
        inpaint_reference_getter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        mask_getter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        initial_state_getter: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        initial_noise_strength: float = 1.0,
        *args,
        shallow_plugins: Optional[Sequence[nn.Module]] = None,
        deep_plugins: Optional[Sequence[nn.Module]] = None,
        metrics: Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]], None] = None,
        global_metrics: Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]], None] = None,
        features_extractor: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        feature_metrics: Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]], None] = None,
        **kwargs
    ):
        """
        Args:
            diffusion_model (DiffusionModel): The diffusion model instance.
            target_data_getter (Callable[[torch.Tensor], torch.Tensor]):
                The target input data for the diffusion.
            source_data_getter (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The source data that may be used to translate the source data into the target data.
                If ``None``, no translation task will be runned.
            data_getter (Optional[Callable[[torch.Tensor], Union[torch.Tensor, Tuple[torch.Tensor, ...]]]]):
                The data that is not regarding to the direct diffusion, but may be for guidance. If ``None``,
                no other data will be used.
            inpaint_reference_getter (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The inpaint data getter for inpainting tasks.
            mask_getter (Optional[Callable[[torch.Tensor], torch.Tensor]]): 
                The mask for inpainting tasks and other mask required tasks.
            initial_state_getter (Optional[Callable[[torch.Tensor], torch.Tensor]]):
                The initial state getter which is used to start the sampling process from this gotten state.
                If ``None``, :param:`initial_noise_strength` should be ``1.0``, and that means the initial
                state will be the pure random.
            initial_noise_strength (float): The initial noise strength value that will be applied to the
                :param:`initial_state_getter`.
            *args
            shallow_plugins (Optional[Sequence[nn.Module]]): The plugins contains modules that return tensors for
                shallow layers of the backbone model.
            deep_plugins (Optional[Sequence[nn.Module]]): The plugins contains modules that return tensors for
                deep layers of the backbone model.
            metrics (Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]], None]):
                The metrics list that contains all metrics calculation function, used after each validation and test step.
            global_metrics (Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]], None]):
                The metrics list that contains all metrics calculation function, used in generated distribution
                and the original distribution. When it is not ``None`` or empty, generation collection will be
                automatically available to obtain.
            features_extractor (Optional[Callable[[torch.Tensor], torch.Tensor]]): The extractor that extracts
                the features from the ground truth and the generated results.
            feature_metrics (Union[Callable[[torch.Tensor], torch.Tensor], Sequence[Callable[[torch.Tensor], torch.Tensor]], None]):
                The metrics that are calculated from the extracted features in the end of the validation and test epoch.
            **kwargs
        
        """
        super().__init__(
            target_data_getter,
            source_data_getter,
            data_getter,
            *args,
            metrics=metrics,
            global_metrics=global_metrics,
            features_extractor=features_extractor,
            feature_metrics=feature_metrics,
            **kwargs
        )
        self.diffusion_model = diffusion_model
        self.inpaint_reference_getter = inpaint_reference_getter
        self.mask_getter = mask_getter
        self.initial_noise_strength = initial_noise_strength
        self.initial_state_getter = initial_state_getter
        self.shallow_plugins = shallow_plugins
        self.deep_plugins = deep_plugins
    
    @overload
    def forward(
        self,
        input_start: torch.Tensor,
        *args,
        shallow_residuals: Optional[Sequence[Optional[torch.Tensor]]] = None,
        deep_residuals: Optional[Sequence[Optional[torch.Tensor]]] = None,
        **kwargs
    ) -> torch.Tensor:
        """Calculates the losses regarding to any timesteps.

        Args:
            input_start (torch.Tensor): The clean input which means it is at the first step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            shallow_residuals (Optional[Sequence[Optional[torch.Tensor]]]): The additional residuals that
                will be added into the shallow layers of the backbone model. If ``None``, no any
                shallow residuals will be added into the backbone.
            deep_residuals (Optional[Sequence[Optional[torch.Tensor]]]): The deep residuals that will be
                added into the deep layers of the backbone model. If ``None``, no any
                deep residuals will be added into the backbone.
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
        shallow_residuals: Optional[Sequence[Optional[torch.Tensor]]] = None,
        deep_residuals: Optional[Sequence[Optional[torch.Tensor]]] = None,
        inpaint_reference: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
        strength: float = 1.0,
        **kwargs
    ) -> torch.Tensor:
        """Samples the degraded input to the predicted input.

        Args:
            input_end (torch.Tensor): The input at the final (or the initial state) step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            shallow_residuals (Optional[Sequence[Optional[torch.Tensor]]]): The additional residuals that
                will be added into the shallow layers of the backbone model. If ``None``, no any
                shallow residuals will be added into the backbone.
            deep_residuals (Optional[Sequence[Optional[torch.Tensor]]]): The deep residuals that will be
                added into the deep layers of the backbone model. If ``None``, no any
                deep residuals will be added into the backbone.
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
    
    def get_initial_state(self, batch) -> Optional[torch.Tensor]:
        if self.initial_noise_strength == 1.:
            if self.initial_state_getter is not None:
                raise ValueError(f"'initial_state_getter' should not be `None` when 'initial_noise_strength' equals to `1`")
            return None
        else:
            if self.initial_noise_strength > 1.:
                raise ValueError(f"'initial_noise_strength' should not be greater then `1`.")
            if self.initial_state_getter is None:
                raise ValueError(f"'initial_state_getter' should be `None` when 'initial_noise_strength' less then `1`")
            else:
                return self.initial_state_getter(batch)
    
    def get_shallow_residuals(self, batch) -> Optional[Sequence[Optional[torch.Tensor]]]:
        """Returns the additional residuals that will be added into the shallow layers of the backbone model.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.

        Returns:
            Sequence[torch.Tensor]: A sequence of additional residuals.
        
        """
        return _combine_residuals(self.shallow_plugins, batch)
    
    def get_deep_residuals(self, batch) -> Optional[Sequence[Optional[torch.Tensor]]]:
        """Returns the deep residuals that will be added into the deep layers of the backbone model.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.

        Returns:
            Sequence[torch.Tensor]: A sequence of skip residuals.
        
        """
        return _combine_residuals(self.deep_plugins, batch)
    
    def get_inpaint_reference(self, batch) -> torch.Tensor:
        return _check_func_none(self.inpaint_reference_getter, batch)
    
    def get_mask(self, batch) -> torch.Tensor:
        return _check_func_none(self.mask_getter, batch)
    
    def select_main_module(self) -> nn.Module:
        return self.diffusion_model.diffuser.backbone
    
    def train_at_step(self, batch, batch_idx) -> torch.Tensor:
        target_input = self.get_target_data(batch)
        source_input = self.get_source_data(batch)
        data = self.get_additional_data(batch)
        if source_input is not None:
            data = (source_input, *data)
        losses: torch.Tensor = self(
            target_input,
            *data,
            shallow_residuals=self.get_shallow_residuals(batch),
            deep_residuals=self.get_deep_residuals(batch)
        )
        return losses
    
    def generate(self, batch, batch_idx, source_input: Optional[torch.Tensor], data: Tuple[torch.Tensor, ...], target_input: torch.Tensor) -> torch.Tensor:
        initial_state = self.get_initial_state(batch)
        if source_input is not None:
            data = (source_input, *data)
        return self(
            target_input.shape,
            *data,
            shallow_residuals=self.get_shallow_residuals(batch),
            deep_residuals=self.get_deep_residuals(batch),
            inpaint_reference=self.get_inpaint_reference(batch),
            mask=self.get_mask(batch),
            initial_state=initial_state,
            strength=self.initial_noise_strength
        )
    
    # @property
    # def _ddp_params_and_buffers_to_ignore(self):
    #     return ('diffusion_model.diffuser.diffusion_schedule', 'diffusion_model.sampler.diffusion_schedule')