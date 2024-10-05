from typing import overload, Callable, Optional, Tuple, Union, Iterable, List, Dict, Mapping, Any
import torch
from torch import nn
from monai.metrics import SSIMMetric
import torchflint as te
from ..model_runner import ModelRunner
from ...diffusion import DiffusionModel


class DiffusionRunner(ModelRunner):
    """The runner that drives the diffusion model, containing the training, validation and test parts.
    """
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        data_getter: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
        target_data_getter: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
        *,
        source_data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        initial_state_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        initial_noise_strength: float = 1.0,
        metrics: Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]] = SSIMMetric(spatial_dims=2, data_range=1.0, win_size=7)
    ):
        """
        Args:
            diffusion_model (DiffusionModel): The diffusion model instance.
            data_getter (Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]): The
                data that is not regarding to the direct diffusion, but may be for guidance.
            target_data_getter (Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]):
                The target input data for the diffusion.
            source_data_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]):
                The source data that may be used to translate the source data into the target data.
                If ``None``, no translation task will be runned.
            initial_state_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]):
                The initial state getter which is used to start the sampling process from this gotten state.
                If ``None``, :param:`initial_noise_strength` should be ``1.0``, and that means the initial
                state will be the pure random.
            initial_noise_strength (float): The initial noise strength value that will be applied to the
                :param:`initial_state_getter`.
            metrics (Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]]):
                The metrics list that contain all metrics calculation function.
        
        """
        super().__init__()
        self.__training_step_outputs = []
        self.__validation_step_outputs = []
        self.__test_step_outputs = []
        self.__test_collection = None
        
        self.diffusion_model = diffusion_model
        self.data_getter = data_getter
        self.target_data_getter = target_data_getter
        if source_data_getter is None:
            self.source_data_getter = lambda _: tuple()
        else:
            self.source_data_getter = lambda batch: (source_data_getter(batch),)
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
        
        self.metrics = metrics if isinstance(metrics, Iterable) else [metrics]
    
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
        return self.diffusion_model.diffuser
    
    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        return self.select_main_module().load_state_dict(state_dict, strict, assign)
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False):
        return self.select_main_module().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        target_input = self.target_data_getter(batch)
        gotten_data = self.data_getter(batch)
        if not isinstance(gotten_data, tuple):
            gotten_data = (gotten_data,)
        losses: torch.Tensor = self(
            target_input,
            *self.source_data_getter(batch),
            *gotten_data,
            additional_residuals=self.get_additional_residuals(batch, batch_idx)
        )
        self.__training_step_outputs.append(losses)
        losses = losses.mean()
        self.log('Step Loss', losses, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("Global Step", self.global_step, prog_bar=True, logger=True, on_step=True, on_epoch=False)
        return losses
    
    def on_train_epoch_end(self):
        loss = torch.vstack(self.__training_step_outputs).mean()
        self.log('Epoch Loss', loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.log('Epoch', self.current_epoch, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        self.__training_step_outputs.clear()
    
    @torch.inference_mode()
    def evaluation_step(self, batch, batch_idx, log_prefix = '', return_images: bool = False) -> Mapping[str, torch.Tensor]:
        """Evaluates the batch step.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            log_prefix (str): The string that tells the evaluation type, for example, 'val' or 'test'.
            return_images (bool): The flag that controls whether adding the sampled and original images into
                the returned dictionary.

        Returns:
            Mapping: A dictionary that contains all metrics and the images if
                :param:`return_images` is ``True``.
        
        """
        initial_state = self.initial_state_getter(batch)
        target_input = self.target_data_getter(batch)
        noisy_input = torch.rand_like(target_input)
        gotten_data = self.data_getter(batch)
        if not isinstance(gotten_data, tuple):
            gotten_data = (gotten_data,)
        sampled = self(
            noisy_input,
            *self.source_data_getter(batch),
            *gotten_data,
            additional_residuals=self.get_additional_residuals(batch, batch_idx),
            initial_state=initial_state,
            strength=self.initial_noise_strength
        )
        target_input = te.map_range(target_input, dim=(1, 2, 3))
        sampled = te.map_range(sampled, dim=(1, 2, 3))
        metric_values = {type(metric).__name__ : metric(sampled, target_input) for metric in self.metrics}
        collection_dict = {f'{log_prefix}-{key}' : value for key, value in metric_values.items()}
        self.log_dict(collection_dict, prog_bar=False, logger=False, on_step=True, on_epoch=True)
        self.log_dict({f'{log_prefix}-{key}' : value.mean() for key, value in collection_dict.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if return_images:
            metric_values.update({'generation' : sampled, 'target': target_input})
        return metric_values
    
    @torch.no_grad()
    def evaluation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], need_clear: bool = True) -> Mapping[str, torch.Tensor]:
        """Gives the evaluation results of the whole dataset after all steps are evaluated, reaching
        to the epoch end.

        Args:
            outputs (List[Dict[str, torch.Tensor]]): The output list containing all step results.
            need_clear (bool): A flag that determines whether clearing the outputs list.

        Returns:
            Mapping: A dictionary that is gathered from the output list.
        
        """
        keys = set()
        for dictionary in outputs:
            keys = keys.union(set(dictionary.keys()))
        output_dict = outputs[0]
        for output in outputs[1:]:
            for key in keys:
                key_output = output.get(key)
                key_result_output = output_dict.get(key)
                if key_result_output is not None:
                    if not isinstance(key_result_output, list):
                        output_dict[key] = [key_result_output]
                    if key_output is not None:
                        output_dict[key].append(key_output)
                elif key_output is not None:
                    output_dict[key] = [key_output]
        for key in keys:
            output_dict[key] = torch.vstack(output_dict[key]).mean()
        if need_clear:
            outputs.clear()
        return output_dict
    
    def validation_step(self, batch, batch_idx):
        metric_values = self.evaluation_step(batch, batch_idx, log_prefix='Val')
        self.__validation_step_outputs.append(metric_values)
    
    def on_validation_epoch_end(self):
        output_dict = self.evaluation_epoch_end(self.__validation_step_outputs)
        self.log_dict(output_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    def use_test_collection(self, use: bool = True):
        """Informs the runner that store the test returned images, and collect them later using
        :attr:`get_test_collection`.

        Args:
            use (bool): The flag that controls whether to use the test collection.
        
        """
        if use:
            self.__test_collection = {'generation' : [], 'target': []}
        else:
            self.__test_collection = None
    
    def get_test_collection(self) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[None, None]]:
        """Returns the test image results of the whole dataset, if already used test collection.

        Returns:
            Tuple: The generations and the original targets if already used test collection, otherwise
                ``None`` and ``None``.
        
        """
        if self.__test_collection is None:
            return None, None
        else:
            generation, target = self.__test_collection.pop('generation'), self.__test_collection.pop('target')
            self.__test_collection = None
            generation = torch.stack(generation, dim=0)
            target = torch.stack(target, dim=0)
            return generation, target
    
    def test_step(self, batch, batch_idx):
        metric_values = self.evaluation_step(batch, batch_idx, log_prefix='Test', return_images=True)
        generation = metric_values.pop('generation')
        target = metric_values.pop('target')
        if self.__test_collection is not None:
            self.__test_collection['generation'].append(generation)
            self.__test_collection['target'].append(target)
        self.__test_step_outputs.append(metric_values)
    
    def on_test_epoch_end(self):
        output_dict = self.evaluation_epoch_end(self.__test_step_outputs)
        self.log_dict(output_dict, prog_bar=False, logger=True, on_step=False, on_epoch=True)