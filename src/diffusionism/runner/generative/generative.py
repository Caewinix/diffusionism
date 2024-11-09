from typing import Callable, Optional, Tuple, Union, Iterable, Mapping, Dict
import numpy as np
import torch
from monai.metrics import SSIMMetric
import torchflint as te
from ..model_runner import ModelRunner


class GenerativeRunner(ModelRunner):
    def __init__(
        self,
        target_data_getter: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
        source_data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        *,
        metrics: Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]] = SSIMMetric(spatial_dims=2, data_range=1.0, win_size=7)
    ):
        """
        Args:
            target_data_getter (Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]):
                The target input data for the diffusion.
            source_data_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]):
                The source data that may be used to translate the source data into the target data.
                If ``None``, no translation task will be runned.
            data_getter (Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]]): The
                data that is not regarding to the direct diffusion, but may be for guidance. If ``None``,
                no other data will be used.
            metrics (Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]]):
                The metrics list that contain all metrics calculation function.
        
        """
        super().__init__()
        self.__validation_collection = None
        self.__test_collection = None
        
        self.target_data_getter = target_data_getter
        if source_data_getter is None:
            self.source_data_getter = lambda _: tuple()
        else:
            self.source_data_getter = source_data_getter
        if data_getter is None:
            self.data_getter = lambda _: tuple()
        else:
            self.data_getter = data_getter
        
        self.metrics = metrics if isinstance(metrics, Iterable) else [metrics]
    
    def get_target_data(self, batch):
        return self.target_data_getter(batch)
    
    def get_source_data(self, batch):
        data = self.source_data_getter(batch)
        if not isinstance(data, tuple):
            data = (data,)
        return data
    
    def get_additional_data(self, batch):
        data = self.data_getter(batch)
        if not isinstance(data, tuple):
            data = (data,)
        return data
    
    def process_training_step_mean_metrics(self, loss):
        self.log('Step Loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log("Global Step", self.global_step, prog_bar=True, logger=False, on_step=True, on_epoch=False)
    
    def end_for_training_epoch(self, losses: torch.Tensor):
        self.log('Epoch Loss', losses.mean(), prog_bar=True, logger=True, on_step=False, on_epoch=True)
        # self.log('Epoch', self.current_epoch, prog_bar=True, logger=True, on_step=False, on_epoch=True)
    
    def generate(self, batch, batch_idx, source_input: Tuple[torch.Tensor, ...], data: Tuple[torch.Tensor, ...], target_input: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass
    
    @torch.inference_mode()
    def __evaluation_step(self, batch, batch_idx, log_prefix = None, return_images: bool = False) -> Mapping[str, torch.Tensor]:
        """Evaluates the batch step.

        Args:
            batch: The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx: The index of this batch.
            log_prefix (Optional[str]): The string that tells the evaluation type, for example, 'val' or 'test'.
            return_images (bool): The flag that controls whether adding the sampled and original images into
                the returned dictionary.

        Returns:
            Mapping: A dictionary that contains all metrics and the images if
                :param:`return_images` is ``True``.
        
        """
        target_input: torch.Tensor = self.get_target_data(batch)
        generation = self.generate(batch, batch_idx, self.get_source_data(batch), self.get_additional_data(batch), target_input)
        map_dim = tuple(np.arange(1, len(target_input.shape)))
        target_input = te.map_range(target_input, dim=map_dim)
        generation = te.map_range(generation, dim=map_dim)
        metric_values = {type(metric).__name__ : metric(generation, target_input) for metric in self.metrics}
        if log_prefix is None:
            collection_dict = metric_values
        else:
            collection_dict = {f'{log_prefix}-{key}' : value for key, value in metric_values.items()}
        # self.log_dict(collection_dict, prog_bar=False, logger=False, on_step=True, on_epoch=True)
        self.log_dict({key : value.mean() for key, value in collection_dict.items()}, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        if return_images:
            return metric_values, generation, target_input
        return metric_values
    
    # @torch.no_grad()
    # def evaluation_epoch_end(self, outputs: List[Dict[str, torch.Tensor]], need_clear: bool = True) -> Mapping[str, torch.Tensor]:
    #     """Gives the evaluation results of the whole dataset after all steps are evaluated, reaching
    #     to the epoch end.

    #     Args:
    #         outputs (List[Dict[str, torch.Tensor]]): The output list containing all step results.
    #         need_clear (bool): A flag that determines whether clearing the outputs list.

    #     Returns:
    #         Mapping: A dictionary that is gathered from the output list.
        
    #     """
    #     keys = set()
    #     for dictionary in outputs:
    #         keys = keys.union(set(dictionary.keys()))
    #     output_dict = outputs[0]
    #     for output in outputs[1:]:
    #         for key in keys:
    #             key_output = output.get(key)
    #             key_result_output = output_dict.get(key)
    #             if key_result_output is not None:
    #                 if not isinstance(key_result_output, list):
    #                     output_dict[key] = [key_result_output]
    #                 if key_output is not None:
    #                     output_dict[key].append(key_output)
    #             elif key_output is not None:
    #                 output_dict[key] = [key_output]
    #     for key in keys:
    #         output_dict[key] = torch.vstack(output_dict[key]).mean()
    #     if need_clear:
    #         outputs.clear()
    #     return output_dict
    
    @torch.no_grad()
    def validate_at_step(self, batch, batch_idx) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.__validation_collection is None:
            metric_values = self.__evaluation_step(batch, batch_idx, log_prefix='Val')
        else:
            metric_values, generation, target = self.__evaluation_step(batch, batch_idx, log_prefix='Val', return_images=True)
            self.__validation_collection[0].append(generation)
            self.__validation_collection[1].append(target)
        return metric_values
    
    def end_for_validation_epoch(self, outputs: dict[str, torch.Tensor]):
        self.log_dict({key : value.mean() for key, value in outputs.items()}, prog_bar=False, logger=True, on_step=False, on_epoch=True)
    
    @torch.no_grad()
    def test_at_step(self, batch, batch_idx) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        if self.__test_collection is None:
            metric_values = self.__evaluation_step(batch, batch_idx, log_prefix='Test')
        else:
            metric_values, generation, target = self.__evaluation_step(batch, batch_idx, log_prefix='Test', return_images=True)
            self.__test_collection[0].append(generation)
            self.__test_collection[1].append(target)
        return metric_values
    
    def end_for_test_epoch(self, outputs: dict[str, torch.Tensor]):
        self.log_dict({key : value.mean() for key, value in outputs.items()}, prog_bar=False, logger=True, on_step=False, on_epoch=True)
        
    def __evaluation_results(self, results, collection) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        if collection is not None:
            generation, target = collection[0], collection[1]
            generation = torch.concat(generation, dim=0).permute(0, 2, 3, 1)
            target = torch.concat(target, dim=0).permute(0, 2, 3, 1)
            error_map = torch.abs(generation - target)
            if isinstance(results, tuple):
                results = results + (generation, target, error_map)
            elif isinstance(results, dict):
                results.update({
                    'generation' : generation,
                    'target' : target,
                    'error' : error_map
                })
            else:
                results = results, generation, target, error_map
        return results
    
    def need_validation_results(self, need = True, need_images = True, *args, **kwargs):
        """Informs the runner to store the validation results, and collect them later by using
        :attr:`take_validation_results`.

        Args:
            need (bool): The flag that controls whether to use the validation results.
            need_images (bool) : The flag that controls whether to use the validation images.
        
        """
        super().need_validation_results(need, *args, **kwargs)
        if need_images:
            self.__validation_collection = ([], [])
        else:
            self.__validation_collection = None
    
    @property
    def __validation_results__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        return self.__evaluation_results(super().__validation_results__, self.__validation_collection)
    
    def take_validation_results(self) -> Union[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        result = super().take_validation_results()
        self.__validation_collection = None
        return result
    
    def need_test_results(self, need = True, need_images = True, *args, **kwargs):
        """Informs the runner to store the test results, and collect them later by using
        :attr:`take_test_results`.

        Args:
            need (bool): The flag that controls whether to use the test results.
            need_images (bool) : The flag that controls whether to use the test images.
        
        """
        super().need_test_results(need, *args, **kwargs)
        if need_images:
            self.__test_collection = ([], [])
        else:
            self.__test_collection = None
    
    @property
    def __test_results__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        return self.__evaluation_results(super().__test_results__, self.__test_collection)
    
    def take_test_results(self) -> Union[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
        result = super().take_test_results()
        self.__test_collection = None
        return result