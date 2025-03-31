from typing import Dict, Callable, Optional, Union, Iterable, Tuple, Mapping, Any, Sequence
from types import MethodType
from itertools import chain
import numpy as np
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
import pytorch_lightning as pl

params_t = Union[Iterable[torch.Tensor], Iterable[Dict[str, Any]]]


class _ResultsNeeded: ...


def _init_optimizer(parameters: params_t, optimizer: Union[Optimizer, Callable[[params_t], Optimizer]]):
    if isinstance(optimizer, Optimizer):
        return optimizer
    elif isinstance(optimizer, Callable):
        return optimizer(params=parameters)
    else:
        raise TypeError(f"'optimizer' should be an instance of either '{Optimizer.__name__}' or '{Callable.__name__}'")


def _init_lr_scheduler(optimizer: Optimizer, lr_scheduler: Union[LRScheduler, Callable[[Optimizer], LRScheduler]]):
    if isinstance(lr_scheduler, LRScheduler):
        return lr_scheduler
    elif isinstance(lr_scheduler, Callable):
        return lr_scheduler(optimizer=optimizer)
    else:
        raise TypeError(f"'lr_scheduler' should be an instance of either '{LRScheduler.__name__}' or '{Callable.__name__}'")


def _dict_key_list(outputs, key):
    try:
        return outputs[key]
    except KeyError:
        list_value = []
        outputs[key] = list_value
        return list_value


def _return_none(_): ...


def _init_outputs_and_method(
    metric_values,
    metric_values_getter,
    tuple_method: MethodType,
    dict_method: MethodType,
    single_method: MethodType,
    final_tuple_method: MethodType,
    final_dict_method: MethodType,
    final_single_method: MethodType
):
    if isinstance(metric_values, tuple):
        outputs = [[] for _ in range(len(metric_values))]
        original_method = tuple_method
        final_method = final_tuple_method
    elif isinstance(metric_values, dict):
        outputs = {key : [] for key in metric_values.keys()}
        original_method = dict_method
        final_method = final_dict_method
    else:
        outputs = []
        original_method = single_method
        final_method = final_single_method
    self = original_method.__self__
    func = original_method.__func__
    method = MethodType(lambda self, batch, batch_idx: func(self, metric_values_getter(self, batch, batch_idx)), self)
    return outputs, method, final_method, original_method


def _append_with_calculation(keep, output, metric_value, each_metric_value_process):
    if keep:
        output.append(metric_value.detach())
    return each_metric_value_process(metric_value)


def _extend_metrics_to_tuple(keep, outputs: list[list], metric_values: tuple[torch.Tensor], each_metric_value_process = _return_none):
    outputs_length = len(outputs)
    metric_values_length = len(metric_values)
    if outputs_length < metric_values_length:
        outputs.extend(tuple([] for _ in range(metric_values_length - outputs_length)))
    final_metrics = tuple(_append_with_calculation(keep, output, metric_value, each_metric_value_process) for output, metric_value in zip(outputs, metric_values))
    return final_metrics


def _extend_metrics_to_dict(keep, outputs: dict[str, list], metric_values: dict[str, torch.Tensor], each_metric_value_process = _return_none):
    final_metrics = {key : _append_with_calculation(keep, _dict_key_list(outputs, key), metric_value, each_metric_value_process) for key, metric_value in metric_values.items()}
    return final_metrics


def _extend_metrics_to_single(keep, outputs: list[torch.Tensor], metric_values: torch.Tensor, each_metric_value_process = _return_none):
    return _append_with_calculation(keep, outputs, metric_values, each_metric_value_process)


def _cat_and_clear_tensor_list(tensor_list: list[torch.Tensor], dim=0):
    first_tensor = tensor_list[0]
    device = first_tensor.device
    if device.type == 'cuda':
        dtype = first_tensor.dtype
        element_size = first_tensor[(0,) * first_tensor.ndim].element_size()
        shape = np.asarray(first_tensor.shape)
        shape[dim] = shape[dim] * (len(tensor_list) - 1) + tensor_list[-1].shape[dim]
        memory_required = int(np.prod(shape) * element_size)
        if memory_required >= torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_allocated(device) - torch.cuda.memory_reserved(device):
            tensor_cpu_list = [tensor.cpu() for tensor in tensor_list]
            tensor_list.clear()
            out = torch.empty(tuple(shape), device=device, dtype=dtype)
            torch.cat(tensor_cpu_list, out=out, dim=dim)
            return out.to(device)
    result = torch.cat(tensor_list, dim=dim)
    tensor_list.clear()
    return result


def _clear_outputs_for_tuple(outputs: list[list], final_method: MethodType):
    gathered_outputs = tuple(_cat_and_clear_tensor_list(output) for output in outputs if len(output) > 0)
    final_method(*gathered_outputs)


def _clear_outputs_for_dict(outputs: dict[str, list], final_method: MethodType):
    gathered_outputs = {key : _cat_and_clear_tensor_list(output) for key, output in outputs.items() if len(output) > 0}
    final_method(gathered_outputs)


def _clear_outputs_for_single(outputs: list, final_method: MethodType):
    if len(outputs) > 0:
        gathered_outputs = _cat_and_clear_tensor_list(outputs)
    else:
        gathered_outputs = None
    final_method(gathered_outputs)


class ModelRunner(pl.LightningModule):
    """The model runner that is used to run any model.
    """
    def __init__(
        self,
        keep_training_outputs: bool = False,
        keep_validation_outputs: bool = True,
        keep_test_outputs: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.__optimizers = []
        self.__optimizer_parameters_getters = []
        self.__lr_schedulers = []
        
        self.keep_training_outputs = keep_training_outputs
        self.keep_validation_outputs = keep_validation_outputs
        self.keep_test_outputs = keep_test_outputs
        
        self.__training_step_outputs = None
        self.__validation_step_outputs = None
        self.__test_step_outputs = None
        
        self.__validation_results = None
        self.__test_results = None
    
    def select_main_module(self) -> Union[nn.Module, Tuple[nn.Module, ...]]:
        """Returns a main module or a tuple of main modules.
        
        This function should return an instance or instances of :class:`nn.Module` stored in the
        :attr:`__init__`.

        Returns:
            nn.Module(s): A chosen module or a tuple of chosen modules.
        
        """
        pass
    
    def load_state_dict(self, state_dict: Union[Mapping[str, Any], Sequence[Mapping[str, Any]]], strict: bool = True, assign: bool = False):
        main_module = self.select_main_module()
        if isinstance(main_module, tuple):
            tuple(each_module.load_state_dict(each_state_dict, strict, assign) for each_module, each_state_dict in zip(main_module, state_dict))
            return
        return main_module.load_state_dict(state_dict, strict, assign)
    
    def state_dict(self, *args, destination=None, prefix='', keep_vars=False) -> Union[Mapping[str, Any], Sequence[Mapping[str, Any]]]:
        main_module = self.select_main_module()
        if isinstance(main_module, tuple):
            return [each.state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars) for each in main_module]
        return self.select_main_module().state_dict(*args, destination=destination, prefix=prefix, keep_vars=keep_vars)
    
    def set_optimizer(
        self,
        optimizer: Union[Optimizer, Callable[[params_t], Optimizer]],
        optimizer_parameters_getter: Optional[Callable[[nn.Module], params_t]] = None
    ):
        """Sets an optimizer that should be configured.

        Args:
            optimizer (Union[Optimizer, Callable[[params_t], Optimizer]]): An optimizer or a function that
                accepts module parameters and returns an optimizer.
            optimizer_parameters_getter (Optional[Callable[[nn.Module], params_t]]): A function that tells
                the runner how to choose the module parameters. If ``None``, using the default selection.
        
        """
        self.__optimizers = [optimizer]
        self.__optimizer_parameters_getters = [optimizer_parameters_getter]
    
    def set_optimizers(
        self,
        optimizers: Iterable[Union[Optimizer, Callable[[params_t], Optimizer]]],
        optimizer_parameters_getters: Optional[Iterable[Optional[Callable[[nn.Module], params_t]]]] = None
    ):
        """Sets the optimizers that should be configured.

        Args:
            optimizers (Iterable[Union[Optimizer, Callable[[params_t], Optimizer]]]): The optimizers
                sequence or a list of functions that accept module parameters and return an optimizer.
            optimizer_parameters_getters (Optional[Iterable[Optional[Callable[[nn.Module], params_t]]]]):
                A sequence of functions that tell the runner how to choose the module parameters. If
                ``None``, using the default selection.
        
        """
        self.__optimizers = optimizers
        if optimizer_parameters_getters is None:
            self.__optimizer_parameters_getters = [None for _ in range(len(self.__optimizers))]
        else:
            self.__optimizer_parameters_getters = optimizer_parameters_getters
    
    def set_lr_scheduler(self, lr_scheduler: Union[LRScheduler, Callable[[Optimizer], LRScheduler]]):
        """Sets the learning rate scheduler that should be configured.

        Args:
            lr_scheduler (Union[LRScheduler, Callable[[Optimizer], LRScheduler]]): The learning rate
                scheduler or a function that receives the optimizer and returns a learning rate scheduler.
        
        """
        self.__lr_schedulers = [lr_scheduler]
    
    def set_lr_schedulers(self, lr_schedulers: Iterable[Union[LRScheduler, Callable[[Optimizer], LRScheduler]]]):
        """Sets the learning rate schedulers that should be configured.

        Args:
            lr_schedulers (Iterable[Union[LRScheduler, Callable[[Optimizer], LRScheduler]]]): A sequence of
            learning rate scheduler or a list of function that receives the optimizer and returns a learning
            rate scheduler.
        
        """
        self.__lr_schedulers = lr_schedulers
    
    def optimization_parameters(self) -> Iterable[params_t]:
        """Returns the parameters for the optimzer(s).

        Returns:
            Iterable[params_t]: The parameters for the optimzer(s).
        
        """
        main_module = self.select_main_module()
        if isinstance(main_module, tuple):
            default_parameters = chain((each_module.parameters() for each_module in main_module))
        else:
            default_parameters = self.select_main_module().parameters()
        return [
            default_parameters if optimizer_parameters_getter is None else optimizer_parameters_getter(self)
            for optimizer_parameters_getter in self.__optimizer_parameters_getters
        ]

    def configure_optimizers(self) -> Union[Iterable[Optimizer], Tuple[Iterable[Optimizer], Iterable[LRScheduler]]]:
        optimization_parameters = self.optimization_parameters()
        optimizers = [
            _init_optimizer(parameters, optimizer)
            for optimizer, parameters in zip(self.__optimizers, optimization_parameters)
        ]
        if self.__lr_schedulers is None or len(self.__lr_schedulers) == 0:
            return optimizers
        else:
            lr_schedulers = [
                _init_lr_scheduler(optimizer, lr_scheduler)
                for optimizer, lr_scheduler in zip(optimizers, self.__lr_schedulers)
            ]
            return optimizers, lr_schedulers
    
    def train_at_step(self, batch: torch.Tensor, batch_idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Here you compute and return the training loss and some additional metrics for e.g. the progress bar or
        logger.

        Args:
            batch (torch.Tensor): The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx (int): The index of this batch.
        
        Returns:
            The losses and other metrics. If you return a dictionary, the losses should be with the key
                "loss".
        
        """
        pass
    
    def process_training_step_metrics(self, *metrics: torch.Tensor):
        """Here you process the metrics that you return in :attr:`train_at_step`, including the losses.
        
        Args:
            *metrics (torch.Tensor): All metrics you return in :attr:`train_at_step`.
        
        """
        pass
    
    def process_training_step_mean_metrics(self, loss: torch.Tensor, *metrics: torch.Tensor):
        """Here you process the loss and metrics that you return in :attr:`train_at_step`.
        
        Args:
            loss (torch.Tensor): The loss that should be used in backward.
            *metrics (torch.Tensor): All metrics you return in :attr:`train_at_step`.
        
        """
        pass
    
    def end_for_training_epoch(self, *outputs: torch.Tensor):
        """Called in the training loop at the very end of the epoch.
        
        Args:
            *outputs (torch.Tensor): All gathered outputs that were returned in the :attr:`train_at_step`.
        
        """
        pass
    
    def validate_at_step(self, batch: torch.Tensor, batch_idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Operates on a single batch of data from the validation set. In this step you'd might generate examples or
        calculate anything of interest like accuracy.

        Args:
            batch (torch.Tensor): The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx (int): The index of this batch.
        
        Returns:
            Metrics.
        
        """
        pass
    
    def process_validation_step_metrics(self, *metrics: torch.Tensor):
        """Here you process the metrics that you return in :attr:`validate_at_step`.
        
        Args:
            *metrics (torch.Tensor): All metrics you return in :attr:`validate_at_step`.
        
        """
        pass
    
    def end_for_validation_epoch(self, *outputs: torch.Tensor):
        """Called in the validation loop at the very end of the epoch.
        
        Args:
            *outputs (torch.Tensor): All gathered outputs that were returned in the :attr:`validate_at_step`.
        
        """
        pass
    
    def test_at_step(self, batch: torch.Tensor, batch_idx: int) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Operates on a single batch of data from the test set. In this step you'd normally generate examples or
        calculate anything of interest such as accuracy.

        Args:
            batch (torch.Tensor): The output of your data iterable, normally a :class:`~torch.utils.data.DataLoader`.
            batch_idx (int): The index of this batch.
        
        Returns:
            Metrics.
        
        """
        pass
    
    def process_test_step_metrics(self, *metrics: torch.Tensor):
        """Here you process the metrics that you return in :attr:`test_at_step`.
        
        Args:
            *metrics (torch.Tensor): All metrics you return in :attr:`test_at_step`.
        
        """
        pass
    
    def end_for_test_epoch(self, *outputs: torch.Tensor):
        """Called in the test loop at the very end of the epoch.
        
        Args:
            *outputs (torch.Tensor): All gathered outputs that were returned in the :attr:`test_at_step`.
        
        """
        pass
    
    @property
    def __validation_results__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """The validation results that users can directly take without clearing.
        
        Returns:
            The validation results.
        
        Raises:
            ValueError:
                If the model has not been validated.
        """
        results = self.__validation_results
        if results is _ResultsNeeded:
            raise ValueError("this model has not been validated, so the validation results cannot be taken")
        return results
    
    def need_validation_results(self, need: bool = True, *args, **kwargs):
        """Informs the runner to store the validation results, and collect them later by using
        :attr:`take_validation_results`.

        Args:
            need (bool): The flag that controls whether to use the validation results.
        
        """
        if need:
            self.__validation_results = _ResultsNeeded
        else:
            self.__validation_results = None
    
    def take_validation_results(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Returns the validation metric values of the whole dataset, if already used validation results
        through :attr:`need_validation_results`. The runner will clear these results after being taken.

        Returns:
            Union: The metric values of the whole validation dataset.
        
        Raises:
            ValueError:
                If `need_validation_results(True)` is not called.
            ValueError:
                If the model has not been validated.
        
        """
        if self.__validation_results is None:
            raise ValueError("the validation results cannot be obtained, please call `need_validation_results(True)`")
        else:
            result = self.__validation_results__
            self.__validation_results = None
            return result
    
    @property
    def __test_results__(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """The test results that users can directly take without clearing.
        
        Returns:
            The test results.
        
        Raises:
            ValueError:
                If the model has not been tested.
        
        """
        results = self.__test_results
        if results is _ResultsNeeded:
            raise ValueError("this model has not been tested, so the test results cannot be taken")
        return results
    
    def need_test_results(self, need: bool = True, *args, **kwargs):
        """Informs the runner to store the test results, and collect them later by using
        :attr:`take_test_results`.

        Args:
            need (bool): The flag that controls whether to use the test results.
        
        """
        if need:
            self.__test_results = _ResultsNeeded
        else:
            self.__test_results = None
    
    def take_test_results(self) -> Union[torch.Tensor, Tuple[torch.Tensor, ...], Dict[str, torch.Tensor]]:
        """Returns the test result values of the whole dataset, if already used test results
        through :attr:`need_test_results`. The runner will clear these results after being taken.

        Returns:
            Union: The metric values of the whole test dataset.
        
        Raises:
            ValueError:
                If `need_test_results(True)` is not called.
            ValueError:
                If the model has not been tested.
        
        """
        if self.__test_results is None:
            raise ValueError("the test results cannot be obtained, please call `need_test_results(True)`")
        else:
            result = self.__test_results__
            self.__test_results = None
            return result
    
    def __training_step_for_tuple(self, metric_values):
        self.process_training_step_metrics(*metric_values)
        mean_metrics = _extend_metrics_to_tuple(self.keep_training_outputs, self.__training_step_outputs, metric_values, torch.mean)
        self.process_training_step_mean_metrics(*mean_metrics)
        loss = mean_metrics[0]
        return loss
    
    def __training_step_for_dict(self, metric_values):
        self.process_training_step_metrics(metric_values)
        mean_metrics = _extend_metrics_to_dict(self.keep_training_outputs, self.__training_step_outputs, metric_values, torch.mean)
        self.process_training_step_mean_metrics(mean_metrics)
        loss = mean_metrics['loss']
        return loss
    
    def __training_step_for_single(self, metric_values):
        self.process_training_step_metrics(metric_values)
        mean_metrics = _extend_metrics_to_single(self.keep_training_outputs, self.__training_step_outputs, metric_values, torch.mean)
        self.process_training_step_mean_metrics(mean_metrics)
        loss = mean_metrics
        return loss
    
    def __training_step(self, batch, batch_idx) -> torch.Tensor:
        metric_values = self.train_at_step(batch, batch_idx)
        self.__training_step_outputs, self.__training_step, self.__on_train_epoch_end, func = _init_outputs_and_method(
            metric_values,
            type(self).train_at_step,
            tuple_method=self.__training_step_for_tuple,
            dict_method=self.__training_step_for_dict,
            single_method=self.__training_step_for_single,
            final_tuple_method=lambda: _clear_outputs_for_tuple(self.__training_step_outputs, self.end_for_training_epoch),
            final_dict_method=lambda: _clear_outputs_for_dict(self.__training_step_outputs, self.end_for_training_epoch),
            final_single_method=lambda: _clear_outputs_for_single(self.__training_step_outputs, self.end_for_training_epoch)
        )
        loss = func(metric_values)
        return loss
        # if isinstance(metric_values, tuple):
        #     self.process_training_step_metrics(*metric_values)
        #     outputs_length = len(self.__training_step_outputs)
        #     metric_values_length = len(metric_values)
        #     if outputs_length < metric_values_length:
        #         self.__training_step_outputs = self.__training_step_outputs + tuple([] for _ in range(metric_values_length - outputs_length))
        #     mean_metrics = tuple((metric_value.mean(), output.append(metric_value))[0] for output, metric_value in zip(self.__training_step_outputs, metric_values))
        #     self.process_training_step_mean_metrics(*mean_metrics)
        #     loss = mean_metrics[0]
        # elif isinstance(metric_values, dict):
        #     self.process_training_step_metrics(metric_values)
        #     mean_metrics = {key : (metric_value.mean(), _dict_key_list(self.__training_step_outputs, key).extend(metric_value))[0] for key, metric_value in metric_values.items()}
        #     self.process_training_step_mean_metrics(mean_metrics)
        #     loss = mean_metrics['loss']
        # else:
        #     self.process_training_step_metrics(metric_values)
        #     self.__training_step_outputs.extend(metric_values)
        #     mean_metrics = metric_values.mean()
        #     self.process_training_step_mean_metrics(mean_metrics)
        #     loss = mean_metrics
        # return loss
    
    def training_step(self, batch, batch_idx) -> torch.Tensor:
        return self.__training_step(batch, batch_idx)
    
    def on_train_epoch_end(self):
        try:
            self.__on_train_epoch_end()
        except AttributeError: ...
        try:
            del self.__training_step
        except AttributeError: ...
        try:
            del self.__on_train_epoch_end
        except AttributeError: ...

    def __validation_step_for_tuple(self, metric_values):
        self.process_validation_step_metrics(*metric_values)
        _extend_metrics_to_tuple(self.keep_validation_outputs, self.__validation_step_outputs, metric_values)
    
    def __validation_step_for_dict(self, metric_values):
        self.process_validation_step_metrics(metric_values)
        _extend_metrics_to_dict(self.keep_validation_outputs, self.__validation_step_outputs, metric_values)
    
    def __validation_step_for_single(self, metric_values):
        self.process_validation_step_metrics(metric_values)
        _extend_metrics_to_single(self.keep_validation_outputs, self.__validation_step_outputs, metric_values)
    
    def __validation_epoch_end_for_tuple(self, *outputs: torch.Tensor):
        self.end_for_validation_epoch(*outputs)
        self.__validation_results = outputs
    
    def __validation_epoch_end_for_others(self, outputs):
        self.end_for_validation_epoch(outputs)
        self.__validation_results = outputs
    
    def __validation_step(self, batch, batch_idx):
        metric_values = self.validate_at_step(batch, batch_idx)
        self.__validation_step_outputs, self.__validation_step, self.__on_validation_epoch_end, func = _init_outputs_and_method(
            metric_values,
            type(self).validate_at_step,
            tuple_method=self.__validation_step_for_tuple,
            dict_method=self.__validation_step_for_dict,
            single_method=self.__validation_step_for_single,
            final_tuple_method=lambda: _clear_outputs_for_tuple(self.__validation_step_outputs, self.__validation_epoch_end_for_tuple),
            final_dict_method=lambda: _clear_outputs_for_dict(self.__validation_step_outputs, self.__validation_epoch_end_for_others),
            final_single_method=lambda: _clear_outputs_for_single(self.__validation_step_outputs, self.__validation_epoch_end_for_others)
        )
        return func(metric_values)
    
    def validation_step(self, batch, batch_idx):
        return self.__validation_step(batch, batch_idx)
    
    def on_validation_epoch_end(self):
        try:
            self.__on_validation_epoch_end()
        except AttributeError: ...
        try:
            del self.__validation_step
        except AttributeError: ...
        try:
            del self.__on_validation_epoch_end
        except AttributeError: ...
    
    def __test_step_for_tuple(self, metric_values):
        self.process_test_step_metrics(*metric_values)
        _extend_metrics_to_tuple(self.keep_test_outputs, self.__test_step_outputs, metric_values)
    
    def __test_step_for_dict(self, metric_values):
        self.process_test_step_metrics(metric_values)
        _extend_metrics_to_dict(self.keep_test_outputs, self.__test_step_outputs, metric_values)
    
    def __test_step_for_single(self, metric_values):
        self.process_test_step_metrics(metric_values)
        _extend_metrics_to_single(self.keep_test_outputs, self.__test_step_outputs, metric_values)
    
    def __test_epoch_end_for_tuple(self, *outputs: torch.Tensor):
        self.end_for_test_epoch(*outputs)
        self.__test_results = outputs
    
    def __test_epoch_end_for_others(self, outputs):
        self.end_for_test_epoch(outputs)
        self.__test_results = outputs
    
    def __test_step(self, batch, batch_idx):
        metric_values = self.test_at_step(batch, batch_idx)
        self.__test_step_outputs, self.__test_step, self.__on_test_epoch_end, func = _init_outputs_and_method(
            metric_values,
            type(self).test_at_step,
            tuple_method=self.__test_step_for_tuple,
            dict_method=self.__test_step_for_dict,
            single_method=self.__test_step_for_single,
            final_tuple_method=lambda: _clear_outputs_for_tuple(self.__test_step_outputs, self.__test_epoch_end_for_tuple),
            final_dict_method=lambda: _clear_outputs_for_dict(self.__test_step_outputs, self.__test_epoch_end_for_others),
            final_single_method=lambda: _clear_outputs_for_single(self.__test_step_outputs, self.__test_epoch_end_for_others)
        )
        return func(metric_values)
    
    def test_step(self, batch, batch_idx):
        return self.__test_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        try:
            self.__on_test_epoch_end()
        except AttributeError: ...
        try:
            del self.__test_step
        except AttributeError: ...
        try:
            del self.__on_test_epoch_end
        except AttributeError: ...