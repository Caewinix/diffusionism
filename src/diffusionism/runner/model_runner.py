from typing import Callable, Optional, Union, Iterable, Tuple
from torch.optim.optimizer import Optimizer, params_t
from torch.optim.lr_scheduler import LRScheduler
from torch import nn
import pytorch_lightning as pl


def _init_optimizer(parameters: params_t, optimizer: Union[Optimizer, Callable[[params_t], Optimizer]]):
    if isinstance(optimizer):
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
        raise TypeError(f"'optimizer' should be an instance of either '{Optimizer.__name__}' or '{Callable.__name__}'")


class ModelRunner(pl.LightningModule):
    """The model runner that is used to run any model.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__optimizers = []
        self.__optimizer_parameters_getters = []
        self.__lr_schedulers = []
    
    def select_main_module(self) -> nn.Module:
        """Returns a main module.
        
        This function should return an instance of :class:`nn.Module` that is stored in the
        :attr:`__init__`.

        Returns:
            nn.Module: A chosen module.
        
        """
        pass
    
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