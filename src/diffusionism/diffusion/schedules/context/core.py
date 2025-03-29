from typing import Callable
from types import MethodType
import weakref
import torch
import numpy as np


leading = True
tailing = False


# class _formal_value:
#     def __init__(self, name):
#         self.name = name


# class formals:
#     def __getattr__(self, name):
#         return _formal_value(name)
# formals = formals()


# def _formal_indice(args: tuple):
#     for i, arg in enumerate(args):
#         if isinstance(arg, _formal_value):
#             yield i


# def _formal_keys(kwargs: dict):
#     for key, arg in kwargs.items():
#         if isinstance(arg, _formal_value):
#             yield key


def _initialize(cls, function, *args, **kwargs):
    instance = object.__new__(cls)
    cls.__init__(instance)
    instance.__function__ = function
    return instance.init(*args, **kwargs)


def initialize(wrapper, *args, **kwargs):
    cls, function = wrapper
    function = function()
    if function is not None:
        del function.__function__
        del function.__leading__
        del function.__reverse__
        del function.initialize
        del function.init
        del function.reversed
        del function.call
    return _initialize(cls, function, *args, **kwargs)


_reversed = reversed
def reversed(wrapper, *args, **kwargs):
    instance = initialize(wrapper, *args, **kwargs)
    instance.__reverse__ = True
    return instance


def call(function, *args, **kwargs):
    function = function()
    if function is not None:
        return function(*args, **kwargs)


class _reverse:
    def __get__(self, instance, owner):
        return None
    
    def __set__(self, instance, owner, value):
        raise ValueError(f"uninitialized '{Context.__qualname__}' cannot be set '__reverse__' attribute")


class Context:
    __function__: Callable
    @staticmethod
    def __new__(cls, function):
        function.__function__ = weakref.proxy(function)
        function.__leading__ = False
        function.__reverse__ = _reverse()
        function.initialize = MethodType(initialize, (cls, weakref.ref(function)))
        function.init = function.initialize
        function.reversed = MethodType(reversed, (cls, weakref.ref(function)))
        function.call = MethodType(call, weakref.ref(function))
        return function
    
    @classmethod
    def init(cls, *args, **kwargs):
        def wrapper(function):
            instance = object.__new__(cls)
            cls.__init__(instance, *args, **kwargs)
            instance.__function__ = function
            return instance.init()
        return wrapper
    
    def __init__(self):
        self.__leading__ = False
        self.__reverse__ = False
        self.call = self.__call
        self.init = self.__init
    
    def initialize(self, *args, **kwargs):
        self.__arguments__ = args
        self.__keyword_arguments__ = kwargs
        return self
    
    def __init(self, *args, **kwargs):
        return self.initialize(*args, **kwargs)
    
    @classmethod
    def call(cls, function, *args, **kwargs):
        return function(*args, **kwargs)
    
    def __call(self, *args, **kwargs):
        return self.__function__(*args, **kwargs)
    
    def __getattribute__(self, name):
        if name == '__name__':
            return self.__function__.__name__
        elif name == '__qualname__':
            return self.__function__.__qualname__
        elif name == '__module__':
            return self.__function__.__module__
        return super().__getattribute__(name)
    
    def __getitem__(self, leading):
        cls = type(self)
        instance = _initialize(cls, self.__function__, *self.__arguments__, **self.__keyword_arguments__)
        instance.__leading__ = leading
        return instance
    
    def reversed(self, *args, **kwargs):
        self.initialize(*args, **kwargs)
        self.__reverse__ = True
        return True
    
    def __call__(self, *args, **kwargs):
        inner_args = args
        inner_kwargs = kwargs
        args = self.__arguments__
        kwargs = self.__keyword_arguments__
        if self.__leading__:
            # formal_indice_start = 0
            in_args = args + inner_args
            in_kwargs = kwargs.copy()
            in_kwargs.update(inner_kwargs)
        else:
            # formal_indice_start = len(inner_args)
            in_args = inner_args + args
            in_kwargs = inner_kwargs.copy()
            in_kwargs.update(kwargs)
        # if len(self.__formal_indice) != 0 or len(self.__formal_keys) != 0:
        #     import inspect
        #     binded = inspect.signature(func).bind(*in_args, **in_kwargs).arguments
        #     for i in self.__formal_indice:
        #         index = formal_indice_start + i
        #         in_args[index] = binded[in_args[index].name]
        #     for key in self.__formal_keys:
        #         in_kwargs[key] = binded[in_kwargs[key].name]
        result = type(self).call(self.__function__, *in_args, **in_kwargs)
        if self.__reverse__:
            if isinstance(result, torch.Tensor):
                result = torch.flip(result)
            elif isinstance(result, np.ndarray):
                result = np.flip(result)
            else:
                result = _reversed(result)
        return result
    
    # @property
    # def arguments(self) -> tuple:
    #     return self.__args
    
    # @arguments.setter
    # def arguments(self, value):
    #     self.__args = value
    #     self.__formal_indice = list(_formal_indice(self.__args))
    
    # @property
    # def keyword_arguments(self) -> dict:
    #     return self.__kwargs
    
    # @keyword_arguments.setter
    # def keyword_arguments(self, value):
    #     self.__kwargs = value
    #     self.__formal_keys = list(_formal_keys(self.__kwargs))