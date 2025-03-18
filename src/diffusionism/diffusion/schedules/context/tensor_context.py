from types import GeneratorType
import torch
from torch import Tensor
from .core import Context


_call_context_error_message = 'unexpected keyword argument'
class TensorContext(Context):
    def __init__(self):
        super().__init__()
        self.__dtype = None
        self.__device = None
    
    def __call__(self, *args, dtype = None, device = None, **kwargs):
        if self.__dtype is None or self.__device is None:
            try:
                value = super().__call__(*args, dtype=dtype, device=device, **kwargs)
                self.__device = self.__dtype = True
                self.__post_process = lambda value, dtype, device: value
            except TypeError as e:
                if _call_context_error_message in e.args[0]:
                    try:
                        value = super().__call__(*args, dtype=dtype, **kwargs)
                        self.__dtype = True
                        self.__device = False
                        if isinstance(value, Tensor):
                            value = value.to(device=device)
                            self.__post_process = lambda tensor, dtype, device: tensor.to(device=device)
                        elif isinstance(value, GeneratorType):
                            value = torch.as_tensor(tuple(value)).to(device=device)
                            self.__post_process = lambda value, dtype, device: torch.as_tensor(tuple(value)).to(device=device)
                        else:
                            value = torch.as_tensor(value).to(device=device)
                            self.__post_process = lambda value, dtype, device: torch.as_tensor(value).to(device=device)
                    except TypeError as e:
                        if _call_context_error_message in e.args[0]:
                            try:
                                value = super().__call__(*args, device=device, **kwargs)
                                self.__dtype = False
                                self.__device = True
                                if isinstance(value, Tensor):
                                    value = value.to(dtype=dtype)
                                    self.__post_process = lambda tensor, dtype, device: tensor.to(dtype=dtype)
                                elif isinstance(value, GeneratorType):
                                    value = torch.as_tensor(tuple(value)).to(dtype=dtype)
                                    self.__post_process = lambda value, dtype, device: torch.as_tensor(tuple(value)).to(dtype=dtype)
                                else:
                                    value = torch.as_tensor(value).to(dtype=dtype)
                                    self.__post_process = lambda value, dtype, device: torch.as_tensor(value).to(dtype=dtype)
                            except TypeError as e:
                                if _call_context_error_message in e.args[0]:
                                    value = super().__call__(*args, **kwargs)
                                    if isinstance(value, Tensor):
                                        value = value.to(dtype=dtype, device=device)
                                        self.__post_process = lambda tensor, dtype, device: tensor.to(dtype=dtype, device=device)
                                    elif isinstance(value, GeneratorType):
                                        value = torch.as_tensor(tuple(value)).to(dtype=dtype, device=device)
                                        self.__post_process = lambda value, dtype, device: torch.as_tensor(tuple(value)).to(dtype=dtype, device=device)
                                    else:
                                        value = torch.as_tensor(value).to(dtype=dtype, device=device)
                                        self.__post_process = lambda value, dtype, device: torch.as_tensor(value).to(dtype=dtype, device=device)
                                else:
                                    raise
                        else:
                            raise
                else:
                    raise
            return value
        else:
            if self.__dtype:
                kwargs.update({'dtype' : dtype})
            if self.__device:
                kwargs.update({'device' : device})
            value = super().__call__(*args, **kwargs)
            return self.__post_process(value, dtype, device)