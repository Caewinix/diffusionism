from typing import Type
from ..buffers.diffusion_buffer import DiffusionBuffer


class MetaDiffusion(type):
    buffer: Type[DiffusionBuffer]
    diffuser: type
    sampler: type
    
    def __new__(metacls, name, bases, attrdict, **kwargs):
        cls = super().__new__(metacls, name, bases, attrdict)
        buffer = kwargs.get('buffer')
        if buffer is not None:
            cls.buffer = buffer
        diffuser = kwargs.get('diffuser')
        if diffuser is not None:
            cls.diffuser = diffuser
        sampler = kwargs.get('sampler')
        if sampler is not None:
            cls.sampler = sampler
        return cls
    
    @staticmethod
    def get_diffusion_buffer(diffusion_type: Type[DiffusionBuffer], *args, error_head: str = 'it', **kwargs) -> DiffusionBuffer:
        """Returns the diffusion buffer from the input forms.

        Args:
            diffusion_type (Type[DiffusionBuffer]): The type of :class:`DiffusionBuffer`.
            *args: The arguments that may be used to construct the instance of :class:`DiffusionBuffer`.
            error_head: The error massage head text, indicating which part raises such exception.
            **kwargs: The keyword arguments that may be used to construct the instance of
                :class:`DiffusionBuffer`.

        Returns:
            Tensor: The diffusion buffer instance.
        
        Raises:
            TypeError:
                If no any arguments are provided and :param:`diffusion_buffer` is missing.
            TypeError:
                If no any arguments are provided and the number of keyword arguments is more
                    than one, which means there should be unexpected keyword argument(s).
            TypeError:
                If the first argument is :class:`DiffusionBuffer`, but also providing other
                    keyword arguments.
            TypeError:
                If multiple values for argument :param:`diffusion_buffer` are given.
            TypeError:
                If there exists (an) unexpected keyword argument(s).
        
        """
        args_length = len(args)
        kwargs_length = len(kwargs)
        diffusion_buffer = kwargs.get('diffusion_buffer')
        if args_length == 0:
            if diffusion_buffer is None:
                raise TypeError(f"missing 1 required positional argument: 'diffusion_buffer'")
            elif kwargs_length != 1:
                raise TypeError(f"{error_head} got an unexpected keyword argument '{kwargs.keys()[1]}'")
        elif args_length == 1 and isinstance(args[0], DiffusionBuffer):
            if kwargs_length != 0:
                raise TypeError(f"{error_head} got an unexpected keyword argument '{kwargs.keys()[0]}'")
            if diffusion_buffer is None:
                diffusion_buffer = args[0]
            else:
                raise TypeError(f"{error_head} got multiple values for argument 'diffusion_buffer'")
        else:
            if diffusion_buffer is None:
                diffusion_buffer = diffusion_type(*args, **kwargs)
            elif kwargs_length != 1:
                raise TypeError(f"{error_head} got an unexpected keyword argument '{kwargs.keys()[1]}'")
        return diffusion_buffer