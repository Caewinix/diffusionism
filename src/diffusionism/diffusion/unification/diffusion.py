from typing import Type
from torchflint import nn
from ..schedules.diffusion_schedule import DiffusionSchedule


class DiffusionBase(nn.Module):
    def __init_subclass__(cls, **kwargs):
        schedule = kwargs.get('schedule')
        if schedule is not None:
            cls.schedule = schedule
        diffuser = kwargs.get('diffuser')
        if diffuser is not None:
            cls.diffuser = diffuser
        sampler = kwargs.get('sampler')
        if sampler is not None:
            cls.sampler = sampler

    @staticmethod
    def get_diffusion_schedule(diffusion_type: Type[DiffusionSchedule], args: list, kwargs: dict, error_head: str = 'it') -> DiffusionSchedule:
        """Returns the diffusion schedule from the input forms.

        Args:
            diffusion_type (Type[DiffusionSchedule]): The type of :class:`DiffusionSchedule`.
            args (list): The arguments that may be used to construct the instance of :class:`DiffusionSchedule`.
            kwargs (dict): The keyword arguments that may be used to construct the instance of
                :class:`DiffusionSchedule`.
            error_head: The error massage head text, indicating which part raises such exception.

        Returns:
            Tensor: The diffusion schedule instance.
        
        Raises:
            TypeError:
                If no any arguments are provided and :param:`diffusion_schedule` is missing.
            TypeError:
                If no any arguments are provided and the number of keyword arguments is more
                    than one, which means there should be unexpected keyword argument(s).
            TypeError:
                If the first argument is :class:`DiffusionSchedule`, but also providing other
                    keyword arguments.
            TypeError:
                If multiple values for argument :param:`diffusion_schedule` are given.
            TypeError:
                If there exists (an) unexpected keyword argument(s).
        
        """
        args_length = len(args)
        diffusion_schedule = kwargs.pop('diffusion_schedule', None)
        # kwarg_keys = list(kwargs.keys())
        if args_length == 0:
            if diffusion_schedule is None:
                raise TypeError(f"missing 1 required positional argument: 'diffusion_schedule'")
            # elif kwargs_length != 1:
            #     raise TypeError(f"{error_head} got an unexpected keyword argument '{kwarg_keys[1]}'")
        elif args_length == 1 and isinstance(args[0], DiffusionSchedule):
            # if kwargs_length != 0:
            #     raise TypeError(f"{error_head} got an unexpected keyword argument '{kwarg_keys[0]}'")
            if diffusion_schedule is None:
                diffusion_schedule = args.pop(0)
            else:
                raise TypeError(f"{error_head} got multiple values for argument 'diffusion_schedule'")
        else:
            if diffusion_schedule is None:
                diffusion_schedule = diffusion_type(*args, **kwargs)
            # elif kwargs_length != 1:
            #     raise TypeError(f"{error_head} got an unexpected keyword argument '{kwarg_keys[1]}'")
        return diffusion_schedule


class Diffusion(DiffusionBase, schedule=DiffusionSchedule):
    def __init__(self, *args, **kwargs):
        args = list(args)
        self.diffusion_schedule = DiffusionBase.get_diffusion_schedule(
            type(self).schedule, args, kwargs, f"{type(self).__name__}.{type(self).__init__.__code__.co_name}()"
        )