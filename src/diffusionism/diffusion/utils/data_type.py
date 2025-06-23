import torch


def is_int_dtype(dtype):
    try:
        torch.iinfo(dtype)
        return True
    except TypeError:
        return False


def is_float_dtype(dtype):
    try:
        torch.finfo(dtype)
        return True
    except TypeError:
        return False


def determine_timesteps_dtype(default_dtype, input_dtype):
    if is_float_dtype(input_dtype):
        if default_dtype is None or is_float_dtype(default_dtype):
            default_dtype = input_dtype
    elif is_int_dtype(input_dtype):
        if default_dtype is not None and is_int_dtype(default_dtype):
            default_dtype = input_dtype
    return default_dtype