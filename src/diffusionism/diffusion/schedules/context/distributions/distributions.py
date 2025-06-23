from typing import Union, Sequence
import math
import torch
from torch import Tensor
from ..tensor_context import TensorContext


class Context(TensorContext):
    def __init__(self, differential_timestep: Union[float, int] = 1.e-3):
        super().__init__()
        self.differential_timestep = differential_timestep


class ScheduledContext(Context):
    ...


class ShapeContext(Context):
    ...


class ScheduledShapeContext(Context):
    ...


@ShapeContext
def uniform(shape: Sequence, low = 0, high = 1, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.rand(shape, dtype=dtype, device=device) * (high - low) + low


@ShapeContext
def normal(shape: Sequence, mean = 0, std = 1, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return mean + torch.randn(shape, dtype=dtype, device=device) * std


@ShapeContext
def uniform_int(shape: Sequence, low = 0, high = 1000, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.randint(high - low, size=shape, dtype=dtype, device=device) + low


@ShapeContext
def inverse_uniform(shape: Sequence, low = 0, high = 1, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return high + torch.rand(shape, dtype=dtype, device=device) * (low - high)


@ShapeContext
def logarithm_uniform_logarithm(shape: Sequence, logarithm_low, logarithm_high, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return logarithm_low * ((logarithm_high / logarithm_low) ** torch.rand(shape, dtype=dtype, device=device))


@ShapeContext
def normal_logarithm(shape: Sequence, mean, std, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.exp_(normal(shape, mean, std, dtype, device))


@ShapeContext
def cosine_interpolated(shape: Sequence, image_d, noise_d_low, noise_d_high, sigma_data=1., min_value=1e-3, max_value=1e3, dtype: torch.dtype = None, device: torch.device = None):
    """Draws samples from an interpolated cosine timestep distribution (from simple diffusion)."""

    def logsnr_schedule_cosine(t, logsnr_min, logsnr_max):
        t_min = math.atan(math.exp(-0.5 * logsnr_max))
        t_max = math.atan(math.exp(-0.5 * logsnr_min))
        return -2 * torch.log(torch.tan(t_min + t * (t_max - t_min))).to(dtype)

    def logsnr_schedule_cosine_shifted(t, image_d, noise_d, logsnr_min, logsnr_max):
        shift = 2 * math.log(noise_d / image_d)
        return logsnr_schedule_cosine(t, logsnr_min - shift, logsnr_max - shift) + shift

    def logsnr_schedule_cosine_interpolated(t, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max):
        logsnr_low = logsnr_schedule_cosine_shifted(t, image_d, noise_d_low, logsnr_min, logsnr_max)
        logsnr_high = logsnr_schedule_cosine_shifted(t, image_d, noise_d_high, logsnr_min, logsnr_max)
        return torch.lerp(logsnr_low, logsnr_high, t).to(dtype)

    logsnr_min = -2 * math.log(min_value / sigma_data)
    logsnr_max = -2 * math.log(max_value / sigma_data)
    u = torch.rand(shape, dtype=dtype, device=device)
    logsnr = logsnr_schedule_cosine_interpolated(u, image_d, noise_d_low, noise_d_high, logsnr_min, logsnr_max)
    return torch.exp(-logsnr / 2) * sigma_data