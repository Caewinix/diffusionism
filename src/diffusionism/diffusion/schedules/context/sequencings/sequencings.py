from typing import Iterable
import math
import torch
from torch import Tensor
from ..tensor_context import TensorContext


class Context(TensorContext):
    ...


class ScheduledContext(Context):
    ...


_range = range
@Context
def range(num_steps: int, start: int, end: int) -> Iterable:
    step = (end - start) // num_steps
    return _range(start, end, step)


@Context
def arange(num_steps: int, start: int, end: int, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    step = (end - start) // num_steps
    return torch.arange(start, end, step, dtype=dtype, device=device)


@Context
def quad(num_steps: int, start: float, end: float, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.linspace(start, math.sqrt((end - start) * .8), num_steps, dtype=dtype, device=device).square_()


@Context
def linear(num_steps: int, start: float, end: float, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.linspace(start, end, num_steps, dtype=dtype, device=device)


@Context
def cosine(num_steps: int, cosine_s: float = 8.e-3, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    timesteps = torch.arange(num_steps + 1, dtype=dtype, device=device) / num_steps + cosine_s
    alphas = timesteps / (1 + cosine_s) * torch.pi / 2
    alphas = torch.cos_(alphas).square_()
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = torch.clip(betas, min=0, max=0.999)
    return betas


@Context
def sqrt_linear_square(num_steps: int, start: float, end: float, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.linspace(start ** 0.5, end ** 0.5, num_steps, dtype=dtype, device=device).square_()


@Context
def linear_sqrt(num_steps: int, start: float, end: float, dtype: torch.dtype = None, device: torch.device = None) -> Tensor:
    return torch.linspace(start, end, num_steps, dtype=dtype, device=device).sqrt_()


@Context
def karras_sigma(num_steps: int, sigma_min: float, sigma_max: float, rho: float = 7., dtype: torch.dtype = None, device: torch.device = None):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, num_steps, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = torch.pow(max_inv_rho + ramp * (min_inv_rho - max_inv_rho), rho).to(dtype=dtype)
    return sigmas


def _betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999):
    """Creates a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].
    
    Args:
        num_diffusion_timesteps: the number of betas to produce.
        alpha_bar: a lambda that takes an argument t from 0 to 1 and
            produces the cumulative product of (1-beta) up to that
            part of the diffusion process.
        max_beta: the maximum beta to use; use values lower than 1 to
            prevent singularities.
    """
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        yield min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta)


@Context
def squaredcos_cap_v2(num_steps: int, max_beta: float = 0.999, dtype: torch.dtype = None, device: torch.device = None) -> Tensor: # used for karlo prior
    # return early
    return torch.as_tensor(list(_betas_for_alpha_bar(
        num_steps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
        max_beta
    )), dtype=dtype, device=device)