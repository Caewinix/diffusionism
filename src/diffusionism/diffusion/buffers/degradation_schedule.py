import torch
from torch import Tensor
import numpy as np


def linear_schedule(start: float = 1e-4, end: float = 2e-2, num_timesteps: int = 1000) -> Tensor:
    return torch.linspace(start ** 0.5, end ** 0.5, num_timesteps, dtype=torch.float64) ** 2


def cosine_schedule(cosine_s: float = 8e-3, num_timesteps: int = 1000) -> Tensor:
    timesteps = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps + cosine_s
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = torch.cos(alphas).pow(2)
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = np.clip(betas, a_min=0, a_max=0.999)
    return torch.from_numpy(betas)


def _betas_for_alpha_bar(num_diffusion_timesteps: int, alpha_bar, max_beta: float = 0.999) -> Tensor:
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
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas)


def squaredcos_cap_v2_schedule(num_timesteps: int = 1000) -> Tensor: # used for karlo prior
    # return early
    import math
    return _betas_for_alpha_bar(
        num_timesteps,
        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
    )


def direct_linear_schedule(start: float = 1e-4, end: float = 2e-2, num_timesteps: int = 1000) -> Tensor:
    return torch.linspace(start, end, num_timesteps, dtype=torch.float64)


def sqrt_linear_schedule(start: float = 1e-4, end: float = 2e-2, num_timesteps: int = 1000) -> Tensor:
    return torch.linspace(start, end, num_timesteps, dtype=torch.float64) ** 0.5