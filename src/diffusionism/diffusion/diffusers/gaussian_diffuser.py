from typing import Optional, Tuple
import torch
from torch import Tensor
from ..functions import extract
from .diffuser import Diffuser
from ..buffers.diffusion_buffer import DiffusionBuffer
from ..buffers.gaussian_diffusion_buffer import GaussianDiffusionBuffer


class GaussianDiffuser(Diffuser, buffer=GaussianDiffusionBuffer):
    diffusion_buffer: GaussianDiffusionBuffer
    
    @classmethod
    def diffuse(
        cls,
        diffusion_buffer: DiffusionBuffer,
        input_start: Tensor,
        timestep: Tensor,
        *args,
        degradation: Optional[Tensor] = None,
        **kwargs
    ) -> Tensor:
        # q_sample
        if degradation is None:
            degradation = cls.degrade(diffusion_buffer, input_start, timestep, *args, **kwargs)
        mean = diffusion_buffer.retention_total_std(input_start, timestep, *args, **kwargs) * input_start
        std = diffusion_buffer.degradation_total_std(input_start, timestep, *args, **kwargs)
        x_t = mean + std * degradation
        return x_t
    
    @classmethod
    def degrade(
        cls,
        diffusion_buffer: DiffusionBuffer,
        input: Tensor,
        timestep: Tensor,
        *args,
        **kwargs
    ) -> Tensor:
        return torch.randn_like(input)