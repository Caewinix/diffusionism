from torch import nn
from .meta_diffusion import MetaDiffusion
from ..buffers.diffusion_buffer import DiffusionBuffer


class Diffusion(nn.Module, metaclass=MetaDiffusion, buffer=DiffusionBuffer):
    pass