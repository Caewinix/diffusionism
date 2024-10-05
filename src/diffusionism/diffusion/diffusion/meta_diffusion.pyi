from typing import Type
from ..buffers.diffusion_buffer import DiffusionBuffer
from ..diffusers.diffuser import Diffuser
from ..samplers.sampler import Sampler


class MetaDiffusion(type):
    buffer: Type[DiffusionBuffer]
    diffuser: Type[Diffuser]
    sampler: Type[Sampler]
    @staticmethod
    def get_diffusion_buffer(diffusion_type: type, *args, error_head: str = 'it', **kwargs) -> DiffusionBuffer: ...