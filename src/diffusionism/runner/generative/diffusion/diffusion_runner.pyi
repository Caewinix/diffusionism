from typing import Callable, Optional, Tuple, Union, Iterable
import torch
from monai.metrics import SSIMMetric
from ..generative import GenerativeRunner
from ....diffusion import DiffusionModel


class DiffusionRunner(GenerativeRunner):
    diffusion_model: DiffusionModel
    initial_state_getter: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]
    initial_noise_strength: float
    
    def __init__(
        self,
        diffusion_model: DiffusionModel,
        target_data_getter: Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]],
        source_data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        data_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        initial_state_getter: Optional[Callable[[Tuple[torch.Tensor, ...]], Tuple[torch.Tensor, ...]]] = None,
        initial_noise_strength: float = 1.0,
        *,
        metrics: Union[Callable[[torch.Tensor], torch.Tensor], Iterable[Callable[[torch.Tensor], torch.Tensor]]] = SSIMMetric(spatial_dims=2, data_range=1.0, win_size=7)
    ): ...
    
    def get_additional_residuals(self, batch, batch_idx) -> Iterable[torch.Tensor]: ...