from typing import Union
import torch
from torch import Tensor
from ..samplers.sampling_progress import OutputStageHandler


class RangeClipper(OutputStageHandler):
    """This class is used to clip the range for the sampled result (step or final).
    """
    def __init__(
        self,
        min: Union[Tensor, int, float, None] = None,
        max: Union[Tensor, int, float, None] = None,
        step_based: bool = False
    ):
        """
        Args:
            min (Union[Tensor, int, float, None]): The lower bound. If ``None``, no
                lower bound will be applied.
            max (Union[Tensor, int, float, None]): The upper bound. If ``None``, no
                upper bound will be applied.
            step_based (bool): The flag that controls whether only applying the
                range to the final sampled result or after each step.
        """
        super().__init__()
        self.min = min
        self.max = max
        self.step_based = step_based
    
    def clip(self, input: Tensor) -> Tensor:
        if self.min is None and self.max is None:
            return input
        return torch.clip_(input, self.min, self.max)
    
    def handle_step(self, input: torch.Tensor) -> torch.Tensor:
        if self.step_based:
            return self.clip(input)
        else:
            return input
    
    def handle_final(self, input: torch.Tensor) -> torch.Tensor:
        return self.clip(input)

NoneRange = RangeClipper()