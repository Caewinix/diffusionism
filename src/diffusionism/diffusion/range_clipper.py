from typing import Union
import torch
from torch import Tensor
from torch import nn


class RangeClipper(nn.Module):
    """This class is used to clip the range for the sampled result (step or final).
    """
    def __init__(
        self,
        min: Union[Tensor, int, float, None] = None,
        max: Union[Tensor, int, float, None] = None,
        is_only_final: bool = True
    ):
        """
        Args:
            min (Union[Tensor, int, float, None]): The lower bound. If ``None``, no
                lower bound will be applied.
            max (Union[Tensor, int, float, None]): The upper bound. If ``None``, no
                upper bound will be applied.
            is_only_final (bool): The flag that controls whether only applying the
                range to the final sampled result, but not the step sampled result.
        """
        super().__init__()
        self.min = min
        self.max = max
        self.is_only_final = is_only_final
    
    def forward(self, input: Tensor):
        if self.min is None and self.max is None:
            return input
        return torch.clip(input, self.min, self.max)


NoneRange = RangeClipper()