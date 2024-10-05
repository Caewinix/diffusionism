from torch import Tensor
import torch.nn.functional as F


def mse_loss(input: Tensor, target: Tensor) -> Tensor:
    return F.mse_loss(input, target, reduction='none')


def mae_loss(input: Tensor, target: Tensor) -> Tensor:
    return (target - input).abs()