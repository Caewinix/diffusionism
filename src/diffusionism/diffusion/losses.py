from torch import Tensor
import torch.nn.functional as F


def mse_loss(input: Tensor, target: Tensor, reduction = 'none') -> Tensor:
    return F.mse_loss(input, target, reduction=reduction)


def mae_loss(input: Tensor, target: Tensor, reduction = 'none') -> Tensor:
    return F.l1_loss(input, target, reduction=reduction)