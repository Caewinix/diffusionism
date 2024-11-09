from typing import Union
import torch
from torch.utils.data import Subset
from torch.utils.data.dataset import Dataset


class RandomDataset(Subset):
    """The dataset randomly messes up indices.
    """
    def __init__(self, dataset: Dataset, limit_num: Union[int, float, None] = None):
        indices = torch.randperm(len(dataset))
        if isinstance(limit_num, int):
            indices = indices[:limit_num]
        elif isinstance(limit_num, float):
            limit_num = int(round(limit_num * len(indices)))
            indices = indices[:limit_num]
        super().__init__(dataset, indices)