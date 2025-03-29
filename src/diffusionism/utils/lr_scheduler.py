from typing import Optional
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler, LRScheduler


class GradualWarmupScheduler(_LRScheduler):
    """This learning rate scheduler warms up the inner one gradually.
    """
    def __init__(
        self,
        optimizer: Optimizer,
        multiplier: int,
        *,
        warm_epochs: Optional[int] = None,
        max_epochs: Optional[int] = None,
        reduction: int = 10,
        after_scheduler: LRScheduler = None
    ):
        self.multiplier = multiplier
        if max_epochs is None:
            self.total_epochs = warm_epochs
        elif warm_epochs is None:
            if reduction is not None:
                self.total_epochs = max_epochs // reduction
            else:
                raise ValueError("when 'max_epochs' is given, 'reduction' should not be `None`")
        else:
            raise ValueError(f"one of 'warm_epochs' and 'max_epochs' should be `None`, but got 'warm_epochs = {warm_epochs}' and 'max_epochs = {max_epochs}'")
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = None
        self.base_lrs = None
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.total_epochs:
            if self.after_scheduler is not None:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_last_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epochs + 1.) for base_lr in self.base_lrs]


    def step(self, epoch: Optional[int] = None, **kwargs):
        if self.finished and self.after_scheduler:
            if epoch is None:
                self.after_scheduler.step(None)
            else:
                self.after_scheduler.step(epoch - self.total_epochs)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)