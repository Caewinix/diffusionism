from .core import Hyperparameters


class ImplicitHyperparameters(Hyperparameters):
    def __init__(self, eta: float = 0.):
        self.eta = eta