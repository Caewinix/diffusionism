from ....diffusion.hyperparameters import Hyperparameters


class ChurnStochasticHyperparameters(Hyperparameters):
    def __init__(
        self,
        stochastic_churn: float = 0,
        stochastic_tmin: float = 0,
        stochastic_tmax: float = float('inf'),
        stochastic_noise: float = 1.,
        alpha: float = 1
    ):
        self.stochastic_churn = stochastic_churn
        self.stochastic_tmin = stochastic_tmin
        self.stochastic_tmax = stochastic_tmax
        self.stochastic_noise = stochastic_noise
        self.alpha = alpha