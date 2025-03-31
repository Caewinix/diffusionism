class Hyperparameters:
    def add_parameters(self, **kwargs):
        for name, value in kwargs.items():
            self.__setattr__(name, value)