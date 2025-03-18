from tqdm import tqdm
import torch


class SamplingProgress:
    """The sampling progress offers a state recorder and a callable iterator.
    """
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.step_index = 0
        self.num_steps = None
    
    def __call__(self, num_steps: int, *args):
        self.num_steps = num_steps
        for i, arg in enumerate(tqdm(zip(*args), leave=False, total=num_steps)):
            if self.num_steps is None:
                return
            self.step_index = i
            yield i, arg
        del self.num_steps


class OutputHandler:
    """This handler handle the output tensor after each step and the final step.
    """
    def __init__(self):
        self.__state = None
    
    def attach(self, state: SamplingProgress):
        self.__state = state
    
    def detach(self):
        self.__state = None
    
    @property
    def __state_getter(self):
        if self.is_available:
            return self.__state
        else:
            raise ValueError(f"this handler should be attached to a '{SamplingProgress.__name__}' instance")
    
    @property
    def is_available(self) -> bool:
        """Checks whether the sampling progress is attached.
        """
        return self.__state is not None
    
    @property
    def step_index(self) -> int:
        """Indicates the index of current step.
        """
        return self.__state_getter.step_index
    
    @property
    def num_steps(self) -> int:
        """Indicates the number of the total steps
        """
        return self.__state_getter.num_steps
    
    @property
    def is_sampling(self) -> bool:
        """Indicates whether the sampling is on going.
        """
        try:
            return self.num_steps is not None
        except AttributeError:
            return False
    
    @property
    def is_completed(self) -> bool:
        """Indicates whether the sampling process is completed.
        """
        try:
            self.__state_getter.num_steps
            return False
        except AttributeError:
            return True
    
    def handle(self, input: torch.Tensor) -> torch.Tensor:
        """The concrete handling process.
        """
        raise NotImplementedError()
    
    def __call__(self, input: torch.Tensor) -> torch.Tensor:
        return self.handle(input)


class OutputStageHandler(OutputHandler):
    def handle_step(self, input: torch.Tensor) -> torch.Tensor:
        """The handling method for each step.
        """
        raise NotImplementedError()
    
    def handle_final(self, input: torch.Tensor) -> torch.Tensor:
        """The handling method for the final result.
        """
        raise NotImplementedError()
    
    def handle(self, input: torch.Tensor) -> torch.Tensor:
        if self.is_completed:
            return self.handle_final(input)
        else:
            return self.handle_step(input)