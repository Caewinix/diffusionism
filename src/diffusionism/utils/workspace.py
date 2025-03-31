from typing import Callable, Any, Optional, Dict
import inspect
import pytorch_lightning as pl


class _Empty: ...
def work(
    function: Callable[..., Any],
    globals: Optional[Dict[str, Any]] = None,
    locals: Optional[Dict[str, Any]] = None
) -> Any:
    keys = inspect.signature(function).parameters.keys()
    parameters = dict()
    if globals is not None:
        parameters.update({key : globals.get(key, _Empty) for key in keys})
    if locals is not None:
        parameters.update({key : locals.get(key, _Empty) for key in keys})
    for key in keys:
        value = parameters[key]
        if value is _Empty:
            parameters.pop(key)
    return function(**parameters)


class Workspace(pl.Trainer):
    def initialize_workspace(
        self,
        globals: Optional[Dict[str, Any]] = None,
        locals: Optional[Dict[str, Any]] = None
    ):
        back_frame = inspect.currentframe().f_back
        if globals is None:
            globals = back_frame.f_globals
        if locals is None:
            locals = back_frame.f_locals
        self.globals = globals
        self.locals = locals
    
    def prepare(self, preparation: Optional[Callable[..., Any]] = None):
        self.preparation = preparation
    
    def finish(self, finishing: Optional[Callable[..., Any]] = None):
        self.finishing = finishing
    
    def set_workshop(
        self,
        preparation: Optional[Callable[..., Any]] = None,
        finishing: Optional[Callable[..., Any]] = None
    ):
        self.preparation = preparation
        self.finishing = finishing
    
    def _run_stage(self):
        if self.preparation is not None:
            work(self.preparation, self.globals, self.locals)
        result = super()._run_stage()
        if self.finishing is not None:
            work(self.finishing, self.globals, self.locals)
        return result