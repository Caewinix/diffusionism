from typing import Callable, overload, Any
class Context:
    __function__: Callable
    __leading__: bool
    __reverse__: bool
    def initialize(self, *args, **kwargs) -> Context:
        """This is used to initialize the fixed arguments before calling the function.
        """
        ...
    @overload
    @classmethod
    def init(cls, *args, **kwargs) -> Context:
        """This is used to initialize the instance, with the arguments that will be passed to
        :attr:`__init__`.
        """
        ...
    @overload
    def init(self, *args, **kwargs) -> Context:
        """This is used to initialize the fixed arguments before calling the function,
        which is the abbreviated version of :attr:`initialize`.
        """
        ...
    def reversed(self, *args, **kwargs) -> Context:
        """This is used to initialize the fixed arguments before calling the function, and
        set the function result as reversed after called.
        """
        ...
    @overload
    @classmethod
    def call(cls, function: Callable, *args, **kwargs) -> Any:
        """This calling is only used in the subclass of :class:`Context`,
        which can be overwritten.
        """
        ...
    @overload
    def call(self, *args, **kwargs) -> Any:
        """This calling is only used out of :class:`Context`, which can
        not be overwritten.
        """
        ...
# class formals: ...
class leading:
    """The label that is used to make :param:`__leading__` ``True`` in :class:`Context`.
    """
    ...
class tailing:
    """The label that is used to make :param:`__leading__` ``False`` in :class:`Context`.
    """
    ...