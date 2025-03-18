from .sequencings import (
    Context,
    ScheduledContext,
    range,
    arange,
    quad,
    linear,
    cosine,
    sqrt_linear_square,
    linear_sqrt,
    karras_sigma,
    squaredcos_cap_v2
)
for cls in (
    Context,
    ScheduledContext
):
    cls.__module__ = f'{__name__}'