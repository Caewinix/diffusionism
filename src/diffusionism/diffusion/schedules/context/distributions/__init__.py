from .distributions import (
    Context,
    ScheduledContext,
    ShapeContext,
    ScheduledShapeContext,
    uniform,
    normal,
    uniform_int,
    inverse_uniform,
    logarithm_uniform_logarithm,
    normal_logarithm,
    cosine_interpolated
)
for cls in (
    Context,
    ScheduledContext,
    ShapeContext,
    ScheduledShapeContext
):
    cls.__module__ = f'{__name__}'