from .schedules import ElucidatedDiffusionModelsSchedule, VariancePreservingSchedule, VarianceExplodingSchedule


class EDMSchedule(ElucidatedDiffusionModelsSchedule): ...
class VPSchedule(VariancePreservingSchedule): ...
class VESchedule(VarianceExplodingSchedule): ...