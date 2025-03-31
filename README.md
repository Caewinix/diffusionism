# Diffusionism: Modular Framework for Customizable Diffusion Models

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.7%2B-green)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![PyTorch Lightning Version](https://img.shields.io/badge/PyTorch_Lightning-2.4.0%2B-blue)](https://lightning.ai/pytorch-lightning)

**Diffusionism** is a PyTorch-based framework for building and experimenting with diffusion models through decoupled forward/reverse processes and configuration-driven workflows. Designed for researchers seeking flexibility, it enables both rapid prototyping of novel diffusion mechanics and production-ready training pipelines.

## Key Features
- **Modular Components**: Independently customize forward diffusion (`Diffuser`) and reverse sampling (`Sampler`) processes
- **Plug-and-Play Architecture**: Combine custom modules via `DiffusionModel` or use standalone components
- **Extensible Base Classes**: Implement new algorithms by overriding targeted methods in base classes
- **Reproducible Pipelines**: Built-in experiment tracking and deterministic training
- **Configuration-Driven Workflows**: Define experiments through YAML configs with PyTorch Lightning integration

## Installation
```bash
pip install diffusionism
```

## Quick Start

### Choise 1: Configuration-Based Execution
Copy `diffusionism/configs` as a template to the project folder, and enter this folder. All YAML files can be modified to meet the task requirements.

For training and validation, just run:
```bash
python -m diffusionism.run --train configs/train.yaml --val configs/val.yaml -m configs/diffusion/diffusion_model.yaml -r configs/diffusion/runner.yaml -n experiment_01
```

After that, run the following to test:
```bash
python -m diffusionism.run --test configs/test.yaml -m configs/diffusion/diffusion_model.yaml -r configs/diffusion/runner.yaml -n experiment_01
```
Make sure the name of the experiment should be kept the same.

### Choise 2: Only Using Predefined Components
Assuming that the `UNet` module has been imported and instantiated as a variable named as `unet`.
```python
import torch
from diffusionism.methods.dpm.diffusers import DPMDiffuser
from diffusionism.methods.dpm.samplers import DDPMSampler
from diffusionism.methods.dpm.schedules import DPMDiscreteSchedule
from diffusionism.methods.dpm.parameterizations import NoiseParameterization
from diffusionism.diffusion.unification import DiffusionModel
from diffusionism.diffusion.schedules.context import distributions
from diffusionism.diffusion.schedules.context import sequencings

# Build an integrated system
model = DiffusionModel(
    diffuser=DPMDiffuser,
    sampler=DDPMSampler,
    timesteps_distribution=distributions.uniform_int.init(low=0, high=1000),
    timesteps=sequencings.arange.reversed(num_steps=1000, start=0, end=1000),
    backbone=unet,
    diffusion_schedule=DPMDiscreteSchedule(betas=torch.linspace(1.e-4, 2.e-2, 1000, dtype=torch.float64)),
    parameterization=NoiseParameterization()
)

# Standalone usage
diffusion_schedule = DPMDiscreteSchedule(betas=torch.linspace(1.e-4, 2.e-2, 1000, dtype=torch.float64))

diffuser = DPMDiffuser(
    backbone=unet,
    diffusion_schedule=diffusion_schedule,
    timesteps_distribution=distributions.uniform_int.init(low=0, high=1000),
    parameterization=NoiseParameterization()
)

sampler = DDPMSampler(
    backbone=unet,
    timesteps=sequencings.arange.reversed(num_steps=1000, start=0, end=1000),
    diffusion_schedule=diffusion_schedule,
    timesteps_distribution=distributions.uniform_int.init(low=0, high=1000),
    parameterization=NoiseParameterization()
)
```

## Customization Guide
This part will use DDPM as the example to show how to implement a custom diffusion model. The predefined one can be found in `diffusionism.methods.dpm`.

Since the diffusion schedule and the parameterization are too complicated to define here, assume that `DiffusionProbabilisticModelsDiscreteSchedule` and `NoiseParameterization` are defined before the following parts.

### 1. Implement the Custom Diffuser
```python
from typing import Callable, Optional
from torch import nn
from torch import Tensor
from diffusionism.diffusion.diffusers.diffuser import Diffuser
from diffusionism.diffusion.parameterizations import Parameterization
from diffusionism.diffusion import losses
from diffusionism.diffusion.schedules.context import distributions
from ..parameterizations import NoiseParameterization
from ..schedules import DiffusionProbabilisticModelsDiscreteSchedule


class DiffusionProbabilisticModelsDiffuser(Diffuser, schedule=DiffusionProbabilisticModelsDiscreteSchedule):
    diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule
    
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        timesteps_distribution: distributions.Context = distributions.uniform_int.init(1000),
        parameterization: Parameterization = NoiseParameterization(),
        loss_function: Callable[[Tensor, Tensor], Tensor] = losses.mse_loss,
        **kwargs
    ):
        super().__init__(
            backbone,
            *args,
            timesteps_distribution=timesteps_distribution,
            parameterization=parameterization,
            loss_function=loss_function,
            **kwargs
        )
    
    @classmethod
    def diffuse(
        cls,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input_start: Tensor,
        timestep: Tensor,
        *diffusion_args,
        noise: Optional[Tensor] = None,
        **diffusion_kwargs
    ) -> Tensor:
        # q_sample
        if noise is None:
            noise = cls.degrade(diffusion_schedule, input_start, timestep, *diffusion_args, **diffusion_kwargs)
        mean = diffusion_schedule.input_scale(input_start, timestep, *diffusion_args, **diffusion_kwargs) * input_start
        std = diffusion_schedule.noise_scale(input_start, timestep, *diffusion_args, **diffusion_kwargs)
        x_t = mean + std * noise
        return x_t
```

### 2. Implement the Custom Sampler
```python
from typing import Optional, Sequence, Tuple, Union, Any, Dict
import torch
from torch import Tensor
from torch import nn
from diffusionism.diffusion.samplers.sampler import Sampler
from diffusionism.diffusion.parameterizations import Parameterization
from diffusionism.diffusion.samplers import OutputHandler
from diffusionism.diffusion.utils.range_clipper import RangeClipper
from ..parameterizations import NoiseParameterization
from ..schedules import DiffusionProbabilisticModelsDiscreteSchedule


class DenoisingDiffusionProbabilisticModelsSampler(Sampler, schedule=DiffusionProbabilisticModelsDiscreteSchedule, diffuser=DiffusionProbabilisticModelsDiffuser):
    diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule

    def __init__(
        self,
        backbone: nn.Module,
        *args,
        parameterization: Parameterization = NoiseParameterization(),
        output_handlers: Sequence[OutputHandler] = [RangeClipper(-1, 1)],
        **kwargs
    ):
        super().__init__(backbone, *args, parameterization=parameterization, output_handlers=output_handlers, **kwargs)
    
    @classmethod
    def posteriorize(
        cls,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input_start: Tensor,
        input_middle: Tensor,
        timestep: Tensor,
        *diffusion_args,
        **diffusion_kwargs
    ) -> Tuple[Tensor, Tensor, Tensor]:
        # q_posterior
        posterior_mean = (
            diffusion_schedule.posterior_mean_start_coefficient(input_start, timestep, *diffusion_args, **diffusion_kwargs) * input_start +
            diffusion_schedule.posterior_mean_current_coefficient(input_middle, timestep, *diffusion_args, **diffusion_kwargs) * input_middle
        )
        posterior_variance = diffusion_schedule.posterior_variance(input_middle, timestep, *diffusion_args, **diffusion_kwargs)
        posterior_log_variance_clipped = diffusion_schedule.posterior_logarithm_variance(input_middle, timestep, *diffusion_args, **diffusion_kwargs)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @classmethod
    def foresee_number_timesteps(cls) -> int:
        return 0
    
    @classmethod
    def refine_prediction(
        cls,
        step_index: int,
        num_steps: int,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        prediction: Tensor,
        input: Tensor,
        timestep: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        **kwargs
    ) -> Union[Tensor, Tuple[Tensor, ...]]:
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        return parameterization.reverse_stochastic_step(
            diffusion_schedule,
            input,
            timestep,
            prediction,
            -1,
            *diffusion_args,
            **diffusion_kwargs
        )
    
    @classmethod
    @torch.inference_mode()
    def sample_step(
        cls,
        step_index: int,
        num_steps: int,
        backbone: nn.Module,
        diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule,
        input_middle: Tensor,
        presampled_middle: Union[Tensor, Tuple[Tensor, ...]],
        timesteps: Sequence[Tensor],
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        variables: Union[Sequence[Any], Dict[str, Any], None] = None,
        parameterization: Parameterization,
        output_handlers: Sequence[OutputHandler],
        **kwargs
    ) -> Tensor:
        timestep: Tensor = timesteps[0]
        
        # p_mean_variance
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        
        step_reconstruction = cls.predict(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            presampled_middle,
            timestep,
            *args,
            inpaint_reference=inpaint_reference,
            mask=mask,
            variables=variables,
            parameterization=parameterization,
            output_handlers=output_handlers,
            **kwargs
        )
        
        model_mean, posterior_variance, posterior_log_variance = cls.posteriorize(
            diffusion_schedule,
            step_reconstruction,
            presampled_middle,
            timestep,
            *diffusion_args,
            **diffusion_kwargs
        )
        model_mean, _, model_log_variance = model_mean, posterior_variance, posterior_log_variance
        
        noise = cls.diffuser.degrade(
            diffusion_schedule,
            presampled_middle,
            timestep,
            *diffusion_args,
            **diffusion_kwargs
        )
        # no noise when t == 0
        nonzero_mask = (1 - (timestep == 0).float()).reshape(presampled_middle.size(0), *((1,) * (len(presampled_middle.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
```
Once again, it is not necessary to implement these as they can be found in `diffusionism.methods.dpm`, since only examples are provided here.

## To Do
- [ ] Inference Code, Distinguished from the Test Code
- [ ] Implement EMA (Exponential Moving Average) Mechanism
- [ ] Latent Diffusion
- [ ] ​Discrete DDPM and DDIM Compatibility Verification Due to the Code Update
- [ ] ​Continuous-Time DDPM and DDIM Implementation
- [ ] Introduce DPM-Solver and DPM-Solver++
- [ ] ...

## Contributing
We welcome contributions!

## Citation
```bibtex
@misc{diffusionism2024,
  author = {Juanhua Zhang},
  title = {Diffusionism: Modular Framework for Customizable Diffusion Models},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Caewinix/diffusionism}}
}
```