from typing import Optional, Sequence, Tuple, Union, Any, Dict
import torch
from torch import Tensor
from torch import nn
from ....diffusion.samplers.sampler import Sampler
from ....diffusion.parameterizations import Parameterization
from ....diffusion.samplers import OutputHandler
from ....diffusion.utils.range_clipper import RangeClipper
from ..diffusers import DiffusionProbabilisticModelsDiffuser
from ..parameterizations import NoiseParameterization
from ..schedules.schedule import DiffusionProbabilisticModelsDiscreteSchedule


class DenoisingDiffusionProbabilisticModelsBasedSampler(Sampler, schedule=DiffusionProbabilisticModelsDiscreteSchedule, diffuser=DiffusionProbabilisticModelsDiffuser):
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


class DenoisingDiffusionProbabilisticModelsSampler(DenoisingDiffusionProbabilisticModelsBasedSampler):
    diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule
    
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
    
    # @classmethod
    # @torch.inference_mode()
    # def sample_step(
    #     cls,
    #     enumeration: int,
    #     length: int,
    #     backbone: nn.Module,
    #     diffusion_schedule: DiffusionProbabilisticModelsSchedule,
    #     input_middle: Tensor,
    #     presampled_middle: Tensor,
    #     timesteps: Iterable[Tensor],
    #     *args,
    #     inpaint_reference: Optional[Tensor] = None,
    #     mask: Optional[Tensor] = None,
    #     variables: Optional[dict] = None,
    #     parameterization: Parameterization,
    #     range_clipper: RangeClipper = RangeClipper(-1, 1),
    #     **kwargs
    # ) -> Tensor:
    #     # p_sample
    #     timestep: Tensor = timesteps[0]
        
    #     # p_mean_variance
    #     diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
    #     backbone_args, backbone_kwargs = diffusion_schedule.get_backbone_arguments(*args, **kwargs)
        
    #     prediction = backbone(presampled_middle, timestep, *backbone_args, **backbone_kwargs)
    #     step_reconstruction = parameterization.reconstruct_step(
    #         diffusion_schedule,
    #         presampled_middle,
    #         timestep,
    #         *diffusion_args,
    #         noise=prediction,
    #         **diffusion_kwargs
    #     )
    #     if not range_clipper.is_only_final:
    #         step_reconstruction = range_clipper(step_reconstruction)

    #     model_mean, posterior_variance, posterior_log_variance = cls.posteriorize(
    #         diffusion_schedule,
    #         step_reconstruction,
    #         presampled_middle,
    #         timestep,
    #         *diffusion_args,
    #         **diffusion_kwargs
    #     )
    #     model_mean, _, model_log_variance = model_mean, posterior_variance, posterior_log_variance
        
    #     noise = cls.diffuser.degrade(
    #         diffusion_schedule,
    #         presampled_middle,
    #         timestep,
    #         *diffusion_args,
    #         **diffusion_kwargs
    #     )
    #     # no noise when t == 0
    #     nonzero_mask = (1 - (timestep == 0).float()).reshape(presampled_middle.size(0), *((1,) * (len(presampled_middle.shape) - 1)))
    #     return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise


class DenoisingDiffusionImplicitModelsSampler(DenoisingDiffusionProbabilisticModelsBasedSampler):
    diffusion_schedule: DiffusionProbabilisticModelsDiscreteSchedule

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
        ddim_alpha = diffusion_schedule.input_scale_square(input, timestep, *diffusion_args, **diffusion_kwargs)
        
        e_t = parameterization.noise(
            diffusion_schedule,
            input,
            timestep,
            prediction,
            *diffusion_args,
            **diffusion_kwargs
        )
        pred_x0 = parameterization.input_start(
            diffusion_schedule,
            input,
            timestep,
            prediction,
            *diffusion_args,
            **diffusion_kwargs
        )
        return pred_x0, e_t, ddim_alpha

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
        diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
        
        t, prev_t = timesteps
        ddim_alpha_prev = diffusion_schedule.input_scale_square(presampled_middle, prev_t, *diffusion_args, **diffusion_kwargs)

        pred_x0, e_t, ddim_alpha = cls.predict(
            step_index,
            num_steps,
            backbone,
            diffusion_schedule,
            presampled_middle,
            t,
            *args,
            **kwargs
        )

        eta = diffusion_schedule.hyperparameters.eta
        if eta == 0:
            ddim_sigma = 0
            sigma_noise = 0
        else:
            ddim_sigma = eta * torch.sqrt((1 - ddim_alpha_prev) / (1 - ddim_alpha) * (1 - ddim_alpha / ddim_alpha_prev))
            noise = cls.diffuser.degrade(
                diffusion_schedule,
                presampled_middle,
                t,
                *diffusion_args,
                **diffusion_kwargs
            )
            sigma_noise = ddim_sigma * noise
        dir_xt = torch.sqrt(1. - ddim_alpha_prev - ddim_sigma ** 2) * e_t
        presampled_middle = torch.sqrt(ddim_alpha_prev) * pred_x0 + dir_xt + sigma_noise
        
        # if returns_start_prediction:
        #     return presampled_middle, pred_x0
        return presampled_middle

    # @classmethod
    # @torch.no_grad()
    # def sample_step(
    #     cls,
    #     enumeration: int,
    #     length: int,
    #     backbone: nn.Module,
    #     diffusion_schedule: DiffusionProbabilisticModelsSchedule,
    #     input_middle: Tensor,
    #     presampled_middle: Tensor,
    #     timesteps: Iterable[Tensor],
    #     *args,
    #     inpaint_reference: Optional[Tensor] = None,
    #     mask: Optional[Tensor] = None,
    #     variables: Optional[dict] = None,
    #     parameterization: Parameterization,
    #     range_clipper: RangeClipper = RangeClipper(-1, 1),
    #     returns_start_prediction: bool = False,
    #     **kwargs
    # ) -> Tensor:
    #     # p_sample
    #     diffusion_args, diffusion_kwargs = diffusion_schedule.get_diffusion_arguments(*args, **kwargs)
    #     backbone_args, backbone_kwargs = diffusion_schedule.get_backbone_arguments(*args, **kwargs)
        
    #     t, prev_t = timesteps
    #     ddim_alpha = diffusion_schedule.mean_square_product(presampled_middle, t, *diffusion_args, **diffusion_kwargs)
    #     ddim_alpha_prev = diffusion_schedule.mean_square_product(presampled_middle, prev_t, *diffusion_args, **diffusion_kwargs)
        
    #     prediction = backbone(presampled_middle, t, *backbone_args, **backbone_kwargs)
    #     e_t, pred_x0 = parameterization.predict_current_and_start(
    #         prediction,
    #         diffusion_schedule,
    #         presampled_middle,
    #         t,
    #         *diffusion_args,
    #         mean_square_product=ddim_alpha,
    #         **diffusion_kwargs
    #     )
    #     if not range_clipper.is_only_final:
    #         pred_x0 = range_clipper(pred_x0)
        
    #     if diffusion_schedule.eta == 0:
    #         ddim_sigma = 0
    #         sigma_noise = 0
    #     else:
    #         ddim_sigma = diffusion_schedule.eta * torch.sqrt((1 - ddim_alpha_prev) / (1 - ddim_alpha) * (1 - ddim_alpha / ddim_alpha_prev))
    #         noise = cls.diffuser.degrade(
    #             diffusion_schedule,
    #             presampled_middle,
    #             t,
    #             *diffusion_args,
    #             **diffusion_kwargs
    #         )
    #         sigma_noise = ddim_sigma * noise
    #     dir_xt = torch.sqrt(1. - ddim_alpha_prev - ddim_sigma ** 2) * e_t
    #     presampled_middle = torch.sqrt(ddim_alpha_prev) * pred_x0 + dir_xt + sigma_noise
        
    #     if returns_start_prediction:
    #         return presampled_middle, pred_x0
    #     return presampled_middle