from typing import overload, Optional, Iterable, Tuple
from tqdm import tqdm
import torch
from torch import Tensor
from torch import nn
from ..buffers.diffusion_buffer import DiffusionBuffer
from ..diffusion import MetaDiffusion, Diffusion
from ..parameterizations import Parameterization, NoiseParameterization
from ..diffusers.diffuser import Diffuser
from ..range_clipper import RangeClipper


class Sampler(Diffusion, buffer=DiffusionBuffer, diffuser=Diffuser):
    """The sampler is an implementation that contains the reverse process part of
    the diffusion model.
    """
    diffusion_buffer: DiffusionBuffer
    diffuser: Diffuser
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        diffusion_buffer: DiffusionBuffer,
        *,
        parameterization: Parameterization = NoiseParameterization(),
        range_clipper: RangeClipper = RangeClipper(-1, 1)
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
        
        """
        ...
    
    @overload
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        parameterization: Parameterization = NoiseParameterization(),
        range_clipper: RangeClipper = RangeClipper(-1, 1),
        **kwargs
    ):
        """
        Args:
            backbone (nn.Module): The backbone model of the diffusion model.
            *args: The arguments that are used to construct the instance of
                :class:`DiffusionBuffer`.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
            **kwargs: The keyword arguments that are used to construct the instance of
                :class:`DiffusionBuffer`.
        
        """
        ...
    
    def __init__(
        self,
        backbone: nn.Module,
        *args,
        parameterization: Parameterization = NoiseParameterization(),
        range_clipper: RangeClipper = RangeClipper(-1, 1),
        **kwargs
    ):
        super(Diffusion, self).__init__()
        self.backbone = backbone
        self.parameterization = parameterization
        self.range_clipper= range_clipper
        self.diffusion_buffer = MetaDiffusion.get_diffusion_buffer(
            type(self).buffer, *args, error_head=f"{type(self).__name__}.{type(self).__init__.__code__.co_name}()", **kwargs
        )
    
    @classmethod
    def initialize_state(
        cls,
        diffusion_buffer: DiffusionBuffer,
        input_end: Tensor,
        initial_state: Optional[Tensor] = None,
        strength: float = 0.8,
        *args,
        **kwargs
    ) -> Tuple[Tensor, Tuple[Iterable[int]], int]:
        """Initializes the initial state of the sampling process, which skips particular timesteps.

        Args:
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            input_end (Tensor): The state at the final timestep.
            initial_state (Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            *args: The arguments that drive both the diffusion process and the backbone model.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tuple: An initialized state at the particular timestep, given by :param:`strength`, 
                a sequence of timesteps that can be iterated when sampling, and the length of the timesteps iteration.
        
        """
        pass
    
    @classmethod
    @torch.inference_mode()
    def sample_step(
        cls,
        backbone: nn.Module,
        diffusion_buffer: DiffusionBuffer,
        parameterization: Parameterization,
        input_middle: Tensor,
        timesteps: Iterable[Tensor],
        *args,
        range_clipper: RangeClipper = RangeClipper(-1, 1),
        **kwargs
    ) -> Tensor:
        """Samples the input to the given timestep.

        Args:
            backbone (nn.Module): The backbone model which predicts the degradation.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            input_middle (Tensor): The input tensor.
            timesteps (Iterable[Tensor]): The required sampling timestep(s), if more than
                one timestep is given, that means this sampling process need more timesteps
                to calculate the sampling result.
            *args: The arguments that drive both the diffusion process and the backbone model.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result at the given timestep.
        
        """
        # p_sample
        pass
    
    @classmethod
    def extract_main_timestep(cls, timesteps: Iterable[Tensor]) -> Tensor:
        """Extracts the main timestep which is relevant to the :param:`timesteps`
        in the method :attr:`sample_step`.

        Args:
            timesteps (Iterable[Tensor]): The required sampling timestep(s).

        Returns:
            Tensor: The main timestep from the given timesteps sequence.

        """
        return timesteps[0]

    @classmethod
    @torch.inference_mode()
    def sample(
        cls,
        backbone: nn.Module,
        diffusion_buffer: DiffusionBuffer,
        parameterization: Parameterization,
        input_end: Tensor,
        *args,
        range_clipper: RangeClipper = RangeClipper(-1, 1),
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 0.8,
        **kwargs
    ):
        """Samples the degraded input to the predicted input.

        Args:
            backbone (nn.Module): The backbone model which predicts the degradation.
            diffusion_buffer (DiffusionBuffer): The buffer of the diffusion, containing
                mathematical variables.
            parameterization (Parameterization): The parameterization situation of the
                diffusion model.
            input_end (Tensor): The input at the final (or the initial state) step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            initial_state (Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result at the given timestep.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        # p_sample_loop "Algorithm 2."
        x_t, timestep_iters, length = cls.initialize_state(diffusion_buffer, input_end, initial_state, strength, *args, **kwargs)
        
        if inpaint_reference is None:
            if mask is None:
                def get_step_state(x_t): return x_t
            else:
                raise ValueError(f"both 'x_inpaint' and 'mask' should be `None` at the same time or not, but 'x_inpaint' is `None` while 'mask' is not")
        else:
            if mask is None:
                raise ValueError(f"both 'x_inpaint' and 'mask' should be `None` at the same time or not, but 'mask' is `None` while 'x_inpaint' is not")
            else:
                def get_step_state(x_t):
                    diffused_inpainting_x = cls.diffuser.diffuse(
                        diffusion_buffer,
                        inpaint_reference,
                        cls.extract_main_timestep(timesteps),
                    ) # TODO: deterministic forward pass?
                    return diffused_inpainting_x * mask + (1. - mask) * x_t
        
        for timesteps in tqdm(zip(*timestep_iters), leave=False, total=length):
            timesteps = [torch.full((input_end.size(0),), timestep, device=input_end.device, dtype=torch.long) for timestep in timesteps]
            x_t = get_step_state(x_t)
            x_t = cls.sample_step(
                backbone,
                diffusion_buffer,
                parameterization,
                x_t,
                timesteps,
                *args,
                range_clipper=range_clipper,
                **kwargs
            )
        
        return range_clipper(x_t)
    
    @torch.inference_mode()
    def forward(
        self,
        input_end: Tensor,
        *args,
        inpaint_reference: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        initial_state: Optional[Tensor] = None,
        strength: float = 0.8,
        **kwargs
    ):
        """Samples the degraded input to the predicted input.

        Args:
            input_end (Tensor): The input at the final (or the initial state) step.
            *args: The arguments that drive both the diffusion process and the backbone model.
            range_clipper (RangeClipper): The method describing how to deal with the range
                of the sampled result.
            inpaint_reference (Optional[Tensor]): The input that users want to inpaint. If ``None``,
                there will not be inpaint mode.
            mask (Optional[Tensor]): The area(s) of the input that users want to inpaint.
                If ``None``, there will not be inpaint mode.
            initial_state (Tensor or None): The initial state that needs to be diffused to
                a given timestep, pretending that it was denoised at that timestep, leaving
                remaining timesteps to denoise. If ``None``, then the diffusion should start
                from the final timestep, and :param:`strength` should be ``1.0``.
            strength (float): How strong the :param:`initial_state` impacts on the final result.
            **kwargs: The keyword arguments that drive both the diffusion process and the
                backbone model.

        Returns:
            Tensor: A sampled result at the given timestep.
        
        Raises:
            ValueError:
                If only one of :param:`inpaint_reference` and :param:`mask` is ``None``.
        
        """
        return self.sample(
            self.backbone,
            self.diffusion_buffer,
            self.parameterization,
            input_end,
            *args,
            range_clipper=self.range_clipper,
            inpaint_reference=inpaint_reference,
            mask=mask,
            initial_state=initial_state,
            strength=strength,
            **kwargs
        )