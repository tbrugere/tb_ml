from typing import ClassVar, Literal, TypeAlias
import torch
import warnings

Timestep_sampling_method: TypeAlias = Literal["uniform", "logit_normal"]

PredictionType: TypeAlias = Literal["v", "noise"]

############### functions directly stolen from 
############### Ting Chen - on the importance of noise scheduling for diffusion models
def simple_linear_schedule(t, clip_min=1e-9):
    """A gamma function that simply is 1-t."""
    return torch.clip(1 - t, clip_min, 1.)

def sigmoid_schedule(t, start=-3, end=3, tau=1.0, clip_min=1e-9):
    """A gamma function based on sigmoid function."""
    v_start = sigmoid(torch.as_tensor(start / tau))
    v_end = sigmoid(torch.as_tensor(end / tau))
    output = sigmoid((t * (end - start) + start) / tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.)

def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    """A gamma function based on cosine function"""
    v_start = torch.cos(torch.as_tensor(start * torch.pi / 2)) ** (2 * tau)
    v_end = torch.cos(torch.as_tensor(end * torch.pi / 2)) ** (2 * tau)
    output = torch.cos((t * (end - start) + start) * torch.pi / 2) ** (2 * tau)
    output = (v_end - output) / (v_end - v_start)
    return torch.clip(output, clip_min, 1.)

def betas_from_alpha_bars(alpha_bars, clip= .999):
    betas = 1 - alpha_bars[1:] / alpha_bars[:-1]
    betas = betas.clip(0, clip)
    return betas

class DiffusionTrainingNoiser():
    def apply_noise(self, x, noise, timesteps):
        del x, noise, timesteps
        raise NotImplementedError

class RectifiedFlowNoise(DiffusionTrainingNoiser):
    def apply_noise(self, x, noise, timesteps):
        return (1 - timesteps) * x + timesteps * noise

    def epsilon_to_v(self, x, epsilon):
        """convert epsilon to v"""
        return x - epsilon

class ExplodingNoise(DiffusionTrainingNoiser):
    def _compute_b_t(self, timesteps):
        return torch.exp(timesteps)

    def _apply_noise(self, x, noise, timesteps):
        b_t = self._compute_b_t(timesteps)

        return x + b_t * noise

class DDPMCosineNoise(DiffusionTrainingNoiser):
    def _apply_noise(self, x, noise, timesteps):
        with torch.no_grad():
            sqrt_alpha_t = torch.cos(timesteps * torch.pi / 2)
            sqrt_1_alpha_t = torch.sin(timesteps * torch.pi / 2)
        return sqrt_alpha_t * x + sqrt_1_alpha_t * noise


class DiffusionTrainingHelperMixin():
    """
    Mixin class for diffusion models that provides trainiing utilities
    """

    _diff_train_prediction_type: PredictionType = "noise"
    _diff_train_timestep_sampling_method: Timestep_sampling_method = "uniform"
    _integer_timesteps: ClassVar[bool] = False
    _n_timesteps: int|None = None
    _noiser: DiffusionTrainingNoiser

    

    def __init__(self, *,  prediction_type: PredictionType = "noise", timestep_sampling_method: Timestep_sampling_method = "uniform", timesteps: int|None = None, noise_type: Literal["cosine", "exploding", "rectified_flow"]):
        self._diff_train_prediction_type = prediction_type
        self._diff_train_timestep_sampling_method = timestep_sampling_method
        if not self._integer_timesteps and timesteps is not None:
                raise ValueError("timesteps should be None if the model does not use integer timesteps (assumed to be [0-1])")
        elif self._integer_timesteps and timesteps is None: 
            raise ValueError("timesteps should be provided if the model uses integer timesteps")
        self._n_timesteps = timesteps

        match noise_type:
            case "cosine": self._noiser = DDPMCosineNoise()
            case "exploding": self._noiser = ExplodingNoise()
            case "rectified_flow": self._noiser = RectifiedFlowNoise()
            case _: raise ValueError(f"Unknown noise type {noise_type}")


    def sample_timesteps(self, batch_size: int, device: torch.device):
        """sample timesteps for the training"""
        match self._diff_train_timestep_sampling_method:
            case "uniform" if self._integer_timesteps:
                assert self._n_timesteps is not None
                return torch.randint(0, self._n_timesteps, (batch_size,), device=device)
            case "uniform":
                return torch.rand(batch_size, device=device)
            case "logit_normal":
                t = torch.sigmoid(torch.randn(batch_size, device=device))
                if self._integer_timesteps:
                    assert self._n_timesteps is not None
                    return torch.round(t * self._n_timesteps)
                return t


    def get_training_objective(self, x, noise, noised_x):
        """return the target value for the model. May be the noise, 
        the v value or the x value"""
        match self._diff_train_prediction_type:
            case "noise":
                return noise
            case "v":
                warnings.warn("Using v as training target, not sure if this is correct")
                return x - noised_x
            case _:
                raise ValueError(f"Unknown prediction type {self._diff_train_prediction_type}")

    
    @torch.no_grad()
    def apply_noise(self, x, noise, timesteps):
        """apply noise to the input point"""
        if self._integer_timesteps: torch.is_floating_point(timesteps)
        timesteps = timesteps[..., None].expand_as(x)
        return self._apply_noise(x, noise, timesteps)

    def _apply_noise(self, x, noise, timesteps):
        """apply noise to the input point. Reimplement this method in the child class if needed"""
        return self._noiser.apply_noise(x, noise, timesteps)


        
        
