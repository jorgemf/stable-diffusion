from typing import Tuple

import torch
import math


def cosine_beta_schedule(timesteps: int,
                         ns=0.0002,
                         ds=0.00025) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps) / timesteps
    alphas_cumprod = torch.cos(((t + ns) / (1 + ds)) * (math.pi / 2)) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    betas = torch.clip(betas, 0, 0.999)
    return betas


class GaussianDiffusion:

    def __init__(self,
                 timesteps: int,
                 sampling_timesteps: int,
                 device: torch.device,
                 dtype: torch.dtype):
        self.timesteps = timesteps
        self.sampling_timesteps = sampling_timesteps
        self.betas = cosine_beta_schedule(timesteps)
        self.betas = self.betas.to(dtype).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, 0)
        self.alphas_cumprod_prev = torch.nn.functional.pad(self.alphas_cumprod[:-1], (1, 0),
                                                           value=1.)
        self.betas = self.betas.to(dtype)

        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.sqrt_betas = torch.sqrt(self.betas)
        self.log_beta = torch.log(self.betas)
        self.sqrt_recip_alphcumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recip_alphcumprod_m1 = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.sqrt_alphacumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / \
                                  (1. - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(self.posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1 = self.betas * torch.sqrt(self.alphas_cumprod_prev) / \
                                    (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * self.sqrt_alphas / \
                                    (1. - self.alphas_cumprod)

        self.signal_noise_ratio = self.alphas_cumprod / (1 - self.alphas_cumprod)

    def q_xt_x0(self, image: torch.Tensor,
                t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean = self.gather(self.sqrt_alphas, t) * image
        var = self.gather(self.betas, t)
        return mean, var

    def predict_start_from_noise(self, x_t: torch.Tensor,
                                 noise: torch.Tensor,
                                 t: torch.Tensor, ) -> torch.Tensor:
        return self.gather(self.sqrt_recip_alphcumprod, t) * x_t - \
               self.gather(self.sqrt_recip_alphcumprod_m1, t) * noise

    def predict_noise_from_start(self, x_t: torch.Tensor,
                                 image: torch.Tensor,
                                 t: torch.Tensor, ) -> torch.Tensor:
        return (self.gather(self.sqrt_recip_alphcumprod, t) * x_t - image) / \
               self.gather(self.sqrt_recip_alphcumprod_m1, t)

    def q_posterior(self, image: torch.Tensor,
                    x_t: torch.Tensor,
                    t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = self.gather(self.posterior_mean_coef1, t) * image + \
                         self.gather(self.posterior_mean_coef2, t) * x_t
        return posterior_mean, \
               self.gather(self.posterior_variance, t), \
               self.gather(self.posterior_log_variance_clipped, t)

    def q_sample(self, image: torch.Tensor,
                 noise: torch.Tensor,
                 t: torch.Tensor, ) -> torch.Tensor:
        return self.gather(self.sqrt_alphacumprod, t) * image + \
               self.gather(self.sqrt_one_minus_alphas_cumprod, t) * noise

    def gather(self, tensor, position, shape=[-1, 1, 1, 1]):
        r = torch.gather(tensor, -1, position)
        r = r.reshape(shape)
        return r
