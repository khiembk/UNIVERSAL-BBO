import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from tqdm.autonotebook import tqdm
import numpy as np

from model.utils import extract, default
# from model.BrownianBridge.base.modules.diffusionmodules.openaimodel import UNetModel
from model.BrownianBridge.base.modules.encoders.modules import SpatialRescaler
from model.BrownianBridge.base.modules.ours.mlp import MLP
# from model.BrownianBridge.base.modules.ours.unet import UNET_1D

class BrownianBridgeModel(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.model_config = model_config
        # model hyperparameters
        model_params = model_config.BB.params
        self.num_timesteps = model_params.num_timesteps
        self.mt_type = model_params.mt_type
        self.max_var = model_params.max_var if model_params.__contains__("max_var") else 1
        self.eta = model_params.eta if model_params.__contains__("eta") else 1
        self.skip_sample = model_params.skip_sample
        self.sample_type = model_params.sample_type
        self.sample_step = model_params.sample_step
        self.steps = None
        self.register_schedule()

        # loss and objective
        self.loss_type = model_params.loss_type
        self.objective = model_params.objective

        
        self.image_size = model_params.MLPParams.image_size
        self.hidden_size = model_params.MLPParams.hidden_size
        self.condition_key = model_params.MLPParams.condition_key
        self.denoise_fn = MLP(input_dim=self.image_size,
                            index_dim=1,
                            hidden_dim=self.hidden_size,
                            )
        

    def register_schedule(self):
        T = self.num_timesteps

        if self.mt_type == "linear":
            m_min, m_max = 0.001, 0.999
            m_t = np.linspace(m_min, m_max, T)
        elif self.mt_type == "sin":
            m_t = 1.0075 ** np.linspace(0, T, T)
            m_t = m_t / m_t[-1]
            m_t[-1] = 0.999
        else:
            raise NotImplementedError
        m_tminus = np.append(0, m_t[:-1])

        variance_t = 2. * (m_t - m_t ** 2) * self.max_var
        variance_tminus = np.append(0., variance_t[:-1])
        variance_t_tminus = variance_t - variance_tminus * ((1. - m_t) / (1. - m_tminus)) ** 2
        posterior_variance_t = variance_t_tminus * variance_tminus / variance_t

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer('m_t', to_torch(m_t))
        self.register_buffer('m_tminus', to_torch(m_tminus))
        self.register_buffer('variance_t', to_torch(variance_t))
        self.register_buffer('variance_tminus', to_torch(variance_tminus))
        self.register_buffer('variance_t_tminus', to_torch(variance_t_tminus))
        self.register_buffer('posterior_variance_t', to_torch(posterior_variance_t))

        if self.skip_sample:
            if self.sample_type == 'linear':
                midsteps = torch.arange(self.num_timesteps - 1, 1,
                                        step=-((self.num_timesteps - 1) / (self.sample_step - 2))).long()
                self.steps = torch.cat((midsteps, torch.Tensor([1, 0]).long()), dim=0)
            elif self.sample_type == 'cosine':
                steps = np.linspace(start=0, stop=self.num_timesteps, num=self.sample_step + 1)
                steps = (np.cos(steps / self.num_timesteps * np.pi) + 1.) / 2. * self.num_timesteps
                self.steps = torch.from_numpy(steps)
        else:
            self.steps = torch.arange(self.num_timesteps-1, -1, -1)

    def apply(self, weight_init):
        self.denoise_fn.apply(weight_init)
        return self

    def get_parameters(self):
        return self.denoise_fn.parameters()

    def forward(self, x_high, y_high, x_low, y_low):
        
        b, d, device, img_size, = *x_high.shape, x_high.device, self.image_size
        # assert d == img_size, f'dimension of data must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(x_high, y_high, x_low, y_low, t)

    def p_losses(self, x_high, y_high, x_low, y_low, t, noise=None):
        """
        model loss
        :param x0: encoded x_ori, E(x_ori) = x0
        :param y: encoded y_ori, E(y_ori) = y
        :param y_ori: original source domain image
        :param t: timestep
        :param noise: Standard Gaussian Noise
        :return: loss
        """
        b, d = x_high.shape
        noise = default(noise, lambda: torch.randn_like(x_high))

        x_t, objective = self.q_sample(x_high, x_low, t, noise)
        #import pdb ; pdb.set_trace()
        objective_recon = self.denoise_fn(x_t, t, y_high, y_low)
        objective_recon = objective_recon.reshape(objective_recon.shape[0],-1)
        #print(objective.shape, objective_recon.shape)
        if self.loss_type == 'l1':
            recloss = (objective - objective_recon).abs().mean()
        elif self.loss_type == 'l2':
            recloss = F.mse_loss(objective, objective_recon)
        else:
            raise NotImplementedError()

        x0_recon = self.predict_x0_from_objective(x_t, y_high, t, objective_recon)
        log_dict = {
            "loss": recloss,
            "x0_recon": x0_recon
        }
        return recloss, log_dict

    def q_sample(self, x_high, x_low, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_high))
        m_t = extract(self.m_t, t, x_high.shape)
        var_t = extract(self.variance_t, t, x_high.shape)
        sigma_t = torch.sqrt(var_t)

        if self.objective == 'grad':
            objective = m_t * (x_low - x_high) + sigma_t * noise
        elif self.objective == 'noise':
            objective = noise
        elif self.objective == 'ysubx':
            objective = x_low - x_high
        else:
            raise NotImplementedError()

        return (
            (1. - m_t) * x_high + m_t * x_low + sigma_t * noise,
            objective
        )

    def predict_x0_from_objective(self, x_t, x_low, t, objective_recon):
        if self.objective == 'grad':
            x0_recon = x_t - objective_recon
        elif self.objective == 'noise':
            m_t = extract(self.m_t, t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            sigma_t = torch.sqrt(var_t)
            x0_recon = (x_t - m_t * x_low - sigma_t * objective_recon) / (1. - m_t)
        elif self.objective == 'ysubx':
            x0_recon = x_low - objective_recon
        else:
            raise NotImplementedError
        return x0_recon

    @torch.no_grad()
    def q_sample_loop(self, x0, y):
        imgs = [x0]
        for i in tqdm(range(self.num_timesteps), desc='q sampling loop', total=self.num_timesteps):
            t = torch.full((y.shape[0],), i, device=x0.device, dtype=torch.long)
            img, _ = self.q_sample(x0, y, t)
            imgs.append(img)
        return imgs

    @torch.no_grad()
    def p_sample(self, x_t, x_low, y_low, y_high, i, clip_denoised=False, classifier_free_guidance_weight=0.):
        b, *_, device = *x_t.shape, x_t.device
        if self.steps[i] == 0:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            objective_recon = (1+classifier_free_guidance_weight)*self.denoise_fn(x_t, t, y_low, y_high) - classifier_free_guidance_weight*self.denoise_fn(x_t, t, torch.zeros_like(y_low), torch.zeros_like(y_high))
            x0_recon = self.predict_x0_from_objective(x_t, x_low, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)
            return x0_recon, x0_recon
        else:
            t = torch.full((x_t.shape[0],), self.steps[i], device=x_t.device, dtype=torch.long)
            n_t = torch.full((x_t.shape[0],), self.steps[i+1], device=x_t.device, dtype=torch.long)

            objective_recon = (1+classifier_free_guidance_weight)*self.denoise_fn(x_t, t, y_low, y_high) - classifier_free_guidance_weight*self.denoise_fn(x_t, t, torch.zeros_like(y_low), torch.zeros_like(y_high))
            x0_recon = self.predict_x0_from_objective(x_t, x_low, t, objective_recon=objective_recon)
            if clip_denoised:
                x0_recon.clamp_(-1., 1.)

            m_t = extract(self.m_t, t, x_t.shape)
            m_nt = extract(self.m_t, n_t, x_t.shape)
            var_t = extract(self.variance_t, t, x_t.shape)
            var_nt = extract(self.variance_t, n_t, x_t.shape)
            sigma2_t = (var_t - var_nt * (1. - m_t) ** 2 / (1. - m_nt) ** 2) * var_nt / var_t
            sigma_t = torch.sqrt(sigma2_t) * self.eta

            noise = torch.randn_like(x_t)
            x_tminus_mean = (1. - m_nt) * x0_recon + m_nt * x_low + torch.sqrt((var_nt - sigma2_t) / var_t) * \
                            (x_t - (1. - m_t) * x0_recon - m_t * x_low)

            return x_tminus_mean + sigma_t * noise, x0_recon

    @torch.no_grad()
    def p_sample_loop(self, x_low, y_low, y_high, clip_denoised=True, sample_mid_step=False, classifier_free_guidance_weight=0.):
        
        img = x_low
        for i in tqdm(range(len(self.steps)), desc=f'sampling loop time step', total=len(self.steps)):
            img, _ = self.p_sample(x_t=img, x_low=x_low, y_low=y_low, y_high=y_high, i=i, clip_denoised=clip_denoised, classifier_free_guidance_weight=classifier_free_guidance_weight)
        return img

    @torch.no_grad()
    def sample(self, x_low, y_low, y_high, clip_denoised=True, sample_mid_step=False, classifier_free_guidance_weight=0.):
        return self.p_sample_loop(x_low, y_low, y_high, clip_denoised, sample_mid_step, classifier_free_guidance_weight=classifier_free_guidance_weight)