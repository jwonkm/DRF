import torch
import numpy as np
from typing import Tuple, Union, Optional, List

def _tensor_size(t):
    return t.size()[1] * t.size()[2] * t.size()[3]

def extract_into_tensor(arr, timesteps, broadcast_shape):
    res = arr[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]

    return res.expand(broadcast_shape)

def tv_loss(x):
    batch_size = x.size()[0]
    h_x = x.size()[2]
    w_x = x.size()[3]
    count_h = _tensor_size(x[:, :, 1:, :])
    count_w = _tensor_size(x[:, :, :, 1:])
    h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, : h_x - 1, :]), 2).sum()
    w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, : w_x - 1]), 2).sum()
    return 2 * (h_tv / count_h + w_tv / count_w) / batch_size

def noise_t2t(scheduler, timestep, timestep_target, x_t, noise=None):
    assert timestep_target >= timestep
    if noise is None:
        noise = torch.randn_like(x_t).to(x_t)
        eps = noise
        
    eps = noise   
    alphas_cumprod = scheduler.alphas_cumprod.to(device=x_t.device, dtype=x_t.dtype)
    
    timestep = timestep.to(torch.long)
    timestep_target = timestep_target.to(torch.long)
    
    alpha_prod_t = alphas_cumprod[timestep]
    alpha_prod_tt = alphas_cumprod[timestep_target]
    alpha_prod = alpha_prod_tt / alpha_prod_t
    
    sqrt_alpha_prod = (alpha_prod ** 0.5).flatten()
    while len(sqrt_alpha_prod.shape) < len(x_t.shape):
        sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)
    
    sqrt_one_minus_alpha_prod = ((1 - alpha_prod) ** 0.5).flatten()
    while len(sqrt_one_minus_alpha_prod.shape) < len(x_t.shape):
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

    x_tt = sqrt_alpha_prod * x_t + sqrt_one_minus_alpha_prod * noise
    return x_tt, eps


# ******************************************************************************************

class DualLoss:

    def __init__(self, t_min, t_max, unet, scale, scheduler, iter_fp, device):
        self.t_min = t_min
        self.t_max = t_max
        self.unet = unet
        self.scale = scale
        self.scheduler = scheduler
        self.device = device
        self.iter_fp = iter_fp


    def noise_input(self, z, eps=None, timestep: Optional[int]= None):
        if timestep is None:
            b = z.shape[0]
            timestep = torch.randint(
                low = self.t_min,
                high = min(self.t_max, 1000) -1,
                size=(b,),
                device=z.device,
                dtype=torch.long
            )

        if eps is None:
            eps = torch.randn_like(z)

        z_t = self.scheduler.add_noise(z, eps, timestep)
        return z_t, eps, timestep
    

    def get_epsilon_prediction(self, z_t, timestep, embedd, timestep_cond, guidance_scale=7.5, cross_attention_kwargs=None, added_cond_kwargs=None):
        
        latent_input = torch.cat([z_t] * 2)
        timestep = timestep.squeeze()
        # timestep = torch.cat([timestep] * 2)
        embedd = embedd.permute(1, 0, 2, 3).reshape(-1, *embedd.shape[2:]) # for concat prompt
        """
        for structure prompt
        : embedd = torch.cat([embedd] * 2)
        """
        # self.unet
        e_t = self.unet(
            latent_input, 
            timestep, 
            encoder_hidden_states = embedd, 
            timestep_cond = timestep_cond, 
            cross_attention_kwargs=cross_attention_kwargs, 
            added_cond_kwargs = added_cond_kwargs,
        ).sample
        
        e_t_uncond, e_t = e_t.chunk(2) # 0.15, e-1 순서
        e_t = e_t_uncond + guidance_scale * (e_t - e_t_uncond)

        assert torch.isfinite(e_t).all()

        return e_t

    def predict_z0(self, model_output, timestep, sample, alpha_prod_t=None):
        if alpha_prod_t is None:
            alphas_cumprod = self.scheduler.alphas_cumprod.to(timestep.device)
            alpha_prod_t = alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod.to(timestep.device)

        alpha_prod_t = alpha_prod_t.to(model_output.dtype)
        beta_prod_t = 1 - alpha_prod_t
        z0_pred = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        return z0_pred


    def update_drf(self, z_src, z_trg, timestep, t_prev, eps, app_embedd, str_embedd, timestep_cond, guidance_scale=7.5, cross_attention_kwargs=None, added_cond_kwargs_1=None, added_cond_kwargs_2=None):
        norm_ = []
        z_trg_prev = z_trg
        k = 5.0
        rho_t = 0.001 

        for i in range(self.iter_fp): 
            with torch.enable_grad():
                eps = eps.requires_grad_()

                z_t_src, _ = noise_t2t(self.scheduler, t_prev, timestep.squeeze(), z_src, eps)
                z_t_trg, _ = noise_t2t(self.scheduler, t_prev, timestep.squeeze(), z_trg, eps)
                # app
                e_src = self.get_epsilon_prediction(z_t_src, timestep, app_embedd, timestep_cond, guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs_1)
                z_0_src_pred = self.predict_z0(e_src, timestep, z_t_src)
                # str
                e_trg = self.get_epsilon_prediction(z_t_trg, timestep, str_embedd, timestep_cond, guidance_scale=guidance_scale, cross_attention_kwargs=cross_attention_kwargs, added_cond_kwargs=added_cond_kwargs_2)
                z_0_trg_pred = self.predict_z0(e_trg, timestep, z_t_trg)

                difference_src = z_src - z_0_src_pred 
                norm_src = torch.linalg.norm(difference_src)
               
                difference_trg = z_trg_prev - z_0_trg_pred 
                norm_trg = torch.linalg.norm(difference_trg)

                ratio = i / float(self.iter_fp - 1) if self.iter_fp > 1 else 1.0
                numerator = torch.exp(torch.Tensor([k * ratio]).to(self.device)) - 1.0
                denominator = torch.exp(torch.Tensor([k]).to(self.device)) - 1.0
                w_iter = numerator / denominator  # 0 ~ 1
  
                dynamic_rho = rho_t * w_iter
                loss_fpr = norm_src + dynamic_rho * norm_trg
                
                norm_grad = torch.autograd.grad(outputs=loss_fpr, inputs=eps)[0]
                eps = eps - norm_grad * self.scale
                eps = eps.detach()
                norm_.append(loss_fpr)

                z_trg_prev = z_0_trg_pred.detach()

        return z_t_src, eps, z_t_trg, norm_ # algorithm 9

    def estimate_eps(self, z_0, z_t, timestep):
        alphas_cumprod = self.scheduler.alphas_cumprod.to(device=z_0.device)
        alphas_cumprod = alphas_cumprod.to(dtype=z_0.dtype)

        sqrt_alpha_prod = alphas_cumprod[timestep] ** 0.5
        sqrt_alpha_prod = sqrt_alpha_prod.flatten()
        while len(sqrt_alpha_prod.shape) < len(z_0.shape):
            sqrt_alpha_prod = sqrt_alpha_prod.unsqueeze(-1)

        sqrt_one_minus_alpha_prod = (1 - alphas_cumprod[timestep]) ** 0.5
        sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.flatten()
        while len(sqrt_one_minus_alpha_prod.shape) < len(z_0.shape):
            sqrt_one_minus_alpha_prod = sqrt_one_minus_alpha_prod.unsqueeze(-1)

        eps = (z_t - sqrt_alpha_prod * z_0) / sqrt_one_minus_alpha_prod
        return eps
