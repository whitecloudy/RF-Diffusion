import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class SignalDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.task_id = params.task_id
        self.input_dim = self.params.sample_rate # input time-series data length, N
        self.extra_dim = self.params.extra_dim # dimension of each data sample, e.g., [S A 2] for complex-valued CSI
        self.max_step_plus_one = self.params.max_step+1 # maximum diffusion steps
        self.beta = np.array([0.0, ]+self.params.noise_schedule) # \beta, [T]
        self.sigma_weights = torch.sqrt(torch.tensor(self.beta.astype(np.float32))) # \sigma_t, [T]
        self.alpha = torch.tensor((1-self.beta).astype(np.float32)) # \alpha_t [T]
        self.alpha_bar = torch.cumprod(self.alpha, dim=0) # \bar{\alpha_t}, [T]
        self.var_blur = torch.tensor(np.array([0.0, ]+self.params.blur_schedule).astype(np.float32)) # var of blur kernels on the frequency domain for each diffusion step
        self.var_blur_bar = torch.cumsum(self.var_blur, dim=0) # var of blur kernels on the frequency domain, [T]
        self.var_kernel = (self.input_dim / self.var_blur).unsqueeze(1) # var of each G_t, [T, 1]
        self.var_kernel_bar = (self.input_dim / self.var_blur_bar).unsqueeze(1) # var of each \bar{G_t}, [T, 1]
        self.gaussian_kernel = self.get_kernel(self.var_kernel) # G_t, [T, N]
        self.gaussian_kernel_bar = self.get_kernel(self.var_kernel_bar) # \bar{G_t}, [T, N]
        # The weight of original information x_0 in degraded data x_t
        self.gamma_weights = self.gaussian_kernel * torch.sqrt(self.alpha).unsqueeze(-1) # [T, N]
        self.gamma_bar_weights = self.gaussian_kernel_bar * torch.sqrt(self.alpha_bar).unsqueeze(-1) # [T, N]
        # The overall weight of gaussian noise \epsilon in degraded data x_t
        self.sigma_bar_weights = self.get_sigma_bar_weights() # [T, N]

        # for calculate mu_wave and sigma_wave
        mu_wave_t_minus_1_left_right_term_weights = [self.get_mu_wave_t_minus_1_terms(1), ]+ [self.get_mu_wave_t_minus_1_terms(t) for t in range(1, self.max_step_plus_one)]

        self.mu_wave_t_minus_1_left_term_weights = torch.stack([mu_wave_t_minus_1_left for mu_wave_t_minus_1_left, _ in mu_wave_t_minus_1_left_right_term_weights], dim=0) # [T, N]
        self.mu_wave_t_minus_1_right_term_weights = torch.stack([mu_wave_t_minus_1_right for _, mu_wave_t_minus_1_right in mu_wave_t_minus_1_left_right_term_weights], dim=0) # [T, N]
        self.sigma_wave_t_minus_1_weights = torch.stack([self.get_sigma_wave_t_minus_1(1), ]+ [self.get_sigma_wave_t_minus_1(t) for t in range(1, self.max_step_plus_one)])
    
    @torch.no_grad()
    def get_loss_multiplier(self, t):
        t = torch.where(t <= 1, 2, t)
        return 1/(2*(self.sigma_wave_t_minus_1_weights[t]**2))

    @torch.no_grad()
    def get_kernel(self, var_kernel):
        samples = torch.arange(0, self.input_dim) # [N]
        gaussian_kernel = torch.exp(-((samples - self.input_dim // 2)**2) / (2 * var_kernel))
        # gaussian_kernel = torch.exp(-((samples - self.input_dim // 2)**2) / (2 * var_kernel)) / torch.sqrt(2 * torch.pi * var_kernel) # G_t, [T, N]
        # gaussian_kernel = self.input_dim * gaussian_kernel / torch.sum(gaussian_kernel, dim=1, keepdim=True) # Normalized G_t, [T, N]
        return gaussian_kernel
    
    @torch.no_grad()
    def get_sigma_bar_weights(self):
        noise_weights = []
        noise_weight_square = torch.zeros(self.input_dim) # [N]

        for t in range(self.max_step_plus_one):
            noise_weight_square *= (self.gamma_weights[t] ** 2)
            noise_weight_square += (torch.ones(self.input_dim) * self.sigma_weights[t]**2)
            noise_weights.append(torch.sqrt(noise_weight_square).clone().detach())

        # <<OLD>>
        # for t in range(self.max_step_plus_one):
        #     upper_bound = t + 1
        #     # one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0:upper_bound]) # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
        #     one_minus_alpha_sqrt = 1 - self.alpha[0:upper_bound] # \sqrt(1-\bar{\alpha_s}), for s in [1, t], [t]
        #     rev_one_minus_alpha_sqrt = torch.flipud(one_minus_alpha_sqrt) # \sqrt(1-\bar{\alpha_s}), for s in [t, 1], [t]
        #     rev_alpha = torch.flipud(self.alpha[0:upper_bound]) # alpha_s, for s in [t, 1], [t]
        #     # rev_alpha_bar_sqrt = torch.sqrt(torch.cumprod(rev_alpha, dim=0) / rev_alpha[-1]) # \sqrt{\bar{\alpha_t} / \bar{\alpha_s}}, for s in [t, 1], [t]
        #     rev_alpha_bar_sqrt = torch.cumprod(rev_alpha, dim=0) # \sqrt{\bar{\alpha_t} / \bar{\alpha_s}}, for s in [t, 1], [t]
        #     rev_var_blur = torch.flipud(self.var_blur[:upper_bound]) # [t] 
        #     rev_var_blur_bar = torch.cumsum(rev_var_blur, dim=0) # [t]
        #     rev_var_kernel_bar = (self.input_dim / rev_var_blur_bar).unsqueeze(1) # [t, 1]
        #     # rev_kernel_bar = self.get_kernel(rev_var_kernel_bar) # \bar{G_t} / \bar{G_s}, for s in [t, 1], [t, N]
        #     rev_kernel_bar = self.get_kernel(rev_var_kernel_bar)**2 # \bar{G_t} / \bar{G_s}, for s in [t, 1], [t, N]
        #     rev_kernel_bar[0, :] = torch.ones(self.input_dim) 
        #     # noise_weights.append(torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1), rev_one_minus_alpha_sqrt)) # [t, N]
        #     noise_weights.append(torch.sqrt(torch.mv((rev_alpha_bar_sqrt.unsqueeze(-1) * rev_kernel_bar).transpose(0, 1), rev_one_minus_alpha_sqrt))) # [t, N]
        return torch.stack(noise_weights, dim=0) # [T, N] 
    
    @torch.no_grad()
    def get_sigma_bar_weights_stats(self):
        noise_weights = []
        one_minus_alpha_sqrt = torch.sqrt(1 - self.alpha[0])
        for t in range(self.max_step_plus_one):
            noise_weights.append((1 - torch.sqrt(self.alpha_bar[t])*self.gaussian_kernel_bar[t, :]) / (1 - torch.sqrt(self.alpha[0]) * self.gaussian_kernel[0, :]))
        return one_minus_alpha_sqrt * torch.stack(noise_weights, dim=0) # [T, N]    
    
    @torch.no_grad()
    def degrade_step(self, x_t_minux_1, t, task_id):
        if torch.any(t < 1) or torch.any(t >= self.max_step_plus_one):
            raise IndexError("t should be in [1, T-1].")
        device = x_t_minux_1.device
        if task_id in [0, 1, 4]:
            noise_weight = self.sigma_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, 1, 1, 1, 1]
            gamma = self.gamma_weights[t, :].unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, N, 1, 1, 1]
        if task_id in [2, 3]:
            noise_weight = self.sigma_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, N, 1, 1, 1]
            gamma = self.gamma_weights[t, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, N, 1, 1, 1]
        noise =  noise_weight * torch.randn_like(x_t_minux_1, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        x_t = gamma * x_t_minux_1 + noise # [B, N, S, A, 2]
        return x_t

    @torch.no_grad()
    def degrade_fn(self, x_0, t, task_id):
        if torch.any(t < 1) or torch.any(t >= self.max_step_plus_one):
            raise IndexError("t should be in [1, T].")
        device = x_0.device
        if task_id in [0, 1, 4]:
            noise_weight = self.sigma_bar_weights[t, :].unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, N, 1, 1, 1]
            gamma = self.gamma_bar_weights[t, :].unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, N, 1, 1, 1]
        if task_id in [2, 3]:
            noise_weight = self.sigma_bar_weights[t, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, N, 1, 1, 1]
            gamma = self.gamma_bar_weights[t, :].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, N, 1, 1, 1]
        # random seed
        # torch.manual_seed(11)
        noise =  noise_weight * torch.randn_like(x_0, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        x_t = gamma * x_0 + noise # [B, N, S, A, 2]
        return x_t
    
    @torch.no_grad()
    def get_mu_wave_t_minus_1_terms(self, t):
        if torch.any(torch.tensor(t)<1) or torch.any(torch.tensor(t)>=self.max_step_plus_one):
            raise IndexError("t should be in [1, T].")
        gamma_t = self.gamma_weights[t]
        gamma_bar_t_minus_1 = self.gamma_bar_weights[t-1]
        sigma_t = self.sigma_weights[t].unsqueeze(-1)
        sigma_bar_t = self.sigma_bar_weights[t]
        sigma_bar_t_minus_1 = self.sigma_bar_weights[t-1]

        multiplier = 1 / (sigma_bar_t**2)
        left_term_wo_x_t = multiplier * gamma_t * (sigma_bar_t_minus_1**2)
        right_term_wo_x_0 = multiplier * gamma_bar_t_minus_1 * (sigma_t**2)

        return left_term_wo_x_t, right_term_wo_x_0

    
    @torch.no_grad()
    def get_mu_wave_t_minus_1(self, x_0, x_t, t, task_id):
        if torch.any(torch.tensor(t)<1) or torch.any(torch.tensor(t)>=self.max_step_plus_one):
            raise IndexError("t should be in [1, T].")

        device = x_0.device

        left_term_wo_x_t = self.mu_wave_t_minus_1_left_term_weights[t, :]
        right_term_wo_x_0 = self.mu_wave_t_minus_1_right_term_weights[t, :]

        if task_id in [0, 1, 4]:
            left_term_wo_x_t = left_term_wo_x_t.unsqueeze(-1).unsqueeze(-1).to(device)
            right_term_wo_x_0 = right_term_wo_x_0.unsqueeze(-1).unsqueeze(-1).to(device)
        if task_id in [2, 3]:
            left_term_wo_x_t = left_term_wo_x_t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)
            right_term_wo_x_0 = right_term_wo_x_0.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device)

        mu_wave_t_minus_1 = left_term_wo_x_t*x_t + right_term_wo_x_0*x_0

        return mu_wave_t_minus_1
    
    @torch.no_grad()
    def get_sigma_wave_t_minus_1(self, t):
        if torch.any(torch.tensor(t)<1) or torch.any(torch.tensor(t)>=self.max_step_plus_one):
            raise IndexError("t should be in [1, T].")
        sigma_bar_t = self.sigma_bar_weights[t,:]
        sigma_bar_t_minus_1 = self.sigma_bar_weights[t-1,:]
        sigma_t = self.sigma_weights[t]

        sigma_wave_t_minus_1 = sigma_t * (sigma_bar_t_minus_1 / sigma_bar_t)

        return sigma_wave_t_minus_1
        
    
    @torch.inference_mode()
    def sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Add batch dimension.
        # cond = torch.view_as_real(torch.from_numpy(cond['cond']).to(torch.complex64)).unsqueeze(0)
        # cond = cond.unsqueeze(0)
        # Construct a mini-batch.
        # cond = cond.repeat((batch_size, 1, 1, 1, 1))
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        for s in range(self.max_step_plus_one-1, 0, -1): # reverse from T to 1
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = D(\hat{x_0}, s)
                x_s = self.degrade_fn(x_0_hat, t=(s)*torch.ones(batch_size, dtype=torch.int64), task_id = self.task_id) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat
    
    @torch.inference_mode()
    def paper_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Add batch dimension.
        # cond = torch.view_as_real(torch.from_numpy(cond['cond']).to(torch.complex64)).unsqueeze(0)
        # cond = cond.unsqueeze(0)
        # Construct a mini-batch.
        # cond = cond.repeat((batch_size, 1, 1, 1, 1))
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        for s in range(self.max_step_plus_one-1, 0, -1): # reverse from t to 1
            mu_theta = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model

            if s > 0:
                sigma_theta = self.sigma_wave_t_minus_1_weights[s*torch.ones(batch_size, dtype=torch.int64)] # [B, N]
                
                if self.task_id in [2,3]:
                    sigma_theta = sigma_theta.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
                else:
                    sigma_theta = sigma_theta.unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1]
                noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
                
                x_s = mu_theta + sigma_theta * noise # degrade \hat{x_s} to x_{s-1
                
        x_0_hat = mu_theta
        return x_0_hat

    
    @torch.inference_mode()
    def robust_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Add batch dimension.
        # cond = torch.view_as_real(torch.from_numpy(cond['cond']).to(torch.complex64)).unsqueeze(0)
        # Construct a mini-batch.
        # cond = cond.repeat((batch_size, 1, 1, 1, 1))
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        for s in range(self.max_step_plus_one-1, -1, -1): # reverse from t to 0
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = x_s - D(\hat{x_0}, s) + D(\hat{x_0}, s-1)
                x_s = x_s - self.degrade_fn(x_0_hat, t=s*torch.ones(batch_size, dtype=torch.int64),task_id = self.task_id) + self.degrade_fn(x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64),task_id = self.task_id) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat
        
    @torch.inference_mode()
    def fast_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat
    
    @torch.inference_mode()
    def fast_step_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        if self.task_id in [2,3]:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        else:
            inf_weight = (self.sigma_bar_weights[batch_max, :] + self.gamma_bar_weights[batch_max, :]).unsqueeze(-1).unsqueeze(-1).to(device) # [B, N, 1, 1, 1]
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        for t in range(self.max_step_plus_one, 0, -1): # [300, 0)
            batch_max = (t-1)*torch.ones(batch_size, dtype=torch.int64)
            x_s = restore_fn(x_s, batch_max, cond)
        
        x_0_hat = x_s
        return x_0_hat
    
    @torch.inference_mode()
    def native_sampling(self, restore_fn, data, cond, device):
        batch_size = cond.shape[0]
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        x_s = self.degrade_fn(data, batch_max,task_id = self.task_id).to(device)
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat
    
    @torch.inference_mode()
    def native_step_sampling(self, restore_fn, data, cond, device):
        batch_size = cond.shape[0]
        batch_max = (self.max_step_plus_one-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        x_s = self.degrade_fn(data, batch_max,task_id = self.task_id).to(device)
        # Restore data from noise.
        for t in range(self.max_step_plus_one, 0, -1): # [300, 0)
            batch_max = (t-1)*torch.ones(batch_size, dtype=torch.int64)
            x_s = restore_fn(x_s, batch_max, cond)
        
        x_0_hat = x_s
        return x_0_hat


class GaussianDiffusion(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.params = params
        self.input_dim = self.params.sample_rate # input time-series data length, N
        self.extra_dim = self.params.extra_dim # dimension of each data sample, e.g., [S A 2] for complex-valued CSI
        self.max_step = self.params.max_step # maximum diffusion steps
        beta = np.array(self.params.noise_schedule) # \beta, [T]
        alpha = torch.tensor((1-beta).astype(np.float32)) # \alpha_t [T]
        self.alpha_bar = torch.cumprod(alpha, dim=0) # \bar{\alpha_t}, [T]
        # The overall weight of gaussian noise \epsilon in degraded data x_t
        self.noise_weights = torch.sqrt(1 - self.alpha_bar) # \sqrt{1 - \bar{\alpha_t}}, [T]
        self.info_weights = torch.sqrt(self.alpha_bar) # \sqrt{\bar{\alpha_t}}, [T]

    def degrade_fn(self, x_0, t):
        device = x_0.device
        noise_weight = self.noise_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent gaussian noise weights, [B, 1, 1, 1]
        info_weight = self.info_weights[t].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) # equivalent original info weights, [B, 1, 1, 1] 
        noise = noise_weight * torch.randn_like(x_0, dtype=torch.float32, device=device) # [B, N, S, 2]
        # noise =  noise_weight.unsqueeze(-1).unsqueeze(-1) * torch.randn_like(x_0, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        x_t = info_weight * x_0 + noise # [B, N, S, A, 2]
        return x_t

    def sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device) # scalar
        x_s = inf_weight * torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, 2]
        # Restore data from noise.
        for s in range(self.max_step-1, -1, -1): # reverse from t to 0
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = D(\hat{x_0}, s-1)
                x_s = self.degrade_fn(x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64)) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat
    
    def robust_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device) # scalar
        x_s = inf_weight * torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        # Restore data from noise.
        for s in range(self.max_step-1, -1, -1): # reverse from t to 0
            x_0_hat = restore_fn(x_s, s*torch.ones(batch_size, dtype=torch.int64), cond) # resotre \hat{x_0} from x_s using trained tfdiff model
            if s > 0:
                # x_{s-1} = x_s - D(\hat{x_0}, s) + D(\hat{x_0}, s-1)
                x_s = x_s - self.degrade_fn(x_0_hat, t=[s]) + self.degrade_fn(self, x_0_hat, t=(s-1)*torch.ones(batch_size, dtype=torch.int64)) # degrade \hat{x_0} to x_{s-1}
        return x_0_hat

    def fast_sampling(self, restore_fn, cond, device):
        batch_size = cond.shape[0] # B
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        data_dim = [batch_size, self.input_dim] + self.extra_dim + [2]
        noise = torch.randn(data_dim, dtype=torch.float32, device=device) # [B, N, S, A, 2]
        inf_weight = (self.noise_weights[self.max_step-1] + self.info_weights[self.max_step-1]).to(device) # scalar
        x_s = inf_weight * noise # [B, N, S, A, 2]
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat
    
    def native_sampling(self, restore_fn, data, cond, device):
        batch_size = cond.shape[0]
        batch_max = (self.max_step-1)*torch.ones(batch_size, dtype=torch.int64)
        # Generate degraded noise.
        x_s = self.degrade_fn(data, batch_max).to(device)
        # Restore data from noise.
        x_0_hat = restore_fn(x_s, batch_max, cond)
        return x_0_hat


if __name__ == "__main__":
    from params import all_params
    task_id = 2
    params = all_params[task_id]
    device = torch.device('cuda', 0)
    diffusion = SignalDiffusion(params)
    batch = 10
    data = torch.ones((batch, 14, 96, 26, 2), dtype=torch.float32, requires_grad=True)

    print(diffusion.alpha_bar[-1]**0.5)
    print(diffusion.gamma_bar_weights[-1])
    print(diffusion.get_loss_multiplier(torch.tensor([2])))
    print(diffusion.sigma_wave_t_minus_1_weights[1])

    # t = torch.ones(batch, dtype=torch.int64)*200
    # t = 200

    # # print(diffusion.sigma_wave_t_minus_1_weights[t,:])
    # print(diffusion.mu_wave_t_minus_1_left_term_weights[t])
    # print(diffusion.mu_wave_t_minus_1_right_term_weights[t])

    # sigma_theta = diffusion.sigma_wave_t_minus_1_weights[t]
    # print(sigma_theta)
    # # data = data.to(device)

    # with torch.no_grad():
    #     from params import all_params
    #     task_id = 2
    #     params = all_params[task_id]
    #     device = torch.device('cuda', 0)
    #     diffusion = SignalDiffusion(params)
    #     batch = 1000
    #     data = torch.ones((batch, 14, 96, 26, 2), dtype=torch.float32)*3.0
    #     data = data.to(device)

    #     step_degrade_data = data.clone().detach()
    #     step_degrade_data2 = data.clone().detach()

    #     where_to = 100
    #     # step_degrade_data = diffusion.degrade_fn(data, [where_to-2 for j in range(batch)], 4)
    #     # step_degrade_data = diffusion.degrade_step(step_degrade_data, [where_to-1 for j in range(batch)], 4)

    #     # for i in range(where_to):
    #     #     step_degrade_data = diffusion.degrade_step_no_noise(step_degrade_data, torch.tensor([i for j in range(batch)]), task_id)
    #     #     degrade_data = diffusion.degrade_fn_no_noise(data, torch.tensor([i for j in range(batch)]), task_id)

    #     #     print(i, " ",
    #     #           float(torch.max(
    #     #               abs(step_degrade_data-degrade_data)/abs(degrade_data))))
    #     #     # print((abs(step_degrade_data-degrade_data)/abs(step_degrade_data)).shape)
    #     dim = (0,)

    #     max_degrade_std_diff = torch.tensor(0).to(device)
    #     max_step_degrade_std_diff = torch.tensor(0).to(device)
    #     max_degrade_minus_1_std_diff = torch.tensor(0).to(device)
    #     max_diff_diff = torch.tensor(0).to(device)

    #     max_degrade_mean_diff = torch.tensor(0).to(device)
    #     max_step_degrade_mean_diff = torch.tensor(0).to(device)

    #     for i in range(1, where_to):
    #         # step_degrade_data = diffusion.degrade_step(step_degrade_data, torch.tensor([i for j in range(batch)]), task_id)
    #         # step_degrade_data2 = diffusion.degrade_step(step_degrade_data2, torch.tensor([i for j in range(batch)]), task_id)
    #         # degrade_data = diffusion.degrade_fn(data, torch.tensor([i for j in range(batch)]), task_id)

    #         degrade_data = diffusion.gamma_bar_weights[torch.tensor([i for j in range(batch)])].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) * data
    #         step_degrade_data = diffusion.gamma_weights[torch.tensor([i for j in range(batch)])].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) * step_degrade_data
    #         step_degrade_data2 = diffusion.gamma_weights[torch.tensor([i for j in range(batch)])].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).to(device) * step_degrade_data2

    #         degrade_std = torch.std(degrade_data, dim=dim)
    #         step_degrade_std = torch.std(step_degrade_data, dim=dim)
    #         step_degrade_std2 = torch.std(step_degrade_data2, dim=dim)

    #         degrade_mean = torch.mean(degrade_data, dim=dim)
    #         step_degrade_mean = torch.mean(step_degrade_data, dim=dim)
    #         step_degrade_mean2 = torch.mean(step_degrade_data2, dim=dim)

    #         degrade_std_diff = torch.mean(abs(degrade_std - step_degrade_std)/abs(step_degrade_std))
    #         step_degrade_std_diff = torch.mean(abs(step_degrade_std - step_degrade_std2)/abs(step_degrade_std))

    #         degrade_mean_diff = torch.mean(abs(degrade_mean - step_degrade_mean)/abs(step_degrade_mean))
    #         step_degrade_mean_diff = torch.mean(abs(step_degrade_mean - step_degrade_mean2)/abs(step_degrade_mean))

    #         if degrade_mean_diff > max_degrade_std_diff:
    #             max_degrade_std_diff = degrade_mean_diff

    #         if step_degrade_mean_diff > max_step_degrade_std_diff:  
    #             max_step_degrade_std_diff = step_degrade_mean_diff

    #         if abs(degrade_mean_diff - step_degrade_mean_diff) > max_diff_diff:
    #             max_diff_diff = abs(degrade_mean_diff - step_degrade_mean_diff)

    #         # if degrade_mean_diff > max_degrade_mean_diff:
    #         #     max_degrade_mean_diff = degrade_mean_diff
            
    #         # if step_degrade_mean_diff > max_step_degrade_mean_diff:
    #         #     max_step_degrade_mean_diff = step_degrade_mean_diff

    #     print()
    #     print("max jump/step degrade std diff : ",max_degrade_std_diff.cpu())
    #     print("max step/step degrade std diff : ",max_step_degrade_std_diff.cpu())
    #     print("max jump/step - step/step std diff : ",max_diff_diff.cpu())


