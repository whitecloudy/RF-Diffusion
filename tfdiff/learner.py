import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import _nested_map
import torch.distributed as dist



class tfdiffLoss(nn.Module):
    def __init__(self, w=0.1):
        super().__init__()
        self.w = w

    def forward(self, target, est, target_noise=None, est_noise=None):
        target_fft = torch.fft.fft(target, dim=1) 
        est_fft = torch.fft(est)
        t_loss = self.complex_mse_loss(target, est)
        f_loss = self.complex_mse_loss(target_fft, est_fft)
        n_loss = self.complex_mse_loss(target_noise, est_noise) if (target_noise and est_noise) else 0.
        return (t_loss + f_loss + self.w * n_loss)

    def complex_mse_loss(self, target, est):
        target = torch.view_as_complex(target)
        est = torch.view_as_complex(est)
        return torch.mean(torch.abs(target-est)**2)
        

class tfdiffLearner:
    def __init__(self, log_dir, model_dir, model, dataset, val_dataset, optimizer, params, *args, **kwargs):
        os.makedirs(model_dir, exist_ok=True)
        self.model_dir = model_dir
        self.task_id = params.task_id
        self.early_stop = params.early_stop
        self.log_dir = log_dir
        self.model = model
        self.dataset = dataset
        self.val_dataset = val_dataset
        self.optimizer = optimizer
        self.device = model.device
        self.diffusion = SignalDiffusion(params) if params.signal_diffusion else GaussianDiffusion(params)
        # self.prof = torch.profiler.profile(
        #     schedule=torch.profiler.schedule(skip_first=1, wait=0, warmup=2, active=1, repeat=1),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler(self.log_dir),
        #     with_modules=True, with_flops=True
        # )
        # eeg
        # self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
        #     self.optimizer, 5, gamma=0.5)
        # mimo
        self.lr_scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, 1, gamma=0.5)
        self.params = params
        self.iter = 0
        self.proc_id = None
        self.is_master = True
        self.loss_fn = nn.MSELoss()
        self.summary_writer = None
        self.summary_val_writer = None
        self.jump_or_step = params.jump_or_step

    def target_degrade_data(self, data, t):
        if self.jump_or_step == 'step':
            degrade_data_t_minus_1 = self.diffusion.degrade_fn(
                data, t-1 ,self.task_id)  # degrade data, x_t-1, [B, N, S*A, 2]
            degrade_data = self.diffusion.degrade_step(
                degrade_data_t_minus_1, t ,self.task_id)    # degrade data, x_t, [B, N, S*A, 2]
            
            return degrade_data_t_minus_1, degrade_data
        elif self.jump_or_step == 'jump':
            degrade_data = self.diffusion.degrade_fn(
                data, t ,self.task_id)  # degrade data, x_t, [B, N, S*A, 2]
            return data, degrade_data

    def state_dict(self):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        return {
            'iter': self.iter,
            'model': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in model_state.items()},
            'optimizer': {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in self.optimizer.state_dict().items()},
            'params': dict(self.params),
        }

    def load_state_dict(self, state_dict):
        if hasattr(self.model, 'module') and isinstance(self.model.module, nn.Module):
            self.model.module.load_state_dict(state_dict['model'])
        else:
            self.model.load_state_dict(state_dict['model'])
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.iter = state_dict['iter']

    def save_to_checkpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/{filename}.pt'
        torch.save(self.state_dict(), save_name)
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.islink(link_name):
                os.unlink(link_name)
            os.symlink(save_basename, link_name)

    def save_to_bestpoint(self, filename='weights'):
        save_basename = f'{filename}-{self.iter}.pt'
        save_name = f'{self.model_dir}/{save_basename}'
        link_name = f'{self.model_dir}/best_{filename}.pt'
        if os.name == 'nt':
            torch.save(self.state_dict(), link_name)
        else:
            if os.path.isfile(save_name):
                if os.path.islink(link_name):
                    os.unlink(link_name)
                os.symlink(save_basename, link_name)
            else:
                print("Counldn't find ", save_basename)

    def restore_from_checkpoint(self, filename='weights'):
        try:
            checkpoint = torch.load(f'{self.model_dir}/{filename}.pt')
            self.load_state_dict(checkpoint)
            return True
        except FileNotFoundError:
            return False

    def train(self, max_iter=None, max_epochs=None):
        device = next(self.model.parameters()).device
        # self.prof.start()
        epochs = 0
        min_loss = float(9999.9)
        early_count = 0
        while True:  # epoch
            # Are we stop here?
            if (epochs is not None) and (epochs >= max_epochs):
                print("max epochs init")
                return
            val_loss = self.validation(device)
            if val_loss < min_loss:
                min_loss = val_loss
                early_count = 0
                if self.is_master:
                    self.save_to_bestpoint()
            else:
                early_count += 1
            if (self.early_stop is not None) and (early_count >= self.early_stop):
                print("early stop init")
                return
            
            # We are not stopping here. Keep training        
            for features in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
                self.iter += 1
                if max_iter is not None and self.iter >= max_iter:
                    # self.prof.stop()
                    return
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                loss = self.train_iter(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at iteration {self.iter}.')
                if self.is_master:
                    if self.iter % 50 == 0:
                        self._write_summary(self.iter, features, loss)
                    if self.iter % (len(self.dataset)) == 0:
                        self.save_to_checkpoint()
                # self.prof.step()

            self.lr_scheduler.step()
            epochs += 1

    def validation(self, device):
        with torch.no_grad():
            self.model.eval()
            loss_data = []

            for features in tqdm(self.val_dataset, desc=f'Validate {(self.iter-1) // len(self.dataset)}') if self.is_master else self.val_dataset:
                features = _nested_map(features, lambda x: x.to(
                    device) if isinstance(x, torch.Tensor) else x)
                loss = self.validation_iter(features)
                if torch.isnan(loss).any():
                    raise RuntimeError(
                        f'Detected NaN loss at iteration {self.iter}.')
                loss = loss.mean()
                
                global_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
                dist.all_gather(global_loss, loss)
                global_loss = torch.tensor(global_loss).to(device).mean()
                loss_data.append(global_loss)
            self.model.train()
            loss_mean = torch.tensor(loss_data).mean()

            if self.is_master:
                self._write_val_summary(self.iter, loss_mean)

            return loss_mean.cpu().item()


    def validation_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']  # orignial data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64)
        target_data, degrade_data = self.target_degrade_data(data, t)
        # degrade_data = self.diffusion.degrade_fn(
        #     data, t ,self.task_id)  # degrade data, tx_, [B, N, S*A, 2]
        predicted = self.model(degrade_data, t, cond)
        if self.task_id==3:
            target_data = target_data.reshape(-1,512,1,2)
        loss = self.loss_fn(target_data, predicted)
        return loss


    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']  # orignial data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        t = torch.randint(0, self.diffusion.max_step, [B], dtype=torch.int64)
        target_data, degrade_data = self.target_degrade_data(data, t)
        predicted = self.model(degrade_data, t, cond)
        if self.task_id==3:
            target_data = target_data.reshape(-1,512,1,2)
        loss = self.loss_fn(target_data, predicted)
        loss.backward()
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()
        return loss

    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.flush()
        self.summary_writer = writer

    def _write_val_summary(self, iter, loss):
        # writer = self.summary_val_writer or SummaryWriter(self.log_dir+"/validation", purge_step=iter)
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        writer.add_scalar('validation/loss', loss, iter)
        writer.flush()
        # self.summary_val_writer = writer
        self.summary_writer = writer

