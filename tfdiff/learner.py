import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from tfdiff.diffusion import SignalDiffusion, GaussianDiffusion
from tfdiff.dataset import _nested_map
import torch.distributed as dist

def cal_SNR_MIMO(predict, truth):
    # Recombine the real and imaginary parts to form complex values
    predict_complex = (predict[:,:,:,:, 0] + 1j * predict[:,:,:,:, 1])
    truth_complex = (truth[:,:,:,:, 0] + 1j * truth[:,:,:,:, 1])
    PS = torch.sum(torch.abs(truth_complex)**2, dim=(-1, -2, -3))  # power of signal
    PN = torch.sum(torch.abs(predict_complex - truth_complex)**2, dim=(-1, -2, -3))  # power of noise
    ratio = PS / PN
    return 10 * torch.log10(ratio)


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

    @torch.no_grad()
    def target_degrade_data(self, data, t):
        degrade_data = self.diffusion.degrade_fn(
            data, t ,self.task_id)  # degrade data, x_t, [B, N, S*A, 2]

        if self.jump_or_step == 'step':
            mu_theta = self.diffusion.get_mu_wave_t_minus_1(data, degrade_data, t, self.task_id)
            return mu_theta, degrade_data
        elif self.jump_or_step == 'jump':
            return data, degrade_data

    @torch.no_grad()
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

    # TODO: 이거 압축해서 데이터 저장해야 할까? 용량이 너무 큰데?
    #       테스트 해본 결과로는 224MB가 40MB 수준으로 줄어든다.
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
        
    def end_phase(self, min_loss):
        device = next(self.model.parameters()).device

        val_loss = self.validation(device)
        if val_loss < min_loss:
            min_loss = val_loss
            if self.is_master:
                self.save_to_bestpoint()

        if self.is_master:
            self.save_to_checkpoint()

        print(f'End phase with min loss: {min_loss}')


    def train(self, max_iter=None, max_epochs=None, is_distributed=False):
        device = next(self.model.parameters()).device
        # self.prof.start()
        epochs = 0
        min_loss = float(9999.9)
        early_count = 0
        while True:  # epoch
            # Are we stop here?
            with torch.no_grad():
                if (max_epochs is not None) and (epochs >= max_epochs):
                    self.end_phase(min_loss)
                    print("max epochs init")
                    return
                
                if epochs % 5 == 0:
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
                    
            if is_distributed:
                self.dataset.sampler.set_epoch(epochs)
            # We are not stopping here. Keep training        
            for features in tqdm(self.dataset, desc=f'Epoch {self.iter // len(self.dataset)}') if self.is_master else self.dataset:
                self.iter += 1
                if max_iter is not None and self.iter >= max_iter:
                    self.end_phase(min_loss)
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

            # TODO: Temporary solution for the learning rate scheduler
            self.lr_scheduler.step()
            epochs += 1

    
    def train_iter(self, features):
        self.optimizer.zero_grad()
        data = features['data']  # orignial data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        t = torch.randint(1, self.diffusion.max_step_plus_one, [B], dtype=torch.int64)
        target_data, degrade_data = self.target_degrade_data(data, t)
        predicted = self.model(degrade_data, t, cond)
        if self.task_id==3:
            target_data = target_data.reshape(-1,512,1,2)
        
        # if self.jump_or_step == 'step':
        #     loss_weight = self.diffusion.get_loss_multiplier(t).to(self.device)
        #     loss_weight = (loss_weight**2).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        # else:
        #     loss_weight = torch.tensor(1.0)
        # loss = self.loss_fn(loss_weight * target_data, loss_weight * predicted)
        loss = self.loss_fn(target_data, predicted)
        loss.backward()
        
        self.grad_norm = nn.utils.clip_grad_norm_(
            self.model.parameters(), self.params.max_grad_norm or 1e9)
        self.optimizer.step()

        # with torch.no_grad():
        #     loss = self.loss_fn(target_data, predicted)
        return loss

    @torch.inference_mode()
    def validation(self, device):
        self.model.eval()

        loss_data = []
        SNR_list = []

        for features in tqdm(self.val_dataset, desc=f'Validate {(self.iter-1) // len(self.dataset)}') if self.is_master else self.val_dataset:
            features = _nested_map(features, lambda x: x.to(
                device) if isinstance(x, torch.Tensor) else x)
            loss, SNR = self.validation_iter(features)
            if torch.isnan(loss).any():
                raise RuntimeError(
                    f'Detected NaN loss at iteration {self.iter}.')
            loss = loss.mean()
            SNR = SNR.mean()
            global_loss = [torch.zeros_like(loss) for _ in range(dist.get_world_size())]
            global_SNR = [torch.zeros_like(SNR) for _ in range(dist.get_world_size())]
            dist.all_gather(global_loss, loss)
            dist.all_gather(global_SNR, SNR)
            global_loss = torch.tensor(global_loss).to(device).mean()
            global_SNR = torch.tensor(global_SNR).to(device).mean()
            loss_data.append(global_loss)
            SNR_list.append(global_SNR)
            loss_mean = torch.tensor(loss_data).mean()
            SNR_mean = torch.tensor(SNR_list).mean()
            loss_mean = float(loss_mean.cpu().item())
            if self.is_master:
                self._write_val_summary(self.iter, loss_mean, SNR_mean)
        loss_mean = torch.tensor(loss_data).mean()
        SNR_mean = torch.tensor(SNR_list).mean()
        loss_mean = float(loss_mean.cpu().item())
        if self.is_master:
            self._write_val_summary(self.iter, loss_mean, SNR_mean)
        self.model.train()

        return loss_mean

    @torch.inference_mode()
    def validation_iter(self, features):
        data = features['data']  # orignial data, x_0, [B, N, S*A, 2]
        cond = features['cond']  # cond, c, [B, C]
        B = data.shape[0]
        # random diffusion step, [B]
        device = data.device
        if self.jump_or_step == 'jump':
            predicted = self.diffusion.sampling(self.model, cond, device)
        elif self.jump_or_step == 'step':
            predicted = self.diffusion.paper_sampling(self.model, cond, device)
        if self.task_id==3:
            data = data.reshape(-1,512,1,2)
        loss = self.loss_fn(data, predicted)
        SNR = cal_SNR_MIMO(predicted, data)
        return loss, SNR
    
    @torch.no_grad()
    def _write_summary(self, iter, features, loss):
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        writer.add_scalar('train/loss', loss, iter)
        writer.add_scalar('train/grad_norm', self.grad_norm, iter)
        writer.flush()
        self.summary_writer = writer

    @torch.no_grad()
    def _write_val_summary(self, iter, loss, SNR):
        # writer = self.summary_val_writer or SummaryWriter(self.log_dir+"/validation", purge_step=iter)
        writer = self.summary_writer or SummaryWriter(self.log_dir, purge_step=iter)
        # writer.add_scalars('feature/csi', features['csi'][0].abs(), step)
        # writer.add_image('feature/stft', features['stft'][0].abs(), step)
        writer.add_scalar('validation/loss', loss, iter)
        writer.add_scalar('validation/SNR', SNR, iter)
        writer.flush()
        # self.summary_val_writer = writer
        self.summary_writer = writer

