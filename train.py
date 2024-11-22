import os

import torch
from torch.cuda import device_count
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel

from argparse import ArgumentParser, BooleanOptionalAction

from tfdiff.params import all_params
from tfdiff.learner import tfdiffLearner
from tfdiff.wifi_model import tfdiff_WiFi
from tfdiff.mimo_model import tfdiff_mimo
from tfdiff.eeg_model import tfdiff_eeg
from tfdiff.fmcw_model import tfdiff_fmcw
from tfdiff.dataset import from_path
import torch.distributed as dist

def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]

def _train_impl(replica_id, model, train_dataset, val_dataset, params):
    opt = torch.optim.AdamW(model.parameters(), lr=params.learning_rate)
    learner = tfdiffLearner(params.log_dir, params.model_dir, model, train_dataset, val_dataset, opt, params)
    learner.proc_id = dist.get_rank()
    learner.is_master = (learner.proc_id == 0)
    if not params.from_start:
        learner.restore_from_checkpoint()
    else:
        if learner.is_master:
            if os.path.exists(params.log_dir):
                import shutil
                shutil.rmtree(params.log_dir)
    learner.train(max_iter=params.max_iter, max_epochs=params.max_epochs)


def train(params):
    train_dataset, val_dataset = from_path(params)
    if params.task_id==0:
        model = tfdiff_eeg(params).cuda()
    elif params.task_id==1:
        model = tfdiff_mimo(params).cuda()
    else:    
        model = tfdiff_WiFi(params).cuda()
    _train_impl(0, model, train_dataset, val_dataset, params)


def train_distributed(replica_id, replica_count, port, params):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group(
        'nccl', rank=replica_id, world_size=replica_count)
    train_dataset, val_dataset = from_path(params, is_distributed=True)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    if params.task_id == 0:
        model = tfdiff_WiFi(params).to(device)
    elif params.task_id == 1:
        model = tfdiff_fmcw(params).to(device)
    elif params.task_id == 2:
        model = tfdiff_mimo(params).to(device)
    elif params.task_id == 3:
        model = tfdiff_eeg(params).to(device)
    elif params.task_id == 4:
        model = tfdiff_WiFi(params).to(device)
    else:    
        raise ValueError("Unexpected task_id.")
    model = DistributedDataParallel(model, device_ids=[replica_id])
    _train_impl(replica_id, model, train_dataset, val_dataset, params)


def log_dir_suffix_name_maker(params):
    name_list = []
    name_list.append(params.random_seed)
    name_list.append(params.jump_or_step)
    return '/'+('-'.join([str(name) for name in name_list]))


def main(args):
    params = all_params[args.task_id]
    if args.batch_size is not None:
        params.batch_size = args.batch_size
    if args.model_dir is not None:
        params.model_dir = args.model_dir
    if args.data_dir is not None:
        params.data_dir = args.data_dir
    if args.log_dir is not None:
        params.log_dir = args.log_dir
    params.max_epochs = args.max_epochs
    if args.early_stop is not None:
        params.early_stop = args.early_stop
    if args.max_iter is not None:
        params.max_iter = args.max_iter
    if args.from_start is not None:
        params.from_start = args.from_start
    if args.random_seed is not None:
        params.random_seed = args.random_seed
    if args.jump_or_step is not None:
        params.jump_or_step = args.jump_or_step

    params.log_dir += log_dir_suffix_name_maker(params)
    params.model_dir += log_dir_suffix_name_maker(params)
    
    torch.manual_seed(args.random_seed)
    replica_count = device_count()
    if replica_count > 1:
        if params.batch_size % replica_count != 0:
            raise ValueError(
                f'Batch size {params.batch_size} is not evenly divisble by # GPUs {replica_count}.')
        params.batch_size = params.batch_size // replica_count
        port = _get_free_port()
        spawn(train_distributed, args=(replica_count, port, params), nprocs=replica_count, join=True)
    else:
        train(params)


# python train.py --task_id [task_id] --model_dir [model_dir] --data_dir [data_dir]
# HF_ENV_NAME=py38-202207 hfai python train.py --task_id [task_id] --model_dir [model_dir] --data_dir [data_dir] --max_iter [iter_num] --batch_size [batch_size] -- -n [node_num] --force
if __name__ == '__main__':
    parser = ArgumentParser(
        description='train (or resume training) a tfdiff model')
    parser.add_argument('--task_id', type=int,
                        help='use case of tfdiff model, 0/1/2/3 for WiFi/FMCW/MIMO/EEG respectively')
    parser.add_argument('--model_dir', default=None,
                        help='directory in which to store model checkpoints and training logs')
    parser.add_argument('--data_dir', default=None, nargs='+',
                        help='space separated list of directories from which to read csi files for training')
    parser.add_argument('--log_dir', default=None)
    parser.add_argument('--max_iter', default=None, type=int,
                        help='maximum number of training iteration')
    parser.add_argument('--max_epochs', default=None, type=int,
                    help='maximum number of training epochs')
    parser.add_argument('--early_stop', default=None, type=int,
                    help='early stopping epoch step')
    parser.add_argument('--batch_size', default=None, type=int)
    parser.add_argument('--random_seed', default=0, type=int)
    parser.add_argument('--jump_or_step', default='jump', type=str)
    parser.add_argument('--from_start', default=False, action=BooleanOptionalAction)
    main(parser.parse_args())
