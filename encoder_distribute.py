# edited from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/distributed.py
import os, sys
# sys.path.append('../')
import math
import time
import subprocess
import argparse
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
# from TTS.utils.generic_utils import load_config, create_experiment_folder
from torch.utils.data.distributed import DistributedSampler
from pathlib import Path

# change name to disable
class DistributedSampler_self(Sampler):
    """
    Non shuffling Distributed Sampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        super(DistributedSampler_self, self).__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= num_gpus
    return rt


def init_distributed(rank, num_gpus, group_name, dist_backend, dist_url):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(
        dist_backend,
        init_method=dist_url,
        world_size=num_gpus,
        rank=rank,
        group_name=group_name)


def apply_gradient_allreduce(module):

    # sync model parameters
    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        if module.needs_reduction:
            module.needs_reduction = False
            # bucketing params based on value types
            buckets = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = type(param.data)
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced, op=dist.ReduceOp.SUM)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(
                        grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):

        def allreduce_hook(*_):
            Variable._execution_engine.queue_callback(allreduce_params)

        if param.requires_grad:
            param.register_hook(allreduce_hook)

    def set_needs_reduction(self, *_):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def main():
    """
    Call train.py as a new process and pass command arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("run_id", type=str, help= \
        "Name for this model instance. If a model state from the same run ID was previously "
        "saved, the training will restart from there. Pass -f to overwrite saved states and "
        "restart from scratch.")
    parser.add_argument("clean_data_root", type=Path, help= \
        "Path to the output directory of encoder_preprocess.py. If you left the default "
        "output directory when preprocessing, it should be <datasets_root>/SV2TTS/encoder/.")
    parser.add_argument("-m", "--models_dir", type=Path, default="encoder/saved_models/", help=\
        "Path to the output directory that will contain the saved model weights, as well as "
        "backups of those weights and plots generated during training.")
    parser.add_argument("-v", "--vis_every", type=int, default=10, help= \
        "Number of steps between updates of the loss and the plots.")
    parser.add_argument("-u", "--umap_every", type=int, default=100, help= \
        "Number of steps between updates of the umap projection. Set to 0 to never update the "
        "projections.")
    parser.add_argument("-s", "--save_every", type=int, default=500, help= \
        "Number of steps between updates of the model on the disk. Set to 0 to never save the "
        "model.")
    parser.add_argument("-b", "--backup_every", type=int, default=7500, help= \
        "Number of steps between backups of the model. Set to 0 to never make backups of the "
        "model.")
    parser.add_argument("-f", "--force_restart", action="store_true", help= \
        "Do not load any saved model.")
    parser.add_argument("--visdom_server", type=str, default="http://localhost")
    parser.add_argument("--no_visdom", action="store_true", help= \
        "Disable visdom.")
    args = parser.parse_args()

    # OUT_PATH = create_experiment_folder(CONFIG.output_path, CONFIG.run_name,
                                        # True)
    # stdout_path = os.path.join(OUT_PATH, "process_stdout/")

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs.")
    group_id = time.strftime("%Y_%m_%d-%H%M%S")

    # set arguments for train.py
    command = ['encoder_train.py']
    command.append(str(args.run_id))
    command.append(str(args.clean_data_root))
    command.append('--models_dir={}'.format(args.models_dir))
    command.append('--vis_every={}'.format(args.vis_every))
    command.append('--umap_every={}'.format(args.umap_every))
    command.append('--save_every={}'.format(args.save_every))
    command.append('--backup_every={}'.format(args.backup_every))
    command.append('--visdom_server={}'.format(args.visdom_server))
    
    if args.force_restart: command.append('-f')
    # if args.no_visdom: command.append('--no_visdom')
    
    command.append('--group_id=group_{}'.format(group_id))
    command.append('')

    # run processes
    processes = []
    for i in range(num_gpus):
        my_env = os.environ.copy()
        my_env["PYTHON_EGG_CACHE"] = "./tmp/tmp{}".format(i)
        command[-1] = '--rank={}'.format(i)
        stdout = None if i == 0 else open(os.devnull, 'w')
        p = subprocess.Popen(['python3.7'] + command, stdout=stdout, env=my_env)
        processes.append(p)
        print(command)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()
