from encoder.visualizations import Visualizations
from encoder.data_objects import SpeakerVerificationDataLoader, SpeakerVerificationDataset
from encoder.params_model import *
from encoder.model import SpeakerEncoder
from utils.profiler import Profiler
from pathlib import Path

from apex.parallel import DistributedDataParallel as DDP
from apex import amp

import torch
import sys

sys.path.append("../")
# added by wuzijun for multi GPU
from torch.utils.data.distributed import DistributedSampler
from encoder_distribute import (DistributedSampler_self, 
                                apply_gradient_allreduce,
                                init_distributed, reduce_tensor)

# added by wuzijun
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)

def sync(device: torch.device):
    # FIXME
    return 
    # For correct profiling (cuda operations are async)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def train(run_id: str, clean_data_root: Path, models_dir: Path, umap_every: int, save_every: int,
          backup_every: int, vis_every: int, force_restart: bool, visdom_server: str,
          no_visdom: bool, group_id: str, rank: int):
    # Create a dataset and a dataloader
    dataset = SpeakerVerificationDataset(clean_data_root)
    # added by wuzijun for multi GPU
    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(rank, num_gpus, group_id,
                         'nccl', 'tcp://localhost:54321')
    # sampler = DistributedSampler_self(dataset) if num_gpus > 1 else None
    sampler = None
    
    loader = SpeakerVerificationDataLoader(
        dataset,
        speakers_per_batch,
        utterances_per_speaker,
        sampler=sampler,
        pin_memory=True,
        num_workers=20,
    )
    
    # Setup the device on which to run the forward pass and the loss. These can be different, 
    # because the forward pass is faster on the GPU whereas the loss is often (depending on your
    # hyperparameters) faster on the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # FIXME: currently, the gradient is None if loss_device is cuda
    loss_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create the model and the optimizer
    # change all to cuda
    model = SpeakerEncoder(device, loss_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate_init)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O0")
    # Added by wuzijun, try multi GPU train
    # DISTRUBUTED
    if num_gpus > 1:
        model = DDP(model)
     
    init_step = 1
    
    # Configure file path for the model
    state_fpath = models_dir.joinpath(run_id + ".pt")
    backup_dir = models_dir.joinpath(run_id + "_backups")

    # Load any existing model
    if not force_restart:
        if state_fpath.exists():
            print("Found existing model \"%s\", loading it and resuming training." % run_id)
            checkpoint = torch.load(state_fpath, map_location='cpu')
            init_step = checkpoint["step"]
            model.load_state_dict(checkpoint["model_state"])
            optimizer.load_state_dict(checkpoint["optimizer_state"])
            optimizer.param_groups[0]["lr"] = learning_rate_init
        else:
            print("No model \"%s\" found, starting training from scratch." % run_id)
    else:
        print("Starting the training from scratch.")
    model.train()
    
    # Initialize the visualization environment
    # modified by wuzijun for multi GPU
    if rank == 0:
        vis = Visualizations(run_id, vis_every, server=visdom_server, disabled=no_visdom)
        vis.log_dataset(dataset)
        vis.log_params()
        device_name = str(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
        vis.log_implementation({"Device": device_name})
 
    # Training loop
    profiler = Profiler(summarize_every=10, disabled=False)
    for step, speaker_batch in enumerate(loader, init_step):
        profiler.tick("Blocking, waiting for batch (threaded)")
        
        # Forward pass
        inputs = torch.from_numpy(speaker_batch.data).to(device)
        sync(device)
        profiler.tick("Data to %s" % device)
        embeds = model(inputs)
        sync(device)
        profiler.tick("Forward pass")
        embeds_loss = embeds.view((speakers_per_batch, utterances_per_speaker, -1)).to(loss_device)
        
        # modified by wuzijun for multi GPU
        loss, eer = model.module.loss(embeds_loss)
        sync(loss_device)
        profiler.tick("Loss")

        # Backward pass
        model.zero_grad()
        # loss.backward()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        profiler.tick("Backward pass")
        model.module.do_gradient_ops()
        
        optimizer.step()
        profiler.tick("Parameter update")
        
        # added by wuzijun, aggregate losses from processes
        if num_gpus > 1:
            loss = reduce_tensor(loss.data, num_gpus)
        
        # Update visualizations
        # learning_rate = optimizer.param_groups[0]["lr"]
        if rank == 0:
            vis.update(loss.item(), eer, step)
        
        # Draw projections and save them to the backup folder
        # add rank==0 by wuzijun
        if umap_every != 0 and step % umap_every == 0 and rank == 0:
            print("Drawing and saving projections (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            projection_fpath = backup_dir.joinpath("%s_umap_%06d.png" % (run_id, step))
            # modified
            embeds = embeds.detach().cpu().numpy()
            vis.draw_projections(embeds, utterances_per_speaker, step, projection_fpath)
            vis.save()

        # Overwrite the latest version of the model
        if save_every != 0 and step % save_every == 0 and rank == 0:
            print("Saving the model (step %d)" % step)
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, state_fpath)
            
        # Make a backup
        if backup_every != 0 and step % backup_every == 0 and rank == 0:
            print("Making a backup (step %d)" % step)
            backup_dir.mkdir(exist_ok=True)
            backup_fpath = backup_dir.joinpath("%s_bak_%06d.pt" % (run_id, step))
            torch.save({
                "step": step + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
            }, backup_fpath)
            
        profiler.tick("Extras (visualizations, saving)")
        