

import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mocov2_data_augmentation_multigpu import get_stl10_dataset
from mocov2_architecture_mutligpu import concat_all_gather, MoCo

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from LARC import LARC


"""
PyTorch DDP (multi-GPU) implementation of:
Improved Baselines with Momentum Contrastive Learning by Xinlei Chen et al.

on STL-10 dataset.


Refer-
https://github.com/facebookresearch/moco
"""


def setup(rank: int, world_size: int) -> None:
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(
        "nccl", rank = rank,
        world_size = world_size
    )

    # Make all '.cuda' calls work properly-
    torch.cuda.set_device(rank)
    
    # Synchronize all threads to reach this point before proceeding-
    dist.barrier()
    
    return None


def cleanup() -> None:
    dist.destroy_process_group()
    return None


class CosineScheduler:
    def __init__(
        self, max_update,
        base_lr = 0.01, final_lr = 0,
        warmup_steps = 0, warmup_begin_lr = 0
    ):
        self.base_lr_orig = base_lr
        self.max_update = max_update
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.warmup_begin_lr = warmup_begin_lr
        self.max_steps = self.max_update - self.warmup_steps


    def get_warmup_lr(self, epoch):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(epoch) / float(self.warmup_steps)
        return self.warmup_begin_lr + increase


    def __call__(self, epoch):
        if epoch < self.warmup_steps:
            return self.get_warmup_lr(epoch)
        if epoch <= self.max_update:
            self.base_lr = self.final_lr + (
                self.base_lr_orig - self.final_lr) * (1 + np.cos(
                np.pi * (epoch - self.warmup_steps) / self.max_steps)) / 2
        return self.base_lr




def main(
    rank: int, world_size: int,
    path_to_stl10:str,
    num_epochs:int = 200, warmup_epochs:int = 15,
    batch_size:int = 256, 
    embed_dim:int = 128, K:int = 65536,
    temp:float = 0.07, momentum:float = 0.999,
    data_augmentation:str = 'mocov1'
):
    # Setup process groups-
    setup(rank, world_size)

    # Get STL-10 unlabeled dataset-
    train_dataset, train_loader = get_stl10_dataset(
        rank = rank, world_size = world_size,
        batch_size = batch_size, num_workers = 0,
        pin_memory = False, path_to_stl10 = path_to_stl10,
        augmentation = data_augmentation
    )

    train_dataset_size = len(train_dataset)

    # number of steps per epoch-
    num_steps_epoch = int(np.round(train_dataset_size / batch_size))
    print(f"\nGPU: {rank}; train data len = {train_dataset_size} & # steps per epoch = {num_steps_epoch}")


    # Initialize MoCo instance-
    moco = MoCo(
        # rank = rank,
        embed_dim = embed_dim, K = K,
        m = momentum, T = temp
    ).to(rank)

    moco = torch.nn.parallel.DistributedDataParallel(
        module = moco, device_ids = [rank],
        output_device = rank,
        # find_unused_parameters = True
    )


    # Python3 dict to contain training metrics-
    train_history = {}

    # Metric to track for saving best params-
    best_train_loss = 10

    
    # Define LARS SGD optimizer and Info-NCE cost function-
    optimizer = torch.optim.SGD(
        # params = model.parameters(), lr = world_size * 0.001,
        params = moco.module.encoder_query.parameters(), lr = 0.0,
        momentum = 0.9, weight_decay = 5e-4
    )
    optimizer = LARC(
        optimizer = optimizer, trust_coefficient = 0.001,
        clip = False
    )

    cost_fn = nn.CrossEntropyLoss()

    # Decay lr in cosine manner unitl 195th epoch-
    scheduler = CosineScheduler(
        max_update = 190, base_lr = 0.03 * world_size,
        final_lr = 0.001, warmup_steps = warmup_epochs,
        warmup_begin_lr = 0.0001
    )


    # Decay lr in cosine manner unitl 195th epoch-
    scheduler = CosineScheduler(
        max_update = 190, base_lr = 0.03 * world_size,
        final_lr = 0.001, warmup_steps = warmup_epochs,
        warmup_begin_lr = 0.0001
    )
    
    
    # TRAIN LOOP-
    for epoch in range(1, num_epochs + 1):
        # Train for one epoch-
        running_loss = 0.0

        # Set model to train mode-
        moco.module.encoder_query.train()

        # Inform DistributedSampler about current epoch-
        train_loader.sampler.set_epoch(epoch)

        for images in train_loader:
            images[0] = images[0].to(rank)
            images[1] = images[1].to(rank)
            
            logits, labels = moco(images[0], images[1])
                          
            # Compute loss-
            loss = cost_fn(logits, labels.to(rank))
        
            '''
            Manually 'allreduce' the losses which sums the losses

            To obtain numerically accurate results, you will have to aggregate the
            running losses on the master node by calling 'reduce(ReduceOp.SUM)'
            operation from the 'torch.distributed' package.
            '''
            
            # Empty accumulated gradients-
            optimizer.zero_grad()
                
            # Perform backprop-
            loss.backward()
                
            # Update parameters-
            optimizer.step()

            # Compute model's performance statistics-
            running_loss += loss.item() * images[0].size(0)

        # Update LR scheduler & LARS params-
        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler(epoch) * world_size

        # to globally reduce local metrics across ranks, they should be Tensors-
        running_loss = torch.tensor([running_loss], device = rank)         
        
        if torch.cuda.is_available():
            dist.reduce(tensor = running_loss, dst = 0, op = torch.distributed.ReduceOp.SUM)

        # will log the aggregated metrics only on the 0th GPU. Make sure "train_dataset" is of type
        # Dataset and not DataLoader to get the size of the full dataset and not of the local shard
        if rank == 0:
            train_loss = running_loss / len(train_dataset)
            
            # print(f"GPU: {rank}, epoch = {epoch}; train loss = {train_loss.item():.4f} & train accuracy = {train_acc.item():.2f}%")
            print(
                f"{'-' * 90}\n[GPU{rank}] (Train) Epoch {epoch:2d} | batchsize: {batch_size} | Steps: {len(train_loader)} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {train_loss.item():.4f}",
                flush = True,
            )

            train_history[epoch] = {
                'loss': train_loss.item(),
                'lr': optimizer.param_groups[0]['lr']
                }

            if train_loss < best_train_loss:
                best_train_loss = train_loss.item()
                print(f"Saving model with lowest train loss = {train_loss.item():.4f}\n")        
                torch.save(moco.state_dict(), "mocov2_stl10_best_trainloss.pth")
                torch.save(moco.module.queue, 'mocov2_stl10_queue_best_trainloss.pth')
                torch.save(optimizer.state_dict(), 'mocov2_lars_sgd_optim_best_trainloss.pth')

            # Write to log file-
            with open("mocov2_STL10_log.txt", "a+") as file:
                file.write(f"\nEpoch = {epoch}, loss = {train_loss.item():.4f} & LR = {optimizer.param_groups[0]['lr']:.7f}\n")


    # Save training metrics-
    if rank == 0:
        # Save training metrics as Python3 history for later analysis-
        with open("MoCov2_STL10_LARS_SGD_train_history.pkl", "wb") as file:
            pickle.dump(train_history, file)


    cleanup()




if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    # Training hyper-parameters-
    batch_size = 64
    num_training_epochs = 200
    warmup_epochs = 15
    
    
    # Final embedding shape-
    embed_dim = 128
    
    # Number of negative samples in queue-
    K = 65536
    
    # Temperature (hyper-param)-
    temp = 0.07
    
    # Momentum for updating key encoder's params-
    momentum = 0.999

    # Data augmentation for MoCo-
    data_augmentation = 'mocov2'

    # Path to STL-10 (unlabeled 100k images) dataset-
    path_to_stl10 = "/home/majumdar/Downloads/.data/stl10_binary/"

    mp.spawn(
         fn = main,
         args = (
             world_size, path_to_stl10,
             num_training_epochs, warmup_epochs,
             batch_size, embed_dim,
             K, temp,
             momentum, data_augmentation
         ),
         nprocs = world_size
    )

