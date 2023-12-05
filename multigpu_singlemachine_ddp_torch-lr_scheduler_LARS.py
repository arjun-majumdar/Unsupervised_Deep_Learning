

import os, pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

from ResNet18_swish_torch import ResNet18, init_weights
from LARC import LARC


"""
EXPERIMENTAL: Multi-GPU, Single Machine DDP PyTorch Training

ResNet-18 CNN + CIFAR-10 dataset


Refer-
https://medium.com/codex/a-comprehensive-tutorial-to-pytorch-distributeddataparallel-1f4b42bb1b51

WRITING DISTRIBUTED APPLICATIONS WITH PYTORCH
https://pytorch.org/tutorials/intermediate/dist_tuto.html
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


def prepare_cifar10_dataset(
    rank: int, world_size: int,
    batch_size = 256, pin_memory = False,
    num_workers = 0, path_dataset = "/home/majumdar/Downloads/.data/"
    ) -> DataLoader:

    # Define transformations for training and test sets-
    transform_train = transforms.Compose(
        [
        transforms.RandomCrop(32, padding = 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4914, 0.4822, 0.4465),
            std = (0.0305, 0.0296, 0.0342)),
        ]
    )

    transform_test = transforms.Compose(
        [
        transforms.ToTensor(),
        transforms.Normalize(
            mean = (0.4942, 0.4846, 0.4498),
            std = (0.0304, 0.0295, 0.0342)),
        ]
    )

    train_dataset = torchvision.datasets.CIFAR10(
        root = path_dataset, train = True,
        download = True, transform = transform_train
        )

    test_dataset = torchvision.datasets.CIFAR10(
        root = path_dataset, train = False,
        download = True, transform = transform_test
    )

    train_sampler = DistributedSampler(
        dataset = train_dataset, num_replicas = world_size,
        rank = rank, shuffle = False,
        drop_last = False
    )

    test_sampler = DistributedSampler(
        dataset = test_dataset, num_replicas = world_size,
        rank = rank, shuffle = False,
        drop_last = False
    )

    train_loader = DataLoader(
        dataset = train_dataset, batch_size = batch_size,
        pin_memory = pin_memory, num_workers = num_workers,
        drop_last = False, 	shuffle = False,
        sampler = train_sampler
    )

    test_loader = DataLoader(
        dataset = test_dataset, batch_size = batch_size,
        pin_memory = pin_memory, num_workers = num_workers,
        drop_last = False, 	shuffle = False,
        sampler = test_sampler
    )

    # return train_loader, test_loader, train_dataset, test_dataset, train_sampler, test_sampler
    return train_loader, test_loader, train_dataset, test_dataset


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
        num_epochs = 50, warmup_epochs = 10,
        batch_size = 256
    ):
    # setup process groups-
    setup(rank, world_size)

    # prepare CIFAR-10 dataloader-
    train_loader, test_loader, train_dataset, test_dataset = prepare_cifar10_dataset(
        rank = rank, world_size = world_size,
        batch_size = batch_size, pin_memory = False,
        num_workers = 0, path_dataset = "/home/majumdar/Downloads/.data/"
    )

    train_dataset_size = len(train_dataset)

    # number of steps per epoch-
    num_steps_epoch = int(np.round(train_dataset_size / batch_size))


    # Instantiate model and move to correct device-
    model = ResNet18(beta = 1.0).to(rank)

    # Convert BatchNorm to SyncBatchNorm-
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # Apply weights initialization-
    model.apply(init_weights)

    # Python3 dict to contain training metrics-
    train_history = {}

    # Wrap model with DDP device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DDP(
        module = model, device_ids = [rank],
        output_device = rank,
        # find_unused_parameters = True
        )
    '''
    Warning: find_unused_parameters=True was specified in DDP constructor, but did not find any unused parameters in the forward pass.
    This flag results in an extra traversal of the autograd graph every iteration,  which can adversely affect performance. If your
    model indeed never has any unused parameters in the forward pass, consider turning this flag off. Note that this warning may be a
    false positive if your model has flow control causing later iterations to have unused parameters. (function operator())
    '''


    # Define cost function and optimizer-
    cost_fn = nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(
        # params = model.parameters(), lr = world_size * 0.001,
        params = model.parameters(), lr = 0.0,
        momentum = 0.9, weight_decay = 5e-4
    )

    optimizer = LARC(
        optimizer = optimizer, trust_coefficient = 0.001,
        clip = False
        )


    # Metric to track for saving best params-
    best_test_acc = 50

    # Decay lr in cosine manner unitl 45th epoch-
    scheduler = CosineScheduler(
        max_update = 45, base_lr = 0.6,
        final_lr = 0.01, warmup_steps = 10,
        warmup_begin_lr = 0.0001
    )


    # TRAIN LOOP (for n epochs):
    for epoch in range(1, num_epochs + 1):

        # Train for one epoch-
        running_loss = 0.0
        running_corrects = 0.0

        # Set model to train mode-
        model.train()

        # Inform DistributedSampler about current epoch-
        train_loader.sampler.set_epoch(epoch)

        for batch_idx, (images, labels) in enumerate(train_loader):
            images = images.to(rank)
            labels = labels.to(rank)

            # Get model predictions-
            outputs = model(images)
              
            # Compute loss-
            loss = cost_fn(outputs, labels)

            '''
            Manually 'allreduce' the losses which sums the losses

            To obtain numerically accurate results, you will have to aggregate the
            running losses on the master node by calling 'reduce(ReduceOp.SUM)'
            operation from the 'torch.distributed' package.
            
            Refer-
            https://discuss.pytorch.org/t/way-to-aggregate-loss-in-ddp-training/176929/3
            https://github.com/NVIDIA/DeepLearningExamples/blob/777d174008c365a5d62799a86f135a4f171f620e/PyTorch/Classification/ConvNets/image_classification/utils.py#L117-L123
            https://github.com/NVIDIA/DeepLearningExamples/blob/777d174008c365a5d62799a86f135a4f171f620e/PyTorch/Classification/ConvNets/image_classification/training.py#L206
            '''
            
            # Empty accumulated gradients-
            optimizer.zero_grad()
                
            # Perform backprop-
            loss.backward()
                
            # Update parameters-
            optimizer.step()

            '''
            # LR linear warmup scheduler-
            # warmup is performed for the initial 10 epochs.
            if epoch > 10:
                e = 10
            else:
                e = epoch
            optimizer.param_groups[0]['lr'] = ((e * 0.01) / 10) * world_size
            '''

            # Update LR scheduler & LARS params-
            for param_group in optimizer.param_groups:
                param_group['lr'] = scheduler(epoch) * world_size

            # Compute model's performance statistics-
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects += torch.sum(predicted == labels.data)

        # to globally reduce local metrics across ranks, they should be Tensors-
        running_loss = torch.tensor([running_loss], device = rank)         
        running_corrects = torch.tensor([running_corrects], device = rank)

        if torch.cuda.is_available():
            dist.reduce(tensor = running_loss, dst = 0, op = torch.distributed.ReduceOp.SUM)
            dist.reduce(tensor = running_corrects, dst = 0, op = torch.distributed.ReduceOp.SUM)

        # will log the aggregated metrics only on the 0th GPU. Make sure "train_dataset" is of type
        # Dataset and not DataLoader to get the size of the full dataset and not of the local shard
        if rank == 0:
            train_loss = running_loss / len(train_dataset)
            train_acc = (running_corrects.double() / len(train_dataset)) * 100

            
            # print(f"GPU: {rank}, epoch = {epoch}; train loss = {train_loss.item():.4f} & train accuracy = {train_acc.item():.2f}%")
            print(
                f"{'-' * 90}\n[GPU{rank}] (Train) Epoch {epoch:2d} | batchsize: {batch_size} | Steps: {len(train_loader)} | "
                f"LR: {optimizer.param_groups[0]['lr']:.6f} | Loss: {train_loss.item():.4f} | Acc: {train_acc.item():.2f}%",
                flush = True,
            )
            

        '''
        Your 'running_loss' and dataloader are local to each GPU in the DDP setup. Calling 'len(dataloader)'
        will give you the size of the local shard of the original dataset (size of the full dataset divided
        by the number of GPUs you are using), since the distributed data sampler shards the dataset into equal
        portions across GPUs.
        Ex: len(train_dataset) = 50000 and for world_size = 4, len(local_shard) = 12500.
        
        The contents of 'running_loss' is a local variable in each separate process (rank): to obtain numerically
        accurate results, you will have to aggregate the running losses on the master node by calling
        'reduce(ReduceOp.SUM)' operation from the 'torch.distributed' package.

        After you do that, you can investigate if you need to change the batch size by dividing by the number of GPUs
        or to alter the learning rate, to achieve the same numerical result as when training on a single GPU.

        TLDR; I am fairly confident the models are nearly identical, you simply forgot to reduce (exchange) local metrics
        across the ranks and to use the full dataset length in your accuracy calculation (len(dataloader.__dataset__) or
        len(dataloader) * world_size).

        Refer-
        https://discuss.pytorch.org/t/torch-ddp-multi-gpu-gives-low-accuracy-metric/189430/6
        '''


        # Test model for one epoch:
        
        # Initialize metric for metric computation, for each epoch-
        running_loss_test = 0.0
        running_corrects_test = 0.0
        
        model.eval()
        
        test_loader.sampler.set_epoch(epoch)
    
        # One epoch of training-
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(rank)
            labels = labels.to(rank)
    
            # Get model predictions-
            outputs = model(images)
                  
            # Compute loss-
            loss_test = cost_fn(outputs, labels)
    
            # Compute model's performance statistics on each rank
            running_loss_test += loss_test.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            running_corrects_test += torch.sum(predicted == labels.data)
    
        # to globally reduce local metrics across ranks, they should be Tensors
        running_loss_test = torch.tensor([running_loss_test], device = rank)         
        running_corrects_test = torch.tensor([running_corrects_test], device = rank)
    
        if torch.cuda.is_available():
            dist.reduce(tensor = running_loss_test, dst = 0, op = torch.distributed.ReduceOp.SUM)
            dist.reduce(tensor = running_corrects_test, dst = 0, op = torch.distributed.ReduceOp.SUM)
    
        # log aggregated metrics only on rank 0 GPU. Make sure "test_dataset" is of type Dataset and not
        # DataLoader to get the size of the full dataset and not of the local shard-
        if rank == 0:
            test_loss = running_loss_test / len(test_dataset)
            test_acc = (running_corrects_test.double() / len(test_dataset)) * 100

            train_loss = running_loss / len(train_dataset)
            train_acc = (running_corrects.double() / len(train_dataset)) * 100

            curr_lr = optimizer.param_groups[0]['lr']
            curr_beta = model.module.beta.item()

            train_history[epoch] = {
                'loss': train_loss.item(), 'acc': train_acc.item(),
                'test_loss': test_loss.item(), 'test_acc': test_acc.item(),
                'lr': curr_lr, 'beta': curr_beta
                }
            
            # print(f"GPU: {rank}, epoch = {epoch}; test loss = {test_loss:.4f} & test accuracy = {test_acc:.2f}%")
            print(
                f"{'-' * 90}\n[GPU{rank}] (Test) Epoch {epoch:2d} | batchsize: {batch_size} | Steps: {len(test_loader)} | "
                f"Loss: {test_loss.item():.4f} | Acc: {test_acc.item():.2f}%",
                flush = True,
            )

            # Save 'best' model using test accuracy as metric-
            if test_acc.item() > best_test_acc:
                best_test_acc = test_acc.item()
                print(f"\nsaving model with best test acc = {best_test_acc:.3f}%\n")
                torch.save(model.module.state_dict(), "ResNet18_CIFAR10_LARS_lrscheduler_multigpu-best_testacc.pth")


    # Save training metrics-
    if rank == 0:
        # Save training metrics as Python3 history for later analysis-
        with open("ResNet18_CIFAR10_LARS_lrscheduler_swish_train_history.pkl", "wb") as file:
            pickle.dump(train_history, file)

    cleanup()




if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    # total number of training epochs-
    num_epochs = 50

    # number of linear warmup epochs-
    warmup_epochs = 10

    batch_size = 256

    mp.spawn(
         fn = main,
         args = (world_size, num_epochs, warmup_epochs, batch_size),
         nprocs = world_size
    )

    # CUDA_VISIBLE_DEVICES=0,1,2 python multigpu_ddp_torch.py

    
