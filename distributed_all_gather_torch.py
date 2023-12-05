

"""
torch.distributed.all_gather() example

torch.distributed.all_gather(tensor_list, tensor, group=None, async_op=False)

Gathers tensors from the whole group in a list.
Complex tensors are supported.


Parameters
    • tensor (Tensor) – Tensor to be broadcast from current process.
    
    • tensor_list (list[Tensor]) – Output list. It should contain correctly-sized tensors
    to be used for output of the collective.
    
    • dst (int, optional) – Destination rank (default is 0)
    
    • group (ProcessGroup, optional) – The process group to work on. If None, the default
    process group will be used.
    
    • async_op (bool, optional) – Whether this op should be an async op

Returns
Async work handle, if async_op is set to True. None, if not async_op or if not part of the
group
"""


import os

import torch
from torch import nn

import torch.distributed as dist
import torch.multiprocessing as mp
# from torch.utils.data.distributed import DistributedSampler
# from torch.nn.parallel import DistributedDataParallel as DDP


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


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


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather() operation on the provided
    input tensors.
    
    *** Warning ***: torch.distributed.all_gather() has no gradient!

    Refer-
    https://github.com/facebookresearch/moco/blob/main/moco/builder.py
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, group = None, async_op = False)

    output = torch.cat(tensors_gather, dim = 0)
    return output


def main(
    rank: int, world_size: int,
    num_epochs: int = 20
    ):

    # setup process groups-
    setup(rank, world_size)
    
    # Create tensor local to each GPU/rank-
    tensor = (torch.arange(2, dtype = torch.int64) + (1 + 2 * rank)).to(rank)

    # Gather tensor values from across all GPUs-
    tensor_all_gpus = concat_all_gather(tensor = tensor)
    
    # Print gathered tensor list only on rank-0 GPU-
    if rank == 0:
        print(f"rank 0: all tensor_list:\n{tensor_all_gpus}")

    
    cleanup()




if __name__ == '__main__':
    world_size = torch.cuda.device_count()

    # total number of training epochs-
    num_epochs = 50

    # number of linear warmup epochs-
    # warmup_epochs = 10

    # batch_size = 256

    mp.spawn(
        fn = main,
        # args = (world_size, num_epochs, warmup_epochs, batch_size),
        args = (world_size, num_epochs),
        nprocs = world_size
    )


"""
Examples:-

CUDA_VISIBLE_DEVICES=4,5,6,7 python distributed_gather_torch.py 
rank 0: all tensor_list:
tensor([1, 2, 3, 4, 5, 6, 7, 8], device='cuda:0')

CUDA_VISIBLE_DEVICES=4,5,6 python distributed_gather_torch.py 
rank 0: all tensor_list:
tensor([1, 2, 3, 4, 5, 6], device='cuda:0')

CUDA_VISIBLE_DEVICES=4,5 python distributed_gather_torch.py 
rank 0: all tensor_list:
tensor([1, 2, 3, 4], device='cuda:0')

CUDA_VISIBLE_DEVICES=4 python distributed_gather_torch.py 
rank 0: all tensor_list:
tensor([1, 2], device='cuda:0')
"""

