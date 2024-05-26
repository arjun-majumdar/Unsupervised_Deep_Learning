import random, os
import torch
import numpy as np


'''
PyTorch - seed everything
Refer-
https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
https://discuss.pytorch.org/t/reproducibility-with-all-the-bells-and-whistles/81097/6
https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
'''


# Set seed value-
seed_val = 47

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(worker_id = seed_val)


train_loader = torch.utils.data.DataLoader(
        eval(dset_string)(root='./data', train=True, transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize,
        ]),download=True),

        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True,
        worker_init_fn=seed_worker
        )


'''
According to the pytorch documentation, torch.backends.cudnn.benchmark should
be set to False, not True:

https://pytorch.org/docs/stable/notes/randomness.html#cuda-convolution-benchmarking

Disabling the benchmarking feature with torch.backends.cudnn.benchmark = False
causes cuDNN to deterministically select an algorithm, possibly at the cost of
reduced performance.
However, if you do not need reproducibility across multiple executions of your
application, then performance might improve if the benchmarking feature is
enabled with torch.backends.cudnn.benchmark = True.
'''

def seed_everything(seed: int):   
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
seed_everything(seed = seed_val)


