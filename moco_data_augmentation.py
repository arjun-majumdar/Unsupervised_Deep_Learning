

import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageFilter
import random

import torch
# import torchvision.datasets as datasets
import torchvision.transforms as transforms
# from torch.utils.data import Subset
import torch.distributed as dist
# import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import Dataset, DataLoader


# Refer-
# https://github.com/facebookresearch/moco/blob/main/main_moco.py
# https://github.com/facebookresearch/moco/blob/main/moco/loader.py


class GaussianBlur:
    """
    Gaussian blur augmentation in SimCLR
    https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma = [0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


def moco_augmentations(augmentation:str = 'mocov1'):

    normalize = transforms.Normalize(
        mean = [0.485, 0.456, 0.406],
        std = [0.229, 0.224, 0.225]
    )

    if augmentation == 'mocov2':
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            # transforms.RandomResizedCrop(size = 224, scale = (0.2, 1.0)),
            transforms.RandomResizedCrop(size = 96, scale = (0.2, 1.0)),
            transforms.RandomApply(
                [
                    # transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)],
                    transforms.ColorJitter(0.3, 0.3, 0.3, 0.3)
                    ],
                p = 0.8  # not strengthened
                ),
            transforms.RandomGrayscale(p = 0.2),
            transforms.RandomApply(
                [
                    GaussianBlur([0.1, 2.0])
                    ],
                p = 0.5
                ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
            ]
        augmentation = transforms.Compose(augmentation)

    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        """
        torchvision.transforms.RandomResizedCrop-
        scale (tuple of python:float) – Specifies the lower and upper
        bounds for the random area of the crop, before resizing. The
        scale is defined with respect to the area of the original image.
        https://pytorch.org/vision/main/generated/torchvision.transforms.RandomResizedCrop.html
        """
        augmentation = [
            transforms.ToPILImage(),
            # transforms.RandomResizedCrop(size = 224, scale = (0.2, 1.0)),
            transforms.RandomResizedCrop(size = 96, scale = (0.2, 1.0)),
            transforms.RandomGrayscale(p = 0.2),
            # transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.ColorJitter(0.3, 0.3, 0.3, 0.3),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    augmentation = transforms.Compose(augmentation)

    return augmentation


class STL10UnlabeledDataset(Dataset):
    '''
    STL-10 dataset for unlabeled dataset.
    '''
    def __init__(self, path_to_data, transform = None):
        super().__init__()

        self.transform = transform

        # Read unlabeled data as .bin file-
        self.data = np.fromfile(path_to_data + "unlabeled_X.bin", dtype = np.uint8)

        # Reshape images as (C, H, W)-
        self.data = np.reshape(self.data, (-1, 3, 96, 96))

        # Rotate array by 90 degrees in the plane specified by axes-
        self.data = np.rot90(m = self.data, k = 3, axes = (2, 3))
        self.data = np.transpose(a = self.data, axes = (0, 2, 3, 1))


    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        image = self.data[idx].copy()
        # print(len(image))

        if self.transform is not None:
            img1 = self.transform(image)
            img2 = self.transform(image)

            return img1, img2


def get_stl10_dataset(
    # rank: int, world_size: int,
    batch_size = 256, pin_memory = False,
    num_workers = 0,
    path_to_stl10 = "/home/majumdar/Downloads/.data/stl10_binary/",
    augmentation = 'mocov1'
    ) -> DataLoader:

    # Choose MoCo styled data augmentation-
    augmentation = moco_augmentations(augmentation = 'mocov1')

    # Create training unlabeled dataset-
    unlabeled_data = STL10UnlabeledDataset(
        path_to_data = path_to_stl10, transform = augmentation
    )

    # print(f"unlabaeled train dataset size = {len(unlabeled_data)}")
    # unlabaeled train dataset size = 100000
    '''
    train_sampler = DistributedSampler(
        dataset = unlabeled_data, num_replicas = world_size,
        rank = rank, shuffle = False,
        drop_last = False
    )
    '''

    # Create train loader-
    train_loader = DataLoader(
        dataset = unlabeled_data, batch_size = batch_size,
        pin_memory = pin_memory, num_workers = num_workers,
        drop_last = True, shuffle = False,
        #sampler = train_sampler
    )

    return unlabeled_data, train_loader


"""
train_dataset, train_loader = get_stl10_dataset(
    path_to_stl10 = "/home/amajumdar/Downloads/.data/stl10_binary/",
    batch_size = 256, augmentation = 'mocov2'
    )

# Sanity check-
x1, x2 = next(iter(train_loader))

print(f"x1.size: {x1.size()} & x2.size: {x2.size()}")
# x1.size: torch.Size([128, 3, 96, 96]) & x2.size: torch.Size([128, 3, 96, 96])


# De-normalize images for visualization-
# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
invTrans = transforms.Compose(
    [
        transforms.Normalize(
            mean = [0., 0., 0.],
            std = [1 / 0.229, 1 / 0.224, 1 / 0.225]
        ),
        transforms.Normalize(
            mean = [-0.485, -0.456, -0.406],
            std = [1., 1., 1.]
        ),
    ]
)

x1_vis = invTrans(x1)
x2_vis = invTrans(x2)

# For visualization-
x1_vis = torch.permute(x1_vis, (0, 2, 3, 1))
x2_vis = torch.permute(x2_vis, (0, 2, 3, 1))

# x1_vis.size(), x2_vis.size()
# (torch.Size([128, 96, 96, 3]), torch.Size([128, 96, 96, 3]))

# Visualize MoCo-v2 augmented images train images-
plt.figure(figsize = (12, 10))

for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.imshow(x1_vis[i])
    # get current axes-
    ax = plt.gca()

    # hide x-axis-
    ax.get_xaxis().set_visible(False)

    # hide y-axis-
    ax.get_yaxis().set_visible(False)

plt.suptitle("Augmented (x1) STL-10 training images")
plt.show()

# Visualize MoCo-v2 augmented images train images-
plt.figure(figsize = (12, 10))

for i in range(50):
    plt.subplot(10, 5, i + 1)
    plt.imshow(x2_vis[i])
    # get current axes-
    ax = plt.gca()

    # hide x-axis-
    ax.get_xaxis().set_visible(False)

    # hide y-axis-
    ax.get_yaxis().set_visible(False)

plt.suptitle("Augmented (x2) STL-10 training images")
plt.show()


https://github.com/facebookresearch/moco/blob/main/moco/builder.py#L178

"--moco-k",
default=65536,
type=int,
help="queue size; number of negative keys (default: 65536)

"--moco-dim", default=128, type=int, help="feature dimension (default: 128)"

"--moco-m",
default=0.999,
type=float,
help="moco momentum of updating key encoder (default: 0.999)",

"--moco-t", default=0.07, type=float, help="softmax temperature (default: 0.07)"
"""

