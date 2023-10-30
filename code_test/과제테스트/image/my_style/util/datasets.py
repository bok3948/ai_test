import os

import torch
from torchvision.datasets import ImageFolder
from torchvision import datasets
from torchvision.transforms import ToTensor

def build_dataet(root=None, transform=None, split='train'):
    #root = os.path.join(root, split)
    #dataset = ImageFolder(root, transform)
    if split == 'train':
        dataset = datasets.FashionMNIST(
    root="/mnt/d/data/image/mnist",
    train=True,
    download=False,
    transform=transform,
)
    else:
       dataset = datasets.FashionMNIST(
    root="/mnt/d/data/image/mnist",
    train=False,
    download=False,
    transform=transform,
)

    return dataset
