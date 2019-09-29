# data loaders
# model building blocks
import os
import sys

import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from data_tiny_img_net import TINDataset

ROOT_PATH = '../../'
DATA_PATH = os.path.join(ROOT_PATH, "data")
FASHION_MNIST_PATH = os.path.join(DATA_PATH, "fashionmnist")
CIFAR10_PATH = os.path.join(DATA_PATH, "cifar10")
SVHN_PATH = os.path.join(DATA_PATH, "svhn")


"""
FashionMNIST:
* 28x28 grayscale images
* 60k images for training, 10k for testing
* 10 classes
"""
def get_fashionmnist_train_loader(batch_size, shuffle=True):
    return data.DataLoader(
        datasets.FashionMNIST(FASHION_MNIST_PATH, train=True, download=True,
                       transform= transforms.Compose(
                       [#transforms.RandomCrop(size=[28,28], padding=4),
                        #transforms.RandomHorizontalFlip(),
                        transforms.ToTensor()])),
        batch_size=batch_size, shuffle=shuffle)


def get_fashionmnist_test_loader(batch_size, shuffle=False):
    return data.DataLoader(
        datasets.FashionMNIST(FASHION_MNIST_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)

"""
CIFAR10:
* 32x32 color images
* 50k for training, 10k for testing
* 10 classes
"""
def get_cifar10_train_loader(batch_size, shuffle=True):
    return data.DataLoader(
        datasets.CIFAR10(CIFAR10_PATH, train=True, download=True,
                       transform= transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)


def get_cifar10_test_loader(batch_size, shuffle=False):
    return data.DataLoader(
        datasets.CIFAR10(CIFAR10_PATH, train=False, download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)


"""
SVHN:
* 32x32 coloar images
* 73257 digits for training, 26032 digits for testing, 531131 additional less difficult digits
* 10 classes
"""
def get_svhn_train_loader(batch_size, shuffle=True):
    return data.DataLoader(
        datasets.SVHN(SVHN_PATH, split='train', download=True,
                       transform= transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)


def get_svhn_test_loader(batch_size, shuffle=False):
    return data.DataLoader(
        datasets.SVHN(SVHN_PATH, split='test', download=True,
                       transform=transforms.ToTensor()),
        batch_size=batch_size, shuffle=shuffle)

"""
Tiny ImageNet:
* 64x64 color images
* 100k images for training (500 per cls), 10k for val (50 per cls), 10k for testing (50 per cls; labels not available)
* 200 classes
"""

def get_tinyImgNet_train_loader(batch_size, shuffle=True, transform_type='none', bootstrap=-1):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  if transform_type == 'all':
    train_transform = transforms.Compose(
                        [transforms.RandomResizedCrop(size=64, scale=(0.2, 1), ratio=(0.8, 1.2)),
                         transforms.RandomHorizontalFlip(),
                         transforms.ToTensor(),
                         normalize])
  elif transform_type == 'flip':
    train_transform = transforms.Compose(
                         [transforms.RandomHorizontalFlip(),
                          transforms.ToTensor(),
                          normalize])
  elif transform_type == 'none':
    train_transform = transforms.Compose([transforms.ToTensor(), normalize])
  else:
    raise ValueError("'transform_type' should be 'none', 'flip', or 'all'. Got {}.".format(transform_type))
    
  dset = TINDataset(is_train=True, transform=train_transform)

  if bootstrap > 0:
    # NOTE: when using a sampler, 'shuffle' has to be False.
    sampler = data.RandomSampler(dset, replacement=True, num_samples=int(min(1, bootstrap)*len(dset)))
    return data.DataLoader(dset, batch_size=batch_size, shuffle=False, sampler=sampler) 
  else:
    return data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)
  
def get_tinyImgNet_test_loader(batch_size, shuffle=False):
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
  test_transform = transforms.Compose([transforms.ToTensor(), normalize]) 
  dset = TINDataset(is_train=False, transform=test_transform)
  return data.DataLoader(dset, batch_size=batch_size, shuffle=shuffle)




"""
L1 ball projection
"""

def l1_proj(x, z=1):
    """
    Implementation of L1 ball projection from:

    https://stanford.edu/~jduchi/projects/DuchiShSiCh08.pdf

    inspired from:

    https://gist.github.com/daien/1272551/edd95a6154106f8e28209a1c7964623ef8397246

    :param x: input data
    :param eps: l1 radius

    :return: tensor containing the projection.
    """

    # Computing the l1 norm of v
    v = torch.abs(x)
    v = v.sum(dim=1)

    # Getting the elements to project in the batch
    indexes_b = torch.nonzero(v > z).view(-1)
    x_b = x[indexes_b]
    batch_size_b = x_b.size(0)

    # If all elements are in the l1-ball, return x
    if batch_size_b == 0:
        return x

    # make the projection on l1 ball for elements outside the ball
    view = x_b
    view_size = view.size(1)
    mu = view.abs().sort(1, descending=True)[0]
    vv = torch.arange(view_size).float().to(x.device)
    st = (mu.cumsum(1)-z)/(vv+1)
    u = (mu-st) > 0
    rho = (1-u).cumsum(dim=1).eq(0).sum(1)-1
    theta = st.gather(1, rho.unsqueeze(1))
    proj_x_b = _thresh_by_magnitude(theta, x_b)

    # gather all the projected batch
    proj_x = x
    proj_x[indexes_b] = proj_x_b
    return proj_x


 
def _thresh_by_magnitude(theta, x):
    return torch.relu(torch.abs(x) - theta) * x.sign()
