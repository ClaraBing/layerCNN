import os
import numpy as np
import cv2
from PIL import Image
import random

import torch
import torch.utils.data as data
# import torchvision.transforms as transforms
import pdb

ROOT = '/home/bingbin/compositionlearning'

class TINDataset(data.Dataset):
  def __init__(self, is_train, transform=None):
    # NOTE: use the val split for testing since test annots are not available
    self.is_train = is_train
    self.split = 'train' if self.is_train else 'val'
    self.img_root = os.path.join(ROOT, 'data/tiny-imagenet-200')
    self.img_path_format = os.path.join(self.img_root, 'train', '{}/images/{}') if self.is_train else os.path.join(self.img_root, 'val/images/{}')
    with open('{}/data/tiny-imagenet-200/{}/{}_annotations.txt'.format(ROOT, self.split, self.split), 'r') as fin:
      lines = [line for line in fin]
      self.img_names = [line.split()[0] for line in lines]
      self.labels = [line.split()[1] for line in lines]

    classes = sorted(set(self.labels))
    self.cls_dict = {cls:i for i,cls in enumerate(classes)}
    self.transform = transform
    
  def __len__(self):
    return len(self.labels)

  def __getitem__(self, idx):
    img_name, label = self.img_names[idx], self.labels[idx]
    img_path = self.img_path_format.format(label, img_name) if self.is_train else self.img_path_format.format(img_name)
    img = cv2.imread(img_path)
    cls = self.cls_dict[label]

    if self.transform is not None:
      img = Image.fromarray(img.astype(np.uint8))
      img = self.transform(img)
    else:
      img = np.transpose(img, (2,0,1)) # (3,w,h)
    return img, cls

