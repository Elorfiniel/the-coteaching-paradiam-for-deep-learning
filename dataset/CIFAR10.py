from dataset.utils import noisify

from torchvision import datasets, transforms
from torch.utils.data import Dataset

import torch as torch
import os.path as osp
import numpy as np


class CIFAR10(Dataset):
  def __init__(self, dataset_root, train=True, transform=None, target_transform=None,
               noise_type='clean', noise_rate=0.2, random_state=0):
    super(CIFAR10, self).__init__()

    self.dataset_root = osp.abspath(dataset_root)
    self.transform = transform
    self.target_transform = target_transform

    self.download_and_process(train, noise_type, noise_rate, random_state)

  def __getitem__(self, index):
    image = self.images[index]
    image = torch.transpose(image, 0, 2)
    image = torch.transpose(image, 1, 2)
    label = self.labels[index]

    if self.transform is not None:
      image = self.transform(image)
    if self.target_transform is not None:
      label = self.target_transform(label)

    return image, label, index

  def __len__(self):
    return len(self.images)

  def download_and_process(self, train, noise_type, noise_rate, random_state):
    # Download MNIST dataset using `torchvision` package
    cifar10_train = datasets.CIFAR10(
      root=self.dataset_root,
      train=True, download=True,
      transform=transforms.ToTensor(),
    )
    cifar10_test = datasets.CIFAR10(
      root=self.dataset_root,
      train=False, download=True,
      transform=transforms.ToTensor(),
    )

    if train and noise_type == 'clean':
      self.images = torch.from_numpy(cifar10_train.data)
      self.labels = torch.tensor(cifar10_train.targets, dtype=torch.long)
      self.noise_rate = 0.0
      self.noise_or_not = self.labels == self.labels

      self.classes = cifar10_train.classes
      self.class_to_idx = cifar10_train.class_to_idx

    if train and noise_type != 'clean':
      self.images = torch.from_numpy(cifar10_train.data)

      clean_labels = np.array(cifar10_train.targets)
      noisy_labels, actual_noise_rate = noisify(
         noise_type=noise_type,
         labels=clean_labels,
         noise_rate=noise_rate,
         n_classes=10,
         random_state=random_state,
      )

      self.labels = torch.tensor(noisy_labels, dtype=torch.long)
      self.noise_rate = actual_noise_rate
      self.noise_or_not = torch.from_numpy(clean_labels == noisy_labels)

      self.classes = cifar10_train.classes
      self.class_to_idx = cifar10_train.class_to_idx

    if not train:
      self.images = torch.from_numpy(cifar10_test.data)
      self.labels = torch.tensor(cifar10_test.targets, dtype=torch.long)
      self.noise_rate = 0.0
      self.noise_or_not = self.labels == self.labels

      self.classes = cifar10_test.classes
      self.class_to_idx = cifar10_test.class_to_idx
