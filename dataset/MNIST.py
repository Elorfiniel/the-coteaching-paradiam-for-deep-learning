from dataset.utils import noisify

from torchvision import datasets, transforms
from torch.utils.data import Dataset

import torch as torch
import os.path as osp


class MNIST(Dataset):
  def __init__(self, dataset_root, train=True, transform=None, target_transform=None,
               noise_type='clean', noise_rate=0.2, random_state=0):
    super(MNIST, self).__init__()

    self.dataset_root = osp.abspath(dataset_root)
    self.transform = transform
    self.target_transform = target_transform

    self.download_and_process(train, noise_type, noise_rate, random_state)

  def __getitem__(self, index):
    image = torch.unsqueeze(self.images[index], 0)
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
    mnist_train = datasets.MNIST(
      root=self.dataset_root,
      train=True, download=True,
      transform=transforms.ToTensor(),
    )
    mnist_test = datasets.MNIST(
      root=self.dataset_root,
      train=False, download=True,
      transform=transforms.ToTensor(),
    )

    # Process downloaded images and labels
    if train and noise_type == 'clean':
      self.images = mnist_train.data
      self.labels = mnist_train.targets
      self.noise_rate = 0.0
      self.noise_or_not = self.labels == self.labels

    if train and noise_type != 'clean':
      self.images = mnist_train.data

      clean_labels = mnist_train.targets.numpy()
      noisy_labels, actual_noise_rate = noisify(
        noise_type=noise_type,
        labels=clean_labels,
        noise_rate=noise_rate,
        n_classes=10,
        random_state=random_state,
      )

      self.labels = torch.from_numpy(noisy_labels)
      self.noise_rate = actual_noise_rate
      self.noise_or_not = torch.from_numpy(clean_labels == noisy_labels)

    if not train:
      self.images = mnist_test.data
      self.labels = mnist_test.targets
      self.noise_rate = 0.0
      self.noise_or_not = self.labels == self.labels
