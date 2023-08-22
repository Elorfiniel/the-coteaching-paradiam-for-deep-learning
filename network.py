import torch as torch
import torch.nn as nn
import torch.nn.functional as F


network_dictionary = {} # All networks defined in this file

def instanciate_network(name, kwargs):
  if not network_dictionary.get(name, False):
    raise RuntimeError(f'No network with name "{name}" exposed, have you decorated the class yet?')
  return network_dictionary[name](**kwargs)


def add_network_to_dictionary(cls_network):
  if hasattr(cls_network, '__name__'):
    cls_name = getattr(cls_network, '__name__')
    network_dictionary[cls_name] = cls_network
  return cls_network


@add_network_to_dictionary
class CnnFromPaper(nn.Module):
  def __init__(self, in_channels=3, n_classes=10, dropout_rate=0.25):
    super(CnnFromPaper, self).__init__()
    self.dropout_rate = dropout_rate

    self.conv_1 = nn.Conv2d(in_channels, 128, kernel_size=3, stride=1, padding=1)
    self.conv_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv_3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
    self.conv_4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
    self.conv_5 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.conv_6 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
    self.conv_7 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0)
    self.conv_8 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0)
    self.conv_9 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0)

    self.logits = nn.Linear(128, n_classes)

    self.bn_1 = nn.BatchNorm2d(128)
    self.bn_2 = nn.BatchNorm2d(128)
    self.bn_3 = nn.BatchNorm2d(128)
    self.bn_4 = nn.BatchNorm2d(256)
    self.bn_5 = nn.BatchNorm2d(256)
    self.bn_6 = nn.BatchNorm2d(256)
    self.bn_7 = nn.BatchNorm2d(512)
    self.bn_8 = nn.BatchNorm2d(256)
    self.bn_9 = nn.BatchNorm2d(128)

  def forward(self, X):
    h = X # (Batch, Channel, Height, Width)

    h = self.bn_1(self.conv_1(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = self.bn_2(self.conv_2(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = self.bn_3(self.conv_3(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = F.max_pool2d(h, kernel_size=2, stride=2)
    h = F.dropout2d(h, p=self.dropout_rate)

    h = self.bn_4(self.conv_4(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = self.bn_5(self.conv_5(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = self.bn_6(self.conv_6(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = F.max_pool2d(h, kernel_size=2, stride=2)
    h = F.dropout2d(h, p=self.dropout_rate)

    h = self.bn_7(self.conv_7(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = self.bn_8(self.conv_8(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = self.bn_9(self.conv_9(h))
    h = F.leaky_relu(h, negative_slope=0.01)
    h = F.avg_pool2d(h, kernel_size=h.shape[2])

    h = h.view(h.size(0), h.size(1))
    logits = self.logits(h)

    return logits

@add_network_to_dictionary
class MyLeNet(nn.Module):
  '''Implementation of LeNet (LeNet-5) with modifications:

  1) Sigmoid activation is replaced with ReLU activaiton.
  2) Average pooling is replaced with max pooling.
  3) The number of neurons in the 2nd FC layer is reduced (84 -> 64).

  Both input shape and output shape are in the (N, C, H, W) format.

  This implementation targets MNIST dataset published by LeCun:

    http://yann.lecun.com/exdb/mnist/'''

  def __init__(self, in_channels=3, n_classes=10):
    super(MyLeNet, self).__init__()

    self.layers = nn.Sequential(
      nn.Conv2d(in_channels, 6, kernel_size=5, stride=1, padding=2),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
      nn.Flatten(),
      nn.Linear(16*5*5, 120),
      nn.ReLU(),
      nn.Linear(120, 64),
      nn.ReLU(),
      nn.Linear(64, n_classes)
    ) # Network Structure

  def forward(self, X: torch.Tensor):
    return self.layers(X)
