from dataset.MNIST import MNIST
from dataset.CIFAR10 import CIFAR10
from dataset.CIFAR100 import CIFAR100
from dataset.utils import ToFloatTensor
from network import instanciate_network
from loss_fn import loss_coteaching, loss_primitive

from argparse import ArgumentParser, Namespace
from datetime import datetime
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import torch as torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os, json


def load_noisy_dataset(dataset, noise_type, noise_rate, random_state):
  assert dataset in ['MNIST', 'CIFAR10', 'CIFAR100']

  if dataset == 'MNIST':
    dataset_train = MNIST('download', train=True, noise_type=noise_type,
                          noise_rate=noise_rate, random_state=random_state,
                          transform=ToFloatTensor())
    dataset_test = MNIST('download', train=False, transform=ToFloatTensor())
    network_kwargs = {'in_channels': 1, 'n_classes': 10}

  if dataset == 'CIFAR10':
    dataset_train = CIFAR10('download', train=True, noise_type=noise_type,
                            noise_rate=noise_rate, random_state=random_state,
                            transform=ToFloatTensor())
    dataset_test = CIFAR10('download', train=False, transform=ToFloatTensor())
    network_kwargs = {'in_channels': 3, 'n_classes': 10}

  if dataset == 'CIFAR100':
    dataset_train = CIFAR100('download', train=True, noise_type=noise_type,
                             noise_rate=noise_rate, random_state=random_state,
                             transform=ToFloatTensor())
    dataset_test = CIFAR100('download', train=False, transform=ToFloatTensor())
    network_kwargs = {'in_channels': 3, 'n_classes': 100}

  return dataset_train, dataset_test, network_kwargs

def adjust_adam_parameters(optimizer, lr, betas_0):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    param_group['betas'] = (betas_0, 0.999)

def configs_from_cmdargs(cmdargs: Namespace):
  configurations = dict(cmdargs._get_kwargs())
  configurations['time'] = datetime.now().strftime('%d/%m/%y %H:%M %a')
  configurations['pwd'] = os.getcwd()
  return configurations

def save_ckpt_and_configs(logging_root, configs, models):
  subdirectory = osp.join(logging_root, 'models')
  if not osp.isdir(subdirectory): os.makedirs(subdirectory)

  for model_name, model in models.items():
    model_path = osp.join(subdirectory, model_name)
    torch.save(model.state_dict(), model_path)

  configs_path = osp.join(subdirectory, 'configs.json')
  with open(configs_path, 'w') as config_file:
    config_file.write(json.dumps(configs, indent=2))


def accuracy(logits, targets, topk=(1, 5)):
  outputs = F.softmax(logits, dim=1)
  _, preds = outputs.topk(max(topk), dim=1)
  results = preds.T == targets

  topk_accuracies = []
  for k in topk:
    correct_k = torch.sum(results[:k], dim=0).sum()
    topk_accuracies.append(100.0 * correct_k / len(targets))

  return topk_accuracies


def train_primitive(dataloader, model, optimizer, ctx):
  def _check_context_dictionary(ctx_dict):
    assert 'writer' in ctx_dict and isinstance(ctx_dict['writer'], SummaryWriter)
    assert 'epoch_idx' in ctx_dict, 'index of the current epoch (staring from 0)'
    assert 'global_step' in ctx_dict, 'total number of batches trained so far'

    return True

  if not _check_context_dictionary(ctx):
    raise RuntimeError(f'Unexpected keyword arguments detected, please see log for details.')

  correctness = 0
  n_total_batch = 0

  for bdx, (images, labels, indices) in enumerate(dataloader):
    logits = model(images)
    prec, _ = accuracy(logits, labels, (1, 5))
    n_total_batch += 1
    correctness += prec

    loss = loss_primitive(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    ctx['global_step'] += 1

  train_accuracy = correctness / n_total_batch

  return train_accuracy

def eval_primitive(dataloader, model):
  model.eval()  # Set as "Eval Mode"
  n_correct, n_total = 0, 0

  for images, labels, _ in dataloader:
    logits = model(images)
    outputs = F.softmax(logits, dim=1)
    _, preds = torch.max(outputs, dim=1)
    n_correct += torch.sum(preds == labels)
    n_total += len(images)

  return 100.0 * n_correct / n_total

def invoke_primitive_paradiam(cmdargs, network_kwargs, loader_train, loader_test, writer):
  # Instanciate model used for training
  model = instanciate_network(cmdargs.network, network_kwargs)
  optimizer = optim.Adam(model.parameters(), lr=cmdargs.learning_rate)

  # Configure schedulers for hyper-parameters
  learning_rates = np.ones(cmdargs.n_epochs) * cmdargs.learning_rate
  betas_0s = np.ones(cmdargs.n_epochs) * 0.9
  for edx in range(cmdargs.decay_start, cmdargs.n_epochs):
    learning_rates[edx] = (cmdargs.n_epochs - edx) / (cmdargs.n_epochs - cmdargs.decay_start) * cmdargs.learning_rate
    betas_0s[edx] = 0.1

  # Create context and start training
  ctx = {'writer': writer, 'epoch_idx': 0, 'global_step': 0}

  train_accuracy, test_accuracy = 0, 0

  test_accuracy = eval_primitive(loader_test, model)
  writer.add_scalars('dashboard', {
    'epoch': 0,
    'train_acc': train_accuracy,
    'test_acc': test_accuracy,
  }, global_step=ctx['global_step'])

  for epoch_idx in range(0, cmdargs.n_epochs):
    model.train() # Set as "Train Mode"
    adjust_adam_parameters(optimizer, learning_rates[epoch_idx], betas_0s[epoch_idx])

    train_accuracy = train_primitive(loader_train, model, optimizer, ctx)
    ctx['epoch_idx'] += 1

    test_accuracy = eval_primitive(loader_test, model)
    writer.add_scalars('dashboard', {
      'epoch': ctx['epoch_idx'],
      'train_acc': train_accuracy,
      'test_acc': test_accuracy,
    }, global_step=ctx['global_step'])

  return model


def train_coteaching(dataloader, model_1, optimizer_1, model_2, optimizer_2, forget_rates, ctx):
  def _check_context_dictionary(ctx_dict):
    assert 'writer' in ctx_dict and isinstance(ctx_dict['writer'], SummaryWriter)
    assert 'epoch_idx' in ctx_dict, 'index of the current epoch (staring from 0)'
    assert 'global_step' in ctx_dict, 'total number of batches trained so far'

    return True

  if not _check_context_dictionary(ctx):
    raise RuntimeError(f'Unexpected keyword arguments detected, please see log for details.')

  correctness_1, correctness_2 = 0, 0
  n_total_batch_1, n_total_batch_2 = 0, 0
  pure_ratio_1_list, pure_ratio_2_list = [], []

  noise_or_not = dataloader.dataset.noise_or_not

  for bdx, (images, labels, indices) in enumerate(dataloader):
    logits_1 = model_1(images)
    prec_1, _ = accuracy(logits_1, labels, topk=(1, 5))
    n_total_batch_1 += 1
    correctness_1 += prec_1

    logits_2 = model_2(images)
    prec_2, _ = accuracy(logits_2, labels, topk=(1, 5))
    n_total_batch_2 += 1
    correctness_2 += prec_2

    forget_rate = forget_rates[ctx['epoch_idx']]
    loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(
      logits_1, logits_2, labels, forget_rate, indices, noise_or_not)

    pure_ratio_1_list.append(pure_ratio_1 * 100.0)
    pure_ratio_2_list.append(pure_ratio_2 * 100.0)

    optimizer_1.zero_grad()
    loss_1.backward()
    optimizer_1.step()
    optimizer_2.zero_grad()
    loss_2.backward()
    optimizer_2.step()

    ctx['global_step'] += 1

  accuracy_1 = correctness_1 / n_total_batch_1
  accuracy_2 = correctness_2 / n_total_batch_2

  pure_ratio_1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
  pure_ratio_2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

  return accuracy_1, accuracy_2, pure_ratio_1, pure_ratio_2

def eval_coteaching(dataloader, model_1, model_2):
  accuracies = [] # acc_1, acc_2

  for model in [model_1, model_2]:
    model.eval()  # Set as "Eval Mode"
    n_correct, n_total = 0, 0

    for images, labels, _ in dataloader:
      logits = model(images)
      outputs = F.softmax(logits, dim=1)
      _, preds = torch.max(outputs, dim=1)
      n_correct += torch.sum(preds == labels)
      n_total += len(images)

    accuracies.append(100.0 * n_correct / n_total)

  return accuracies

def invoke_coteaching_paradiam(cmdargs, network_kwargs, loader_train, loader_test, writer):
  # Instanciate models with differenct initial parameters
  model_1 = instanciate_network(cmdargs.network, network_kwargs)
  optimizer_1 = optim.Adam(model_1.parameters(), lr=cmdargs.learning_rate)
  model_2 = instanciate_network(cmdargs.network, network_kwargs)
  optimizer_2 = optim.Adam(model_2.parameters(), lr=cmdargs.learning_rate)

  # Configure schedulers for hyper-parameters
  learning_rates = np.ones(cmdargs.n_epochs) * cmdargs.learning_rate
  betas_0s = np.ones(cmdargs.n_epochs) * 0.9
  for edx in range(cmdargs.decay_start, cmdargs.n_epochs):
    learning_rates[edx] = (cmdargs.n_epochs - edx) / (cmdargs.n_epochs - cmdargs.decay_start) * cmdargs.learning_rate
    betas_0s[edx] = 0.1

  forget_rates = np.ones(cmdargs.n_epochs) * cmdargs.forget_rate
  forget_rates[:cmdargs.num_gradual] = np.linspace(0, cmdargs.forget_rate ** cmdargs.exponent, cmdargs.num_gradual)

  # Create context and start training
  ctx = {'writer': writer, 'epoch_idx': 0, 'global_step': 0}

  train_accuracy_1, train_accuracy_2 = 0, 0
  test_accuracy_1, test_accuracy_2 = 0, 0
  pure_ratio_1, pure_ratio_2 = 0, 0

  test_accuracy_1, test_accuracy_2 = eval_coteaching(loader_test, model_1, model_2)
  writer.add_scalars('dashboard', {
    'epoch': 0,
    'train_acc_1': train_accuracy_1,
    'train_acc_2': train_accuracy_2,
    'test_acc_1': test_accuracy_1,
    'test_acc_2': test_accuracy_2,
    'pure_ratio_1': pure_ratio_1,
    'pure_ratio_2': pure_ratio_2,
  }, global_step=ctx['global_step'])

  for epoch_idx in range(0, cmdargs.n_epochs):
    model_1.train() # Set as "Train Mode"
    adjust_adam_parameters(optimizer_1, learning_rates[epoch_idx], betas_0s[epoch_idx])
    model_2.train() # Set as "Train Mode"
    adjust_adam_parameters(optimizer_2, learning_rates[epoch_idx], betas_0s[epoch_idx])

    train_accuracy_1, train_accuracy_2, pure_ratio_1, pure_ratio_2 = train_coteaching(
      loader_train, model_1, optimizer_1, model_2, optimizer_2, forget_rates, ctx)
    ctx['epoch_idx'] += 1

    test_accuracy_1, test_accuracy_2 = eval_coteaching(loader_test, model_1, model_2)
    writer.add_scalars('dashboard', {
      'epoch': ctx['epoch_idx'],
      'train_acc_1': train_accuracy_1,
      'train_acc_2': train_accuracy_2,
      'test_acc_1': test_accuracy_1,
      'test_acc_2': test_accuracy_2,
      'pure_ratio_1': pure_ratio_1,
      'pure_ratio_2': pure_ratio_2,
    }, global_step=ctx['global_step'])

  return model_1, model_2


def main_procedure(cmdargs: Namespace):
  # Sets the seed for generating random numbers
  torch.manual_seed(cmdargs.random_seed)

  # Download, process and load noisy dataset
  dataset_train, dataset_test, network_kwargs = load_noisy_dataset(
    cmdargs.dataset, cmdargs.noise_type, cmdargs.noise_rate, cmdargs.random_seed)
  loader_train = DataLoader(dataset_train, cmdargs.batch_size, True, num_workers=cmdargs.n_workers, drop_last=True)
  loader_test = DataLoader(dataset_test, cmdargs.batch_size, False, num_workers=cmdargs.n_workers, drop_last=True)

  # Instanciate summary writer for logging
  logging_root = osp.abspath(cmdargs.logging_root)
  writer = SummaryWriter(logging_root)

  if cmdargs.paradiam == 'co-teaching':
    model_1, model_2 = invoke_coteaching_paradiam(cmdargs, network_kwargs, loader_train, loader_test, writer)

  if cmdargs.paradiam == 'primitive':
    model = invoke_primitive_paradiam(cmdargs, network_kwargs, loader_train, loader_test, writer)

  # Close summary writer which will flush cached logs
  writer.close()

  # Save last model checkpoint and experiment configurations
  configurations = configs_from_cmdargs(cmdargs)

  if cmdargs.paradiam == 'co-teaching':
    models = {'model_1.pth': model_1, 'model_2.pth': model_2}
    save_ckpt_and_configs(logging_root, configurations, models)

  if cmdargs.paradiam == 'primitive':
    models = {'model.pth': model}
    save_ckpt_and_configs(logging_root, configurations, models)



if __name__ == '__main__':
  parser = ArgumentParser(description='Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.')

  parser.add_argument('--paradiam', type=str, default='co-teaching', choices=['co-teaching', 'primitive'], help='Select deep learning paradiam.')
  parser.add_argument('--network', type=str, required=True, help='Name of the network.')
  parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per batch.')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
  parser.add_argument('--logging_root', type=str, default='logging/default', help='Root directory for logging during this run.')
  parser.add_argument('--noise_rate', type=float, default=0.2, help='Corruption rate, in range (0.0, 1.0).')
  parser.add_argument('--forget_rate', type=float, default=0.2, help='Forget rate, in range (0.0, 1.0).')
  parser.add_argument('--noise_type', type=str, default='clean', choices=['clean', 'pair_flip', 'symmetric', 'uniform_random'], help='Noise type.')
  parser.add_argument('--num_gradual', type=int, default=10, choices=[5, 10, 15], help='"Tk" for R(T) in Co-teaching paper.')
  parser.add_argument('--exponent', type=float, default=1.0, choices=[0.5, 1.0, 2.0], help='"c in Tc" for R(T) in Co-teaching paper.')
  parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'], help='Dataset.')
  parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--random_seed', type=int, default=42, help='Seed for random generators.')
  parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loading.')
  parser.add_argument('--decay_start', type=int, default=80, help='Start lr & betas decaying from this epoch, in range [0, n_epochs).')

  main_procedure(parser.parse_args())
