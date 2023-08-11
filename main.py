from dataset.MNIST import MNIST
from dataset.utils import ToFloatTensor
from network import CnnFromPaper, MyLeNet
from loss_fn import loss_coteaching

from argparse import ArgumentParser, Namespace
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter

import torch as torch
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


def load_noisy_dataset(dataset, noise_type, noise_rate, random_state):
  assert dataset in ['MNIST', 'CIFAR10', 'CIFAR100']

  if dataset == 'MNIST':
    dataset_train = MNIST('download', train=True, noise_type=noise_type,
                          noise_rate=noise_rate, random_state=random_state,
                          transform=ToFloatTensor())
    dataset_test = MNIST('download', train=False, transform=ToFloatTensor())
    dataset_spec = {'in_channels': 1, 'n_classes': 10}

  if dataset == 'CIFAR10':
    raise NotImplementedError(f'Framework has not implemented dataset "{dataset}" yet.')

  if dataset == 'CIFAR100':
    raise NotImplementedError(f'Framework has not implemented dataset "{dataset}" yet.')

  return dataset_train, dataset_test, dataset_spec

def adjust_adam_parameters(optimizer, lr, betas_0):
  for param_group in optimizer.param_groups:
    param_group['lr'] = lr
    param_group['betas'] = (betas_0, 0.999)

def accuracy(logits, targets, topk=(1, 5)):
  outputs = F.softmax(logits, dim=1)
  _, preds = outputs.topk(max(topk), dim=1)
  results = preds.T == targets

  topk_accuracies = []
  for k in topk:
    correct_k = torch.sum(results[:k], dim=0).sum()
    topk_accuracies.append(100.0 * correct_k / len(targets))

  return topk_accuracies

def train(dataloader, epoch_idx, step, model_1, optimizer_1, model_2, optimizer_2, forget_rates):
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

    loss_1, loss_2, pure_ratio_1, pure_ratio_2 = loss_coteaching(
      logits_1, logits_2, labels, forget_rates[epoch_idx], indices, noise_or_not)

    pure_ratio_1_list.append(pure_ratio_1 * 100.0)
    pure_ratio_2_list.append(pure_ratio_2 * 100.0)

    optimizer_1.zero_grad()
    loss_1.backward()
    optimizer_1.step()
    optimizer_2.zero_grad()
    loss_2.backward()
    optimizer_2.step()

    step += 1 # Increase global step by 1

  accuracy_1 = correctness_1 / n_total_batch_1
  accuracy_2 = correctness_2 / n_total_batch_2

  pure_ratio_1 = sum(pure_ratio_1_list) / len(pure_ratio_1_list)
  pure_ratio_2 = sum(pure_ratio_2_list) / len(pure_ratio_2_list)

  return accuracy_1, accuracy_2, pure_ratio_1, pure_ratio_2, step

def eval(dataloader, model_1, model_2):
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


def main_procedure(cmdargs: Namespace):
  # Sets the seed for generating random numbers
  torch.manual_seed(cmdargs.random_seed)

  # Download, process and load noisy dataset
  d_train, d_test, spec = load_noisy_dataset(cmdargs.dataset, cmdargs.noise_type, cmdargs.noise_rate, cmdargs.random_seed)
  l_train = DataLoader(d_train, cmdargs.batch_size, True, num_workers=cmdargs.n_workers, drop_last=True)
  l_test = DataLoader(d_test, cmdargs.batch_size, False, num_workers=cmdargs.n_workers, drop_last=True)

  # Instanciate models with differenct initial parameters
  # model_1 = CnnFromPaper(spec['in_channels'], spec['n_classes'])
  model_1 = MyLeNet(spec['in_channels'], spec['n_classes'])
  optimizer_1 = optim.Adam(model_1.parameters(), lr=cmdargs.learning_rate)

  # model_2 = CnnFromPaper(spec['in_channels'], spec['n_classes'])
  model_2 = MyLeNet(spec['in_channels'], spec['n_classes'])
  optimizer_2 = optim.Adam(model_2.parameters(), lr=cmdargs.learning_rate)

  # Configure schedulers for hyper-parameters
  learning_rates = np.ones(cmdargs.n_epochs) * cmdargs.learning_rate
  betas_0s = np.ones(cmdargs.n_epochs) * 0.9
  for edx in range(cmdargs.decay_start, cmdargs.n_epochs):
    learning_rates[edx] = (cmdargs.n_epochs - edx) / (cmdargs.n_epochs - cmdargs.decay_start) * cmdargs.learning_rate
    betas_0s[edx] = 0.1

  forget_rates = np.ones(cmdargs.n_epochs) * cmdargs.forget_rate
  forget_rates[:cmdargs.num_gradual] = np.linspace(0, cmdargs.forget_rate ** cmdargs.exponent, cmdargs.num_gradual)

  writer = SummaryWriter('logging/MNIST')
  global_training_step = 0

  # Start training using co-teaching paradiam
  train_accuracy_1, train_accuracy_2 = 0, 0
  test_accuracy_1, test_accuracy_2 = 0, 0
  pure_ratio_1, pure_ratio_2 = 0, 0

  # Evaluate before start training
  test_accuracy_1, test_accuracy_2 = eval(l_test, model_1, model_2)
  writer.add_scalars('dashboard', {
    'epoch': 0,
    'train_acc_1': train_accuracy_1,
    'train_acc_2': train_accuracy_2,
    'test_acc_1': test_accuracy_1,
    'test_acc_2': test_accuracy_2,
    'pure_ratio_1': pure_ratio_1,
    'pure_ratio_2': pure_ratio_2,
  }, global_step=global_training_step)

  for epoch_idx in range(0, cmdargs.n_epochs):
    model_1.train() # Set as "Train Mode"
    adjust_adam_parameters(optimizer_1, learning_rates[epoch_idx], betas_0s[epoch_idx])
    model_2.train() # Set as "Train Mode"
    adjust_adam_parameters(optimizer_2, learning_rates[epoch_idx], betas_0s[epoch_idx])

    train_accuracy_1, train_accuracy_2, pure_ratio_1, pure_ratio_2, global_training_step = train(
      l_train, epoch_idx, global_training_step, model_1, optimizer_1, model_2, optimizer_2, forget_rates)

    test_accuracy_1, test_accuracy_2 = eval(l_test, model_1, model_2)

    writer.add_scalars('dashboard', {
      'epoch': epoch_idx + 1,
      'train_acc_1': train_accuracy_1,
      'train_acc_2': train_accuracy_2,
      'test_acc_1': test_accuracy_1,
      'test_acc_2': test_accuracy_2,
      'pure_ratio_1': pure_ratio_1,
      'pure_ratio_2': pure_ratio_2,
    }, global_step=global_training_step)

  writer.close()

  # Save the trained models
  torch.save(model_1.state_dict(), 'checkpoints/model_1.pth')
  torch.save(model_2.state_dict(), 'checkpoints/model_2.pth')


if __name__ == '__main__':
  parser = ArgumentParser(description='Co-teaching: Robust Training of Deep Neural Networks with Extremely Noisy Labels.')

  parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per batch.')
  parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate.')
  # Not implemented: result_dir
  parser.add_argument('--noise_rate', type=float, default=0.2, help='Corruption rate, in range (0.0, 1.0).')
  parser.add_argument('--forget_rate', type=float, default=0.2, help='Forget rate, in range (0.0, 1.0).')
  parser.add_argument('--noise_type', type=str, default='clean', choices=['clean', 'pair_flip', 'symmetric'], help='Noise type.')
  parser.add_argument('--num_gradual', type=int, default=10, choices=[5, 10, 15], help='"Tk" for R(T) in Co-teaching paper.')
  parser.add_argument('--exponent', type=float, default=1.0, choices=[0.5, 1.0, 2.0], help='"c in Tc" for R(T) in Co-teaching paper.')
  parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST', 'CIFAR10', 'CIFAR100'], help='Dataset.')
  parser.add_argument('--n_epochs', type=int, default=200, help='Number of epochs.')
  parser.add_argument('--random_seed', type=int, default=42, help='Seed for random generators.')
  # Not implemented: print_freq
  parser.add_argument('--n_workers', type=int, default=4, help='Number of workers for data loading.')
  # Not implemented: num_iter_per_epoch
  parser.add_argument('--decay_start', type=int, default=80, help='Start lr & betas decaying from this epoch, in range [0, n_epochs).')

  main_procedure(parser.parse_args())
