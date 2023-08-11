import torch as torch
import numpy as np


class ToFloatTensor():
  def __init__(self, bits=8, dtype=torch.float32):
    self.max_value = 2**bits - 1
    self.dtype = dtype

  def __call__(self, image: torch.Tensor):
    result = image.div(self.max_value)
    return result.to(self.dtype)

  def __repr__(self):
    return f'{self.__class__.__name__}()'


def apply_transition_matrix(M, labels, random_state=42):
  assert np.all(M >= 0.0) # Assertion for noise transition matrix
  np.testing.assert_array_almost_equal(np.sum(M, axis=1), np.ones(len(M)))

  noisy_labels = labels.copy()
  f = np.random.RandomState(random_state)
  for idx in range(len(labels)):
    class_id = labels[idx]
    result = f.multinomial(1, M[class_id, :], 1)[0]
    noisy_labels[idx] = np.where(result == 1)[0]

  return noisy_labels

def noisify_pair_flip(labels, noise_rate, n_classes, random_state=42):
  assert noise_rate > 0.0 and noise_rate < 1.0 and n_classes > 1
  assert np.min(labels) >= 0 and np.max(labels) <= n_classes - 1

  M_transition = np.zeros(shape=(n_classes, n_classes))
  for row in range(n_classes):
    M_transition[row, row] = 1.0 - noise_rate
    M_transition[row, (row + 1) % n_classes] = noise_rate

  noisy_labels = apply_transition_matrix(M_transition, labels, random_state)
  actual_noise_rate = np.average(noisy_labels != labels)
  assert actual_noise_rate >= 0.0

  return noisy_labels, actual_noise_rate

def noisify_symmetric(labels, noise_rate, n_classes, random_state=42):
  assert noise_rate > 0.0 and noise_rate < 1.0 and n_classes > 1
  assert np.min(labels) >= 0 and np.max(labels) <= n_classes - 1

  M_transition = np.ones(shape=(n_classes, n_classes))
  M_transition = M_transition * (noise_rate / (n_classes - 1))

  for row in range(n_classes):
    M_transition[row, row] = 1.0 - noise_rate

  noisy_labels = apply_transition_matrix(M_transition, labels, random_state)
  actual_noise_rate = np.average(noisy_labels != labels)
  assert actual_noise_rate >= 0.0

  return noisy_labels, actual_noise_rate

def noisify(noise_type, labels, noise_rate, n_classes, random_state):
  assert noise_type in ['pair_flip', 'symmetric']

  if noise_type == 'pair_flip':
    noisy_labels, actual_noise_rate = noisify_pair_flip(labels, noise_rate, n_classes, random_state)

  if noise_type == 'symmetric':
    noisy_labels, actual_noise_rate = noisify_symmetric(labels, noise_rate, n_classes, random_state)

  return noisy_labels, actual_noise_rate
