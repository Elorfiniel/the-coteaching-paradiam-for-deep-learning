import torch as torch
import torch.nn.functional as F

def loss_coteaching(y_1, y_2, t, forget_rate, ind, noise_or_not):
  loss_1 = F.cross_entropy(y_1, t, reduction='none')
  ind_1_sorted = torch.argsort(loss_1.data)

  loss_2 = F.cross_entropy(y_2, t, reduction='none')
  ind_2_sorted = torch.argsort(loss_2.data)

  remember_rate = 1.0 - forget_rate
  n_remembered = int(remember_rate * len(t))

  pure_ratio_1 = torch.sum(noise_or_not[ind[ind_1_sorted[:n_remembered]]]) / n_remembered
  pure_ratio_2 = torch.sum(noise_or_not[ind[ind_2_sorted[:n_remembered]]]) / n_remembered

  ind_1_update = ind_1_sorted[:n_remembered]
  ind_2_update = ind_2_sorted[:n_remembered]

  loss_1_update = F.cross_entropy(y_1[ind_2_update], t[ind_2_update], reduction='mean')
  loss_2_update = F.cross_entropy(y_2[ind_1_update], t[ind_1_update], reduction='mean')

  return loss_1_update, loss_2_update, pure_ratio_1, pure_ratio_2
