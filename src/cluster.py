import torch
import numpy as np
from tqdm import tqdm
from sklearn import cluster
from functools import partial
from sklearn.svm import LinearSVC

from loss import MaximalCodingRateReduction

from typing import Tuple


def bcss(x: np.ndarray, pred_labels: np.ndarray) -> float:
  n_labels = len(set(pred_labels))
  extra_disp = 0.0
  mean = x.mean(0)
  for k in range(n_labels):
    cluster_k = x[pred_labels == k]
    mean_k = cluster_k.mean(0)
    extra_disp += len(cluster_k) * ((mean_k - mean)**2).sum()
  return float(extra_disp)


def wcss(x: np.ndarray, pred_labels: np.ndarray) -> float:
  n_labels = len(set(pred_labels))
  intra_disp = 0.0
  for k in range(n_labels):
    cluster_k = x[pred_labels == k]
    mean_k = cluster_k.mean(0)
    intra_disp += ((cluster_k - mean_k)**2).sum()
  return float(intra_disp)


def cluster_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
  return (pred_labels == true_labels).mean()


def maximal_coding_rate(activations: np.ndarray, labels: np.ndarray) -> Tuple[float, ...]:
  criterion = MaximalCodingRateReduction()
  delta_R, R, Rc = 0, 0, 0
  step = 128
  x = torch.from_numpy(activations)
  y = torch.from_numpy(labels)
  n = np.ceil(len(x) / step)
  for i in tqdm(range(0, int(n))):
    w = x[i * step:(i + 1) * step].T
    pi = criterion.label_to_membership(y[i * step:(i + 1) * step])
    r = criterion.compute_discrimn_loss_empirical(w) / n
    rc = criterion.compute_compress_loss_empirical(w, pi) / n
    delta_R += r - rc
    R += r
    Rc += rc
  return delta_R, R, Rc


def svm(activations: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
  model = LinearSVC(**kwargs)
  model.fit(activations, labels)
  pred_labels = model.predict(activations)
  return pred_labels


def spectral(activations: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
  return cluster.SpectralClustering(**kwargs).fit_predict(activations)


def get_clustering_func(method: str, **kwargs):
  if method == "svm":
    return partial(svm, **kwargs)
  elif method == "spectral":
    return partial(spectral, **kwargs)
  else:
    raise NotImplementedError(method)
