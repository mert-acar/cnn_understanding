import re
import os
import math
import numpy as np
from glob import glob
from scipy.io import loadmat
from kneed import KneeLocator
from typing import List, Union


def load_MNIST_labels() -> np.ndarray:
  return loadmat("../data/mnist/labels.mat")["labels"][0]


def combine_scores(data):
  cov_matrix = np.cov(data.T)
  eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
  max_eigenvalue_idx = np.argmax(eigenvalues)
  weights = eigenvectors[:, max_eigenvalue_idx]
  weights = -weights / np.linalg.norm(weights)
  combined_feature = data @ weights
  return combined_feature, weights


def select_random_samples(
  labels: Union[list[int], np.ndarray], num_samples_per_label: int, seed: int = 9001
) -> np.ndarray:
  unique_labels = np.unique(labels)
  selected_indices = []
  np.random.seed(seed)
  for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) < num_samples_per_label:
      raise ValueError(f"Not enough samples for label {label}. Only {len(indices)} available.")
    selected = np.random.choice(indices, num_samples_per_label, replace=False)
    selected_indices.extend(selected)
  return np.array(selected_indices)


def normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
  norm = np.linalg.norm(x, axis=axis, keepdims=True)
  return x / (norm + eps)


def rescale(x: np.ndarray) -> np.ndarray:
  x = x - x.min()
  x = x / x.max()
  return x


def extract_epoch(key: str) -> int:
  match = re.search(r'epoch_(\d+)', key)
  return int(match.group(1)) if match else 0


def get_filenames(experiment_path: str, reverse: bool = True, ext: str = "mat") -> List[str]:
  return sorted(glob(os.path.join(experiment_path, f"*.{ext}")), key=extract_epoch, reverse=reverse)


def find_non_zero_idx(data: np.ndarray, beta: float = 0.95) -> np.ndarray:
  _, C, H, W = data.shape
  stat = np.reshape(np.abs(data).mean(0), (C, H * W)).mean(-1)
  y = np.sort(stat)[::-1]
  x = list(range(len(y)))
  kn = KneeLocator(x, y, curve='convex', direction='decreasing').knee
  non_zero_idx = stat >= y[kn] * beta
  return non_zero_idx


def closest_factors(n: int) -> tuple[int, int]:
  root = int(math.isqrt(n))
  for i in range(root, 0, -1):
    if n % i == 0:
      return (i, n // i)
  return (n, 1)


def svd_reduction(
  activations: np.ndarray,
  n_components: Union[None, int] = 10,
  threshold: Union[None, float] = None
) -> np.ndarray:
  assert (n_components is None) != (threshold is None), "Either rank or threshold should be specified"
  u, s, _ = np.linalg.svd(activations, full_matrices=False)

  if threshold is not None:
    s2 = s**2
    energies = np.cumsum(s2) / np.sum(s2)
    k = np.argmax(energies > threshold) + 1
  else:
    k = n_components

  u_k = u[:, :k]
  s_k = s[:k]
  recon = np.dot(u_k, np.diag(s_k))
  return recon
