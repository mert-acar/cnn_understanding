import os
import torch
import numpy as np
from math import isqrt
from shutil import rmtree

from typing import Union, List, Dict


def combine_scores(data):
  cov_matrix = np.cov(data.T)
  eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
  max_eigenvalue_idx = np.argmax(eigenvalues)
  weights = eigenvectors[:, max_eigenvalue_idx]
  weights = -weights / np.linalg.norm(weights)
  combined_feature = data @ weights
  return combined_feature, weights


def create_dir(output_path: str):
  if os.path.exists(output_path):
    c = input(f"Output path {output_path} is not empty! Do you want to delete the folder [y / n]: ")
    if "y" == c.lower():
      rmtree(output_path, ignore_errors=True)
    else:
      print("Exit!")
      raise SystemExit
  os.makedirs(os.path.join(output_path, "checkpoints"))


def get_device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  elif torch.backends.mps.is_available():
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = (
      "1"  # fallback to cpu if an mps-incompatible op is tried
    )
    return torch.device("mps")
  else:
    return torch.device("cpu")


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


def closest_factors(n: int) -> tuple[int, int]:
  root = int(isqrt(n))
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


def calculate_accuracy(output: torch.Tensor, target: torch.Tensor) -> float:
  pred = torch.nn.functional.log_softmax(output, dim=1)
  acc = pred.argmax(1).eq(target).sum().item() / output.shape[0]
  return acc


def get_metric_scores(metric_list: List[str], output: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
  out = {}
  for metric in metric_list:
    if metric.lower() == "accuracy":
      out[metric] = calculate_accuracy(output, target)
    else:
      print(f"Metric [{metric}] is not implemented, skipping...")
  return out
