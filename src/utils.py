import re
import os
import numpy as np
import pandas as pd
from glob import glob
from scipy.io import loadmat
from kneed import KneeLocator
from dim_reduction import svd_reduction
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def normalize_cols(x, axis=1, eps=1e-12):
  norm = np.linalg.norm(x, axis=axis, keepdims=True)
  return x / (norm + eps)


# Function to extract epoch number from the key
def extract_epoch(key):
  match = re.search(r'epoch_(\d+)', key)
  return int(match.group(1)) if match else 0


def create_dataloader(data_root="../data/", batch_size=1, num_workers=4, split="train", **kwargs):
  transform = Compose([ToTensor(), Normalize((0.1307, ), (0.3081, ))])
  return DataLoader(
    MNIST(root=data_root, download=True, train=split == 'train', transform=transform),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=split == 'train'
  )


def get_filenames(
  layer,
  experiment_path=os.path.join("..", "logs", "resnet18_run4", "activations"),
  reverse=True,
  ext="mat"
):
  return sorted(
    glob(os.path.join(experiment_path, layer, f"*.{ext}")), key=extract_epoch, reverse=reverse
  )


def cluster_matrix_to_df(matrices, titles):
  col = []
  for mat, title in zip(matrices, titles):
    b = np.full((13, 11), "").astype(object)
    b[1, 1] = title
    b[2, 0] = "label"
    b[2, 1:] = np.linspace(0, 9, 10, dtype=int)
    b[3:, 0] = np.linspace(0, 9, 10, dtype=int)
    b[3:, 1:] = mat
    col.append(b)
  col = np.vstack(col)
  return pd.DataFrame(col)


def select_random_samples(labels, num_samples_per_label):
  unique_labels = np.unique(labels)
  selected_indices = []
  for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) < num_samples_per_label:
      raise ValueError(f"Not enough samples for label {label}. Only {len(indices)} available.")
    selected = np.random.choice(indices, num_samples_per_label, replace=False)
    selected_indices.extend(selected)
  return np.array(selected_indices)


def find_non_zero_idx(data, beta=0.95):
  _, C, H, W = data.shape
  stat = np.reshape(np.abs(data).mean(0), (C, H * W)).mean(-1)
  y = np.sort(stat)[::-1]
  x = list(range(len(y)))
  kn = KneeLocator(x, y, curve='convex', direction='decreasing').knee
  non_zero_idx = stat >= y[kn] * beta
  return non_zero_idx


def read_to_cluster(file_path, reshape="avg", norm=True, svd_dim=None, threshold=0.98):
  data = loadmat(file_path)
  # Ground truth labels of shape [N]
  y = data["labels"][0]
  # Layer activations of shape [N, C, H, W]
  X = data["activations"]

  if reshape == "avg":
    # [N, C, H, W] -> [N, C]
    # where the spatial dimensions are averaged out
    X = X.mean((-2, -1))
  elif reshape == "flatten":
    # [N, C, H, W] -> [N, C*H*W]
    X = X.reshape(X.shape[0], -1)
  else:
    raise ValueError("'reshape' argument has to be one of 'avg' or 'flatten'")

  if norm:
    # Normalize between [-1, 1]
    X = X / np.abs(X).max()

  # Center the data and reduce dimensions using SVD
  X = svd_reduction(X - X.mean(0), n_components=svd_dim, threshold=threshold)

  return X, y
