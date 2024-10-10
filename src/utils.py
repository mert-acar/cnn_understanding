import re
import os
import numpy as np
import pandas as pd
from glob import glob
from sklearn import metrics
from scipy.io import loadmat
from kneed import KneeLocator
from dim_reduction import svd_reduction
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def performance_scores(data, cluster_labels, labels):
  return {
    "silhouette": metrics.silhouette_score(data, cluster_labels),
    "calinski_harabasz_score": metrics.calinski_harabasz_score(data, cluster_labels),
    "davies_bouldin_score": metrics.davies_bouldin_score(data, cluster_labels),
    "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
    "completeness": metrics.completeness_score(labels, cluster_labels),
    "v_measure": metrics.v_measure_score(labels, cluster_labels),
    "mutual_information": metrics.adjusted_mutual_info_score(labels, cluster_labels),
    "num_clusters": len(np.unique(cluster_labels[cluster_labels != -1])),
    "n_noisy_samples": 100 * sum(cluster_labels == -1) / len(cluster_labels)
  }


def normalize(x, axis=1, eps=1e-12):
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


def read_to_cluster(file_path, svd_dim=86, threshold=None, norm=True):
  data = loadmat(file_path)
  y = data["labels"][0]
  X = data["activations"]
  if X.sum() == 0:
    return None, y
  X = X.reshape(X.shape[0], -1)
  # normalize the columns to length 1 -> project the manifold onto a hyper-sphere
  if norm:
    X = normalize(X)
  X = svd_reduction(X - X.mean(0), n_components=svd_dim, threshold=threshold)
  return X, y
