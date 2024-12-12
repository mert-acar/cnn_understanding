import torch
import pickle as p
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from sklearn import metrics, cluster
from scipy.optimize import linear_sum_assignment


def cluster_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
  return (pred_labels == true_labels).mean()


def bcss_wcss(x: np.ndarray,
              cluster_labels: np.ndarray,
              return_chi: bool = False) -> Tuple[float, float, Optional[float]]:
  n_labels = len(set(cluster_labels))
  extra_disp, intra_disp = 0.0, 0.0
  mean = np.mean(x, axis=0)
  for k in range(n_labels):
    cluster_k = x[cluster_labels == k]
    mean_k = np.mean(cluster_k, axis=0)
    extra_disp += len(cluster_k) * np.sum((mean_k - mean)**2)
    intra_disp += np.sum((cluster_k - mean_k)**2)
  if return_chi:
    chi = extra_disp * (len(x) - n_labels) / (intra_disp * (n_labels - 1.0))
    return extra_disp, intra_disp, chi
  return extra_disp, intra_disp, None


def map_clusters(label_a: np.ndarray, labels_b: np.ndarray) -> Tuple[np.ndarray, dict[int, int]]:
  unique_clusters = np.unique(label_a)
  unique_true_labels = np.unique(labels_b)
  cost_matrix = np.zeros((len(unique_clusters), len(unique_true_labels)))
  for i, cluster in enumerate(unique_clusters):
    for j, label in enumerate(unique_true_labels):
      count = np.sum((label_a == cluster) & (labels_b == label))
      cost_matrix[i, j] = -count
  _, col_ind = linear_sum_assignment(cost_matrix)
  cluster_to_label_map = {}
  for i, cluster in enumerate(unique_clusters):
    mapped_label = unique_true_labels[col_ind[i]]
    cluster_to_label_map[cluster] = mapped_label
  new_cluster_labels = np.array([cluster_to_label_map[cluster] for cluster in label_a])
  return new_cluster_labels, cluster_to_label_map


def normalized_minkowski(x: np.ndarray, y: np.ndarray) -> float:
  return np.linalg.norm(x - y) / np.sqrt(x.shape[0])


if __name__ == "__main__":
  import os
  import pickle as p
  import pandas as pd
  from tqdm import tqdm
  from time import time
  from tabulate import tabulate
  from collections import defaultdict
  from sklearn.decomposition import PCA
  from sklearn.preprocessing import normalize

  from mlr import train_mlr
  from model import HOOK_TARGETS
  from visualize import plot_scores
  from utils import load_ImageNet_labels, load_MNIST_labels, load_CIFAR10_labels

  model_name = "resnet18"
  dataset = "MNIST"
  identifier = "kmeans"
  low, high = 5, 15
  idx = None
  if dataset == "MNIST":
    labels = load_MNIST_labels()
  elif dataset == "CIFAR10":
    labels = load_CIFAR10_labels()
  elif dataset == "IMAGENET":
    labels = load_ImageNet_labels()
  else:
    raise NotImplementedError

  if idx is not None:
    labels = labels[idx]

  exp_dir = f"../logs/{model_name}_{dataset}_CIL"
  out_path = os.path.join(exp_dir, "clusters")
  os.makedirs(out_path, exist_ok=True)

  vars = HOOK_TARGETS[model_name]

  scores = defaultdict(list)
  print(f"+ Working on: {exp_dir}")
  for var in vars:
    out = defaultdict(list)
    print(var, "\n----------------")
    file_path = os.path.join(out_path, f"{var.replace('.', '_')}.p")
    if os.path.exists(file_path):
      with open(file_path, "rb") as f:
        x = p.load(f)
    else:
      with open(os.path.join(exp_dir, "activations", f"{var.replace('.', '_')}_act.p"), "rb") as f:
        x = p.load(f)

      # x = torch.load(
      #   os.path.join(exp_dir, "activations", f"{var.replace('.', '_')}_act.pt"), weights_only=False
      # ).numpy()

      if idx is not None:
        x = x[idx]

      # [N, C, H, W] -> [N, C*H*W]
      x = x.reshape(x.shape[0], -1)

      print(f"Before PCA: {x.shape}")
      tick = time()
      x = PCA(n_components=0.95, whiten=True).fit_transform(x)
      print(f"PCA took {time() - tick:.3f} seconds")
      print(f"After PCA: {x.shape}")
      # x = normalize(x, norm='l2', axis=1)
      with open(file_path, "wb") as f:
        p.dump(x, f, protocol=p.HIGHEST_PROTOCOL)

    for k in tqdm(range(low, high)):
      cluster_labels = cluster.MiniBatchKMeans(n_clusters=k, batch_size=2048).fit_predict(x)
      perf = {
        "k": k,
        "silhouette": metrics.silhouette_score(x, cluster_labels),
        "nmi": metrics.normalized_mutual_info_score(labels, cluster_labels),
        "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
        "completeness": metrics.completeness_score(labels, cluster_labels),
        "fowlkes_mallows": metrics.fowlkes_mallows_score(labels, cluster_labels),
        "chi": metrics.calinski_harabasz_score(x, cluster_labels),
      }
      for key in perf:
        out[key].append(perf[key])

    print(tabulate(out, headers=list(out.keys())))
    scores["layer"].extend([var] * len(out[key]))
    for key in out:
      scores[key].extend(out[key])

  df = pd.DataFrame(scores)
  df.to_csv(os.path.join(out_path, f"{identifier}_scores.csv"))

  df = pd.read_csv(os.path.join(out_path, f"{identifier}_scores.csv"))
  out_path = os.path.join(exp_dir, "figures")
  os.makedirs(out_path, exist_ok=True)
  # df["silhouette"] = (df["silhouette"] + 1) / 2
  plot_scores(df, identifier, out_path)
