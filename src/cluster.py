import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn import cluster
from typing import Callable, Tuple
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import ParameterGrid


def select_random_samples(labels: np.ndarray, num_samples_per_label: int) -> np.ndarray:
  unique_labels = np.unique(labels)
  selected_indices = []
  for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) < num_samples_per_label:
      raise ValueError(f"Not enough samples for label {label}. Only {len(indices)} available.")
    selected = np.random.choice(indices, num_samples_per_label, replace=False)
    selected_indices.extend(selected)
  return np.array(selected_indices)


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


def parameter_search(
  data: np.ndarray,
  labels: np.ndarray,
  params: dict,
  algo: Callable = cluster.AgglomerativeClustering,
  optimize_over: str = "calinski_harabasz_score",
  max: bool = True
) -> Tuple[np.ndarray, dict, dict]:
  best = None
  comparator = (lambda x, y: x > y) if max else (lambda x, y: x < y)
  for param in tqdm(ParameterGrid(params), desc="Parameter Searching...", ncols=94):
    try:
      cluster_labels = algo(**param).fit(data).labels_
      scores = performance_scores(data, cluster_labels, labels)
      print()
      pprint(param)
      pprint(scores)
    except ValueError:
      continue
    score = scores[optimize_over]
    if best is None or comparator(score, best):
      best = score
      best_scores = scores
      best_labels = cluster_labels
      best_params = param
  return best_labels, best_params, best_scores


def performance_scores(data: np.ndarray, cluster_labels: np.ndarray, labels: np.ndarray) -> dict:
  return {
    "silhouette": metrics.silhouette_score(data, cluster_labels),
    "calinski_harabasz_score": metrics.calinski_harabasz_score(data, cluster_labels),
    "davies_bouldin_score": metrics.davies_bouldin_score(data, cluster_labels),
    "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
    "completeness": metrics.completeness_score(labels, cluster_labels),
    "v_measure": metrics.v_measure_score(labels, cluster_labels),
    "mutual_information": metrics.adjusted_mutual_info_score(labels, cluster_labels),
    "num_clusters": len(np.unique(cluster_labels[cluster_labels != -1])),
  }


if __name__ == "__main__":
  import os
  import pickle
  from pprint import pprint
  from scipy.io import loadmat
  from dim_reduction import svd_reduction
  from scipy.spatial.distance import cdist

  exp_dir = "../logs/customnet_run2/"
  labels = loadmat("../data/labels.mat")["labels"][0]
  epoch = 33
  out = {}
  for i in range(0, 9, 2):
    vars = [f"features.{i}_input", f"features.{i}_output"]
    for var in vars:
      data = loadmat(
        os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"),
        variable_names=[var],
      )
      print(var)
      x = data[var]
      n = 24000 // (10 * x.shape[1])
      print("num samples:", n)
      idx = select_random_samples(labels, n)
      x = x[idx]
      l = labels[idx]
      l = np.repeat(l, x.shape[1])
      x = x.reshape(-1, x.shape[-1])
      print(f"Activations: {x.shape}")
      print(f"Labels: {l.shape}")
      x = x - x.mean(0)
      x = x / np.abs(x).max()
      x = svd_reduction(x, n_components=None, threshold=0.98)
      # d = cdist(x, x).mean()
      print(f"After SVD: {x.shape}")
      params = {
        "n_clusters": [None],
        "distance_threshold": [k for k in np.linspace(5, 200, 20)],
      }
      clusters, p, scores = parameter_search(x, l, params)
      # p.update({"mean_l2_dist": d})
      pprint(p)
      pprint(scores)
      print()
      out[var] = {
        "cluster_labels": clusters,
        "params": p,
        "scores": scores,
        "idx": idx,
      }

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}_v2.p"), "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
