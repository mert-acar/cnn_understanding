import numpy as np
from sklearn import metrics
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
  algo: Callable,
  params: dict,
  optimize_over: str = "calinski_harabasz_score",
  max: bool = True
) -> Tuple[np.ndarray, dict, dict]:
  best = None
  comparator = (lambda x, y: x > y) if max else (lambda x, y: x < y)
  for param in ParameterGrid(params):
    try:
      cluster_labels = algo(**param).fit(data).labels_
      scores = performance_scores(data, cluster_labels, labels)
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
  from sklearn import cluster
  from scipy.io import loadmat
  from scipy.spatial.distance import cdist

  vars = ["features.8_input", "features.8_output"]
  data = loadmat("../logs/customnet_run2/activations/patches_epoch_33.mat", variable_names=vars)
  metric = "calinski_harabasz_score"
  i = 1
  labels = loadmat("../data/labels.mat")["labels"][0]
  d = data[vars[i]]

  n = 30
  idx = select_random_samples(labels, n)
  d = d[idx]
  labels = labels[idx]

  print(f"{vars[i]} with n = {n}")

  print(f"Data shape: {d.shape}")
  labels = np.repeat(labels, d.shape[1])
  d = d.reshape(-1, d.shape[-1])
  print(f"Aggregated shape: {d.shape}")
  d = d / np.abs(d).max()

  cluster_labels = cluster.AgglomerativeClustering(n_clusters=10).fit(d).labels_
  scores = performance_scores(d, cluster_labels, labels)
  dist = cdist(d, d).mean()

  params = {"n_clusters": [None], "distance_threshold": [i * dist for i in np.linspace(0.1, 5, 20)]}
  cluster_labels, best_param, scores = parameter_search(
    d, labels, cluster.AgglomerativeClustering, params, optimize_over=metric
  )

  print("-" * 10)
  for key, score in scores.items():
    print(f"{key}: {score:.3f}")
