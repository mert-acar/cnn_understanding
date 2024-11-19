import numpy as np
from tqdm import tqdm
from sklearn import metrics
from sklearn import cluster
from typing import Callable, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import ParameterGrid


def bcss_wcss(
  x: np.ndarray,
  cluster_labels: np.ndarray,
  return_chi: bool = False
) -> Tuple[float, float, Optional[float]]:
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


def parameter_search(
  data: np.ndarray,
  labels: np.ndarray,
  params: dict,
  algo: Callable = cluster.AgglomerativeClustering,
) -> Tuple[np.ndarray, dict, dict]:
  best = -1
  comparator = lambda x, y: x > y
  for param in tqdm(ParameterGrid(params), desc="Parameter Searching...", ncols=94):
    try:
      cluster_labels = algo(**param).fit(data).labels_
      scores = performance_scores(data, cluster_labels, labels)
    except ValueError:
      continue
    score = scores["silhouette"]
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
