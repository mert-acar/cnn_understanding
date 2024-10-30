import numpy as np
from scipy.sparse import random
from tqdm import tqdm
from sklearn import metrics
from sklearn import cluster
from typing import Callable, Tuple
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import ParameterGrid

from dim_reduction import pca_reduction


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
  for i, param in tqdm(
    list(enumerate(ParameterGrid(params))), desc="Parameter Searching...", ncols=94
  ):
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
  import os
  from time import perf_counter
  from scipy.io import loadmat
  from sklearn.mixture import GaussianMixture
  from sklearn.neighbors import kneighbors_graph
  from sklearn.preprocessing import StandardScaler

  exp_dir = "../logs/customnet_run2/"
  labels = loadmat("../data/labels.mat")["labels"][0]
  epoch = 33
  random_state = 9001

  var = "features.8_output"
  x = loadmat(
    os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"),
    variable_names=[var],
  )[var]

  # n = 40000 // (10 * x.shape[1])
  # idx = select_random_samples(labels, n)
  # x = x[idx]
  # l = labels[idx]
  labels = np.repeat(labels, x.shape[1])
  x = x.reshape(-1, x.shape[-1])
  x = StandardScaler().fit_transform(x)
  print(f"Activations: {x.shape}")
  tick = perf_counter()
  x = pca_reduction(x, n_components=None, threshold=0.98)
  print(f"PCA took: {perf_counter() - tick:.3f} seconds")
  print("After PCA:", x.shape)

  connectivity = kneighbors_graph(
      x, n_neighbors=4, include_self=False
  )
  connectivity = 0.5 * (connectivity + connectivity.T)

  algos = {
      "k_means": cluster.MiniBatchKMeans(n_clusters=10, random_state=random_state, batch_size=2048),
      "agglomerative": cluster.AgglomerativeClustering(n_clusters=10, connectivity=connectivity),
      "spectral": cluster.SpectralClustering(n_clusters=10, affinity="nearest_neighbors", random_state=random_state),
      "gmm": GaussianMixture(n_components=10, random_state=random_state)
  }

  for name, algo in algos.items():
    tick = perf_counter()
    y_pred = algo.fit(x).labels_
    print(f"{name} took {perf_counter() - tick:.3f} seconds")
    y_pred, _ = map_clusters(y_pred, labels)
    print(f"{name} | Adjusted RAND Index: {metrics.adjusted_rand_score(labels, y_pred):.3f}") # | Silhouette: {metrics.silhouette_score(x, y_pred):.3f}")
