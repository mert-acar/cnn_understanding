import numpy as np
import pandas as pd
from sklearn import metrics
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import ParameterGrid


def map_clusters(label_a, labels_b):
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


def parameter_search(data, labels, algo, params, optimizer_over="silhouette", max=True):
  best = None
  comparator = (lambda x, y: x > y) if max else (lambda x, y: x < y)
  for param in ParameterGrid(params):
    try:
      cluster_labels = algo(**param).fit(data).labels_
      scores = performance_scores(data, cluster_labels, labels)
    except ValueError:
      continue
    score = scores[optimizer_over]
    if best is None or comparator(score, best):
      best = score
      best_scores = scores
      best_labels = cluster_labels
      best_params = param
  return best_labels, best_params, best_scores


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
  }
