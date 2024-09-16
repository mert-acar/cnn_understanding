import re
import sklearn.metrics as metrics
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import ParameterGrid


def create_cluster_matrix(cluster_indices, class_labels):
  num_classes = len(np.unique(class_labels))
  cluster_matrix = np.zeros((num_classes, num_classes), dtype=int)
  valid_clusters = np.unique(cluster_indices[cluster_indices != -1])

  for cluster in valid_clusters:
    points_in_cluster = np.where(cluster_indices == cluster)[0]
    classes_in_cluster = np.unique(class_labels[points_in_cluster])
    for i in range(len(classes_in_cluster)):
      for j in range(i, len(classes_in_cluster)):
        class_i = classes_in_cluster[i]
        class_j = classes_in_cluster[j]
        cluster_matrix[class_i, class_j] += 1
        if class_i != class_j:
          cluster_matrix[class_j, class_i] += 1
  return cluster_matrix


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


# Function to extract epoch number from the key
def extract_epoch(key):
  match = re.search(r'epoch_(\d+)', key)
  return int(match.group(1)) if match else 0


def parameter_search(activations, labels, param_grid):
  best_score = -9999
  best_param = None
  for param in ParameterGrid(param_grid):
    try:
      clusters = HDBSCAN(**param).fit(activations).labels_
    except TypeError:
      continue
    score = metrics.homogeneity_score(labels, clusters)
    if score > best_score:
      best_score = score
      best_param = param
  return best_param, best_score


if __name__ == "__main__":
  import os
  import numpy as np
  import pickle as p
  import pandas as pd
  from glob import glob
  from scipy.io import loadmat
  from kneed import KneeLocator
  from collections import defaultdict

  manifold = True

  if manifold:
    from manifold import get_facet_proj
  else:
    from dim_reduction import low_rank_approximation

  experiment_path = "../logs/resnet18_run1/activations/"
  layer = "layer4.1.conv2"
  filenames = list(
    reversed(sorted(glob(os.path.join(experiment_path, layer, "*.mat")), key=extract_epoch))
  )
  non_zero_idx = None
  param = None
  df = defaultdict(list)
  np.random.seed(9001)

  for fname in filenames:
    print(f"Working on {layer} -> {os.path.basename(fname)}")
    data = loadmat(fname)
    labels = data["labels"][0]

    data = data[layer]

    if non_zero_idx is None:
      N, C, H, W = data.shape
      stat = np.reshape(np.abs(data).mean(0), (C, H * W)).mean(-1)
      y = np.sort(stat)[::-1]
      x = list(range(len(y)))
      kn = KneeLocator(x, y, curve='convex', direction='decreasing').knee
      non_zero_idx = stat >= y[kn] * 0.95
      print(f"+ Remaining: {non_zero_idx.sum()} / {data.shape[1]}")

    data = data[:, non_zero_idx.squeeze()].reshape(data.shape[0], -1)

    if manifold:
      facet_planes, plane_centers, transformed_data, _ = get_facet_proj(data)
      with open(os.path.splitext(fname)[0] + '_manifold.pkl', "wb") as f:
        p.dump(
          {
            "plane_basis": facet_planes,
            "plane_centers": plane_centers
          },
          f,
          protocol=p.HIGHEST_PROTOCOL
        )
    else:
      transformed_data, _, _ = low_rank_approximation(data, 0.95, False)

    if param is None:
      print("+ Parameter searching...")
      param, score = parameter_search(
        transformed_data, labels, {
          "cluster_selection_epsilon": [0, 0.1, 0.5],
          "min_cluster_size": [5, 10, 20, 30, 50, 60, 70],
        }
      )
      print(f"+ Best parameters {param}")

    clusters = HDBSCAN(**param).fit(transformed_data).labels_
    n_clusters = len(set(clusters)) - 1
    n_noisy_samples = 100 * sum(clusters == -1) / len(clusters)
    homogeneity_score = metrics.homogeneity_score(labels, clusters)
    completeness_score = metrics.completeness_score(labels, clusters)
    silhouette_score = metrics.silhouette_score(transformed_data, clusters)

    print(
      f"+ Num_clusters: {n_clusters} | Noisy Points: {n_noisy_samples:.2f}% | Homogeneity: {homogeneity_score:.3f} | Completeness: {completeness_score:.3f}"
    )
    df["n_clusters"].append(n_clusters)
    df["n_noisy_samples"].append(n_noisy_samples)
    df["homogeneity_score"].append(homogeneity_score)
    df["completeness_score"].append(completeness_score)
    df["silhouette_score"].append(silhouette_score)
    mat = create_cluster_matrix(clusters, labels)
    np.save(os.path.splitext(fname)[0] + "_class_vs_cluster_matrix.npy", mat)

  df = pd.DataFrame(df)
  df.to_csv(f"{os.path.join(experiment_path, layer.replace('.', '_') + '.csv')}")
