import os
import numpy as np
import pickle as p
import pandas as pd
from tqdm import tqdm
from scipy.io import loadmat
from functools import partial
import sklearn.metrics as metrics
import sklearn.cluster as cluster
from collections import defaultdict
from scipy.spatial.distance import pdist
from manifold import get_facet_proj, transform
from dim_reduction import low_rank_approximation
from sklearn.model_selection import ParameterGrid
from utils import get_filenames, find_non_zero_idx, normalize


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


def main(experiment_path, layer, manifold, create_matrix, algo):
  filenames = get_filenames(layer, experiment_path=experiment_path)
  non_zero_idx = None
  param = None
  df = defaultdict(list)
  np.random.seed(9001)
  print(f"+ Working on {layer}")
  pbar = tqdm(filenames)
  for fname in pbar:
    pbar.set_description(os.path.basename(fname))
    data = loadmat(fname)
    labels = data["labels"][0]

    data = data[layer]

    if non_zero_idx is None:
      non_zero_idx = find_non_zero_idx(data)
      print(f"Remaining: {non_zero_idx.sum()} / {data.shape[1]}")

    data = data[:, non_zero_idx.squeeze()].reshape(data.shape[0], -1)
    data = normalize(data)

    if manifold:
      pkl = os.path.splitext(fname)[0] + '_manifold.pkl'
      if os.path.exists(pkl):
        with open(pkl, "rb") as f:
          facet_dict = p.load(f)
        transformed_data, _ = transform(
          facet_dict["plane_basis"], facet_dict["plane_centers"], data
        )
      else:
        facet_planes, plane_centers, transformed_data, _ = get_facet_proj(data)
        with open(pkl, "wb") as f:
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

    if algo == "HDBSCAN":
      cluster_algorithm = cluster.HDBSCAN
      param_grid = {"min_cluster_size": [5, 10, 20, 30, 50, 60, 70]}
    else:
      cluster_algorithm = partial(cluster.AgglomerativeClustering, n_clusters=None)
      dist = pdist(transformed_data, metric='euclidean').mean()
      param_grid = {"distance_threshold": [i * dist for i in np.linspace(6, 8, 8)]}

    best_score = -9999
    best_param = None
    for param in ParameterGrid(param_grid):
      try:
        clusters = cluster_algorithm(**param).fit(transformed_data).labels_
      except TypeError:
        continue
      h = metrics.homogeneity_score(labels, clusters)
      c = metrics.completeness_score(labels, clusters)
      try:
        s = metrics.silhouette_score(transformed_data, clusters)
      except ValueError:
        s = -1
      score = (h + c + s) / 3
      if score > best_score:
        best_score = score
        best_param = param
        best_clusters = clusters
        best_h = h
        best_c = c
        best_s = s
        best_n = (np.unique(clusters) != -1).sum()
        n_noisy_samples = 100 * sum(clusters == -1) / len(clusters)

    print(f"Best parameters: {best_param}")
    print(
      f"+ Num_clusters: {best_n} | Homogeneity: {best_h:.3f} | Completeness: {best_c:.3f} | Sillhouette: {best_s:.3f}"
    )

    df["n_clusters"].append(best_n)
    df["n_noisy_samples"].append(n_noisy_samples)
    df["homogeneity_score"].append(best_h)
    df["completeness_score"].append(best_c)
    df["silhouette_score"].append(best_s)
    if create_matrix:
      mat = create_cluster_matrix(best_clusters, labels)
      np.save(os.path.splitext(fname)[0] + f"_{algo}_manifold_{manifold}_cc.npy", mat)

  df = pd.DataFrame(df)
  df.to_csv(
    os.path.join(experiment_path, f"{layer.replace('.', '_')}_{algo}_manifold_{manifold}.csv")
  )


if __name__ == "__main__":
  from fire import Fire
  # main(experiment_path, layer, manifold, create_matrix, algo):
  Fire(main)
