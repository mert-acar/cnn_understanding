import os
import numpy as np
import pandas as pd
from time import time
from scipy.io import loadmat
from collections import defaultdict
from sklearn import cluster, metrics
from sklearn.model_selection import ParameterGrid

from dim_reduction import pca_reduction
from cluster import select_random_samples

if __name__ == "__main__":
  exp_dir = "../logs/customnet_run2/"
  labels = loadmat("../data/labels.mat")["labels"][0]
  idx = select_random_samples(labels, 700)
  labels = labels[idx]
  epoch = 33
  out = defaultdict(list)
  metric = "rbf"
  for i in range(0, 9, 2):
    var = f"features.{i}"
    print(var, "\n----------------")

    x = loadmat(
      os.path.join(exp_dir, "activations", f"act_epoch_{epoch}.mat"), variable_names=[var]
    )[var]

    x = x[idx]

    x = x.reshape(x.shape[0], -1)
    x = x - x.mean(0)
    x = x / np.abs(x).max()

    print(f"Before PCA: {x.shape}")
    o = x.shape[-1]
    x = pca_reduction(x, n_components=None, threshold=0.95)
    r = x.shape[-1]
    print(f"After PCA: {x.shape}")

    params = {
      "n_clusters": [8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18],
      "affinity": [metric],
      "n_jobs": [-1],
      "n_neighbors": [20],
      "eigen_solver": ["amg"],
      "eigen_tol": [5e-5],
      "assign_labels": ["cluster_qr"]
    }

    for param in ParameterGrid(params):
      tick = time()
      cluster_labels = cluster.SpectralClustering(**param).fit(x).labels_
      t = time() - tick
      h, c = metrics.homogeneity_score(labels, cluster_labels), metrics.completeness_score(
        labels, cluster_labels
      )
      s = metrics.silhouette_score(x, cluster_labels)
      out["layer"].append(var)
      out["original_dimension"].append(o)
      out["reduced_dimension"].append(r)
      out["n_clusters"].append(param["n_clusters"])
      out["affinity"].append(param["affinity"])
      out["homogeneity_score"].append(float(h))
      out["completeness_score"].append(float(c))
      out["silhouette_score"].append(float(s))
      print(
        f"\t + N: {param['n_clusters']}, Affinity: {param['affinity']}, -> Homogeneity Score: {h:.3f} | Completeness Score: {c:.3f} | Silhouette Score: {s:.3f} | time: {t:.3f}s"
      )
    print()

  out = pd.DataFrame(out)
  out.to_csv(f"spectral_results_{metric}.csv")
