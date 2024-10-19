import os
import pickle
import numpy as np
from pprint import pprint
from sklearn import cluster
from scipy.io import loadmat
from cluster import parameter_search
from dim_reduction import svd_reduction
from scipy.spatial.distance import cdist
from cluster import select_random_samples

if __name__ == "__main__":
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
      d = cdist(x, x).mean()
      x = svd_reduction(x, n_components=None, threshold=0.98)
      print(f"After SVD: {x.shape}")
      params = {
        "n_clusters": [None],
        "distance_threshold": [k * d for k in np.linspace(0.2, 20, 20)],
      }
      clusters, p, scores = parameter_search(x, l, cluster.AgglomerativeClustering, params)
      pprint(p)
      pprint(scores)
      print()
      out[var] = {
        "cluster_labels": clusters,
        "params": p,
        "scores": scores,
        "idx": idx,
      }

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}.p"), "wb") as f:
    pickle.dump(out, f, protocol=pickle.HIGHEST_PROTOCOL)
