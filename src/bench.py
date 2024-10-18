import os
import pickle as p
import numpy as np
from pprint import pprint
from scipy.spatial.distance import cdist
from sklearn import cluster
from scipy.io import loadmat
from cluster import parameter_search
from dim_reduction import svd_reduction
from cluster import select_random_samples

if __name__ == "__main__":
  exp_dir = "../logs/customnet_run2/"
  # out_dir = os.path.join(exp_dir, "figures")
  # os.makedirs(out_dir, exist_ok=True)

  labels = loadmat("../data/labels.mat")["labels"][0]

  epoch = 33
  out = {}
  # mds = manifold.MDS(n_components=2, metric=False, max_iter=1000, eps=1e-8, n_jobs=-1)
  for i in range(0, 9, 2):
    vars = [f"features.{i}_input", f"features.{i}_output"]
    for var in vars:
      data = loadmat(
        os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"), variable_names=[var]
      )
      print(var)
      x = data[var]
      n = 18000 // (10 * x.shape[1])
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
      # embedded = mds.fit_transform(x)
      d = cdist(x, x).mean()
      x = svd_reduction(x, n_components=None, threshold=0.98)
      print(f"After SVD: {x.shape}")
      params = {"n_clusters": [None], "distance_threshold": [k * d for k in np.linspace(0.2, 20, 20)]}
      clusters, p, scores = parameter_search(x, l, cluster.AgglomerativeClustering, params)
      pprint(p)
      pprint(scores)
      print()
      out[var] = {"cluster_labels": clusters, "params": p, "scores": scores}

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}.p"), "wb") as f:
    p.dump(out, f, protocol=p.HIGHEST_PROTOCOL)

      # _, axs = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6))
      # axs[0].scatter(embedded[:, 0], embedded[:, 1], c=l)
      # axs[0].grid(True)
      # axs[0].axis("equal")
      # axs[0].set_title(f"Original Labels (n = {len(set(l))})")
      # axs[1].scatter(embedded[:, 0], embedded[:, 1], c=clusters)
      # axs[1].grid(True)
      # axs[1].axis("equal")
      # axs[1].set_title(f"Cluster Labels (n = {len(set(clusters))})")
      # plt.suptitle(var)
      # plt.savefig(os.path.join(out_dir, var.replace(".", "_") + ".png"), bbox_inches="tight")
      # plt.clf()
      # plt.close()
