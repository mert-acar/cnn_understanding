import sys

sys.path.append("../")

import os
import numpy as np
import pickle as p
import pandas as pd
from time import time
from collections import defaultdict
from sklearn import cluster, metrics
from scipy.io import loadmat, savemat
from scipy.spatial.distance import cdist

from cluster import bcss_wcss
from utils import load_MNIST_labels
from dim_reduction import pca_reduction
from dataset import select_random_samples, create_dataloader


def normalized_minkowski(x: np.ndarray, y: np.ndarray) -> float:
  return np.linalg.norm(x - y) / np.sqrt(x.shape[0])


if __name__ == "__main__":
  # exp_dir = "../logs/resnet18_IMAGENET/"
  # vars = ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]
  # exp_dir = "../logs/densenet121_IMAGENET/"
  # vars = ["features.conv0"] + [f"features.denseblock{i}" for i in range(1, 5)]
  # exp_dir = "../logs/efficientnetb3_IMAGENET/"
  # vars = [f"features.{i}" for i in range(1, 8)]
  # exp_dir = "../logs/customnet_IMAGENET/"
  # vars = [f"features.{i}" for i in range(1, 8)]

  # labels = create_dataloader("imagenet", "../data/ImageNet", "val").dataset.targets
  # idx = None


  # exp_dir = "../logs/resnet18_MNIST/"
  # vars = ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)]
  exp_dir = "../logs/densenet121_MNIST/"
  vars = ["features.conv0"] + [f"features.denseblock{i}" for i in range(1, 5)]
  # exp_dir = "../logs/efficientnetb2_MNIST/"
  # vars = [f"features.{i}" for i in range(1, 8)]
  # exp_dir = "../logs/customnet_MNIST/"
  # vars = [f"features.{i}" for i in range(1, 8)]

  labels = load_MNIST_labels()
  idx = select_random_samples(labels, 700)
  labels = labels[idx]
  
  param = {
    "affinity": "precomputed",
    "n_jobs": -1,
    "eigen_solver": "amg",
    "eigen_tol": 5e-5,
    "assign_labels": "cluster_qr"
  }

  out = {}
  scores = defaultdict(list)
  for var in vars:
    print(var, "\n----------------")
    out[var] = []

    x = loadmat(
      os.path.join(exp_dir, "activations", f"act_pretrained.mat"),
      variable_names=[var]
    )[var]

    if idx is not None:
      x = x[idx]

    x = x.reshape(x.shape[0], -1)
    x = x - x.mean(0)
    x = x / np.abs(x).max()

    print(f"Before PCA: {x.shape}")
    o = x.shape[-1]
    tick = time()
    x = pca_reduction(x, n_components=None, threshold=0.95)
    print(f"PCA took {time() - tick:.3f} seconds")
    r = x.shape[-1]
    print(f"After PCA: {x.shape}")

    os.makedirs(os.path.join(exp_dir, "clusters"), exist_ok=True)
    affinity_path = os.path.join(
      exp_dir, "clusters", f"{var.replace('.', '_')}_custom_metric_affinity.mat"
    )
    if os.path.exists(affinity_path):
      affinity = loadmat(affinity_path)["affinity"]
    else:
      print(f"Calculating new affinity matrix for {var}...")
      tick = time()
      affinity = np.exp(-cdist(x, x, metric=normalized_minkowski)**2 / 4)
      print(f"Affinity matrix is calculated in {time() - tick:.3f} seconds")
      savemat(affinity_path, {"affinity": affinity})

    for n in [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]:
      tick = time()
      algo = cluster.SpectralClustering(n_clusters=n, **param)
      cluster_labels = algo.fit(affinity).labels_
      t = time() - tick

      s = metrics.silhouette_score(x, cluster_labels)
      h = metrics.homogeneity_score(labels, cluster_labels)
      c = metrics.completeness_score(labels, cluster_labels)
      bcss, wcss, chi = bcss_wcss(x, cluster_labels, True)

      out[var].append({
        "idx": None,  # idx,
        "cluster_labels": cluster_labels,
        "params": param,
      })
      scores["layer"].append(var)
      scores["metric"].append("custom")
      scores["original_dim"].append(o)
      scores["reduced_dim"].append(r)
      scores["n_clusters"].append(n)
      scores["silhouette"].append(s)
      scores["homogeneity"].append(h)
      scores["completeness"].append(c)
      scores["chi"].append(chi)
      scores["bcss"].append(bcss)
      scores["wcss"].append(wcss)

      print(
        f"\t + N: {n}, Affinity: {param['affinity']}, -> Homogeneity Score: {h:.3f} | Completeness Score: {c:.3f} | Silhouette Score: {s:.3f} | time: {t:.3f}s"
      )
    print()

  out_path = os.path.join(exp_dir, "clusters")
  os.makedirs(out_path, exist_ok=True)

  with open(os.path.join(out_path, "spectral_custom_metric.p"), "wb") as f:
    p.dump(out, f, protocol=p.HIGHEST_PROTOCOL)

  df = pd.DataFrame(scores)
  df.to_csv(os.path.join(out_path, "spectral_custom_metric_scores.csv"))
