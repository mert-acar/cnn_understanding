import os
import numpy as np
import pickle as p
import pandas as pd
from scipy.spatial.distance import cdist
from yaml import full_load
from scipy.io import loadmat
from utils import load_labels
import matplotlib.pyplot as plt
from collections import defaultdict
from dim_reduction import svd_reduction


def chi(data, cluster_labels):
  extra_disp = bcss(data, cluster_labels)
  intra_disp = wcss(data, cluster_labels)
  n_samples = len(data)
  n_labels = len(np.unique(cluster_labels))
  return extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))


def wcss(data, cluster_labels):
  unique_clusters = np.unique(cluster_labels)
  score = 0
  for c in unique_clusters:
    cluster_points = data[cluster_labels == c]
    cluster_center = cluster_points.mean(0)
    score += np.sum((cluster_points - cluster_center)**2)
  return score


def bcss(data, cluster_labels):
  data_center = data.mean(0)
  unique_clusters = np.unique(cluster_labels)
  score = 0
  for c in unique_clusters:
    cluster_points = data[cluster_labels == c]
    cluster_center = cluster_points.mean(0)
    score += len(cluster_points) * ((cluster_center - data_center)**2).sum()
  return score


if __name__ == "__main__":
  from tqdm import tqdm
  exp_dir = "../logs/customnet_run2/"
  epoch = 33

  labels = load_labels()

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}_v2.p"), "rb") as f:
    clusters = p.load(f)

  in_data = defaultdict(list)
  out_data = defaultdict(list)
  for i in tqdm(range(0, 9, 2)):
    for arr, s in zip([in_data, out_data], ["input", "output"]):
      var = f"features.{i}_" + s
      # data = loadmat(
      #   os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"),
      #   variable_names=[var]
      # )[var][clusters[var]["idx"]]
      # data = data.reshape(-1, data.shape[-1])
      # data = data - data.mean(0)
      # data = data / np.abs(data).max()
      # data = svd_reduction(data, n_components=None, threshold=0.98)
      # cluster_labels = clusters[var]["cluster_labels"]
      # d = cdist(data, data).mean()

      arr["var"].append(f"features.{i}")
      arr["num_clusters"].append(clusters[var]["scores"]["num_clusters"])
      arr["homogeneity"].append(clusters[var]["scores"]["homogeneity"])
      arr["completeness"].append(clusters[var]["scores"]["completeness"])
      # arr["dist_thresh"].append(clusters[var]["params"]["distance_threshold"])
      # arr["mean_dist"].append(d)
      # arr["k"].append(clusters[var]["params"]["distance_threshold"] / d)
      arr["CHI"].append(clusters[var]["scores"]["calinski_harabasz_score"])
      arr["silhouette"].append(clusters[var]["scores"]["silhouette"])

  print(pd.DataFrame(in_data))
  print(pd.DataFrame(out_data))
