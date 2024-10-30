import os
import numpy as np
import pickle as p
from scipy.io import loadmat
from utils import load_labels
import matplotlib.pyplot as plt
import seaborn as sns

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

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}.p"), "rb") as f:
    clusters = p.load(f)


  pbar = tqdm(total=10)
  for i in range(0, 9, 2):
    for var in [f"features.{i}_input", f"features.{i}_output"]:
      data = loadmat(
        os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"), variable_names=[var]
      )[var][clusters[var]["idx"]]
      l = np.repeat(labels[clusters[var]["idx"]], data.shape[1])
      data = data.reshape(-1, data.shape[-1])
      data = data - data.mean(0)
      data = data / np.abs(data).max()
      cluster_labels = clusters[var]["cluster_labels"]
      unique_clusters = np.unique(cluster_labels)
      data_center = data.mean(0)
      heatmap = np.zeros((10, len(unique_clusters)), dtype=int)
      for c in unique_clusters:
        for label in range(10):
          heatmap[label, c] = np.sum((l == label) & (cluster_labels == c))


      plt.figure(figsize=(10, 8))
      sns.heatmap(heatmap, annot=True, fmt='d', xticklabels=unique_clusters, yticklabels=list(range(10)), cmap="Blues")
      plt.xlabel('Cluster Labels')
      plt.ylabel('True Labels')
      plt.title('Heatmap of Label vs Cluster Distribution')
      plt.savefig(var + "_heatmap.png", bbox_inches="tight")
      plt.clf()
      plt.close("all")
      pbar.update(1)
