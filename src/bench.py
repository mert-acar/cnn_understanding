import os
import re
import numpy as np
from tqdm import tqdm
from glob import glob
from scipy.io import loadmat
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from sklearn.metrics import homogeneity_score
from dim_reduction import low_rank_approximation
from sklearn.model_selection import ParameterGrid


# Function to extract epoch number from the key
def extract_epoch(key):
  match = re.search(r'epoch_(\d+)', key)
  return int(match.group(1)) if match else 0


def parameter_search(activations, labels, param_grid):
  best_score = -9999
  best_param = None
  for param in ParameterGrid(param_grid):
    clusters = HDBSCAN(**param).fit(activations).labels_
    score = homogeneity_score(labels, clusters)
    if score > best_score:
      best_score = score
      best_param = param
  return best_param, best_score


if __name__ == "__main__":
  varname = "layer1.1.conv2"
  experiment_path = "../logs/resnet18_run1/"
  filenames = sorted(glob(os.path.join(experiment_path, varname, "*.mat")), key=extract_epoch)
  filenames = list(reversed(filenames))
  scores = []
  n_clusters = []
  n_noisy_samples = []
  param = None
  non_zero_idx = None
  pbar = tqdm(filenames)
  for fname in pbar:
    pbar.set_description(f"Processing: {os.path.basename(fname)}")
    data = loadmat(fname)
    activations = data[varname]
    if activations.sum() == 0:
      n_clusters.append(0)
      n_noisy_samples.append(100)
      scores.append(0)
      continue
    labels = data["labels"][0]

    N, C, H, W = activations.shape
    stat = np.reshape(np.abs(activations).mean(0), (C, H * W)).mean(-1)
    y = np.sort(stat)[::-1]
    x = list(range(len(y)))
    kn = KneeLocator(x, y, curve='convex', direction='decreasing').knee
    act_threshold = y[kn] * 0.95
    non_zero_idx = stat >= act_threshold
    print("+ Num non-zero idx:", non_zero_idx.sum())

    if sum(non_zero_idx) == 0:
      non_zero_idx[:] = True
    activations = activations[:, non_zero_idx]
    activations = activations.reshape(activations.shape[0], -1)

    recon, _, _ = low_rank_approximation(activations, 0.95, False)
    print("Parameter searching...")
    param, score = parameter_search(
      recon, labels, {
        "min_cluster_size": [5, 10, 20, 30, 50, 60, 70],
      }
    )
    print(f"Best parameters {param} | Best Score: {score:.3f}")

    clusters = HDBSCAN(**param).fit(recon).labels_
    score = homogeneity_score(labels, clusters)
    scores.append(score)
    n_clusters.append(len(set(clusters)) - 1)
    n_noisy_samples.append(100 * sum(clusters == -1) / len(clusters))

  _, axs = plt.subplots(1, 2, figsize=(20, 7), tight_layout=True)
  epochs = list(reversed((range(1, len(scores) + 1))))
  axs[0].plot(epochs, scores, marker="o")
  axs[0].set_ylabel("Purity Score")
  axs[0].set_xlabel("Epoch")
  axs[0].grid(True)
  axs[0].set_title("Purity Score Across Epochs")
  axs[1].errorbar(epochs, n_clusters, yerr=n_noisy_samples, marker="x", ecolor="red", color="blue")
  axs[1].set_xlabel("Epoch")
  axs[1].grid(True)
  axs[1].set_ylabel("Number of Clusters")
  axs[1].set_title("Number of Clusters Across Epochs (with noisy sample percentage)")
  for epoch, n_cluster, yerr in zip(epochs, n_clusters, n_noisy_samples):
    axs[1].annotate(
      f'{n_cluster}', (epoch, n_cluster + yerr),
      textcoords="offset points",
      xytext=(0, 5),
      ha='center'
    )
  plt.suptitle(varname)
  plt.show()
