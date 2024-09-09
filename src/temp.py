import os
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from scipy.io import loadmat
from kneed import KneeLocator
import sklearn.metrics as metrics
from sklearn.cluster import HDBSCAN
from collections import defaultdict
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
  experiment_path = "../logs/resnet18_run1/"
  for l_num in range(2, 4):
    varname = f"layer{l_num}.1.conv1"
    print("Processing:", varname)
    filenames = sorted(glob(os.path.join(experiment_path, varname, "*.mat")), key=extract_epoch)
    filenames = list(reversed(filenames))
    df = defaultdict(list)
    param = None
    non_zero_idx = None
    # pbar = tqdm(filenames)
    for fname in filenames:
      # pbar.set_description(f"Processing: {os.path.basename(fname)}")
      print(f"+ {os.path.basename(fname)}")
      data = loadmat(fname)
      activations = data[varname]
      df["epoch"].append(int(os.path.basename(fname).split(".")[0].split("_")[-1]))
      if activations.sum() == 0:
        print("lol2")
        df["n_clusters"].append(0)
        df["n_noisy_samples"].append(0)
        df["homogeneity_score"].append(0)
        df["completeness_score"].append(0)
        df["silhouette_score"].append(0)
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
        print("lol")
        continue

      activations = activations[:, non_zero_idx]
      activations = activations.reshape(activations.shape[0], -1)
      print(f"Before dimensionality reduction: {activations.shape}")

      recon, _, _ = low_rank_approximation(activations, 0.95, False)
      print(f"After dimensionality reduction: {recon.shape}")

      if param is None:
        print("Parameter searching...")
        param, score = parameter_search(
          recon, labels, {
            "cluster_selection_epsilon": [0, 0.1, 0.5, 1],
            "min_cluster_size": [5, 10, 20, 30, 50, 60, 70],
          }
        )
        print(f"Best parameters {param} | Best Score: {score:.3f}")

      clusters = HDBSCAN(**param).fit(recon).labels_
      print(
        "+ Num_clusters:",
        len(set(clusters)) - 1, f"| Noisy Points: {100 * sum(clusters == -1) / len(clusters):.2f}"
      )
      df["n_clusters"].append(len(set(clusters)) - 1)
      df["n_noisy_samples"].append(100 * sum(clusters == -1) / len(clusters))
      df["homogeneity_score"].append(metrics.homogeneity_score(labels, clusters))
      df["completeness_score"].append(metrics.completeness_score(labels, clusters))
      df["silhouette_score"].append(metrics.silhouette_score(recon, clusters))

    df = pd.DataFrame(df)
    df.to_csv(f"{varname.replace('.', '_')}.csv")
