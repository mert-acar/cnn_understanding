import os
import pickle
import numpy as np
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat
import matplotlib.gridspec as gridspec
from dim_reduction import pca_reduction
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
  exp_dir = "../logs/customnet_run2/"
  epoch = 33

  data = loadmat(os.path.join(exp_dir, "activations", f"act_epoch_{epoch}.mat"))
  labels = loadmat("../data/labels.mat")["labels"][0]
  l = list(map(str, range(10)))

  # adj = {}
  # for i in range(10):
  #   var = f"features.{i}"
  #   adj[var] = np.zeros((10, 10))
  #   x = data[var]
  #   x = x.reshape(x.shape[0], -1)
  #   x = x - x.mean(0)
  #   x = x / np.abs(x).max()
  #   x = pca_reduction(x, n_components=None, threshold=0.95)
  #   print(var, "->", x.shape)
  #   dim = np.sqrt(x.shape[-1])
  #   for j in range(10):
  #     for k in range(j + 1, 10):
  #       lbl_j = x[labels == j]
  #       lbl_k = x[labels == k]
  #       d = (cdist(lbl_j, lbl_k) / dim).mean()
  #       adj[var][j, k] = d
  #       adj[var][k, j] = d

  adj = loadmat(os.path.join(exp_dir, "clusters", "adjacency_matrices.mat"))
  np.set_printoptions(precision=2, suppress=True)
  # for i in range(10):
  #   key = f"features.{i}"
  #   print(key)
  #   print(adj[key])
  #   print()

  vmin, vmax = 0, 2
  for i in range(0, 10, 2):
    # Create a new figure for each pair
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05])  # Third column for color bar

    # Create subplots for the heatmaps
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    cbar_ax = fig.add_subplot(gs[0, 2])

    # Plot each matrix as a heatmap
    sns.heatmap(
      adj[f"features.{i}"],
      annot=True,
      fmt=".2f",
      ax=ax1,
      vmin=vmin,
      vmax=vmax,
      xticklabels=l,
      yticklabels=l,
      cmap="viridis",
      cbar=True,
      cbar_ax=cbar_ax
    )
    sns.heatmap(
      adj[f"features.{i + 1}"],
      annot=True,
      fmt=".2f",
      ax=ax2,
      vmin=vmin,
      vmax=vmax,
      xticklabels=l,
      yticklabels=l,
      cmap="viridis",
      cbar=False
    )

    # # Add the color bar on the third subplot
    # sns.heatmap(
    #   adj[f"features.{i}"], ax=ax1, cmap="viridis", cbar=True, vmin=vmin, vmax=vmax, cbar_ax=cbar_ax
    # )

    # Set titles or labels as needed
    ax1.set_title(f"features.{i} [conv]")
    ax1.set_xlabel("Labels")
    ax1.set_ylabel("Labels")
    ax2.set_title(f"features.{i + 1} [ReLU]")
    ax2.set_xlabel("Labels")
    ax2.set_ylabel("Labels")

    # Adjust layout
    plt.tight_layout()
    plt.savefig(f"features_{i}.png", bbox_inches="tight")
