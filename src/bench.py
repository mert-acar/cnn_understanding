import os
import pickle as p
import numpy as np
from yaml import full_load
from scipy.io import loadmat
import matplotlib.pyplot as plt

if __name__ == "__main__":
  exp_dir = "../logs/customnet_run2/"
  epoch = 33

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}_v2.p"), "rb") as f:
    clusters = p.load(f)

  with open(os.path.join(exp_dir, "ExperimentSummary.yaml"), "r") as f:
    model_config = [x[1:] for x in full_load(f)["model_config"]["features"]]

  feat_idx = 0
  H = 28
  fi, fo, k, s, p = model_config[feat_idx // 2]

  points = loadmat(
    os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"),
    variable_names=[f"features.{feat_idx}_input", f"features.{feat_idx}_output"]
  )
  idx = clusters[f"features.{feat_idx}_input"]["idx"]
  data = points[f"features.{feat_idx}_input"][idx]
  N, D, C = data.shape
  h = (H + 2 * p - k) // s + 1
  ns = len(idx) // 10

  positions_y = np.arange(0, H - k + 1, s)
  positions_x = np.arange(0, H - k + 1, s)
  grid_y, grid_x = np.meshgrid(positions_y, positions_x, indexing='ij')
  center_offset = (k - 1) // 2
  center_y = grid_y + center_offset
  center_x = grid_x + center_offset
  patch_centers = np.stack((center_y.flatten(), center_x.flatten()), axis=1)

  cluster_labels = clusters[f"features.{feat_idx}_input"]["cluster_labels"]
  unique_clusters = np.unique(cluster_labels)

  cluster_labels = cluster_labels.reshape(N, D)
  inp_plots = np.zeros((10, len(unique_clusters), H, H))
  for i in range(10):
    labels = cluster_labels[i * ns:(i + 1) * ns].flatten()
    for j, lbl in enumerate(labels):
      k = j % (h * h)
      y, x = patch_centers[k]
      inp_plots[i, lbl, y, x] = inp_plots[i, lbl, y, x] + 1
  _, axs = plt.subplots(10, len(unique_clusters), tight_layout=True)
  for row in range(10):
    for col in range(len(unique_clusters)):
      axs[row, col].imshow(inp_plots[row, col], cmap='gray')
      axs[row, col].axis(False)
      if row == 0:
        axs[row, col].set_title(f"cluster {col + 1}")
  plt.show()

  cluster_labels = clusters[f"features.{feat_idx}_input"]["cluster_labels"]
  inp_plots = np.zeros((len(unique_clusters), H, H))
  for i, lbl in enumerate(cluster_labels):
    j = i % (h * h)
    y, x = patch_centers[j]
    inp_plots[lbl, y, x] = inp_plots[lbl, y, x] + 1
  first_row = np.ceil(len(unique_clusters) / 2).astype(int)
  second_row = len(unique_clusters) - first_row
  _, axs = plt.subplots(2, max(first_row, second_row), tight_layout=True)
  for i in range(first_row):
    axs[0, i].imshow(inp_plots[i], cmap='gray')
    axs[0, i].axis(False)
    axs[0, i].set_title(f"cluster {i + 1}")
  for i in range(second_row):
    axs[1, i].imshow(inp_plots[i + first_row], cmap='gray')
    axs[1, i].axis(False)
    axs[1, i].set_title(f"cluster {i + 1 + first_row}")
  if len(unique_clusters) % 2 != 0:
    axs[1, -1].axis("off")
  plt.show()

  cluster_labels = clusters[f"features.{feat_idx}_output"]["cluster_labels"]
  unique_clusters = np.unique(cluster_labels)

  cluster_labels = cluster_labels.reshape(N, D)
  out_plots = np.zeros((10, len(unique_clusters), h, h))
  for i in range(10):
    labels = cluster_labels[i * ns:(i + 1) * ns].flatten()
    for j, lbl in enumerate(labels):
      k = j % (h * h)
      y, x = k // h, k % h
      out_plots[i, lbl, y, x] = out_plots[i, lbl, y, x] + 1
  _, axs = plt.subplots(10, len(unique_clusters), tight_layout=True)
  for row in range(10):
    for col in range(len(unique_clusters)):
      axs[row, col].imshow(out_plots[row, col].squeeze(), cmap='gray')
      axs[row, col].axis(False)
      if row == 0:
        axs[row, col].set_title(f"cluster {col + 1}")
  plt.show()

  cluster_labels = clusters[f"features.{feat_idx}_output"]["cluster_labels"]
  out_plots = np.zeros((len(unique_clusters), h, h))
  for i, lbl in enumerate(cluster_labels):
    j = i % (h * h)
    y, x = j // h, j % h
    out_plots[lbl, y, x] = out_plots[lbl, y, x] + 1
  first_row = np.ceil(len(unique_clusters) / 2).astype(int)
  second_row = len(unique_clusters) - first_row
  _, axs = plt.subplots(2, max(first_row, second_row), tight_layout=True)
  for i in range(first_row):
    axs[0, i].imshow(out_plots[i], cmap='gray')
    axs[0, i].axis(False)
    axs[0, i].set_title(f"cluster {i + 1}")
  for i in range(second_row):
    axs[1, i].imshow(out_plots[i + first_row], cmap='gray')
    axs[1, i].axis(False)
    axs[1, i].set_title(f"cluster {i + 1 + first_row}")
  if len(unique_clusters) % 2 != 0:
    axs[1, -1].axis("off")
  plt.show()
