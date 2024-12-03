import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from utils import combine_scores

def add_pos_cube(ax, l=0.4):
  vertices = [
    [0, 0, 0],
    [l, 0, 0],
    [l, l, 0],
    [0, l, 0],
    [0, 0, l],
    [l, 0, l],
    [l, l, l],
    [0, l, l],
  ]
  faces = [
    [vertices[0], vertices[1], vertices[5], vertices[4]],
    [vertices[1], vertices[2], vertices[6], vertices[5]],
    [vertices[2], vertices[3], vertices[7], vertices[6]],
    [vertices[0], vertices[3], vertices[7], vertices[4]],
    [vertices[0], vertices[1], vertices[2], vertices[3]],
    [vertices[4], vertices[5], vertices[6], vertices[7]],
  ]
  cube = Poly3DCollection(faces, color='blue', alpha=0.05, edgecolor='k')
  ax.add_collection3d(cube)


def plot_3d_derivatives(data, cube=True):
  l = 0.4
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')
  x = np.diff(data[:, 0])
  y = np.diff(data[:, 1])
  z = np.diff(data[:, 2])
  ax.scatter(x, y, z, alpha=0.7, marker='o', label=exp_path.split("/")[-2])
  if cube:
    add_pos_cube(ax, l)
  ax.set_xlabel('d[silhouette]')
  ax.set_ylabel('d[homogeneity]')
  ax.set_zlabel('d[completeness]')
  ax.set_xlim([-l, l])
  ax.set_ylim([-l, l])
  ax.set_zlim([-l, l])
  ax.quiver(-l, 0, 0, 2 * l, 0, 0, color="black", arrow_length_ratio=0.05)
  ax.quiver(0, -l, 0, 0, 2 * l, 0, color="black", arrow_length_ratio=0.05)
  ax.quiver(0, 0, -l, 0, 0, 2 * l, color="black", arrow_length_ratio=0.05)
  ax.legend()
  plt.subplots_adjust(0, 0, 1, 1)
  plt.show()


def plot_combined_score(data, layers, title=""):
  combined, _ = combine_scores(data[:, 1:])
  _, axs = plt.subplots(2, 1, tight_layout=True, figsize=(7, 8))
  axs[0].plot(layers, combined, label="combined feature")
  axs[0].set_xlabel("Layers")
  axs[0].set_xticks(layers)
  axs[0].set_ylabel("Score")
  axs[0].grid()
  axs[0].set_title("Combined Score vs. Layers")
  diff = np.diff(combined)
  axs[1].plot(layers[1:], diff, label="combined feature")
  axs[1].set_xlabel("Layers")
  axs[1].set_xticks(layers[1:])
  axs[1].set_ylabel("Score")
  axs[1].grid()
  axs[1].set_ylim([min(-0.02, min(diff) * 1.1), max(0.02, max(diff) * 1.1)])
  axs[1].set_title("Derivative of Combined Score vs. Layers")
  axs[1].axhline(y=0, linestyle="--", color='red', linewidth=2)
  plt.subplots_adjust(0, 0, 1, 1)
  plt.suptitle(title)
  plt.show()


def plot_3d_scores(data, max_var_arrow=True):
  l = 1.5
  fig = plt.figure(figsize=(8, 8))
  ax = fig.add_subplot(111, projection='3d')
  x = data[:, 0]
  y = data[:, 1]
  z = data[:, 2]
  ax.scatter(x, y, z, alpha=0.7, marker='o', label=exp_path.split("/")[-2])
  if max_var_arrow:
    _, weights = combine_scores(data)
    ax.quiver(
      0,
      0,
      0,
      weights[0],
      weights[1],
      weights[2],
      color='r',
      arrow_length_ratio=0.1,
      label="direction of highest variance"
    )
  ax.set_xlabel('silhouette')
  ax.set_ylabel('homogeneity')
  ax.set_zlabel('completeness')
  ax.set_xlim([-l, l])
  ax.set_ylim([-l, l])
  ax.set_zlim([-l, l])
  ax.quiver(-l, 0, 0, 2 * l, 0, 0, color="black", arrow_length_ratio=0.05)
  ax.quiver(0, -l, 0, 0, 2 * l, 0, color="black", arrow_length_ratio=0.05)
  ax.quiver(0, 0, -l, 0, 0, 2 * l, color="black", arrow_length_ratio=0.05)
  ax.legend()
  plt.subplots_adjust(0, 0, 1, 1)
  plt.show()


def vis2d(activations: np.ndarray):
  _, axs = plt.subplots(2, 4, figsize=(11, 10), tight_layout=True)
  for i, ax in enumerate(axs.ravel()):
    ax.imshow(activations[i], cmap='gray')
    ax.axis(False)
    ax.set_title(f"{np.abs(activations[i]).sum():.3f}")
  plt.suptitle(f"Avg Activation")
  plt.show()


def vis3d(activations: np.ndarray):
  fig = plt.figure(figsize=(11, 10))
  ax = fig.add_subplot(111, projection='3d')
  x, y, z = np.indices(activations.shape)
  x = x.flatten()
  y = y.flatten()
  z = z.flatten()
  values = np.abs(activations.flatten())
  scatter = ax.scatter(
    x[values > 0], y[values > 0], z[values > 0], c=values[values > 0], cmap='viridis'
  )
  color_bar = plt.colorbar(scatter, ax=ax)
  color_bar.set_label('Intensity')
  ax.set_xlabel('Channels')
  ax.set_ylabel('H')
  ax.set_zlabel('W')
  plt.show()


def vis(embedded: np.ndarray, clusters: np.ndarray, labels: np.ndarray):
  _, axs = plt.subplots(1, 2, figsize=(15, 7), tight_layout=True)
  colormap = mpl.colormaps['tab20']
  for j, lbls in enumerate([clusters, labels]):
    ax = axs[j]
    for i, c in enumerate(reversed(list(set(lbls)))):
      idx = lbls == c
      if c == -1:
        lbl = "Noisy Samples"
        color = 'gray'
      else:
        lbl = f"Cluster {c}"
        color = colormap(i)
      ax.scatter(embedded[idx, 0], embedded[idx, 1], color=color, label=lbl, alpha=0.3)
    ax.grid(True)
    ax.set_title(f"{'predicted' if j == 0 else 'label'} clusters")
    ax.legend()
  plt.suptitle(f"Activations")
  plt.show()


def create_multilayer_sankey(cluster_labels_layers: np.ndarray):
  num_layers = len(cluster_labels_layers)
  labels = []
  sources = []
  targets = []
  values = []
  current_label_idx = 0
  label_indices = []
  for layer_idx in range(num_layers):
    unique_clusters = np.unique(cluster_labels_layers[layer_idx])
    layer_labels = [f"L{layer_idx+1}_{cluster}" for cluster in unique_clusters]
    labels.extend(layer_labels)
    label_indices.append({
      cluster: current_label_idx + i
      for i, cluster in enumerate(unique_clusters)
    })
    current_label_idx += len(unique_clusters)

  for layer_idx in range(num_layers - 1):
    clusters_A = cluster_labels_layers[layer_idx]
    clusters_B = cluster_labels_layers[layer_idx + 1]
    contingency_table = pd.crosstab(clusters_A, clusters_B)
    label_idx_A = label_indices[layer_idx]
    label_idx_B = label_indices[layer_idx + 1]
    for cluster_A in contingency_table.index:
      for cluster_B in contingency_table.columns:
        count = contingency_table.loc[cluster_A, cluster_B]
        if count > 0:
          sources.append(label_idx_A[cluster_A])
          targets.append(label_idx_B[cluster_B])
          values.append(count)

  fig = go.Figure(
    go.Sankey(
      node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
      link=dict(source=sources, target=targets, value=values)
    )
  )
  fig.update_layout(title_text="Cluster Transitions Across Layers", font_size=10)
  fig.show()


def main(mode: str, exp_file: str, lbl: str, threshold: float = 0.1):
  inp_path = Path(exp_file)
  data = loadmat(inp_path, variable_names=["activations", "labels"])
  activations = data["activations"]
  labels = data["labels"][0]
  activations = activations[labels == lbl][0]
  activations[np.abs(activations) < threshold] = 0
  print(f"[INFO] Plotting a random activation for the label {lbl}")
  if mode == '2d':
    vis2d(activations)
  else:
    vis3d(activations)


if __name__ == "__main__":
  import os

  exp_path = "../logs/resnet18_MNIST/"
  data_path = "clusters/spectral_custom_metric_scores.csv"
  scores = ["silhouette", "homogeneity", "completeness", "bcss", "wcss", "chi"]

  out_path = os.path.join(exp_path, "figures")
  os.makedirs(out_path, exist_ok=True)
  df = pd.read_csv(os.path.join(exp_path, data_path))
  df["silhouette"] = (df["silhouette"] + 1) / 2

  vars = df["layer"].unique()
  clusters = df["n_clusters"].unique()
  colors = mpl.color_sequences['tab10']
  mins, maxes = [], []
  for score in scores:
    mins.append(df[score].min())
    maxes.append(df[score].max())

  h, w = 2, 3
  best_idx = None
  best_vals = {}
  for var in vars:
    _, axs = plt.subplots(h, w, tight_layout=True, figsize=(24, 12))
    best_vals[var] = {}
    data = df[df["layer"] == var]
    for i in range(h):
      for j in range(w):
        flat_idx = (i * w) + j
        score = scores[flat_idx]
        best_idx = np.argmax(data[score])
        best_val = data[score].tolist()[best_idx]
        best_vals[var][score] = (best_val, clusters[best_idx])
        axs[i, j].plot(clusters, data[score].tolist(), color=colors[flat_idx], label=score)
        axs[i, j].plot([clusters[best_idx]], [best_val], "x", label='best')
        axs[i, j].set_xlabel("Number of Epochs")
        axs[i, j].set_xticks(clusters)
        axs[i, j].set_ylabel("Score")
        axs[i, j].set_ylim(mins[flat_idx] * 0.9, maxes[flat_idx] * 1.1)
        axs[i, j].grid()
        axs[i, j].legend()
        axs[i, j].set_title(score)
    plt.suptitle(var)
    # plt.show()
    plt.savefig(os.path.join(out_path, f"{var.replace('.','_')}.png"), bbox_inches="tight")
    plt.clf()
    plt.close("all")

  _, axs = plt.subplots(h, w, tight_layout=True, figsize=(24, 12))
  for i in range(h):
    for j in range(w):
      flat_idx = (i * w) + j
      score = scores[flat_idx]
      data = [v[score][0] for v in best_vals.values()]
      axs[i, j].plot(vars, data, color=colors[flat_idx])
      for k, (y, l) in enumerate(zip(data, [v[score][1] for v in best_vals.values()])):
        axs[i, j].text(k, y, f"N={l}", ha='center', va='bottom')
      axs[i, j].set_xlabel("Layers")
      axs[i, j].set_xticks(vars)
      axs[i, j].set_ylabel("Score")
      axs[i, j].set_ylim(mins[flat_idx] * 0.9, maxes[flat_idx] * 1.1)
      axs[i, j].grid()
      axs[i, j].set_title(score)
  plt.suptitle("Best Scores")
  # plt.show()
  plt.savefig(os.path.join(out_path, f"best.png"), bbox_inches="tight")
  plt.clf()
  plt.close("all")

  # ------------------------------------------

  # data = np.zeros((len(vars), len(scores)))
  # for i, layer in enumerate(vars):
  #   ldf = df[df["layer"] == layer]
  #   for j, score in enumerate(scores):
  #     best_idx = np.argmax(ldf["silhouette"])
  #     data[i, j] = max(ldf[score])
  # plot_3d_derivatives(data)
  # plot_3d_scores(data)
  # plot_combined_score(data, vars, exp_path.split("/")[-2])
