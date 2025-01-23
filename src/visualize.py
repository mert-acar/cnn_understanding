import os
import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from utils import closest_factors
from typing import Dict, List


def plot_performance_curves(metrics: Dict[str, Dict[str, List[float]]], output_path: str):
  fig, axs = plt.subplots(1, len(metrics), tight_layout=True, figsize=(10, 5))
  epochs = list(range(1, len(metrics["loss"]["train"]) + 1))
  for i, (metric, arr) in enumerate(metrics.items()):
    for phase, val in arr.items():
      axs[i].plot(epochs, val, label=phase)
    axs[i].set_xlabel("Epochs")
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)
  fig.suptitle("Model Performance Across Epochs")
  plt.savefig(os.path.join(output_path, "performance_curves.png"), bbox_inches="tight")


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
  scatter = ax.scatter(x[values > 0], y[values > 0], z[values > 0], c=values[values > 0], cmap='viridis')
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
    label_indices.append({cluster: current_label_idx + i for i, cluster in enumerate(unique_clusters)})
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


def plot_scores(df, identifier, out_path):
  scores = df.columns[3:]
  print(scores)

  vars = df["layer"].unique()
  clusters = df["k"].unique()
  colors = mpl.color_sequences['tab10']
  mins, maxes = [], []
  for score in scores:
    mins.append(df[score].min())
    maxes.append(df[score].max())

  h, w = closest_factors(len(scores))
  best_idx = None
  best_vals = {}
  for var in vars:
    _, axs = plt.subplots(h, w, tight_layout=True, figsize=(24, 12), squeeze=False)
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
    plt.savefig(os.path.join(out_path, f"{var.replace('.','_')}_{identifier}.png"), bbox_inches="tight")

    plt.clf()
    plt.close("all")

  _, axs = plt.subplots(h, w, tight_layout=True, figsize=(24, 12), squeeze=False)
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
  plt.savefig(os.path.join(out_path, f"best_{identifier}.png"), bbox_inches="tight")
  plt.clf()
  plt.close("all")


if __name__ == "__main__":
  import os
  from math import sqrt
  from model import load_model
  from dataset import get_labels

  model_name = "smallnet"
  dataset = "MNIST"
  labels = get_labels(dataset, "test")

  model = load_model("../logs/smallnet_MNIST_ATTN/")
  attn_map = model.attention
  k = int(sqrt(attn_map.shape[0] // 32))
  attn_map = attn_map.view(32, k, k).detach().numpy()
  fig, axs = plt.subplots(4, 8, tight_layout=True)
  for i in range(32):
    k, l = i // 8, i % 8
    axs[k, l].imshow(attn_map[i], cmap='gray')
    axs[k, l].set_title(f"Map {i + 1}")
    axs[k, l].axis(False)
  plt.show()
