import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def vis2d(activations):
  _, axs = plt.subplots(2, 4, figsize=(11, 10), tight_layout=True)
  for i, ax in enumerate(axs.ravel()):
    ax.imshow(activations[i], cmap='gray')
    ax.axis(False)
    ax.set_title(f"{np.abs(activations[i]).sum():.3f}")
  plt.suptitle(f"Avg Activation")
  plt.show()


def vis3d(activations):
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


def vis(embedded, clusters, labels):
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


def create_multilayer_sankey(cluster_labels_layers):
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


def main(mode, exp_file, lbl, threshold=0.1):
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
  from fire import Fire
  Fire(main)
