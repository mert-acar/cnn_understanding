import numpy as np
import pandas as pd
import matplotlib as mpl
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt
import plotly.graph_objects as go


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
  # from fire import Fire
  # Fire(main)

  import os
  import pickle as p
  from sklearn import manifold

  exp_dir = "../logs/customnet_run2/"
  out_dir = os.path.join(exp_dir, "figures")
  epoch = 33
  os.makedirs(out_dir, exist_ok=True)

  labels = loadmat("../data/labels.mat")["labels"][0]
  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}.p"), "rb") as f:
    cluster_data = p.load(f)

  # mds = manifold.MDS(n_components=2, metric=False, max_iter=1000, eps=1e-8, n_jobs=-1)
  mds = manifold.TSNE(n_components=2)
  for i in range(0, 9, 2):
    vars = [f"features.{i}_input", f"features.{i}_output"]
    for var in vars:
      data = loadmat(
        os.path.join(exp_dir, "activations", f"patches_epoch_{epoch}.mat"), variable_names=[var]
      )
      print(var)
      x = data[var]
      o = 4000 // x.shape[1]
      idx = np.random.choice(cluster_data[var]["idx"], size=o, replace=False)
      x = x[idx]
      l = labels[idx]
      l = np.repeat(l, x.shape[1])
      x = x.reshape(-1, x.shape[-1])
      print(f"Activations: {x.shape}")
      print(f"Labels: {l.shape}")
      x = x - x.mean(0)
      x = x / np.abs(x).max()
      embedded = mds.fit_transform(x)
      embedded = embedded / np.abs(embedded).max()
      clusters = cluster_data[var]["cluster_labels"][:len(l)]
      _, axs = plt.subplots(1, 2, tight_layout=True, figsize=(12, 6))
      lim = 1.4
      axs[0].scatter(embedded[:, 0], embedded[:, 1], c=l, alpha=0.3)
      axs[0].grid(True)
      axs[0].axis("equal")
      axs[0].set_title(f"Original Labels (n = {len(set(l))})")
      axs[0].set_xlim((-lim, lim))
      axs[0].set_ylim((-lim, lim))
      axs[1].scatter(embedded[:, 0], embedded[:, 1], c=clusters, alpha=0.3)
      axs[1].grid(True)
      axs[1].axis("equal")
      axs[1].set_title(f"Cluster Labels (n = {len(set(clusters))})")
      axs[1].set_xlim((-lim, lim))
      axs[1].set_ylim((-lim, lim))

      axs[1].text(-1, 1, f"CHI: {cluster_data[var]['scores']['calinski_harabasz_score']:.3f}")
      axs[1].text(-1, 0.88, f"Silhouette: {cluster_data[var]['scores']['silhouette']:.3f}")
      axs[1].text(0.2, 1, f"Homogeneity: {cluster_data[var]['scores']['homogeneity']:.3f}")
      axs[1].text(0.2, 0.88, f"Completeness: {cluster_data[var]['scores']['completeness']:.3f}")
      plt.suptitle(var)
      # plt.show()
      plt.savefig(os.path.join(out_dir, var.replace(".", "_") + ".png"), bbox_inches="tight")
      plt.clf()
      plt.close()
