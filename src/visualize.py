import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from typing import Dict, List
from utils import closest_factors


def plot_pca(act: np.ndarray, labels: np.ndarray, n_comp: int = 30):
  n_classes = len(set(labels))
  pca = PCA(n_components=n_comp).fit(act)
  sig_vals = [pca.singular_values_]
  for c in range(n_classes):
    pca = PCA(n_components=n_comp).fit(act[labels == c])
    sig_vals.append((pca.singular_values_))
  fig, ax = plt.subplots(ncols=1, nrows=1)
  ax.plot(
    np.arange(n_comp),
    sig_vals[0][:n_comp],
    "-p",
    label="all",
    color='tomato',
    alpha=0.6,
  )
  for c, sig_val in enumerate(sig_vals[1:]):
    if c == 0:
      ax.plot(
        np.arange(n_comp),
        sig_val[:n_comp],
        "-o",
        color="#89CFF0",
        label="classes",
        alpha=0.6,
      )
    else:
      ax.plot(
        np.arange(n_comp),
        sig_val[:n_comp],
        "-o",
        color="#89CFF0",
        alpha=0.6,
      )
  ax.set_xticks(np.arange(0, n_comp, 5))
  ax.set_xlabel("components")
  ax.set_ylabel("sigular values")
  ax.grid(True)
  ax.legend()
  fig.tight_layout()
  plt.show()


def plot_performance_curves(metrics: Dict[str, Dict[str, List[float]]], output_path: str):
    # Calculate number of rows and columns for subplots
    n_metrics = len(metrics)
    n_rows, n_cols = closest_factors(n_metrics)
    
    # Create figure with subplots
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows), squeeze=False)
    epochs = list(range(1, len(next(iter(metrics.values()))["train"]) + 1))
    
    # Plot each metric
    for idx, (metric_name, phases) in enumerate(metrics.items()):
        row = idx // n_cols
        col = idx % n_cols
        
        for phase, values in phases.items():
            axs[row, col].plot(epochs, values, label=phase, 
                             marker='o' if phase == 'test' else None,
                             markersize=2)
            
        axs[row, col].set_xlabel("Epochs")
        axs[row, col].set_ylabel(metric_name.replace('_', ' ').title())
        axs[row, col].legend()
        axs[row, col].grid(True)
        
        # Set title for each subplot
        axs[row, col].set_title(f"{metric_name.replace('_', ' ').title()} vs Epochs")
    
    # Remove empty subplots if any
    for idx in range(len(metrics), n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        fig.delaxes(axs[row, col])
    
    # Add main title and adjust layout
    fig.suptitle("Training Metrics Across Epochs", y=1.02, fontsize=16)
    plt.tight_layout()
    
    # Save the figure
    save_path = os.path.join(output_path, "performance_curves.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=300)
    plt.close()


def vis(embedded: np.ndarray, clusters: np.ndarray, labels: np.ndarray):
  fig = plt.figure(figsize=(14, 8))
  for j, lbls in enumerate([clusters, labels]):
    ax = fig.add_subplot(121 + j, projection='3d')
    ax.scatter(embedded[:, 0], embedded[:, 1], embedded[:, 2], c=lbls, cmap='viridis', marker='o', alpha=0.3)
    ax.grid(True)
    ax.set_title(f"{'predicted' if j == 0 else 'label'} clusters")
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
  plt.suptitle(f"Activations")
  plt.show()


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
  from dataset import get_labels
  from models import HOOK_TARGETS
  from sklearn.manifold import TSNE

  exp_dir = "../logs/smallnet_MNIST_CBAM/"
  var = HOOK_TARGETS["smallnet"][-1]
  labels = get_labels("MNIST", "test")
  activations = np.load(os.path.join(exp_dir, "activations", f"{var}_act.npy"))
  cluster_labels = np.load(os.path.join(exp_dir, "clusters", f"{var}_cluster_labels_svm.npy"))
  # embedded = PCA(n_components=3).fit_transform(activations)
  embedded = TSNE(n_components=3).fit_transform(activations)

  # plot_pca(activations, labels, 30)
  vis(embedded, cluster_labels, labels)
