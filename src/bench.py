import os
import re
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib as mpl
from tsnecuda import TSNE
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.cluster import HDBSCAN
from dimensionality_reduction import low_rank_approximation


def extract_epoch_number(filename):
  match = epoch_pattern.search(filename)
  return int(match.group(1)) if match else float('inf')


if __name__ == "__main__":
  exp = "resnet18_run4"
  var_name = "conv1"
  energy_threshold = 0.95
  act_threshold = 0.05

  exp_path = os.path.join("../logs/", exp)
  out_path = os.path.join("../figures/", exp)
  filenames = glob(os.path.join(exp_path, "*.mat"))
  epoch_pattern = re.compile(r'act_epoch_(\d+)\.mat')
  filenames = sorted(filenames, key=extract_epoch_number, reverse=True)
  pbar = tqdm(filenames)
  project = None
  for fname in pbar:
    pbar.set_description(os.path.basename(fname))
    data = loadmat(fname)
    activations = data[var_name]
    labels = np.squeeze(data["labels"])
    non_zero_idx = [
      np.abs(activations[labels == 0]).mean(0)[j].sum() > act_threshold
      for j in range(activations.shape[1])
    ]
    activations = activations[:, non_zero_idx]
    activations = activations.reshape(activations.shape[0], -1)

    if project is None:
      recon, project, reconstruct = low_rank_approximation(activations, energy_threshold)
    else:
      recon = project(activations)

    # recon = low_rank_approximation(activations, energy_threshold)
    recon = TSNE(n_components=2).fit_transform(activations)
    clusters = HDBSCAN(min_cluster_size=50).fit(recon).labels_
    # clusters, _ = cluster(recon, labels)
    if len(set(clusters)) < 3:
      continue

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
        ax.scatter(recon[idx, 0], recon[idx, 1], color=color, label=lbl, alpha=0.3)
      ax.grid(True)
      ax.set_title(f"{'predicted' if j == 0 else 'label'} clusters")
      # ax.legend()
    epoch = int(os.path.basename(fname).split('.')[0].split('_')[-1])
    plt.suptitle(f"Epoch {epoch} Activations")
    plt.savefig(os.path.join(out_path, f"epoch_{epoch}.png"), bbox_inches='tight')
    plt.clf()
    plt.close("all")
