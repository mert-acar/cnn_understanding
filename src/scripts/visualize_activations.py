import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


def visualize_activations(exp_file, variable_name):
  inp_path = Path(exp_file)
  activations = loadmat(inp_path, variable_names=variable_name)[variable_name]
  labels = loadmat("../../data/labels.mat")["labels"][0]

  out_path = Path(f"../../figures/{inp_path.stem}_{variable_name.replace('.', '_')}")
  out_path.mkdir(parents=True, exist_ok=True)

  k = np.random.randint(500)
  for lbl in tqdm(set(labels), desc=f"Single activation #{k}"):
    fig, axs = plt.subplots(8, 8, tight_layout=True)
    act = activations[labels == lbl][k]
    for j, ax in enumerate(axs.ravel()):
      ax.imshow(act[j], cmap='Reds')
      ax.axis(False)
    plt.suptitle(f"Class {lbl} Activation {k}")
    plt.savefig(out_path / f"cls_{lbl}_act_{j}_{k}.png", bbox_inches="tight")
    plt.clf()
    plt.close(fig)

  for lbl in tqdm(set(labels), desc="Avg Activations"):
    fig, axs = plt.subplots(8, 8, tight_layout=True)
    act = activations[labels == lbl].mean(0)
    for i, ax in enumerate(axs.ravel()):
      ax.imshow(act[i], cmap='Reds')
      ax.axis(False)
    plt.suptitle(f"Class {lbl} Avg Activation")
    plt.savefig(out_path / f"cls_{lbl}_act_{i}_avg.png", bbox_inches="tight")
    plt.clf()
    plt.close(fig)
  print("Figures are saved to", str(out_path))


if __name__ == "__main__":
  from fire import Fire
  Fire(visualize_activations)
