import numpy as np
from tqdm import tqdm
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


def visualize_activations(exp_file, variable_name, lbl=0):
  inp_path = Path(exp_file)
  activations = loadmat(inp_path, variable_names=variable_name)[variable_name]
  labels = loadmat("../../data/labels.mat")["labels"][0]
  _, axs = plt.subplots(8, 8, tight_layout=True)
  act = activations[labels == lbl][0]
  print(f"[INFO] Plotting a random activation for the label {lbl}")
  for i, ax in enumerate(axs.ravel()):
    ax.imshow(act[i], cmap='Reds')
    ax.axis(False)
  plt.suptitle(f"Class {lbl} Avg Activation")
  plt.show()


def vis3d(exp_file, variable_name, lbl=0):
  inp_path = Path(exp_file)
  activations = loadmat(inp_path, variable_names=variable_name)[variable_name]
  labels = loadmat("../../data/labels.mat")["labels"][0]
  print(f"[INFO] Plotting a random activation for the label {lbl}")
  activations = activations[labels == lbl][0]

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  x, y, z = np.indices(activations.shape)

  x = x.flatten()
  y = y.flatten()
  z = z.flatten()
  values = activations.flatten()

  scatter = ax.scatter(x[values > 0], z[values > 0], y[values > 0], c=values[values > 0], cmap='viridis')

  color_bar = plt.colorbar(scatter, ax=ax)
  color_bar.set_label('Intensity')
  ax.set_xlabel('Channels')
  ax.set_ylabel('H')
  ax.set_zlabel('W')
  plt.show()


def main(mode, exp_file, variable_name, lbl):
  if mode == '2d':
    visualize_activations(exp_file, variable_name, int(lbl))
  else:
    vis3d(exp_file, variable_name, int(lbl))


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
