import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


def vis2d(activations):
  _, axs = plt.subplots(8, 8, figsize=(11, 10), tight_layout=True)
  for i, ax in enumerate(axs.ravel()):
    ax.imshow(activations[i], cmap='gray')
    ax.axis(False)
    ax.set_title(f"{np.abs(activations[i]).sum():.3f}")
  plt.suptitle(f"Avg Activation")
  plt.show()


def vis3d(activations):
  fig = plt.figure(figsize=(11,10))
  ax = fig.add_subplot(111, projection='3d')
  x, y, z = np.indices(activations.shape)
  x = x.flatten()
  y = y.flatten()
  z = z.flatten()
  values = np.abs(activations.flatten())
  scatter = ax.scatter(
    x[values > 0],
    y[values > 0],
    z[values > 0],
    c=values[values > 0],
    cmap='viridis'
  )
  color_bar = plt.colorbar(scatter, ax=ax)
  color_bar.set_label('Intensity')
  ax.set_xlabel('Channels')
  ax.set_ylabel('H')
  ax.set_zlabel('W')
  plt.show()


def main(mode, exp_file, variable_name, lbl, threshold=0.1):
  inp_path = Path(exp_file)
  data = loadmat(inp_path, variable_names=[variable_name, "labels"])
  activations = data[variable_name]
  labels = data["labels"][0]
  activations = activations[labels == lbl][0]
  activations[np.sum(np.abs(activations), axis=(1, 2)) < threshold] = 0
  print(f"[INFO] Plotting a random activation for the label {lbl}")
  if mode == '2d':
    vis2d(activations)
  else:
    vis3d(activations)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
