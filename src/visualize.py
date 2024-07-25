import numpy as np
from pathlib import Path
from scipy.io import loadmat
import matplotlib.pyplot as plt


def vis2d(activations):
  _, axs = plt.subplots(8, 8, tight_layout=True)
  for i, ax in enumerate(axs.ravel()):
    ax.imshow(activations[i], cmap='gray')
    ax.axis(False)
  plt.suptitle(f"Avg Activation")
  plt.show()


def vis3d(activations, threshold=1):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  x, y, z = np.indices(activations.shape)
  x = x.flatten()
  y = y.flatten()
  z = z.flatten()
  values = np.abs(activations.flatten())
  scatter = ax.scatter(
    x[values > threshold],
    y[values > threshold],
    z[values > threshold],
    c=values[values > threshold],
    cmap='viridis'
  )
  color_bar = plt.colorbar(scatter, ax=ax)
  color_bar.set_label('Intensity')
  ax.set_xlabel('Channels')
  ax.set_ylabel('H')
  ax.set_zlabel('W')
  plt.show()


def main(mode, exp_file, variable_name, lbl, threshold=1):
  inp_path = Path(exp_file)
  data = loadmat(inp_path, variable_names=[variable_name, "labels"])
  activations = data[variable_name]
  labels = data["labels"][0]
  activations = activations[labels == lbl][0]
  print(f"[INFO] Plotting a random activation for the label {lbl}")
  if mode == '2d':
    vis2d(activations)
  else:
    vis3d(activations, threshold)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
