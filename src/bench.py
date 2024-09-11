import torch
from scipy.io import loadmat
import torch.nn.functional as F
import matplotlib.pyplot as plt
from kneed import KneeLocator

if __name__ == "__main__":
  data = torch.from_numpy(
    loadmat("../logs/resnet18_run1/layer1.1.conv2/act_epoch_34.mat")["layer1.1.conv2"]
  ).reshape(10000, -1)
  data = F.normalize(data)
  dist = torch.cdist(data, data)
  sample = dist[0]
  y, _ = torch.sort(sample, descending=True)
  y = y.numpy()
  x = list(range(len(y)))
  # Find the knee point using KneeLocator
  kn = KneeLocator(x, y, curve='concave', direction='decreasing')

  # Plotting
  plt.figure(figsize=(10, 6))
  plt.plot(x, y, label='Distances')
  plt.axvline(x=kn.knee, color='r', linestyle='--', label=f'Knee point at x={kn.knee}')
  plt.title("Knee Point Visualization")
  plt.xlabel("Index")
  plt.ylabel("Distance")
  plt.legend()
  plt.grid()
  plt.show()
