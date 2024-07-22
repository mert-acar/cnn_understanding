import os
import numpy as np
from glob import glob
from tqdm import tqdm
from scipy.io import loadmat
import matplotlib.pyplot as plt

if __name__ == "__main__":
  exp_path = "../logs/resnet18_run2"
  var_name = "conv1"
  threshold = 0.95
  filenames = glob(os.path.join(exp_path, "*.mat"))
  fname = filenames[0]
  print(f"[INFO] Processing file: {os.path.basename(fname)}")
  data = loadmat(fname)
  activations = data[var_name]
  labels = np.squeeze(data["labels"])
  non_zero_idx = [
    np.abs(activations[labels == 0]).mean(0)[j].sum() > 0 for j in range(activations.shape[1])
  ]
  activations = activations[:, non_zero_idx]
  activations = activations.reshape(activations.shape[0], -1)
  print(f"[INFO] Flat activations shape: {activations.shape}")

  # Center the data
  mean = np.mean(activations, axis=0)
  u, s, _ = np.linalg.svd(activations - mean, full_matrices=False)
  s2 = s**2

  # Rank becomes: r for 95% of all energy [75]
  energies = np.cumsum(s2) / np.sum(s2)
  k = np.argmax(energies > threshold) + 1
  print(f"[INFO] Rank [{k} / {len(s)}] explain {energies[k-1] * 100:.2f}% of the total energy")

  recon = u[:, :k] @ np.diag(s[:k])
  print(f"[INFO] Recon shape: {recon.shape}")
