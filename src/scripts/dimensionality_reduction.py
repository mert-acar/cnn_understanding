import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def low_rank_approximation(act_path, variable_name, threshold=0.95):
  data_path = Path(act_path)
  activations = loadmat(data_path, variable_names=variable_name)[variable_name]
  print(f"[INFO] Activations shape: {activations.shape}")
  labels = loadmat("/Users/smol/Desktop/phd/uiuc/cnn_understanding/data/labels.mat")["labels"][0]
  non_zero_idx = [
    np.abs(activations[labels == 0]).mean(0)[j].sum() > 0 for j in range(activations.shape[1])
  ]
  print(f"[INFO] # of non-zero channels: {sum(non_zero_idx)}")

  # Shape: [10000, k << 64, 14, 14]
  activations = activations[:, non_zero_idx]

  # Shape: [10000, k << 64 * 14 * 14]
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

  out_path = Path(
    f"/Users/smol/Desktop/phd/uiuc/cnn_understanding/data/svd_low_rank_recon_{data_path.stem.split('_')[-1]}_{variable_name}.mat"
  )
  savemat(out_path, {"activations": recon})
  print(f"[INFO] Recon is saved to {out_path}")
  print(f"[INFO] Recon shape: {recon.shape}")


def pca_reduction(act_path, variable_name, threshold=0.95):
  data_path = Path(act_path)
  activations = loadmat(data_path, variable_names=variable_name)[variable_name]
  print(f"[INFO] Activations shape: {activations.shape}")
  labels = loadmat("/Users/smol/Desktop/phd/uiuc/cnn_understanding/data/labels.mat")["labels"][0]
  non_zero_idx = [
    np.abs(activations[labels == 0]).mean(0)[j].sum() > 0 for j in range(activations.shape[1])
  ]
  print(f"[INFO] # of non-zero channels: {sum(non_zero_idx)}")

  # Shape: [10000, k << 64, 14, 14]
  activations = activations[:, non_zero_idx]

  # Shape: [10000, k << 64 * 14 * 14]
  activations = activations.reshape(activations.shape[0], -1)
  print(f"[INFO] Flat activations shape: {activations.shape}")

  # Standardize the data
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(activations)

  # Initialize PCA with enough components to cover all variance
  pca = PCA()
  pca.fit(scaled_data)

  # Determine the number of components needed to explain the threshold variance
  cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
  num_components = np.argmax(cumulative_explained_variance >= threshold) + 1

  pca = PCA(n_components=num_components)
  pca_result = pca.fit_transform(scaled_data)

  print(
    f"[INFO] [{num_components} / {activations.shape[1]}] principal components explain >={threshold} of the total variance"
  )

  out_path = Path(
    f"/Users/smol/Desktop/phd/uiuc/cnn_understanding/data/pca_recon_{data_path.stem.split('_')[-1]}_{variable_name}.mat"
  )
  savemat(out_path, {"activations": pca_result})
  print(f"[INFO] Recon is saved to {out_path}")
  print(f"[INFO] Recon shape: {pca_result.shape}")


def dim_reduction(method, act_path, variable_name, threshold=0.95):
  if method == "pca":
    pca_reduction(act_path, variable_name, threshold)
  elif method == "svd":
    low_rank_approximation(act_path, variable_name, threshold)
  else:
    print(f"[ERROR] {method} method is not implemented!")


if __name__ == "__main__":
  from fire import Fire
  Fire(dim_reduction)
