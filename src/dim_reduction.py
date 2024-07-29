import numpy as np
from pathlib import Path
from scipy.io import loadmat, savemat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def low_rank_approximation(activations, threshold=0.95, verbose=False):
  # Center the data
  mean = np.mean(activations, axis=0)
  u, s, vt = np.linalg.svd(activations - mean, full_matrices=False)
  s2 = s**2

  # Rank becomes: r for 95% of all energy [75]
  energies = np.cumsum(s2) / np.sum(s2)
  k = np.argmax(energies > threshold) + 1
  if verbose:
    print(f"[INFO] Rank [{k} / {len(s)}] explain {energies[k-1] * 100:.2f}% of the total energy")

  u_k = u[:, :k]
  s_k = s[:k]
  vt_k = vt[:k, :]
  recon = u_k @ np.diag(s_k)

  # Function to project new data points
  def project(new_data):
    return (new_data - mean) @ vt_k.T

  # Function to reconstruct data from reduced space
  def reconstruct(recon):
    return (recon @ vt_k) + mean

  return recon, project, reconstruct


def pca_reduction(activations, threshold=0.95, verbose=False):
  # Standardize the data
  scaler = StandardScaler().fit(activations)
  scaled_data = scaler.transform(activations)
  pca = PCA(n_components=threshold).fit(scaled_data)
  pca_result = pca.transform(scaled_data)

  if verbose:
    print(
      f"[INFO] [{pca_result.shape[1]} / {activations.shape[1]}] principal components explain >={threshold} of the total variance"
    )

  def project(new_data):
    return pca.transform(scaler.transform(new_data))

  return pca_result, project


def main(method, act_path, variable_name, act_threshold=0.1, energy_threshold=0.95, verbose=True):
  data_path = Path(act_path)
  data = loadmat(data_path, variable_names=[variable_name, "labels"])
  activations = data[variable_name]
  if verbose:
    print(f"[INFO] Activations shape: {activations.shape}")
  labels = data["labels"][0]
  non_zero_idx = [
    np.abs(activations[labels == 0]).mean(0)[j].sum() > act_threshold
    for j in range(activations.shape[1])
  ]
  if verbose:
    print(f"[INFO] # of non-zero channels: {sum(non_zero_idx)}")
  activations = activations[:, non_zero_idx]
  activations = activations.reshape(activations.shape[0], -1)
  if verbose:
    print(f"[INFO] Flat activations shape: {activations.shape}")

  if method == "pca":
    recon, _ = pca_reduction(activations, energy_threshold, verbose)
  elif method == "svd":
    recon, _, _ = low_rank_approximation(activations, energy_threshold, verbose)
  else:
    print(f"[ERROR] {method} method is not implemented!")

  out_path = Path(
    f"{data_path.parent}/{method}_recon_epoch_{data_path.stem.split('_')[-1]}_{variable_name}.mat"
  )
  savemat(out_path, {variable_name: recon, "labels": labels})
  if verbose:
    print(f"[INFO] Recon shape: {recon.shape}")
    print(f"[INFO] Recon is saved to {out_path}")


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
