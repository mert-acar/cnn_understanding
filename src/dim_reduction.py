import numpy as np
from sklearn.decomposition import PCA


def svd_reduction(activations, n_components=10, threshold=None):
  assert (n_components
          is None) != (threshold is None), "Either rank or threshold should be specified"
  u, s, _ = np.linalg.svd(activations, full_matrices=False)

  if threshold is not None:
    s2 = s**2
    energies = np.cumsum(s2) / np.sum(s2)
    k = np.argmax(energies > threshold) + 1
  else:
    k = n_components

  u_k = u[:, :k]
  s_k = s[:k]
  recon = np.dot(u_k, np.diag(s_k))
  return recon


def pca_reduction(activations, n_components=5, threshold=None):
  assert (n_components
          is None) != (threshold is None), "Either n_components or threshold should be specified"

  if threshold is not None:
    pca = PCA(n_components=threshold)
  else:
    pca = PCA(n_components=n_components)

  pca_result = pca.fit_transform(activations)
  return pca_result
