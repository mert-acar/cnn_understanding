import numpy as np
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler


def low_rank_approximation(activations, rank=10, threshold=None):
  assert (rank is None) != (threshold is None), "Either rank or threshold should be specified"

  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(activations)

  if threshold is not None:
    full_svd = TruncatedSVD(n_components=min(activations.shape[1] - 1, activations.shape[0] - 1))
    full_svd.fit(scaled_data)
    cumulative_variance = np.cumsum(full_svd.explained_variance_ratio_)
    n_components = np.searchsorted(cumulative_variance, threshold) + 1

  # Now initialize and apply SVD with the determined number of components
  svd = TruncatedSVD(n_components=n_components)
  transformed_data = svd.fit_transform(scaled_data)
  return transformed_data


def pca_reduction(activations, n_components=5, threshold=None):
  assert (n_components
          is None) != (threshold is None), "Either n_components or threshold should be specified"

  # Standardize the data
  scaled_data = StandardScaler().fit_transform(activations)

  if threshold is not None:
    pca = PCA(n_components=threshold)
  else:
    pca = PCA(n_components=n_components)

  pca_result = pca.fit_transform(scaled_data)
  return pca_result
