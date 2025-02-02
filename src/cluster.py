import numpy as np
from sklearn import cluster
from functools import partial
from sklearn.svm import LinearSVC


def svm(activations: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
  model = LinearSVC(**kwargs)
  model.fit(activations, labels)
  pred_labels = model.predict(activations)
  return pred_labels


def spectral(activations: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
  return cluster.SpectralClustering(**kwargs).fit_predict(activations)


def get_clustering_func(method: str, **kwargs):
  if method == "svm":
    return partial(svm, **kwargs)
  elif method == "spectral":
    return partial(spectral, **kwargs)
  else:
    raise NotImplementedError(method)
