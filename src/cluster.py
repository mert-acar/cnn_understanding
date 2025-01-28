import numpy as np
from sklearn import cluster
from functools import partial
from sklearn.svm import LinearSVC


def bcss(x: np.ndarray, pred_labels: np.ndarray) -> float:
  n_labels = len(set(pred_labels))
  extra_disp = 0.0
  mean = x.mean(0)
  for k in range(n_labels):
    cluster_k = x[pred_labels == k]
    mean_k = cluster_k.mean(0)
    extra_disp += len(cluster_k) * ((mean_k - mean)**2).sum()
  return float(extra_disp)


def wcss(x: np.ndarray, pred_labels: np.ndarray) -> float:
  n_labels = len(set(pred_labels))
  intra_disp = 0.0
  for k in range(n_labels):
    cluster_k = x[pred_labels == k]
    mean_k = cluster_k.mean(0)
    intra_disp += ((cluster_k - mean_k)**2).sum()
  return float(intra_disp)


def cluster_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
  return (pred_labels == true_labels).mean()


def svm(activations: np.ndarray, labels: np.ndarray, **kwargs) -> np.ndarray:
  model = LinearSVC(**kwargs)
  model.fit(activations, labels)
  pred_labels = model.predict(activations)
  return pred_labels


def get_clustering_func(method: str, k: int = 10, **kwargs):
  if method == "svm":
    return partial(svm, **kwargs)
  else:
    raise NotImplementedError(method)
