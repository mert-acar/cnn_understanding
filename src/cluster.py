import numpy as np
import matplotlib as mpl
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score, v_measure_score, make_scorer


class HDBSCANClusterer(BaseEstimator, ClusterMixin):
  def __init__(
    self, min_cluster_size=5, min_samples=None, alpha=1.0, cluster_selection_epsilon=0.0
  ):
    self.min_cluster_size = min_cluster_size
    self.min_samples = min_samples
    self.alpha = alpha
    self.cluster_selection_epsilon = cluster_selection_epsilon
    self.hdbscan = HDBSCAN(
      min_cluster_size=self.min_cluster_size,
      min_samples=self.min_samples,
      alpha=self.alpha,
      cluster_selection_epsilon=self.cluster_selection_epsilon
    )

  def fit(self, X, y=None):
    self.hdbscan.fit(X)
    return self

  def fit_predict(self, X, y=None):
    return self.hdbscan.fit_predict(X)


def main(data_path, variable_name="conv1"):
  data = loadmat(data_path, variable_names=[variable_name, "labels"])
  activations = data[variable_name]
  labels = data["labels"]
  clusters = HDBSCAN(min_cluster_size=50).fit(activations).labels_
  print(f"Number of clusters: {len(set(clusters)) - 1}")
  print(
    f"Number of noise samples: {(clusters == -1).sum()}/{activations.shape[0]} [{100 * (clusters == -1).sum() / activations.shape[0]:.2f}%]"
  )
  sscore = silhouette_score(activations, clusters)
  vmeasure = v_measure_score(labels, clusters)
  print(f"Silhouette Score: {sscore:.3f} [-1.0 - 1.0]")
  print(f"V-measure Score: {vmeasure:.3f} [ 0.0 - 1.0]")

  if data.shape[1] > 2:
    embedded = TSNE(n_components=2).fit_transform(activations)
  else:
    embedded = data

  _, axs = plt.subplots(1, 2, figsize=(15, 7), tight_layout=True)
  colormap = mpl.colormaps['tab20']
  for j, lbls in enumerate([clusters, labels]):
    ax = axs[j]
    for i, c in enumerate(reversed(list(set(lbls)))):
      idx = lbls == c
      if c == -1:
        lbl = "Noisy Samples"
        color = 'gray'
      else:
        lbl = f"Cluster {c}"
        color = colormap(i)
      ax.scatter(embedded[idx, 0], embedded[idx, 1], color=color, label=lbl, alpha=0.3)
    ax.grid(True)
    ax.set_title(f"{'predicted' if j == 0 else 'label'} clusters")
    ax.legend()
  plt.suptitle(f"{data_path.split('_')[-1].split('.')[0]} Activations")
  plt.show()


def silhouette_scorer(estimator, X):
  labels = estimator.fit_predict(X)
  # Check if all labels are the same or if there are more than one label before calculating silhouette_score
  if len(set(labels)) == 1 or len(set(labels)) == len(labels):
    return -1  # A bad score if all labels are the same or each point is its own cluster
  return silhouette_score(X, labels)


if __name__ == "__main__":
  # from fire import Fire
  # Fire(main)

  from dimensionality_reduction import low_rank_approximation

  # Setting up the grid search
  data_path = "../logs/resnet18_run4/act_epoch_37.mat"
  variable_name = "conv1"
  data = loadmat(data_path, variable_names=[variable_name, "labels"])
  activations = data[variable_name]
  activations = activations.reshape(activations.shape[0], -1)
  X, _, _ = low_rank_approximation(activations, verbose=True)

  param_grid = {
    'min_cluster_size': [20, 30, 40, 50, 60],
    'alpha': [0.1, 0.5, 1.0, 1.5, 3],
    'cluster_selection_epsilon': [0.0, 0.1, 0.2, 0.9]
  }
  grid_search = GridSearchCV(HDBSCANClusterer(), param_grid, scoring=silhouette_scorer, cv=3)
  grid_search.fit(X)

  print("Best parameters:", grid_search.best_params_)
  print("Best silhouette score:", grid_search.best_score_)
