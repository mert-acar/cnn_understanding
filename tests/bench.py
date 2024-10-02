import numpy as np
from sklearn import cluster
from scipy.io import loadmat
from tabulate import tabulate
import dim_reduction as reduction
from collections import defaultdict
from scipy.spatial.distance import cdist
from sklearn.model_selection import ParameterGrid
from utils import get_filenames, performance_scores


def parameter_search(data, algo, params, score_type):
  best = -1
  for param in ParameterGrid(params):
    try:
      cluster_labels = algo(**param).fit(data).labels_
      scores = performance_scores(data, cluster_labels, labels)
    except ValueError:
      print(param, "Error")
      continue
    if scores[score_type] > best:
      best = scores[score_type]
      best_scores = scores
      best_param = param
      best_labels = cluster_labels
  print(tabulate([key, value] for key, value in best_scores.items()))
  return best_labels, best_param, best_scores


if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/activations/"
  scores_data = defaultdict(list)
  score_type = "silhouette"

  layers = ["pool", "features.8", "features.6", "features.4", "features.2", "features.0"]
  metric = "euclidean"
  linkage = "ward"
  l, h, n = 5, 30, 30

  data = {}

  for layer in layers:
    fname = get_filenames(layer, experiment_path=experiment_path)[0]
    data = loadmat(fname)
    activations = data[layer]
    labels = data["labels"][0]
    print(f"+ {layer}: {activations.shape}")

    act = activations.reshape(activations.shape[0], -1)
    print(f"+ Before Transform Shape: {act.shape}")

    act = reduction.low_rank_approximation(act, n_components=None, threshold=0.95, norm=True)
    print(f"+ Transformed Shape: {act.shape}")

    d = cdist(act, act, metric=metric).mean()
    cluster_labels, _, scores = parameter_search(
      act, cluster.AgglomerativeClustering, {
        "n_clusters": [None],
        "metric": [metric],
        "linkage": [linkage],
        "distance_threshold": [i * d for i in np.linspace(l, h, n)]
      }, score_type
    )
    data[layer] = {"cluser_labels": cluster_labels, "scores": scores}
