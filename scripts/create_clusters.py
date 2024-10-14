# System Imports
import pickle as p

# Library Imports
import numpy as np
from sklearn import cluster
from scipy.spatial.distance import cdist

# Internal Imports
from utils import read_to_cluster
from cluster import parameter_search


if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [f"features.{i}" for i in range(0, 9, 2)] + ["pool"]
  output_path = "./data/simple_flat_e{epoch}.p"
  epochs = [33, 17, 1]
  threshold = 0.98
  svd_dim = None
  metric = "calinski_harabasz_score"
  print(f"+ Optimizing over: {metric}")

  for epoch in epochs:
    print(f"\n--------------- Epoch: {epoch} ---------------")
    out = {}
    for layer in layers:
      X, y = read_to_cluster(
        file_path=experiment_path.format(epoch=epoch, layer=layer),
        reshape="flatten",
        norm=True,
        svd_dim=svd_dim,
        threshold=threshold
      )
      d = cdist(X, X).mean()
      params = {"n_clusters": [None], "distance_threshold": [i * d for i in np.linspace(2, 30, 20)]}
      cluster_labels, best_params, scores = parameter_search(
        X, y, cluster.AgglomerativeClustering, params, optimize_over=metric
      )

      print(
        f"Layer: {layer} | Num Clusters: {len(set(cluster_labels))} | Score: {scores[metric]:.3f}"
      )
      out[layer] = {
        "cluster_labels": cluster_labels,
        "params": best_params,
        "scores": scores,
        "data": [X, y]
      }

    with open(output_path.format(epoch=epoch), "wb") as f:
      p.dump(out, f, protocol=p.HIGHEST_PROTOCOL)
