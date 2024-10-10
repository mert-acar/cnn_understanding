import numpy as np
from sklearn import cluster
from utils import read_to_cluster

if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [f"features.{i}" for i in range(9)] + ["pool"]

  # svd_dim = 86
  # threshold = None
  epoch = 33
  svd_dim = None
  threshold = 0.98
  n = 10

  for layer in layers:
    X, y = read_to_cluster(
      file_path=experiment_path.format(layer=layer, epoch=epoch),
      svd_dim=svd_dim,
      threshold=threshold,
    )
    cluster_labels = cluster.AgglomerativeClustering(n_clusters=n).fit(X).labels_
    v = {i: np.var(X[cluster_labels == i]) for i in range(n)}
