import pickle as p
import numpy as np
from sklearn import cluster as c
from utils import read_to_cluster
from cluster import parameter_search
from scipy.spatial.distance import cdist

if __name__ == "__main__":
  # experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  # layers = [f"features.{i}" for i in range(9)] + ["pool"]
  experiment_path = "../logs/resnet18_run1/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [
    "conv1",
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.1.conv1",
    "layer4.1.conv2",
  ]
  epoch = 34

  # svd_dim = 86
  # threshold = None
  svd_dim = None
  threshold = 0.98

  param_data = {}
  for layer in layers:
    X, y = read_to_cluster(
      file_path=experiment_path.format(layer=layer, epoch=epoch),
      svd_dim=svd_dim,
      threshold=threshold,
      norm=True
    )
    if X is None:
      continue
    d = cdist(X, X).mean()
    print(f"Layer: {layer}")
    print(f"    Shape: {X.shape}")
    print(f"    Mean Eulidean distance: {d:.4f}")
    params = {"n_clusters": [None], "distance_threshold": [i * d for i in np.linspace(5, 45, 20)]}
    cluster_labels, params, score = parameter_search(X, y, c.AgglomerativeClustering, params)
    print(f"    Optimum params: {params['distance_threshold'] / d:.4f} times the mean distance")
    for k, v in score.items():
      print(f"    {k}: {v:.4f}")
    print()
    param_data[layer] = {
      "scores": score,
      "params": params,
      "cluster_labels": cluster_labels,
      "labels": y
    }

  with open(f"./data/resnet_th98_ps_AC_ep{epoch}.p", "wb") as f:
    p.dump(param_data, f, protocol=p.HIGHEST_PROTOCOL)
