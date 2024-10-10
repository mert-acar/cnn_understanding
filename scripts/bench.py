import numpy as np
import pickle as p
from tqdm import tqdm
from sklearn import cluster
from scipy.io import loadmat
from utils import svd_reduction
from cluster import parameter_search
from scipy.spatial.distance import cdist


if __name__ == "__main__":
  # experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  # layers = [f"features.{i}" for i in range(0, 9, 2)] + ["pool"]
  # epoch = 33

  experiment_path = "../logs/resnet18_run1/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [
    "conv1", 
    "layer1.0", "layer1.1", "layer2.0", "layer2.1", 
    "layer3.0", "layer3.1", "layer4.0", "layer4.1",
    "avgpool"
  ]
  epoch = 34

  threshold = 0.98
  svd_dim = None
  out = {}
  for layer in tqdm(layers):
    data = loadmat(experiment_path.format(layer=layer, epoch=epoch))
    y = data["labels"][0]
    X = data["activations"]
    X = X.mean((-2, -1))
    X = X / np.abs(X).max()
    X = svd_reduction(X - X.mean(0), n_components=svd_dim, threshold=threshold)
    d = cdist(X, X).mean()
    params = {"n_clusters": [None], "distance_threshold": [i * d for i in np.linspace(2, 30, 20)]}
    cluster_labels, best_params, scores = parameter_search(
      X, y, cluster.AgglomerativeClustering, params
    )
    out[layer] = {
      "params": best_params,
      "cluster_labels": cluster_labels,
      "scores": scores,
      "svd_param": {"n_components": svd_dim, "threshold": threshold}
    }

  with open("./data/resnet_clusters.p", "wb") as f:
  # with open("./data/customnet_clusters.p", "wb") as f:
    p.dump(out, f, protocol=p.HIGHEST_PROTOCOL)
