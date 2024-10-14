import pickle as p
import numpy as np
from pprint import pprint
from scipy.io import loadmat
from utils import svd_reduction
from cluster import map_clusters

if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [f"features.{i}" for i in range(0, 9, 2)] + ["pool"]

  threshold = 0.98
  svd_dim = None
  epoch = 33

  with open(f"./data/simple_clusters_e{epoch}.p", "rb") as f:
    clusters = p.load(f)

  compactness = {}
  for i, layer in enumerate(layers):
    compactness[layer] = {}
    print(f"Layer: {layer} | Epoch: {epoch}")
    # data = loadmat(experiment_path.format(layer=layer, epoch=epoch))
    # y = data["labels"][0]
    # X = data["activations"]
    # X = X.mean((-2, -1))
    # X = X / np.abs(X).max()
    # X = svd_reduction(X - X.mean(0), n_components=svd_dim, threshold=threshold)
    # cluster_labels = clusters[layer]["cluster_labels"]
    # cluster_labels, _ = map_clusters(cluster_labels, y)
    # for lbl in np.unique(cluster_labels):
    #   cluster = X[cluster_labels == lbl]
    #   cluster_mean = cluster.mean(0)
    #   wcss = np.linalg.norm(cluster - cluster_mean, ord=2)
    #   compactness[layer][int(lbl)] = float(wcss)
