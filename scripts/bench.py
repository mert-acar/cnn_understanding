import numpy as np
from pprint import pprint
from cluster import parameter_search
from sklearn import cluster
from scipy.io import loadmat
from dim_reduction import svd_reduction

if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/new_act/features.{idx}_epoch_33.mat"
  data = loadmat(experiment_path.format(idx=0))
  num_points_per_sample = data["input"].shape[1]
  inp_dim = data["input"].shape[-1]
  labels = np.repeat(data["labels"][0], num_points_per_sample)
  inp = data["input"].reshape(-1, inp_dim)
  # out_dim = data["output"].shape[-1]
  # out = data["output"].reshape(-1, out_dim)
  del data

  metric = "calinski_harabasz_score"
  inp = inp / np.abs(inp).max()
  inp = svd_reduction(inp - inp.mean(0), n_components=None, threshold=0.98)
  params = {"n_clusters": [None], "distance_threshold": [i  for i in np.linspace(2, 30, 10)]}
  cluster_labels, best_params, scores = parameter_search(
    inp, labels, cluster.AgglomerativeClustering, params, optimize_over=metric
  )
  pprint(scores)
