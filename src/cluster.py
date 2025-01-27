import pickle as p
import numpy as np
from tqdm import tqdm
from typing import Tuple, Optional
from sklearn import metrics, cluster
from scipy.optimize import linear_sum_assignment

from utils import normalize
from manifold import get_facet_planes, find_facets, pw_similarity


if __name__ == "__main__":
  import os
  import pickle as p
  import pandas as pd
  from tqdm import tqdm
  from tabulate import tabulate
  from collections import defaultdict
  from sklearn.decomposition import PCA

  from model import HOOK_TARGETS
  from visualize import plot_scores
  from utils import load_MNIST_labels, load_CIFAR10_labels

  model_name = "resnet18mcr"
  dataset = "CIFAR10"
  labels = load_CIFAR10_labels()
  # labels = load_MNIST_labels()
  low, high = 5, 15
  idx = None # np.random.randint(0, len(labels), 2000)

  if idx is not None:
    labels = labels[idx]

  exp_dir = f"../logs/resnet18_CIFAR10_MCR/"
  out_path = os.path.join(exp_dir, "clusters")
  os.makedirs(out_path, exist_ok=True)

  vars = HOOK_TARGETS[model_name]

  scores = defaultdict(list)
  print(f"+ Working on: {exp_dir}")
  for var in vars:
    out = defaultdict(list)
    print(var, "\n----------------")
    with open(os.path.join(exp_dir, "activations", f"{var.replace('.', '_')}_act.p"), "rb") as f:
      x = p.load(f)

    if idx is not None:
      x = x[idx]

    # [N, C, H, W] -> [N, C]
    if x.ndim == 4:
      x = x.mean(-1).mean(-1)

    # [N, C, H, W] -> [N, C*H*W]
    # x = x.reshape(x.shape[0], -1)

    # print(f"Before PCA: {x.shape}")
    # tick = time()
    # x = StandardScaler().fit_transform(x)
    # x = PCA(n_components=0.95).fit_transform(x)
    # print(f"PCA took {time() - tick:.3f} seconds")
    # print(f"After PCA: {x.shape}")

    x = normalize(x, 1)

    # planes = get_facet_planes(x, 3, 0.05)

    # chosen_facets = find_facets(planes, x)

    # sim_mat = np.zeros((len(x), len(x)))
    # for i in tqdm(range(len(x)), desc="finding similarities"):
    #   for j in range(i, len(x)):
    #     if i == j:
    #       sim_mat[i, j] = 1
    #       continue
    #     kx, ky = x[i], x[j]
    #     x_facet, y_facet = planes[chosen_facets[i]], planes[chosen_facets[j]]
    #     similarity = pw_similarity(kx, x_facet, ky, y_facet)
    #     sim_mat[i, j] = similarity
    #     sim_mat[j, i] = similarity

    for k in tqdm(range(low, high)):
      cluster_labels = cluster.MiniBatchKMeans(n_clusters=k, batch_size=2048).fit_predict(x)

      # cluster_labels = cluster.SpectralClustering(
      #   n_clusters=k, affinity="precomputed", assign_labels="cluster_qr", n_jobs=-1
      # ).fit_predict(sim_mat)

      perf = {
        "k": k,
        "silhouette": metrics.silhouette_score(x, cluster_labels),
        "nmi": metrics.normalized_mutual_info_score(labels, cluster_labels),
        "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
        "completeness": metrics.completeness_score(labels, cluster_labels),
        "fowlkes_mallows": metrics.fowlkes_mallows_score(labels, cluster_labels),
        "chi": metrics.calinski_harabasz_score(x, cluster_labels),
      }
      for key in perf:
        out[key].append(perf[key])

    print(tabulate(out, headers=list(out.keys())))
    scores["layer"].extend([var] * len(out[key]))
    for key in out:
      scores[key].extend(out[key])

  df = pd.DataFrame(scores)
  df.to_csv(os.path.join(out_path, f"{identifier}_scores.csv"))

  df = pd.read_csv(os.path.join(out_path, f"{identifier}_scores.csv"))
  out_path = os.path.join(exp_dir, "figures")
  os.makedirs(out_path, exist_ok=True)
  plot_scores(df, identifier, out_path)
