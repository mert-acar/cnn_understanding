import numpy as np
from tqdm import tqdm
from sklearn import metrics, cluster
from typing import Callable, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.model_selection import ParameterGrid


def bcss_wcss(x: np.ndarray,
              cluster_labels: np.ndarray,
              return_chi: bool = False) -> Tuple[float, float, Optional[float]]:
  n_labels = len(set(cluster_labels))
  extra_disp, intra_disp = 0.0, 0.0
  mean = np.mean(x, axis=0)
  for k in range(n_labels):
    cluster_k = x[cluster_labels == k]
    mean_k = np.mean(cluster_k, axis=0)
    extra_disp += len(cluster_k) * np.sum((mean_k - mean)**2)
    intra_disp += np.sum((cluster_k - mean_k)**2)
  if return_chi:
    chi = extra_disp * (len(x) - n_labels) / (intra_disp * (n_labels - 1.0))
    return extra_disp, intra_disp, chi
  return extra_disp, intra_disp, None


def map_clusters(label_a: np.ndarray, labels_b: np.ndarray) -> Tuple[np.ndarray, dict[int, int]]:
  unique_clusters = np.unique(label_a)
  unique_true_labels = np.unique(labels_b)
  cost_matrix = np.zeros((len(unique_clusters), len(unique_true_labels)))
  for i, cluster in enumerate(unique_clusters):
    for j, label in enumerate(unique_true_labels):
      count = np.sum((label_a == cluster) & (labels_b == label))
      cost_matrix[i, j] = -count
  _, col_ind = linear_sum_assignment(cost_matrix)
  cluster_to_label_map = {}
  for i, cluster in enumerate(unique_clusters):
    mapped_label = unique_true_labels[col_ind[i]]
    cluster_to_label_map[cluster] = mapped_label
  new_cluster_labels = np.array([cluster_to_label_map[cluster] for cluster in label_a])
  return new_cluster_labels, cluster_to_label_map


def parameter_search(
  data: np.ndarray,
  labels: np.ndarray,
  params: dict,
  algo: Callable = cluster.AgglomerativeClustering,
) -> Tuple[np.ndarray, dict, dict]:
  best = -1
  comparator = lambda x, y: x > y
  for param in tqdm(ParameterGrid(params), desc="Parameter Searching...", ncols=94):
    try:
      cluster_labels = algo(**param).fit(data).labels_
      scores = performance_scores(data, cluster_labels, labels)
    except ValueError:
      continue
    score = scores["silhouette"]
    if best is None or comparator(score, best):
      best = score
      best_scores = scores
      best_labels = cluster_labels
      best_params = param
  return best_labels, best_params, best_scores


def performance_scores(data: np.ndarray, cluster_labels: np.ndarray, labels: np.ndarray) -> dict:
  return {
    "silhouette": metrics.silhouette_score(data, cluster_labels),
    "calinski_harabasz_score": metrics.calinski_harabasz_score(data, cluster_labels),
    "davies_bouldin_score": metrics.davies_bouldin_score(data, cluster_labels),
    "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
    "completeness": metrics.completeness_score(labels, cluster_labels),
    "v_measure": metrics.v_measure_score(labels, cluster_labels),
    "mutual_information": metrics.adjusted_mutual_info_score(labels, cluster_labels),
  }


def normalized_minkowski(x: np.ndarray, y: np.ndarray) -> float:
  return np.linalg.norm(x - y) / np.sqrt(x.shape[0])


if __name__ == "__main__":
  import os
  import pickle as p
  import pandas as pd
  from time import time
  import matplotlib as mpl
  import matplotlib.pyplot as plt
  from collections import defaultdict
  from scipy.io import loadmat, savemat
  from scipy.spatial.distance import cdist

  from model import HOOK_TARGETS
  from dim_reduction import pca_reduction
  from utils import select_random_samples, load_MNIST_labels

  model_name = "resnet18"
  dataset = "MNIST"
  identifier = "channel_neuron_max"
  idx = None
  labels = load_MNIST_labels()

  idx = select_random_samples(labels, 700)

  exp_dir = f"../logs/{model_name}_{dataset}"
  out_path = os.path.join(exp_dir, "clusters")
  os.makedirs(out_path, exist_ok=True)

  vars = HOOK_TARGETS[model_name]
  if idx is not None:
    labels = labels[idx]

  method = cluster.SpectralClustering
  low, high = 2, 15
  param = {
    "affinity": "precomputed",
    "n_jobs": -1,
    "eigen_solver": "amg",
    "eigen_tol": 5e-5,
    "assign_labels": "cluster_qr"
  }

  print(f"+ Working on: {exp_dir}")
  out = {}
  scores = defaultdict(list)
  for var in vars:
    print(var, "\n----------------")
    out[var] = []
    x = loadmat(
      os.path.join(exp_dir, "activations", f"act_pretrained.mat"), variable_names=[var]
    )[var]
    if idx is not None:
      x = x[idx]

    # x = x.reshape(x.shape[0], -1)

    x = x.max(-2).max(-1)
    x = x - x.mean(0)
    x = x / np.abs(x).max()
    print(f"Data shape: {x.shape}")

    # print(f"Before PCA: {x.shape}")
    # o = x.shape[-1]
    # tick = time()
    # x = pca_reduction(x, n_components=None, threshold=0.95)
    # print(f"PCA took {time() - tick:.3f} seconds")
    # r = x.shape[-1]
    # print(f"After PCA: {x.shape}")

    os.makedirs(os.path.join(exp_dir, "clusters"), exist_ok=True)
    affinity_path = os.path.join(
      exp_dir, "clusters", f"{var.replace('.', '_')}_{identifier}_affinity.mat"
    )

    if os.path.exists(affinity_path):
      affinity = loadmat(affinity_path)["affinity"]
    else:
      print(f"Calculating new affinity matrix for {var}...")
      tick = time()
      affinity = np.exp(-cdist(x, x, metric=normalized_minkowski)**2 / 4)
      print(f"Affinity matrix is calculated in {time() - tick:.3f} seconds")
      savemat(affinity_path, {"affinity": affinity})

    for n in range(low, high):
      tick = time()
      algo = method(n_clusters=n, **param)
      cluster_labels = algo.fit(affinity).labels_
      t = time() - tick

      s = metrics.silhouette_score(x, cluster_labels)
      h = metrics.homogeneity_score(labels, cluster_labels)
      c = metrics.completeness_score(labels, cluster_labels)
      bcss, wcss, chi = bcss_wcss(x, cluster_labels, True)

      out[var].append({
        "idx": idx,
        "cluster_labels": cluster_labels,
        "params": param,
      })
      scores["layer"].append(var)
      scores["metric"].append("custom")
      scores["n_clusters"].append(n)
      scores["silhouette"].append(s)
      scores["homogeneity"].append(h)
      scores["completeness"].append(c)
      scores["chi"].append(chi)
      scores["bcss"].append(bcss)
      scores["wcss"].append(wcss)

      print(
        f"\t + N: {n}, Affinity: {param['affinity']} -> Homogeneity Score: {h:.3f} | Completeness Score: {c:.3f} | Silhouette Score: {s:.3f} | time: {t:.3f}s"
      )
    print()

  with open(os.path.join(out_path, f"{identifier}.p"), "wb") as f:
    p.dump(out, f, protocol=p.HIGHEST_PROTOCOL)

  df = pd.DataFrame(scores)
  df.to_csv(os.path.join(out_path, f"{identifier}_scores.csv"))

  scores = ["silhouette", "homogeneity", "completeness", "bcss", "wcss", "chi"]

  out_path = os.path.join(exp_dir, "figures")
  os.makedirs(out_path, exist_ok=True)

  df["silhouette"] = (df["silhouette"] + 1) / 2

  vars = df["layer"].unique()
  clusters = df["n_clusters"].unique()
  colors = mpl.color_sequences['tab10']
  mins, maxes = [], []
  for score in scores:
    mins.append(df[score].min())
    maxes.append(df[score].max())

  h, w = 2, 3
  best_idx = None
  best_vals = {}
  for var in vars:
    _, axs = plt.subplots(h, w, tight_layout=True, figsize=(24, 12))
    best_vals[var] = {}
    data = df[df["layer"] == var]
    for i in range(h):
      for j in range(w):
        flat_idx = (i * w) + j
        score = scores[flat_idx]
        best_idx = np.argmax(data[score])
        best_val = data[score].tolist()[best_idx]
        best_vals[var][score] = (best_val, clusters[best_idx])
        axs[i, j].plot(clusters, data[score].tolist(), color=colors[flat_idx], label=score)
        axs[i, j].plot([clusters[best_idx]], [best_val], "x", label='best')
        axs[i, j].set_xlabel("Number of Epochs")
        axs[i, j].set_xticks(clusters)
        axs[i, j].set_ylabel("Score")
        axs[i, j].set_ylim(mins[flat_idx] * 0.9, maxes[flat_idx] * 1.1)
        axs[i, j].grid()
        axs[i, j].legend()
        axs[i, j].set_title(score)
    plt.suptitle(var)
    # plt.show()
    plt.savefig(os.path.join(out_path, f"{var.replace('.','_')}_{identifier}.png"), bbox_inches="tight")
    plt.clf()
    plt.close("all")

  _, axs = plt.subplots(h, w, tight_layout=True, figsize=(24, 12))
  for i in range(h):
    for j in range(w):
      flat_idx = (i * w) + j
      score = scores[flat_idx]
      data = [v[score][0] for v in best_vals.values()]
      axs[i, j].plot(vars, data, color=colors[flat_idx])
      for k, (y, l) in enumerate(zip(data, [v[score][1] for v in best_vals.values()])):
        axs[i, j].text(k, y, f"N={l}", ha='center', va='bottom')
      axs[i, j].set_xlabel("Layers")
      axs[i, j].set_xticks(vars)
      axs[i, j].set_ylabel("Score")
      axs[i, j].set_ylim(mins[flat_idx] * 0.9, maxes[flat_idx] * 1.1)
      axs[i, j].grid()
      axs[i, j].set_title(score)
  plt.suptitle("Best Scores")
  # plt.show()
  plt.savefig(os.path.join(out_path, f"best_{identifier}.png"), bbox_inches="tight")
  plt.clf()
  plt.close("all")

