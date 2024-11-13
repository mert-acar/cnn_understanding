from enum import member
import numpy as np
from pprint import pprint
from scipy.stats import binom
from sklearn.metrics import silhouette_score, homogeneity_score, completeness_score, calinski_harabasz_score


def find_threhsold(V, p, epsilon=0.0005):
  T = V
  # Iterate downwards to find the smallest T such that P(X >= T) <= epsilon
  while T > 0:
    # Compute P(X <= T - 1)
    prob_less_than_T = binom.cdf(T - 1, V, p)
    # Check if P(X >= T) = 1 - P(X <= T - 1) <= epsilon
    if 1 - prob_less_than_T >= epsilon:
      return T + 1
    # Move to the next lower T
    T -= 1
  return -1


def chernoff_bound(z, k, delta, p=1):
  return (z * k * p) + (k * p * np.log(1 / delta)) + (
    k * p * np.sqrt((np.log(1 / delta)**2) + (2 * z * np.log(1 / delta)))
  )


if __name__ == "__main__":
  import os
  from scipy.io import loadmat

  exp_dir = "../logs/customnet_run2/"
  labels = loadmat("../data/labels.mat")["labels"][0]
  epoch = 33
  var = "features.8"
  x = loadmat(
    os.path.join(exp_dir, "activations", f"act_epoch_{epoch}.mat"),
    variable_names=[var],
  )[var]
  x = x.reshape(x.shape[0], -1)
  x = x - x.mean(0)
  x = x / np.max(np.abs(x), 0, keepdims=True)

  # PARAMETERS
  min_points_per_cluster = 30
  min_cluster_size = 320
  distance_threshold = 150
  n_neighbors = 20
  rng = np.random.default_rng(seed=9001)
  value_range = x.max() - x.min()

  # SAMPLING
  k = len(x) // min_cluster_size
  len_S = int(chernoff_bound(min_points_per_cluster, k, 0.01))
  len_M = int(chernoff_bound(1, k, 0.01))
  sample_idx = rng.choice(np.arange(len(x)), size=len_S + len_M, replace=False)
  S = x[sample_idx[:len_S]]
  M = x[sample_idx[len_S:]]
  n_dim = M.shape[-1]

  eps_list = [i * value_range / 100 for i in range(1, 26)]
  best_soundness = 0
  best_medoid = None
  best_kd_masks = None

  for eps in eps_list:

    # DIMENSION VOTING
    voting_threshold = find_threhsold(V=n_neighbors, p=2 * eps / value_range)
    kd_masks = np.zeros_like(M, dtype=bool)
    for i in range(len(M)):
      m = M[i:i + 1]
      dimension_distances = (np.abs(m - S) <= eps).astype(int)  # dimension_distances: [len_S, n_dim]
      idx = np.argpartition(n_dim - dimension_distances.sum(1), n_neighbors - 1)[:n_neighbors]
      kd_masks[i] = dimension_distances[idx].sum(0) > voting_threshold

    # CLUSTER SIMULATING
    memberships = np.zeros(len_S, dtype=int)
    mask_cardinalities = kd_masks.sum(1)
    for i, s in enumerate(S):
      distances = np.zeros(len(M), dtype=int)
      for j in range(len(M)):
        m = M[j]
        kd = kd_masks[j]
        distances[j] = mask_cardinalities[j] - (np.abs(m[kd] - s[kd]) <= eps).sum()
      zero_indices = np.where(distances == 0)[0]
      if len(zero_indices) == 0:
        memberships[i] = -1
        continue
      else:
        memberships[i] = zero_indices[np.argmax(mask_cardinalities[zero_indices])]

    # MEDOID CLUSTERING
    def dod(i, j):
      ci = mask_cardinalities[i]
      cj = mask_cardinalities[j]
      common_mask = kd_masks[i] & kd_masks[j]
      distance = (np.abs(M[i][common_mask] - M[j][common_mask]) <= eps).sum()
      if distance <= 2:
        return n_dim
      return max(ci, cj) - distance

    medoid_distance_matrix = np.zeros((len_M, len_M))
    for i in range(len_M):
      for j in range(i + 1, len_M):
        distance = dod(i, j)
        medoid_distance_matrix[i, j] = distance
        medoid_distance_matrix[j, i] = distance

    # Initialize medoid_clusters (each point starts in its own cluster)
    medoid_clusters = [[i] for i in range(len_M)]

    while True:
      min_dist = float('inf')
      merge_pair = None
      for i in range(len(medoid_clusters)):
        for j in range(i + 1, len(medoid_clusters)):
          cluster_dist = 0
          counti = 0
          countj = 0
          up = True
          for idx1 in medoid_clusters[i]:
            mi = (memberships == idx1).sum() + 1
            counti += mi
            for idx2 in medoid_clusters[j]:
              mj = (memberships == idx2).sum() + 1
              if up:
                countj += mj
              cluster_dist += (mi * mj * medoid_distance_matrix[idx1, idx2])
            up = False
          cluster_dist /= (counti * countj)

          if cluster_dist < min_dist:
            min_dist = cluster_dist
            merge_pair = (i, j)

      if min_dist > distance_threshold:
        break

      # Merge the closest medoid_clusters
      i, j = merge_pair
      new_cluster = medoid_clusters[i] + medoid_clusters[j]

      # Remove old medoid_clusters and add the merged one
      medoid_clusters = [c for idx, c in enumerate(medoid_clusters) if idx not in (i, j)]
      medoid_clusters.append(new_cluster)

    # MC REFINING
    for l, cluster in enumerate(medoid_clusters):
      dims = [False] * n_dim
      for d in range(n_dim):
        nom = 0
        denom = 0
        for i in cluster:
          denom += (memberships == i).sum()
          if d in np.where(kd_masks[i] != False)[0]:
            nom += (memberships == i).sum()
        if denom == 0:
          continue
        ratio = nom / denom
        if ratio >= 0.95:
          dims[d] = True
      dims = np.array(dims)
      for i in cluster:
        kd_masks[i] = dims

    medoid_distance_matrix = np.zeros((len_M, len_M))
    for i in range(len_M):
      for j in range(i + 1, len_M):
        distance = dod(i, j)
        medoid_distance_matrix[i, j] = distance
        medoid_distance_matrix[j, i] = distance

    while True:
      min_dist = float('inf')
      merge_pair = None
      for i in range(len(medoid_clusters)):
        for j in range(i + 1, len(medoid_clusters)):
          cluster_dist = 0
          counti = 0
          countj = 0
          up = True
          for idx1 in medoid_clusters[i]:
            mi = (memberships == idx1).sum() + 1
            counti += mi
            for idx2 in medoid_clusters[j]:
              mj = (memberships == idx2).sum() + 1
              if up:
                countj += mj
              cluster_dist += (mi * mj * medoid_distance_matrix[idx1, idx2])
            up = False
          cluster_dist /= (counti * countj)

          if cluster_dist < min_dist:
            min_dist = cluster_dist
            merge_pair = (i, j)

      if min_dist > distance_threshold:
        break

      # Merge the closest medoid_clusters
      i, j = merge_pair
      new_cluster = medoid_clusters[i] + medoid_clusters[j]

      # Remove old medoid_clusters and add the merged one
      medoid_clusters = [c for idx, c in enumerate(medoid_clusters) if idx not in (i, j)]
      medoid_clusters.append(new_cluster)

    for l, cluster in enumerate(medoid_clusters):
      dims = [False] * n_dim
      for d in range(n_dim):
        nom = 0
        denom = 0
        for i in cluster:
          denom += (memberships == i).sum()
          if d in np.where(kd_masks[i] != False)[0]:
            nom += (memberships == i).sum()
        if denom == 0:
          continue
        ratio = nom / denom
        if ratio >= 0.95:
          dims[d] = True
      dims = np.array(dims)
      for i in cluster:
        kd_masks[i] = dims

    medoid_clusters = [cluster for cluster in medoid_clusters if sum((memberships == i).sum() for i in cluster) >  min_points_per_cluster]
    soundness = 0
    for cluster in medoid_clusters:
      size = sum((memberships == i).sum() for i in cluster)
      soundness += size * kd_masks[cluster[0]].sum()  
    print(f"{eps:.5f} -> Soundness: {soundness}")

    if soundness > best_soundness:
      best_soundness = soundness
      best_kd_masks = kd_masks
      best_medoid = medoid_clusters

  # CLUSTER SIMULATING
  memberships = np.zeros(len(x), dtype=int)
  mask_cardinalities = best_kd_masks.sum(1)
  cluster_map = {}
  for key, value in enumerate(medoid_clusters):
    for v in value:
      cluster_map[v] = key

  for i, data in enumerate(x):
    distances = np.ones(len(M), dtype=int)
    for j in range(len(medoid_clusters)):
      for c, cluster in enumerate(medoid_clusters[j]):
        m = M[c]
        kd = best_kd_masks[c]
        distances[c] = mask_cardinalities[c] - (np.abs(m[kd] - data[kd]) <= eps).sum()
    zero_indices = np.where(distances == 0)[0]
    if len(zero_indices) == 0:
      memberships[i] = -1
    else:
      key = int(zero_indices[np.argmax(mask_cardinalities[zero_indices])])
      if key in cluster_map:
        memberships[i] = cluster_map[key]
      else:
        memberships[i] = -1


  idx = np.where(memberships != -1)[0]
  cluster_labels = memberships[idx]
  pprint({
    "noise_ratio": (memberships == -1).sum() / len(memberships),
    "n_clusters": len(np.unique(cluster_labels)),
    "silhouette": silhouette_score(x[idx], cluster_labels),
    "homogeneity": homogeneity_score(labels[idx], cluster_labels),
    "completeness": completeness_score(labels[idx], cluster_labels),
    "chi": calinski_harabasz_score(x[idx], cluster_labels),
  })

