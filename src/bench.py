import numpy as np
from tqdm import tqdm
from scipy.io import loadmat
from sklearn.cluster import HDBSCAN, OPTICS
from sklearn.metrics import silhouette_score

if __name__ == "__main__":
  data_path = "../data/svd_low_rank_recon_run1_conv1.mat"
  # data_path = "../data/pca_recon_run1_conv1.mat"
  data = loadmat(data_path, variable_names=["activations"])["activations"]
  labels = loadmat("../data/labels.mat")["labels"][0]

  print("HDBSCAN")
  cluster_sizes = [10, 15, 20, 25, 30, 35]
  selection_epsilons = [0.00, 0.02, 0.04, 0.06, 0.08, 0.1]
  results = np.zeros((len(cluster_sizes), len(selection_epsilons))) - 1
  pbar = tqdm(total=results.size, desc="Grid-Search")
  for i, mcs in enumerate(cluster_sizes):
    for j, cse in enumerate(selection_epsilons):
      clusters = HDBSCAN(min_cluster_size=mcs, cluster_selection_epsilon=cse).fit(data).labels_
      num_clusters = len(set(clusters)) - 1
      noisy_samples = (clusters == -1).sum() / data.shape[0]
      if 2 < num_clusters < 11:
        sscore = silhouette_score(data, clusters)
        results[i] = sscore
        pbar.set_description(
          f"[{num_clusters}] - [{noisy_samples:.4f}] - [{mcs} / {cse}] - {sscore:.3f}"
        )
      pbar.update(1)

  print(f"Best silhouette score: {results.max():.3f}")
  print(
    f"Best parameters: mcs={cluster_sizes[np.unravel_index(results.argmax(), results.shape)[0]]} cse={selection_epsilons[np.unravel_index(results.argmax(), results.shape)[1]]}"
  )

  print("PCA")
  cluster_sizes = [10, 15, 20, 25, 30, 35]
  results = np.zeros(len(cluster_sizes)) - 1
  pbar = tqdm(total=results.size, desc="Grid-Search")
  for i, mcs in enumerate(cluster_sizes):
    clusters = OPTICS(min_samples=mcs).fit(data).labels_
    num_clusters = len(set(clusters)) - 1
    noisy_samples = (clusters == -1).sum() / data.shape[0]
    if 2 < num_clusters < 11:
      sscore = silhouette_score(data, clusters)
      results[i] = sscore
      pbar.set_description(
        f"[{num_clusters}] - [{noisy_samples:.4f}] - [{mcs}] - {sscore:.3f}"
      )
    pbar.update(1)

  print(f"Best silhouette score: {results.max():.3f}")
  print(f"Best parameter: mcs={cluster_sizes[results.argmax()]}")
