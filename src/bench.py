import torch
import pickle as p
import numpy as np
from tqdm import tqdm
from yaml import full_load
from time import perf_counter
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from utils import create_dataloader, create_model


def hook_fn(model, input, output):
  activations.append(output.detach())


if __name__ == "__main__":
  with open("../data/class_activations_conv1.pkl", "rb") as f:
    activations = p.load(f)

  activations = np.concatenate(list(activations.values()), 0)
  flat_activations = activations.reshape(10000, -1)


  # scaler = StandardScaler()
  # normalized_activations = scaler.fit_transform(flat_activations)
  # reduced_activations = PCA(n_components=118).fit_transform(normalized_activations)
  # db = HDBSCAN(min_cluster_size=13).fit(reduced_activations)
  # clusters = db.labels_
  # cmap = plt.get_cmap('viridis')
  # colors = cmap(np.linspace(0, 1, len(set(clusters))))
  # x, y, z = np.indices((64, 14, 14))
  # fig = plt.figure()
  # ax = fig.add_subplot(111, projection='3d')
  # threshold = 1
  # print(len(set(clusters)))
  # for i in range(len(set(clusters)) - 1):
  #   m = clusters == i
  #   act = activations[m].mean(0)
  #   ax.scatter(
  #     x[act > threshold],
  #     y[act > threshold],
  #     z[act > threshold],
  #     color=colors[i],
  #     label=f"Cluster {i+1}"
  #   )
  # ax.set_xlabel('Filters')
  # ax.set_ylabel('H')
  # ax.set_zlabel('W')
  # ax.legend()
  # plt.savefig("3dspace.png", bbox_inches="tight")

  # n_components = list(range(100, 131, 2))
  # min_cluster_sizes = list(range(5, 41, 4))
  # scores = np.zeros((len(n_components), len(min_cluster_sizes)))
  # oscores = np.zeros((len(n_components), len(min_cluster_sizes)))
  # pbar = tqdm(total=scores.size, ncols=90)
  # for i, n in enumerate(n_components):
  #   reduced_activations = MDS(n_components=n).fit_transform(flat_activations)
  #   # reduced_activations = PCA(n_components=n).fit_transform(normalized_activations)
  #   for j, mcs in enumerate(min_cluster_sizes):
  #     db = HDBSCAN(min_cluster_size=mcs).fit(reduced_activations)
  #     clusters = db.labels_
  #     if len(set(clusters)) > 1:
  #       score = silhouette_score(reduced_activations, clusters)
  #       scores[i, j] = score
  #       score = silhouette_score(flat_activations, clusters)
  #       oscores[i, j] = score
  #     pbar.update(1)
  # print(f"Best Silhouette Score: {scores.max():.3f}")
  # print(f"Best Silhouette Score Parameters: {np.unravel_index(np.argmax(scores), scores.shape)}")
  # print("--")
  # print(f"Best Silhouette Score [Original]: {oscores.max():.3f}")
  # print(
  #   f"Best Silhouette Score Parameters [Original]: {np.unravel_index(np.argmax(oscores), oscores.shape)}"
  # )
