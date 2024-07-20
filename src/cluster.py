import matplotlib as mpl
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, v_measure_score


def cluster(data, labels):
  lol = TSNE(n_components=2).fit_transform(data)
  clusters = HDBSCAN(min_cluster_size=50).fit(lol).labels_
  print(f"Number of clusters: {len(set(clusters)) - 1}")
  print(
    f"Number of noise samples: {(clusters == -1).sum()}/{data.shape[0]} [{100 * (clusters == -1).sum() / data.shape[0]:.2f}%]"
  )

  sscore = silhouette_score(data, clusters)
  vmeasure = v_measure_score(labels, clusters)
  print(f"Silhouette Score: {sscore:.3f} [-1.0 - 1.0]")
  print(f"V-measure Score: {vmeasure:.3f} [ 0.0 - 1.0]")
  return clusters, (sscore, vmeasure)


def main(data_path):
  data = loadmat(data_path, variable_names=["activations"])["activations"]
  print(f"Data shape: {data.shape}")
  print(f"Data mean: {data.mean():.4f}")
  labels = loadmat("../data/labels.mat")["labels"][0]
  clusters, _ = cluster(data, labels)

  if data.shape[1] > 2:
    embedded = TSNE(n_components=2).fit_transform(data)
  else:
    embedded = data

  _, axs = plt.subplots(1, 2, figsize=(15, 7), tight_layout=True)
  colormap = mpl.colormaps['tab20']
  for j, lbls in enumerate([clusters, labels]):
    ax = axs[j]
    for i, c in enumerate(reversed(list(set(lbls)))):
      idx = lbls == c
      if c == -1:
        lbl = "Noisy Samples"
        color = 'gray'
      else:
        lbl = f"Cluster {c}"
        color = colormap(i)
      ax.scatter(embedded[idx, 0], embedded[idx, 1], color=color, label=lbl, alpha=0.3)
    ax.grid(True)
    ax.set_title(f"{'predicted' if j == 0 else 'label'} clusters")
    ax.legend()
  plt.suptitle(f"{data_path.split('_')[-1].split('.')[0]} Activations")
  plt.show()


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
