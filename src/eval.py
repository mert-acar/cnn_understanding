import numpy as np
from sklearn.svm import LinearSVC
from sklearn import metrics, cluster


def bcss(x: np.ndarray, pred_labels: np.ndarray) -> float:
  n_labels = len(set(pred_labels))
  extra_disp = 0.0
  mean = x.mean(0)
  for k in range(n_labels):
    cluster_k = x[pred_labels == k]
    mean_k = cluster_k.mean(0)
    extra_disp += len(cluster_k) * ((mean_k - mean)**2).sum()
  return float(extra_disp)


def wcss(x: np.ndarray, pred_labels: np.ndarray) -> float:
  n_labels = len(set(pred_labels))
  intra_disp = 0.0
  for k in range(n_labels):
    cluster_k = x[pred_labels == k]
    mean_k = cluster_k.mean(0)
    intra_disp += ((cluster_k - mean_k)**2).sum()
  return float(intra_disp)


def cluster_accuracy(pred_labels: np.ndarray, true_labels: np.ndarray) -> float:
  return (pred_labels == true_labels).mean()


def svm(activations: np.ndarray, labels: np.ndarray) -> np.ndarray:
  model = LinearSVC(random_state=10)
  model.fit(activations, labels)
  pred_labels = model.predict(activations)
  return pred_labels


if __name__ == "__main__":
  import os
  from model import HOOK_TARGETS
  from dataset import get_labels

  model = "smallnet"
  dataset = "MNIST"
  iden = ""
  exp = "_".join([model, dataset]) + (f"_{iden}" if iden != "" else "")
  exp_dir = os.path.join("../logs", exp)
  vars = HOOK_TARGETS[model]

  var = vars[0]
  activations = np.load(os.path.join(exp_dir, "activations", f"{var.replace('.', '_')}_act.npy"))
  if activations.ndim != 2:
    activations = activations.reshape(activations.shape[0], -1)

  labels = get_labels(dataset, "test")

  print(f"Working on: {exp_dir} [{var}]\n----------")
  cluster_labels = svm(activations, labels)
  scores = {
    "Accuracy": cluster_accuracy(cluster_labels, labels) * 100,
    "Silhouette": metrics.silhouette_score(activations, cluster_labels),
    "BCSS": bcss(activations, cluster_labels),
    "WCSS": wcss(activations, cluster_labels),
    "CHI": metrics.calinski_harabasz_score(activations, cluster_labels),
    "nmi": metrics.normalized_mutual_info_score(labels, cluster_labels),
    "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
    "completeness": metrics.completeness_score(labels, cluster_labels),
  }
  for key, score in scores.items():
    print(f"{key:10}: {score:.2f}")
