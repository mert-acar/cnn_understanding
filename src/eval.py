import numpy as np
from sklearn import metrics

from cluster import get_clustering_func, cluster_accuracy, bcss, wcss, maximal_coding_rate

if __name__ == "__main__":
  import os
  from model import HOOK_TARGETS
  from dataset import get_labels
  from yaml import dump, full_load

  model = "smallnet"
  dataset = "CIFAR10"
  iden = "CBAM"
  exp = "_".join([model, dataset]) + (f"_{iden}" if iden != "" else "")
  exp_dir = os.path.join("../logs", exp)
  labels = get_labels(dataset, "test")

  vars = HOOK_TARGETS[model]
  var = vars[-1]

  print(f"Working on: {exp_dir} [{var}]\n----------")
  activations = np.load(os.path.join(exp_dir, "activations", f"{var.replace('.', '_')}_test_act.npy"))
  if activations.ndim != 2:
    activations = activations.reshape(activations.shape[0], -1)

  cluster_root = os.path.join(exp_dir, "clusters")
  os.makedirs(cluster_root, exist_ok=True)

  method = "svm"
  func_kwargs = {"random_state": 9001}
  func = get_clustering_func(method, **func_kwargs)

  cluster_label_path = os.path.join(cluster_root, f"{var}_cluster_labels_{method}.npy")
  if os.path.exists(cluster_label_path):
    cluster_labels = np.load(cluster_label_path)
  else:
    cluster_labels = func(activations, labels)
    np.save(cluster_label_path, cluster_labels)

  dR, R, Rc = maximal_coding_rate(activations, labels)
  scores = {
    "Accuracy": cluster_accuracy(cluster_labels, labels) * 100,
    "Silhouette": metrics.silhouette_score(activations, cluster_labels),
    "BCSS": bcss(activations, cluster_labels),
    "WCSS": wcss(activations, cluster_labels),
    "CHI": metrics.calinski_harabasz_score(activations, cluster_labels),
    "NMI": metrics.normalized_mutual_info_score(labels, cluster_labels),
    "homogeneity": metrics.homogeneity_score(labels, cluster_labels),
    "completeness": metrics.completeness_score(labels, cluster_labels),
    "ΔR": dR,
    "R": R,
    "Rc": Rc,
  }
  for key, score in scores.items():
    print(f"{key:10}: {score:.2f}")

  scores = {var: {key: round(float(value), 3) for key, value in scores.items()}}
  score_path = os.path.join(cluster_root, f"scores_{method}.yaml")
  if os.path.exists(score_path):
    with open(score_path, "r") as f:
      all_scores = full_load(f)
  else:
    all_scores = {}
  all_scores.update(scores)
  with open(score_path, "w") as f:
    dump(all_scores, f)
