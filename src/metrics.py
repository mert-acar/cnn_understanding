import torch
import numpy as np
from sklearn import metrics
from typing import Dict, Union, List
from scipy.optimize import linear_sum_assignment

from loss import MaximalCodingRateReduction


def softmax(x: np.ndarray) -> np.ndarray:
  return np.exp(x) / sum(np.exp(x))


class MetricCalculator:
  def __init__(self, metric_names: List[str]):
    self.mcr = MaximalCodingRateReduction(1, 1, 0.05, 10)
    self.metric_names = metric_names

  def _register_data(
    self, outputs: Union[torch.Tensor, np.ndarray], targets: Union[torch.Tensor, np.ndarray]
  ):
    if isinstance(outputs, torch.Tensor):
      self.outputs_np = outputs.detach().cpu().numpy()
    else:
      self.outputs_np = outputs

    if isinstance(targets, torch.Tensor):
      self.targets_np = targets.detach().cpu().numpy()
    else:
      self.targets_np = targets

  def _get_cluster_assignments(self) -> np.ndarray:
    if self.outputs_np.sum(1).sum() != len(self.outputs_np):
      probs = softmax(self.outputs_np)
    else:
      probs = self.outputs_np
    return np.argmax(probs, axis=1)

  def accuracy(self) -> float:
    pred_labels = self._get_cluster_assignments()
    true_labels = self.targets_np

    pred_classes = list(range(10))
    true_classes = list(range(10))

    n_classes = max(len(pred_classes), len(true_classes))
    cost_matrix = np.zeros((n_classes, n_classes))

    for i in range(len(pred_classes)):
      for j in range(len(true_classes)):
        pred_idx = pred_labels == pred_classes[i]
        true_idx = true_labels == true_classes[j]
        cost_matrix[i, j] = -np.sum(pred_idx & true_idx)

    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    mapping = {pred_classes[i]: true_classes[j] for i, j in zip(row_ind, col_ind)}
    aligned_preds = np.array([mapping[label] for label in pred_labels])
    accuracy = np.sum(aligned_preds == true_labels) / len(true_labels)
    return accuracy

  def completeness_score(self) -> float:
    return float(metrics.completeness_score(self.targets_np, self._get_cluster_assignments()))

  def homogeneity_score(self) -> float:
    return float(metrics.homogeneity_score(self.targets_np, self._get_cluster_assignments()))

  def normalized_mutual_information(self) -> float:
    return float(
      metrics.normalized_mutual_info_score(self.targets_np, self._get_cluster_assignments())
    )

  def calinski_harabasz_index(self) -> float:
    return float(metrics.calinski_harabasz_score(self.outputs_np, self.targets_np))

  def bcss(self) -> float:
    n_labels = len(set(self.targets_np))
    extra_disp = 0.0
    mean = self.outputs_np.mean(0)
    for k in range(n_labels):
      cluster_k = self.outputs_np[self.targets_np == k]
      if len(cluster_k) == 0:
        continue
      mean_k = cluster_k.mean(0)
      extra_disp += len(cluster_k) * ((mean_k - mean)**2).sum()
    return float(extra_disp)

  def wcss(self) -> float:
    n_labels = len(set(self.targets_np))
    intra_disp = 0.0
    for k in range(n_labels):
      cluster_k = self.outputs_np[self.targets_np == k]
      if len(cluster_k) == 0:
        continue
      mean_k = cluster_k.mean(0)
      intra_disp += ((cluster_k - mean_k)**2).sum()
    return float(intra_disp)

  def maximal_coding_rate(self) -> float:
    delta_R = 0
    step = 128
    x = torch.from_numpy(self.outputs_np)
    y = torch.from_numpy(self.targets_np)
    n = np.ceil(len(x) / step)
    for i in range(0, int(n)):
      w = x[i * step:(i + 1) * step].T
      pi = self.mcr.label_to_membership(y[i * step:(i + 1) * step])
      r = self.mcr.compute_discrimn_loss_empirical(w) / n
      rc = self.mcr.compute_compress_loss_empirical(w, pi) / n
      delta_R += r - rc
    return delta_R

  def calculate_metrics(self, outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    self._register_data(outputs, targets)
    results = {}
    for metric_name in self.metric_names:
      metric_fn = getattr(self, metric_name, None)
      if metric_fn is None:
        raise ValueError(f"Metric {metric_name} not implemented")
      results[metric_name] = float(metric_fn())

    return results


def clustering_accuracy(y_true, y_pred):
  """
    Calculate clustering accuracy after using the Hungarian algorithm to find the
    best matching between true and predicted labels.
    
    Args:
        y_true: true labels, numpy.array of shape (n_samples,)
        y_pred: predicted labels, numpy.array of shape (n_samples,)
        
    Returns:
        accuracy: float, clustering accuracy
    """
  y_true = np.array(y_true)
  y_pred = np.array(y_pred)

  assert y_pred.size == y_true.size, "Size of y_true and y_pred must be equal"

  D = max(y_pred.max(), y_true.max()) + 1
  w = np.zeros((D, D), dtype=int)

  # Count the intersection between y_true and y_pred
  for i in range(y_pred.size):
    w[y_pred[i], y_true[i]] += 1

  # Use Hungarian algorithm to find the best matching
  row_ind, col_ind = linear_sum_assignment(w.max() - w)

  # Calculate accuracy
  count = sum([w[row_ind[i], col_ind[i]] for i in range(len(row_ind))])

  return count / y_pred.size
