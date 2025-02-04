import torch
import numpy as np
from sklearn import metrics
from typing import Dict, Union, List

from loss import MaximalCodingRateReduction


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
    return np.argmax(self.outputs_np, axis=1)

  def accuracy(self) -> float:
    return (self._get_cluster_assignments() == self.targets_np).mean()

  def completeness_score(self) -> float:
    return float(metrics.completeness_score(self.targets_np, self._get_cluster_assignments()))

  def homogeneity_score(self) -> float:
    return float(metrics.homogeneity_score(self.targets_np, self._get_cluster_assignments()))

  def normalized_mutual_information(self) -> float:
    return float(metrics.normalized_mutual_info_score(self.targets_np, self._get_cluster_assignments()))

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
