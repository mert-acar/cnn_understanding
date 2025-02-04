import torch
import torch.nn.functional as F

from typing import Callable, Any


class LossRegistry:
  def __init__(self):
    self.loss_classes = {}

  def register(self, name: str):
    def decorator(cls: Callable):
      self.loss_classes[name] = cls
      return cls

    return decorator

  def get(self, name: str, **kwargs):
    if name in self.loss_classes:
      return self.loss_classes[name](**kwargs)
    else:
      raise ValueError(f"Loss class '{name}' not found in the registry.")


loss_registry = LossRegistry()


@loss_registry.register("cross_entropy_loss")
class CrossEntropyLoss(torch.nn.Module):
  def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


@loss_registry.register("cluster_inducing_loss")
class ClusterInducingLoss(torch.nn.Module):
  def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = F.softmax(logits, dim=1)
    k = pred.shape[-1]
    ri = torch.sqrt(torch.sum(pred, dim=0))
    num = pred / ri
    q = num / torch.sum(num, dim=0)
    loss = -1 * torch.mean(q * torch.log(pred)) / k
    return loss


@loss_registry.register("maximal_coding_rate_reduction")
class MaximalCodingRateReduction(torch.nn.Module):
  def __init__(self, gam1: float = 1.0, gam2: float = 1.0, eps: float = 0.01, num_classes: int = 10):
    super(MaximalCodingRateReduction, self).__init__()
    self.gam1 = gam1
    self.gam2 = gam2
    self.eps = eps
    self.num_classes = num_classes

  def compute_discrimn_loss_empirical(self, embed: torch.Tensor) -> torch.Tensor:
    p, m = embed.shape
    I = torch.eye(p, device=embed.device)
    scalar = p / (m * self.eps)
    logdet = torch.logdet(I + self.gam1 * scalar * embed.matmul(embed.T))
    return logdet / 2.

  def compute_compress_loss_empirical(self, embed: torch.Tensor, pi: torch.Tensor) -> torch.Tensor:
    p, m = embed.shape
    k, _, _ = pi.shape
    I = torch.eye(p, device=embed.device)
    compress_loss = 0.0
    for j in range(k):
      trPi = torch.trace(pi[j]) + 1e-8
      scalar = p / (trPi * self.eps)
      log_det = torch.logdet(I + scalar * embed.matmul(pi[j]).matmul(embed.T))
      compress_loss += log_det * trPi / m
    return compress_loss / 2.0

  def label_to_membership(self, labels: torch.Tensor) -> torch.Tensor:
    num_samples = labels.size(0)
    pi = torch.zeros((self.num_classes, num_samples, num_samples), dtype=torch.float32, device=labels.device)
    row_indices = torch.arange(num_samples, device=labels.device)
    pi[labels, row_indices, row_indices] = 1.0
    return pi

  def forward(self, embed: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    w = embed.T
    pi = self.label_to_membership(targets)
    discrimn_loss_empi = self.compute_discrimn_loss_empirical(w)
    compress_loss_empi = self.compute_compress_loss_empirical(w, pi)
    total_loss_empi = self.gam2 * -discrimn_loss_empi + compress_loss_empi
    return total_loss_empi


@loss_registry.register("calinski_harabasz_index")
class CalinskiHarabaszLoss(torch.nn.Module):
  def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    labels = labels.long().view(-1)
    n = embeddings.size(0)
    unique_labels, inverse_indices = torch.unique(labels, return_inverse=True)
    k = unique_labels.size(0)

    # Handle edge cases (no variance or single cluster)
    if k < 2 or n == k:
      return torch.tensor(0.0, device=embeddings.device, requires_grad=True)

    sum_per_cluster = torch.zeros((k, embeddings.size(1)), device=embeddings.device)
    sum_per_cluster.index_add_(0, inverse_indices, embeddings)
    counts = torch.bincount(inverse_indices, minlength=k).float().view(-1, 1)
    cluster_means = sum_per_cluster / counts
    overall_mean = embeddings.mean(dim=0, keepdim=True)

    # Between-cluster variance (BCSS)
    diff_b = cluster_means - overall_mean
    bcss = (diff_b.pow(2).sum(dim=1) * counts.squeeze()).sum()

    # Within-cluster variance (WCSS)
    cluster_means_expanded = cluster_means[inverse_indices]
    diff_w = embeddings - cluster_means_expanded
    wcss = diff_w.pow(2).sum(dim=1).sum()

    ch_score = (bcss * (n - k)) / ((k - 1) * wcss + 1e-10)
    return -ch_score


@loss_registry.register("completenesss_homogeneity_loss")
class CompletenessHomogeneityLoss(torch.nn.Module):
  def __init__(self, num_classes: int = 10, lambda_h: float = 1.0, lambda_c: float = 1.0):
    super().__init__()
    self.num_classes = num_classes
    self.lambda_h = lambda_h
    self.lambda_c = lambda_c
    self.eps = 1e-10
    self.alpha = 5e-2

  def entropy(self, probs: torch.Tensor) -> torch.Tensor:
    freq = probs.sum(0)
    freq_s = freq.sum()
    return -((freq / freq_s) * (torch.log(freq) - torch.log(freq_s))).sum()

  def get_soft_contingency_matrix(self, pred_probs: torch.Tensor, true_probs: torch.Tensor) -> torch.Tensor:
    return torch.einsum('ij,ik->ijk', true_probs, pred_probs).sum(0)

  def get_mutual_info(self, contingency: torch.Tensor) -> torch.Tensor:
    c_sum = contingency.sum()
    pi = torch.ravel(contingency.sum(1))
    pj = torch.ravel(contingency.sum(0))
    log_contingency_nm = torch.log(contingency).flatten()
    contingency_nm = contingency.flatten() / c_sum
    outer = torch.outer(pi, pj).flatten()
    log_outer = -torch.log(outer) + torch.log(pi.sum()) + torch.log(pj.sum())
    mi = contingency_nm * (log_contingency_nm - torch.log(c_sum)) + contingency_nm * log_outer
    mi = torch.clamp(mi.sum(), min=0.0)
    return mi

  def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred_probs = F.softmax(logits / 0.05, dim=1)
    true_probs = F.one_hot(targets, num_classes=self.num_classes).float()
    entropy_C = self.entropy(true_probs)
    entropy_K = self.entropy(pred_probs)
    contingency_m = self.get_soft_contingency_matrix(pred_probs, true_probs)
    mutual_info = self.get_mutual_info(contingency_m)
    homogeneity = mutual_info / (entropy_C + self.eps)
    completeness = mutual_info / (entropy_K + self.eps)
    print(homogeneity, completeness)
    return self.lambda_h * (1.0 - homogeneity) + self.lambda_c * (1.0 - completeness)


class CompositeLoss(torch.nn.Module):
  def __init__(self, modules: dict[str, dict[str, Any]]):
    super().__init__()
    self.losses = []
    for loss_name, args in modules.items():
      kwargs = args.get("args", {})
      weight = args.get("weight", 1)
      self.losses.append((loss_registry.get(loss_name, **kwargs), weight))

  def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    total_loss = 0
    for loss_fn, w in self.losses:
      total_loss += loss_fn(logits, targets) * w
    return total_loss
