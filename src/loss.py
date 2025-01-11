import torch
import torch.nn.functional as F


def group_lasso_penalty(model: torch.nn.Module) -> torch.Tensor:
  penalty = 0
  for name, param in model.named_parameters():
    if 'features' in name:
      penalty += torch.norm(torch.norm(param.view(param.shape[0], -1), p=2, dim=1), p=1)
  return penalty


def cluster_inducing_loss(output: torch.Tensor) -> torch.Tensor:
  pred = F.softmax(output, dim=1)
  k = pred.shape[-1]
  ri = torch.sqrt(torch.sum(pred, dim=0))
  num = pred / ri
  q = num / torch.sum(num, dim=0)
  loss = -1 * torch.mean(q * torch.log(pred)) / k
  return loss


def contrastive_loss(
  features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07
) -> torch.Tensor:

  batch_size = features.shape[0]

  labels = labels.view(-1, 1)
  mask = torch.eq(labels, labels.T).float()

  logits = torch.div(torch.matmul(features, features.T), temperature)
  # numerical stability trick
  logits = logits - torch.max(logits, dim=1, keepdim=True)[0].detach()

  self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=features.device)
  mask *= self_mask

  exp_logits = torch.exp(logits) * self_mask
  log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

  mask_pos_pairs = mask.sum(1)
  mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)

  loss = (-(mask * log_prob).sum(1) / mask_pos_pairs).mean()
  return loss
