import torch
import torch.nn.functional as F

class ReshapeNormalize(torch.nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.view(x.size(0), -1))

class Reshape(torch.nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.view(x.size(0), -1)

class Normalize(torch.nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x)
