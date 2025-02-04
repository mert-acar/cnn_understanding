import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import Normalize
from .attention import SAM, CBAM


def build_layer(
  in_ch: int,
  out_ch: int,
  kernel_size: int = 3,
  padding: int = 1,
  stride: int = 1,
  attention: str = "none",
  relu: bool = False,
  batch_norm: bool = True,
  bias: bool = False
):
  assert attention.lower() in ["none", "sam", "cbam"]
  layer = [
    nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias),
  ]

  if batch_norm:
    layer.append(nn.BatchNorm2d(out_ch))

  if relu:
    layer.append(nn.ReLU())

  if attention.lower() == "sam":
    layer.append(SAM())
  elif attention.lower() == "cbam":
    layer.append(CBAM(out_ch))
  else:
    layer.append(nn.Identity())

  return nn.Sequential(*layer)


class CustomCNN(nn.Module):
  def __init__(
    self,
    num_layers: int = 1,
    in_ch: int = 3,
    out_dim: int = 256,
    attention: str = "none",
    relu: bool = False,
    start_ch: int = 32,
    batch_norm: bool = False
  ):
    super().__init__()
    k, p, s = 7, 2, 3
    layers = [build_layer(in_ch, start_ch, k, p, s, attention, relu, batch_norm)]
    out = (32 + (2 * p) - k) // s + 1
    prev = start_ch
    for _ in range(num_layers - 1):
      layers.append(build_layer(prev, prev * 2, 3, 0, 1, attention, relu, batch_norm))
      prev = prev * 2
      out = out - 2
    layers.append(nn.Flatten())
    if out_dim != 0:
      layers.append(nn.Linear(prev * out * out, out_dim, bias=False))
    layers.append(Normalize())
    self.layers = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers(x)


class ClassificationHead(nn.Module):
  def __init__(self, num_classes: int = 10, latent_dim: int = 256):
    super().__init__()
    self.fc = nn.Linear(latent_dim, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)


class ClusterHead(nn.Module):
  def __init__(self, num_clusters: int = 10, latent_dim: int = 256, temperature: float = 1.0):
    super().__init__()
    self.temperature = temperature
    self.cluster_centroids = nn.Parameter(torch.randn(num_clusters, latent_dim))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    centers = F.normalize(self.cluster_centroids)
    similarities = torch.mm(x, centers.t())  # cosine similarities
    return similarities


class ClassifyingCNN(nn.Module):
  def __init__(
    self,
    num_classes: int = 10,
    num_layers: int = 1,
    in_ch: int = 3,
    out_dim: int = 256,
    attention: str = "none",
    relu: bool = False,
    start_ch: int = 32,
    batch_norm: bool = False
  ):
    super().__init__()
    self.feature_extractor = CustomCNN(num_layers, in_ch, out_dim, attention, relu, start_ch, batch_norm)
    self.classification_head = ClassificationHead(num_classes, out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classification_head(self.feature_extractor(x))


class ClusteringCNN(nn.Module):
  def __init__(
    self,
    num_clusters: int = 10,
    temperature: float = 1.0,
    num_layers: int = 1,
    in_ch: int = 3,
    out_dim: int = 256,
    attention: str = "none",
    relu: bool = False,
    start_ch: int = 32,
    batch_norm: bool = False
  ):
    super().__init__()
    self.feature_extractor = CustomCNN(num_layers, in_ch, out_dim, attention, relu, start_ch, batch_norm)
    self.cluster_head = ClusterHead(num_clusters, out_dim, temperature)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.cluster_head(self.feature_extractor(x))
