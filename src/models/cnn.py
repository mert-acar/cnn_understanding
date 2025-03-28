import torch
import torch.nn as nn

from .attention import SAM, CBAM


def build_layer(
  in_ch: int,
  out_ch: int,
  kernel_size: int = 3,
  padding: int = 1,
  stride: int = 1,
  attention: str = "none",
  relu: bool = True,
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


class FeatureCNN(nn.Module):
  def __init__(
    self,
    num_layers: int = 1,
    in_ch: int = 3,
    out_dim: int = 256,
    attention: str = "none",
    relu: bool = True,
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
      layers.append(nn.ReLU())
    self.layers = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.layers(x)


class ClassificationHead(nn.Module):
  def __init__(self, num_classes: int = 10, latent_dim: int = 256):
    super().__init__()
    self.fc = nn.Linear(latent_dim, num_classes)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.fc(x)


<<<<<<< HEAD
=======
class ClusterHead(nn.Module):
  def __init__(self, num_clusters: int = 10, latent_dim: int = 256):
    super().__init__()

    self.cluster_centroids = nn.Parameter(
      self.init_points(num_clusters, latent_dim), requires_grad=True
    )

  def init_points(self, num_clusters: int = 10, latent_dim: int = 256) -> torch.Tensor:
    matrix = torch.randn(latent_dim, latent_dim)
    q, _ = torch.linalg.qr(matrix)
    points = q[:num_clusters]
    return F.normalize(points, p=2, dim=1)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return torch.mm(x, F.normalize(self.cluster_centroids).t())


>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33
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
<<<<<<< HEAD
    self.feature_extractor = FeatureCNN(num_layers, in_ch, out_dim, attention, relu, start_ch, batch_norm)
=======
    self.feature_extractor = CustomCNN(
      num_layers, in_ch, out_dim, attention, relu, start_ch, batch_norm
    )
>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33
    self.classification_head = ClassificationHead(num_classes, out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.classification_head(self.feature_extractor(x))
<<<<<<< HEAD
=======


class ClusteringCNN(nn.Module):
  def __init__(
    self,
    num_clusters: int = 10,
    num_layers: int = 1,
    in_ch: int = 3,
    out_dim: int = 256,
    attention: str = "none",
    relu: bool = False,
    start_ch: int = 32,
    batch_norm: bool = False
  ):
    super().__init__()
    self.feature_extractor = CustomCNN(
      num_layers, in_ch, out_dim, attention, relu, start_ch, batch_norm
    )
    self.cluster_head = ClusterHead(num_clusters, out_dim)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.cluster_head(self.feature_extractor(x))
>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33
