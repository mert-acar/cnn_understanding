import torch
import torch.nn as nn
from torchvision import models
import torch.nn.functional as F

from .utils import ReshapeNormalize


class ClusteringResNet18(nn.Module):
  def __init__(self, in_ch: int = 1, weights: str = "DEFAULT"):
    super().__init__()
    model = models.get_model("resnet18", weights=weights)
    self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    self.bn1 = model.bn1
    self.maxpool = model.maxpool
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4
    self.avgpool = model.avgpool
    self.reshape_norm = ReshapeNormalize()
    self.temperature = 1.0
    self.cluster_centroids = nn.Parameter(torch.randn(10, 512))

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = self.reshape_norm(out)
    centers = F.normalize(self.cluster_centroids)
    similarities = torch.mm(out, centers.t())
    q = F.softmax(similarities / self.temperature, dim=1)
    return q


class ResNet18CTRL(nn.Module):
  def __init__(self, in_ch: int = 1, weights: str = "DEFAULT"):
    super().__init__()
    model = models.get_model("resnet18", weights=weights)
    self.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    self.bn1 = model.bn1
    self.maxpool = model.maxpool
    self.layer1 = model.layer1
    self.layer2 = model.layer2
    self.layer3 = model.layer3
    self.layer4 = model.layer4
    self.avgpool = model.avgpool
    self.reshape_norm = ReshapeNormalize()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = self.reshape_norm(out)
    return out
