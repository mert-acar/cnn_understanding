import os
import torch
import torch.nn as nn
from yaml import full_load
from torchvision import models
import torch.nn.functional as F

from typing import Optional, Union, Dict, Any, Tuple

HOOK_TARGETS = {
  "densenet121": ["features.conv0"] + [f"features.denseblock{i}" for i in range(1, 5)],
  "resnet18": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)],
  "smallnet": ["layer", "attention", "reshape_norm"],
  "resnet18ctrl": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)] + ["reshape_norm"],
  "efficientnetb2": [f"features.{i}" for i in range(1, 8)],
  "efficientnetb3": [f"features.{i}" for i in range(1, 8)]
}


class SAM(nn.Module):
  def __init__(self):
    super(SAM, self).__init__()
    self.conv = nn.Conv2d(
      in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    max = torch.max(x, 1)[0].unsqueeze(1)
    avg = torch.mean(x, 1).unsqueeze(1)
    concat = torch.cat((max, avg), dim=1)
    return F.sigmoid(self.conv(concat)) * x


class CAM(nn.Module):
  def __init__(self, channels: int, r: int = 4):
    super(CAM, self).__init__()
    self.linear = nn.Sequential(
      nn.Linear(in_features=channels, out_features=channels // r, bias=True), nn.ReLU(inplace=True),
      nn.Linear(in_features=channels // r, out_features=channels, bias=True)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    max = F.adaptive_max_pool2d(x, output_size=1)
    avg = F.adaptive_avg_pool2d(x, output_size=1)
    b, c, _, _ = x.size()
    linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
    linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
    return F.sigmoid(linear_max + linear_avg) * x


class CBAM(nn.Module):
  def __init__(self, channels: int, r: int = 4):
    super(CBAM, self).__init__()
    self.sam = SAM()
    self.cam = CAM(channels, r)

  def forward(self, x):
    return self.sam(self.cam(x))


class ReshapeNormalize(nn.Module):
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x.view(x.size(0), -1))

class SingleLayer(nn.Module):
  def __init__(
    self,
    in_ch: int = 1,
    num_filters: int = 32,
    kernel_size: int = 7,
    stride: int = 3,
    padding: int = 1,
    attention: Optional[str] = None,
    **kwargs
  ):
    super().__init__()
    self.layer = torch.nn.Conv2d(in_ch, num_filters, kernel_size, stride, padding, bias=False)
    if attention is None:
      self.attention = nn.Identity()
    elif attention.lower() == "sam":
      self.attention = SAM()
    elif attention.lower() == "cbam":
      self.attention = CBAM(num_filters)
    else:
      print(f"{attention} is not implemented. Reverting to identity")
      self.attention = nn.Identity()
    self.reshape_norm = ReshapeNormalize()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    act = self.layer(x)
    act = self.attention(act)
    act = self.reshape_norm(act)
    return act


class ResNet18CTRL(nn.Module):
  def __init__(self, in_ch: int = 1, **kwargs):
    super().__init__()
    model = create_model("resnet18", in_ch=in_ch)
    self.conv1 = model.conv1
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


def create_model(
  model_name: str, weights: str = "DEFAULT", in_ch: int = 1, out_ch: int = 10, **kwargs
) -> torch.nn.Module:
  if model_name.lower() == "resnet18":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.conv1 = nn.Conv2d(in_ch, 64, 3, 2, 3, bias=False)
    model.fc = nn.Linear(512, out_ch, bias=True)
  elif model_name.lower() == "resnet18ctrl":
    model = ResNet18CTRL(in_ch, **kwargs)
  elif model_name.lower() == "smallnet":
    model = SingleLayer(in_ch, **kwargs)
  elif model_name.lower() == "densenet121":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.features.conv0 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    model.classifier = nn.Linear(1024, out_ch, bias=True)
  elif model_name.lower() == "efficientnet_b2":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
    model.classifier[1] = nn.Linear(1408, out_ch, bias=True)
  elif model_name.lower() == "efficientnet_b3":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.features[0][0] = nn.Conv2d(in_ch, 40, 3, 2, 1, bias=False)
    model.classifier[1] = nn.Linear(1536, out_ch, bias=True)
  else:
    raise NotImplementedError(model_name)
  return model


def load_model(experiment_path: str,
               return_config: bool = False) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Dict[str, Any]]]:
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  model = create_model(**config["model"])

  state = torch.load(
    os.path.join(experiment_path, f"best_state.pt"), map_location=torch.device("cpu"), weights_only=True
  )
  model.load_state_dict(state)
  if return_config:
    return model, config
  return model


if __name__ == "__main__":
  x = torch.randn(1, 1, 28, 28)
  model = create_model("smallnet")
  model.eval()
  with torch.inference_mode():
    out = model(x)
  print(f"Out: {out.shape}")
