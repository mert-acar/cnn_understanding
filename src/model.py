import os
import torch
import torch.nn as nn
from yaml import full_load
from torchvision import models
import torch.nn.functional as F

HOOK_TARGETS = {
  "densenet121": ["features.conv0"] + [f"features.denseblock{i}" for i in range(1, 5)],
  "resnet18": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)],
  "resnet18ctrl": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)],
  "efficientnetb2": [f"features.{i}" for i in range(1, 8)],
  "efficientnetb3": [f"features.{i}" for i in range(1, 8)]
}


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

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    out = self.maxpool(F.relu(self.bn1(self.conv1(x))))
    out = self.layer1(out)
    out = self.layer2(out)
    out = self.layer3(out)
    out = self.layer4(out)
    out = self.avgpool(out)
    out = out.view(out.size(0), -1)
    return F.normalize(out)


def create_model(
  model_name: str, weights: str = "DEFAULT", in_ch: int = 1, out_ch: int = 10, **kwargs
) -> torch.nn.Module:
  if model_name.lower() == "resnet18":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.conv1 = nn.Conv2d(in_ch, 64, 3, 2, 3, bias=False)
    model.fc = nn.Linear(512, out_ch, bias=True)
  elif model_name.lower() == "resnet18ctrl":
    model = ResNet18CTRL(in_ch)
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


def load_model(experiment_path: str) -> torch.nn.Module:
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  model = create_model(**config["model"])

  state = torch.load(
    os.path.join(experiment_path, f"best_state.pt"), map_location=torch.device("cpu"), weights_only=True
  )
  model.load_state_dict(state)
  return model


if __name__ == "__main__":
  x = torch.randn(1, 1, 28, 28)
  model = create_model("resnet18ctrl")
  model.eval()
  with torch.inference_mode():
    out = model(x)
  print(f"Out: {out.shape}")
