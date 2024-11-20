import os
import torch
import torch.nn as nn
from yaml import full_load
from torchvision import models

from dataset import create_dataloader

# VGG_HOOK_IDX: [0, 2, 5, 7, 10, 12, 14, 17, 19, 21, 24, 26, 28]

def create_model(model_name: str, config: dict) -> nn.Module:
  if model_name.lower() == "convnet":
    model =  ConvNet(config)
  elif model_name.lower() == "resnet18":
    model = models.get_model(model_name, **config)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, 10, bias=True)
  elif model_name.lower() == "densenet121":
    model = models.get_model(model_name, **config)
    model.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, 10, bias=True)
  else:
    raise NotImplementedError(model_name)
  return model


def conv2d(
  in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0
) -> list[torch.nn.Module]:
  return [
    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    nn.ReLU(inplace=False)
  ]


class ConvNet(nn.Module):
  def __init__(self, config: dict):
    super(ConvNet, self).__init__()
    layers = []
    for c in config["features"]:
      if c[0] == "C":
        layers = layers + conv2d(*c[1:])
      elif c[0] == "M":
        layers.append(nn.MaxPool2d(*c[1:]))
      elif c[0] == "A":
        layers.append(nn.AvgPool2d(*c[1:]))
      else:
        raise NotImplemented

    self.features = nn.Sequential(*layers)
    self.pool = None
    if "pool" in config:
      if config["pool"][0] == "A":
        self.pool = nn.AdaptiveAvgPool2d(config["pool"][1])
      elif config["pool"][0] == "M":
        self.pool = nn.AdaptiveMaxPool2d(config["pool"][1])
      else:
        raise NotImplemented
    self.flat = nn.Flatten()
    self.classifier = nn.Linear(*config["classifier"])

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.features(x)
    if self.pool is not None:
      x = self.pool(x)
    x = self.flat(x)
    return self.classifier(x)


def load_model(experiment_path: str, checkpoint_number: int) -> torch.nn.Module:
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  model = create_model(**config["model"])
  
  state = torch.load(
    os.path.join(experiment_path, "checkpoints", f"checkpoint_{checkpoint_number}.pt"),
    map_location=torch.device("cpu"),
    weights_only=True
  )
  model.load_state_dict(state)
  return model
