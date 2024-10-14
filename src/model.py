import os
import torch
import torch.nn as nn
from yaml import full_load


def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
  return [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding), nn.ReLU(inplace=True)]


class ConvNet(nn.Module):
  def __init__(self, config):
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

  def forward(self, x):
    x = self.features(x)
    if self.pool is not None:
      x = self.pool(x)
    x = self.flat(x)
    return self.classifier(x)


def load_model(experiment_path, checkpoint_number):
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)["model_config"]
  model = ConvNet(config)
  state = torch.load(
    os.path.join(experiment_path, "checkpoints", f"checkpoint_{checkpoint_number}.pt"),
    map_location=torch.device("cpu"),
    weights_only=True
  )
  model.load_state_dict(state)
  return model
