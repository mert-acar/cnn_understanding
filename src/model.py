import os
import timm
import torch
import torch.nn as nn
from yaml import full_load
from torchvision import models

HOOK_TARGETS = {
  "customnet": [f"features.{i}" for i in range(0, 9, 2)],
  "densenet121": ["features.conv0"] + [f"features.denseblock{i}" for i in range(1, 5)],
  "resnet18": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)],
  "efficientnet_b2": [f"features.{i}" for i in range(1, 8)],
  "efficientnet_b3": [f"features.{i}" for i in range(1, 8)]
}


def load_library_model(model_name: str, weights: str = "", **kwargs: dict) -> nn.Module:
  if model_name == "customnet":
    return load_model(f"../logs/{model_name}_{weights.upper()}", kwargs.get("checkpoint_number", 33))
  if weights.lower() == "imagenet":
    return create_model(model_name, {"weights": "DEFAULT"})
  elif weights.lower() == "mnist":
    return load_model(
      f"../logs/{model_name}_{weights.upper()}/", kwargs.get("checkpoint_number", 1)
    )
  elif weights.lower() == "cifar10":
    if model_name == "resnet18":
      model = timm.create_model("resnet18", pretrained=False)
      model.conv1 = torch.nn.Conv2d(
        3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
      )
      model.maxpool = torch.nn.Identity()
      model.fc = torch.nn.Linear(512, 10)
      model.load_state_dict(
        torch.hub.load_state_dict_from_url(
          "https://huggingface.co/edadaltocg/resnet18_cifar10/resolve/main/pytorch_model.bin",
          map_location="cpu",
          file_name="resnet18_cifar10.pth",
        )
      )
      return model
    else:
      raise NotImplementedError(f"{model_name} / {weights}")
  else:
    raise NotImplementedError(f"{model_name} / {weights}")


def create_model(model_name: str, config: dict, in_ch: int = 1) -> nn.Module:
  if model_name.lower() == "convnet":
    model = ConvNet(config)
  elif model_name.lower() == "resnet18":
    model = models.get_model(model_name, **config)
    model.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, 10, bias=True)
  elif model_name.lower() == "densenet121":
    model = models.get_model(model_name, **config)
    model.features.conv0 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    model.classifier = nn.Linear(1024, 10, bias=True)
  elif model_name.lower() == "efficientnet_b2":
    model = models.get_model(model_name, **config)
    model.features[0][0] = nn.Conv2d(in_ch, 32, 3, 2, 1, bias=False)
    model.classifier[1] = nn.Linear(1408, 10, bias=True)
  elif model_name.lower() == "efficientnet_b3":
    model = models.get_model(model_name, **config)
    model.features[0][0] = nn.Conv2d(in_ch, 40, 3, 2, 1, bias=False)
    model.classifier[1] = nn.Linear(1536, 10, bias=True)
  else:
    raise NotImplementedError(model_name)
  return model


def conv2d(
  in_channels: int,
  out_channels: int,
  kernel_size: int,
  stride: int = 1,
  padding: int = 0
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
