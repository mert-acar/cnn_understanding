import os
import torch
import torch.nn as nn
from yaml import full_load
from torchvision import models
from typing import Union, Dict, Any, Tuple

from .resnet import ResNet18CTRL
from .cnn import ClusteringCNN, CustomCNN

HOOK_TARGETS = {
  "densenet121": ["features.conv0"] + [f"features.denseblock{i}" for i in range(1, 5)],
  "resnet18": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)],
  "customcnn": ["layers"],
  "clustercnn": ["feature_extractor", "cluster_head"],
  "resnet18ctrl": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)] + ["reshape_norm"],
  "efficientnetb2": [f"features.{i}" for i in range(1, 8)],
  "efficientnetb3": [f"features.{i}" for i in range(1, 8)]
}


def create_model(
  model_name: str, weights: str = "DEFAULT", in_ch: int = 1, out_ch: int = 10, **kwargs
) -> torch.nn.Module:
  if model_name.lower() == "resnet18":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.conv1 = nn.Conv2d(in_ch, 64, 3, 2, 3, bias=False)
    model.fc = nn.Linear(512, out_ch, bias=True)
  elif model_name.lower() == "resnet18ctrl":
    model = ResNet18CTRL(in_ch, weights)
  elif model_name.lower() == "customcnn":
    model = CustomCNN(in_ch=in_ch, **kwargs)
  elif model_name.lower() == "clustercnn":
    model = ClusteringCNN(in_ch=in_ch, **kwargs)
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
