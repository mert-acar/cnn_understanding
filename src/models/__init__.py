import os
import torch
import torch.nn as nn
from yaml import full_load
from torchvision import models
from typing import Union, Dict, Any, Tuple

<<<<<<< HEAD
from .resnet import *
from .cnn import ClassifyingCNN, FeatureCNN
=======
from .resnet import ResNet18CTRL, ClusteringResNet18, SelfExpressiveCNN
from .cnn import ClassifyingCNN, ClusteringCNN, CustomCNN
>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33

HOOK_TARGETS = {
  "resnet18": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)],
  "resnet18ctrl": ["conv1"] + [f"layer{i}.{j}" for i in range(1, 5) for j in range(2)] + ["reshape_norm"],
<<<<<<< HEAD
  "featurecnn": ["layers"],
  "classificationcnn": ["feature_extractor", "classification_head "],
=======
  "customcnn": ["layers"],
  "clustercnn": ["feature_extractor", "cluster_head"],
  "classificationcnn": ["feature_extractor", "classification_head "],
  "efficientnetb2": [f"features.{i}" for i in range(1, 8)],
  "efficientnetb3": [f"features.{i}" for i in range(1, 8)]
>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33
}


def create_model(
  model_name: str,
  weights: str = "DEFAULT",
  in_ch: int = 1,
<<<<<<< HEAD
  num_classes: int = 10,
=======
  out_ch: int = 10,
>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33
  **kwargs
) -> torch.nn.Module:
  if model_name.lower() == "resnet18":
    model = models.get_model(model_name, weights=weights)
    if in_ch == 1:
      model.conv1 = nn.Conv2d(in_ch, 64, 7, 2, 3, bias=False)
    model.fc = nn.Linear(512, num_classes, bias=True)
  elif model_name.lower() == "resnet18l1":
    model = ResNet18L1()
  elif model_name.lower() == "resnet18l2":
    model = ResNet18L2()
  elif model_name.lower() == "resnet18l3":
    model = ResNet18L3()
  elif model_name.lower() == "resnet18l4":
    model = ResNet18L4()
  elif model_name.lower() == "resnet18ctrl":
    model = ResNet18CTRL(in_ch, weights)
<<<<<<< HEAD
  elif model_name.lower() == "featurecnn":
    model = FeatureCNN(in_ch=in_ch, **kwargs)
=======
  elif model_name.lower() == "selfexpressivecnn":
    model = SelfExpressiveCNN(in_ch, weights, **kwargs)
  elif model_name.lower() == "cluster_resnet18":
    model = ClusteringResNet18(in_ch, weights)
  elif model_name.lower() == "customcnn":
    model = CustomCNN(in_ch=in_ch, **kwargs)
  elif model_name.lower() == "clustercnn":
    model = ClusteringCNN(in_ch=in_ch, **kwargs)
>>>>>>> 3ac7afde059c0a006656fbf50103bbddb5b1aa33
  elif model_name.lower() == "classificationcnn":
    model = ClassifyingCNN(in_ch=in_ch, num_classes=num_classes, **kwargs)
  else:
    raise NotImplementedError(model_name)
  return model


def load_model(
  experiment_path: str,
  return_config: bool = False
) -> Union[torch.nn.Module, Tuple[torch.nn.Module, Dict[str, Any]]]:
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)
  model = create_model(**config["model"])
  state = torch.load(
    os.path.join(experiment_path, f"best_state.pt"),
    map_location=torch.device("cpu"),
    weights_only=True
  )
  model.load_state_dict(state)
  if return_config:
    return model, config
  return model
