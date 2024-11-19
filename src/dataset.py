import os
import json
import torch
import numpy as np
from glob import glob
from PIL import Image
from typing import Union
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms

IMAGENET_CLASS_IDX = [37, 55, 69, 256, 259, 275, 308, 525, 551, 577, 582, 620, 630, 647, 649, 676, 729, 776, 839, 914]

def create_dataloader(
  dataset: str,
  data_root: str,
  split: str,
  batch_size: int = 1,
  num_workers: int = 4,
  **kwargs
) -> DataLoader:
  if dataset.lower() == "mnist":
    return create_mnist_dataloader(data_root, batch_size, num_workers, split, **kwargs)
  elif dataset.lower() == "imagenet":
    return create_imagenet_dataloader(data_root, batch_size, num_workers, split, **kwargs)
  else:
    raise NotImplementedError(dataset)


def select_random_samples(
  labels: Union[list[int], np.ndarray], num_samples_per_label: int, seed: int = 9001
) -> np.ndarray:
  unique_labels = np.unique(labels)
  selected_indices = []
  np.random.seed(seed)
  for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) < num_samples_per_label:
      raise ValueError(f"Not enough samples for label {label}. Only {len(indices)} available.")
    selected = np.random.choice(indices, num_samples_per_label, replace=False)
    selected_indices.extend(selected)
  return np.array(selected_indices)


def create_mnist_dataloader(
  data_root: str = "../data/",
  batch_size: int = 1,
  num_workers: int = 4,
  split: str = "train",
  **kwargs
) -> DataLoader:
  transform = transforms.Compose([
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize((0.1307, ), (0.3081, ))
  ])
  return DataLoader(
    datasets.MNIST(root=data_root, download=True, train=split == 'train', transform=transform),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=split == 'train'
  )


def create_imagenet_dataloader(
  data_root: str = "../data/ImageNet",
  batch_size: int = 1,
  num_workers: int = 4,
  split: str = "val",
  **kwargs
) -> DataLoader:
  transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
  ])
  return DataLoader(
    ImageNet(data_root, transform=transform),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=split == 'train'
  )


class ImageNet(torch.utils.data.Dataset):
  def __init__(
    self,
    data_path: str = "../data/ImageNet/",
    transform=None,
    class_idx: list[int] = IMAGENET_CLASS_IDX,
    **kwargs
  ):
    with open(os.path.join(data_path, "idx_to_wind.json"), "r") as f:
      idx_to_wind = json.load(f)
    self.wind = [idx_to_wind[str(idx)] for idx in class_idx]

    data_paths, targets = [], []
    for i, w in enumerate(self.wind):
      paths = glob(os.path.join(data_path, "val", w, "*.JPEG"))
      data_paths.extend(paths)
      targets.extend([i] * len(paths))
    self.data_paths = data_paths
    self.targets = targets
    self.transform = transform

  def __len__(self):
    return len(self.data_paths)

  def __getitem__(self, idx: int) -> tuple[Image.Image, int]:
    path = self.data_paths[idx]
    img = Image.open(path)
    if self.transform:
      img = self.transform(img)
    label = self.targets[idx]
    return img, label
