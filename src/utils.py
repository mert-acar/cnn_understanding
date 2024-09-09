import torch
import numpy as np
from torchvision import models
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from torchvision.transforms import Compose, ToTensor, Normalize


def create_dataloader(data_root="../data/", batch_size=1, num_workers=4, split="train", **kwargs):
  transform = Compose([ToTensor(), Normalize((0.1307, ), (0.3081, ))])
  return DataLoader(
    MNIST(root=data_root, download=True, train=split == 'train', transform=transform),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=split == 'train'
  )


def create_model(model_name, model_weights="DEFAULT", **kwargs):
  model = getattr(models, model_name)(weights=model_weights)
  model.conv1 = torch.nn.Conv2d(
    1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
  )
  model.fc = torch.nn.Linear(512, 10, bias=True)
  return model


def average_distance(data):
  n = data.shape[0]
  distances = np.linalg.norm(data[:, np.newaxis] - data, axis=2)
  total_distance = np.sum(np.triu(distances, k=1))
  num_pairs = (n * (n - 1)) // 2
  return total_distance / num_pairs


def analyze_data(data):
  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(data)
  avg_mean_distance = np.mean(np.mean(np.abs(scaled_data), axis=0))
  avg_max_distance = np.mean(np.max(np.abs(scaled_data), axis=0))
  return {"avg_mean_distance": avg_mean_distance, "avg_max_distance": avg_max_distance}
