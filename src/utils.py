import re
import os
import torch
import numpy as np
import pandas as pd
from glob import glob
from kneed import KneeLocator
from torchvision import models
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def normalize(x, axis=1, eps=1e-12):
  norm = np.linalg.norm(x, axis=axis, keepdims=True)
  return x / (norm + eps)


# Function to extract epoch number from the key
def extract_epoch(key):
  match = re.search(r'epoch_(\d+)', key)
  return int(match.group(1)) if match else 0


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


def get_filenames(
  layer,
  experiment_path=os.path.join("..", "logs", "resnet18_run1", "activations"),
  reverse=True,
  ext="mat"
):
  return sorted(
    glob(os.path.join(experiment_path, layer, f"*.{ext}")), key=extract_epoch, reverse=reverse
  )


def cluster_matrix_to_df(matrices, titles):
  col = []
  for mat, title in zip(matrices, titles):
    b = np.full((13, 11), "").astype(object)
    b[1, 1] = title
    b[2, 0] = "label"
    b[2, 1:] = np.linspace(0, 9, 10, dtype=int)
    b[3:, 0] = np.linspace(0, 9, 10, dtype=int)
    b[3:, 1:] = mat
    col.append(b)
  col = np.vstack(col)
  return pd.DataFrame(col)


def select_random_samples(labels, num_samples_per_label):
  unique_labels = np.unique(labels)
  selected_indices = []
  for label in unique_labels:
    indices = np.where(labels == label)[0]
    if len(indices) < num_samples_per_label:
      raise ValueError(f"Not enough samples for label {label}. Only {len(indices)} available.")
    selected = np.random.choice(indices, num_samples_per_label, replace=False)
    selected_indices.extend(selected)
  return np.array(selected_indices)


def find_non_zero_idx(data, beta=0.95):
  _, C, H, W = data.shape
  stat = np.reshape(np.abs(data).mean(0), (C, H * W)).mean(-1)
  y = np.sort(stat)[::-1]
  x = list(range(len(y)))
  kn = KneeLocator(x, y, curve='convex', direction='decreasing').knee
  non_zero_idx = stat >= y[kn] * beta
  return non_zero_idx


if __name__ == "__main__":
  from openpyxl import Workbook
  from openpyxl.utils.dataframe import dataframe_to_rows

  layers = ["layer1.1.conv2", "layer2.1.conv1", "layer3.1.conv1", "layer4.1.conv2"]
  for layer in layers:
    filenames = get_filenames(layer)
    ss = [
      "_AgglomerativeClustering_manifold_0_cc.npy", "_AgglomerativeClustering_manifold_1_cc.npy",
      "_HDBSCAN_manifold_0_cc.npy", "_HDBSCAN_manifold_1_cc.npy"
    ]
    for s in ss:
      matrices = [np.load(os.path.splitext(fname)[0] + s) for fname in filenames[:5]]
      titles = [f"epoch {i}" for i in reversed(range(25, 35))]
      df = cluster_matrix_to_df(matrices, titles)

      wb = Workbook()
      ws = wb.active

      for r_idx, row in enumerate(dataframe_to_rows(df, index=False, header=False), 1):
        ws.append(row)

      for i in range(len(matrices)):
        start_row = 13 * i + 2
        ws.merge_cells(start_row=start_row, start_column=2, end_row=start_row, end_column=11)

      output_filename = f"{layer}{os.path.splitext(s)[0]}.xlsx"
      wb.save(output_filename)
