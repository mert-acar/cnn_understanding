import torch.nn as nn
from torchvision import models
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize


def create_dataloader(data_root, batch_size, num_workers, split="train", **kwargs):
  transform = Compose([ToTensor(), Normalize((0.1307, ), (0.3081, ))])

  return DataLoader(
    MNIST(root=data_root, download=True, train=split == 'train', transform=transform),
    batch_size=batch_size,
    num_workers=num_workers,
    shuffle=split == 'train'
  )


def create_model(model_name, model_weights, **kwargs):
  model = getattr(models, model_name)(weights=model_weights)
  model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  model.fc = nn.Linear(512, 10, bias=True)
  return model
