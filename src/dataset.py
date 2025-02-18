import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms.v2 as transforms


def get_labels(dataset: str, split: str = "test") -> np.ndarray:
  assert dataset.lower() in [
    "mnist", "cifar10"
  ], f"dataset name must be one of ['mnist', 'cifar10'], got {dataset}"
  assert split in ["train", "test"], f"split must be one of ['train', 'test'], got {split}"
  return np.load(f"../data/{dataset.lower()}/{split}_labels.npy")


def get_transforms(dataset: str, split: str = "train") -> torch.nn.Module:
  dataset = dataset.lower()
  transform_list = [
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
  ]
  if split == "test":
    return transforms.Compose(transform_list)

  if dataset == "mnist":
    augmentations = [
      transforms.RandomChoice(
        [
          transforms.RandomAffine((-90, 90)),
          transforms.RandomAffine(0, translate=(0.2, 0.4)),
          transforms.RandomAffine(0, scale=(0.8, 1.1)),
          transforms.RandomAffine(0, shear=(-20, 20))
        ]
      ),
    ]
  elif dataset == "cifar10":
    augmentations = [
      transforms.RandomCrop(32, padding=8),
      transforms.RandomHorizontalFlip(),
      transforms.RandomGrayscale(p=0.2)
    ]

  return transforms.Compose(transform_list + augmentations)


def get_dataloader(
  dataset: str, split: str, batch_size: int = 1, num_workers: int = 1, **kwargs
) -> DataLoader:
  dataset = dataset.lower()
  isTrain = split == "train"
  transform = get_transforms(dataset, split)
  if dataset == "mnist":
    dataset_cls = datasets.MNIST(
      "../data/mnist/", train=isTrain, transform=transform, download=True
    )
  if dataset == "cifar10":
    dataset_cls = datasets.CIFAR10(
      "../data/cifar10/", train=isTrain, transform=transform, download=True
    )
  return DataLoader(
    dataset_cls, batch_size=batch_size, num_workers=num_workers, shuffle=isTrain, drop_last=True
  )
