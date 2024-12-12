import os
import torch
import pickle as p
import numpy as np
from tqdm import tqdm
import matplotlib as mpl
from model import HOOK_TARGETS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
from utils import load_MNIST_labels
from sklearn.preprocessing import normalize

if __name__ == "__main__":
  # with open("../data/MNIST/test_data.p", "rb") as f:
  #   x = p.load(f)

  model_name = "resnet18"
  weights = "MNIST"
  layers = HOOK_TARGETS[model_name]
  experiment_path = f"../logs/{model_name}_{weights}_CIL/"
  with open(os.path.join(experiment_path, "activations", f'{layers[-1].replace(".", "_")}_act.p'), "rb") as f:
    x = p.load(f)

  x = x.reshape(x.shape[0], -1)
  x = PCA(n_components=2, whiten=True).fit_transform(x)  # SHAPE [10000, 2]
  print(x.shape)
  # x = normalize(x, norm='l2', axis=1)

  labels = load_MNIST_labels()

  colors = np.array(mpl.color_sequences['tab10'])
  plt.figure(figsize=(10,8))
  plt.scatter(x[:, 0], x[:, 1], c=colors[labels], alpha=0.3)
  for label in np.unique(labels):
    plt.scatter([], [], color=colors[label], label=str(label))

  plt.grid(True)
  plt.legend(title="Classes")
  plt.xlabel("PC1")
  plt.ylabel("PC2")
  plt.title("Cluster Induced Activations")
  plt.show()
