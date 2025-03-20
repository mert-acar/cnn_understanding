import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from metrics import clustering_accuracy
from sklearn import cluster as C
from sklearn.neighbors import NearestNeighbors

from utils import lr_schedule, normalize
from loss import MaximalCodingRateReduction


def cluster(activations: np.ndarray, n_clusters: int = 10) -> np.ndarray:
  return C.AgglomerativeClustering(n_clusters=n_clusters).fit_predict(activations)


class MCRTraining:
  def __init__(
    self,
    n_iter: int,
    start_lr: float = 0.01,
    factor: float = 0.8,
    gam1: float = 1.0,
    gam2: float = 1.0,
    eps: float = 0.1
  ):
    self.n_iter = n_iter
    self.lr_schedule = lr_schedule(start_lr, n_iter, factor)
    self.scores = {"R": [], "Rc": [], "dR": []}
    self.mcr = MaximalCodingRateReduction(gam1, gam2, eps)

  def step(self, z_l: torch.Tensor, pi: torch.Tensor, lr: float) -> torch.Tensor:
    w = z_l.T.detach().requires_grad_(True)
    disc = self.mcr.compute_discrimn_loss_empirical(w)
    self.scores["R"].append(disc.item())
    comp = self.mcr.compute_compress_loss_empirical(w, pi)
    self.scores["Rc"].append(comp.item())
    mcr = disc - comp
    self.scores["dR"].append(mcr.item())

    mcr.backward()
    grad = w.grad.T
    if grad is None:
      raise ValueError(f"There is no grad on the tensor")

    z_l1 = w.T + (lr * grad)
    z_l1 = F.normalize(z_l1, p=2, dim=1)
    return z_l1.detach()

  def train(self, activations: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    pi = self.mcr.label_to_membership(labels).requires_grad_(False)
    z_l = F.normalize(activations, p=2, dim=1)
    for i in tqdm(range(self.n_iter)):
      z_l = self.step(z_l, pi, self.lr_schedule[i])
    return cluster(z_l.detach().cpu().numpy())


class Projector(torch.nn.Module):
  def __init__(self, input_dim: int, output_dim: int = 128):
    super().__init__()
    self.proj = torch.nn.Parameter(torch.randn(input_dim, output_dim), requires_grad=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x @ self.proj


class CHITraining:
  def __init__(self, n_iter: int, lr: float = 0.01):
    self.n_iter = n_iter
    self.lr = lr
    self.scores = {"BCSS": [], "WCSS": [], "CHI": [], "GRAD_NORM": []}

  def calculate_chi(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    n, _ = z.shape
    k = len(torch.unique(y))
    one_hot = F.one_hot(y, num_classes=k).float()

    n_j = one_hot.sum(dim=0) + 1e-8
    mu_j = torch.mm(z.T, one_hot) / n_j.unsqueeze(0)
    mu = z.mean(dim=0)

    # Between-cluster variance (BCSS)
    diff = mu_j - mu.unsqueeze(1)
    b = torch.sum(n_j * torch.sum(diff**2, dim=0))
    self.scores["BCSS"].append(b.item())

    # Within-cluster variance (WCSS)
    sample_mu = mu_j.T[y]
    w = torch.sum((z - sample_mu)**2)
    self.scores["WCSS"].append(w.item())

    # calinski-harabasz index
    chi = (b / w) * (n - k) / (k - 1 + 1e-8)
    self.scores["CHI"].append(chi.item())
    return chi

  def train2(self, activations: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    z_l = activations.detach()
    p = Projector(z_l.shape[-1]).to(activations.device)
    optimizer = torch.optim.AdamW(p.parameters(), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for _ in tqdm(range(self.n_iter)):
      optimizer.zero_grad()
      z_proj = p(z_l)
      chi = self.calculate_chi(z_proj, labels)
      if torch.isnan(chi):
        print("NaN chi! Break...")
        break
      (-chi).backward()
      self.scores["GRAD_NORM"].append(torch.norm(p.proj.grad).item())
      optimizer.step()
      scheduler.step()
    return cluster(z_proj.detach().cpu().numpy())

  def train(self, activations: torch.Tensor, labels: torch.Tensor) -> np.ndarray:
    z_l = activations.requires_grad_(True)
    # p = Projector(z_l.shape[-1]).to(activations.device)
    optimizer = torch.optim.AdamW([z_l], lr=self.lr)
    # optimizer = torch.optim.AdamW([z_l] + list(p.parameters()), lr=self.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    for _ in range(self.n_iter):
      # for _ in tqdm(range(self.n_iter)):
      optimizer.zero_grad()
      chi = self.calculate_chi(z_l, labels)
      # chi = self.calculate_chi(p(z_l), labels)
      if torch.isnan(chi):
        print("NaN chi! Break...")
        break
      (-chi).backward()
      self.scores["GRAD_NORM"].append(torch.norm(z_l.grad).item())
      # self.scores["GRAD_NORM"].append(torch.norm(p.proj.grad).item())
      optimizer.step()
      scheduler.step()
    return cluster(z_l.detach().cpu().numpy())
    # return cluster(p(z_l).detach().cpu().numpy())


def compute_intrinsic_dim(data):
  nbrs = NearestNeighbors(n_neighbors=5).fit(data)
  distances, _ = nbrs.kneighbors(data)
  return np.mean(distances[:, -1] / (distances[:, 0] + 1e-8))


if __name__ == "__main__":
  from pprint import pprint
  from dataset import get_labels
  from visualize import plot_scores
  from utils import select_random_samples, get_device

  dataset = "cifar10"
  num_sample_per_label = 100
  n_iter = 100
  start_lr = 0.0001
  factor = 0.8

  device = torch.device('cpu')
  # device = get_device()

  labels = get_labels(dataset)
  idx = select_random_samples(labels, num_sample_per_label)
  labels = labels[idx]
  labels = torch.from_numpy(labels).to(device)
  for layer_name in ["layer1", "layer2", "layer3", "layer4"]:
    print(f"Layer Name: {layer_name}")
    print(f"----")
    activations = np.load(f"../data/{dataset}/{layer_name}_test_act.npy")
    activations = activations[idx]

    activations = activations.reshape(labels.shape[0], -1)
    activations = normalize(activations)

    acc = clustering_accuracy(labels, cluster(activations))
    print(f"Accuracy (Before CHI Optimization): {100 * acc:.3f}%")

    activations = torch.from_numpy(activations).to(device)

    trainer = CHITraining(n_iter, start_lr)
    # trainer = MCRTraining(n_iter, start_lr, factor, eps=0.5)

    out = trainer.train(activations, labels)
    acc = clustering_accuracy(labels.detach().cpu().numpy(), out)
    print(f"Accuracy (After CHI Optimization): {100 * acc:.3f}%")
    pprint({key: {"step 0": value[0], f"step {n_iter}": value[-1]} for key, value in trainer.scores.items()})
    print()
  # plot_scores(trainer.scores)
