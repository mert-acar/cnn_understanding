import os
import torch
from tqdm import tqdm
from yaml import full_load
import torch.nn.functional as F
from train import create_dataloader
from extract_activations import get_patches

if __name__ == "__main__":
  exp_dir = "../logs/customnet_lasso_run_0.01/"
  epoch = 50

  with open(os.path.join(exp_dir, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  model_feat_args = [x[3:] for x in config["model_config"]["features"]]

  device = torch.device("cpu")
  dataloader = create_dataloader(batch_size=1, split="test")
  state = torch.load(
    os.path.join(exp_dir, "checkpoints", f"checkpoint_{epoch}.pt"),
    map_location=device,
    weights_only=True
  )

  accuracy = 0
  with torch.inference_mode():
    for X, y in tqdm(dataloader):
      X, y = X.to(device), y.to(device)

      # Convolutions
      for i, (k, s, p) in enumerate(model_feat_args):
        W = state[f"features.{i * 2}.weight"]
        f = W.abs().reshape(W.shape[0], -1).mean(-1).sort(descending=True).values
        
        b = state[f"features.{i * 2}.bias"].unsqueeze(0).unsqueeze(1)

        C = W.shape[0]
        h = ((X.shape[-1] + (2 * p) - k) // s) + 1
        X = get_patches(X, k, s, p)

        W = W.reshape(C, -1)
        X = X.mT
        X = X @ W.T + b

        X = X.mT.reshape(X.shape[0], C, h, h)

        # ReLU
        X[X <= 0] = 0

      # AvgPool
      X = X.mean(-1).mean(-1)

      # Classifier
      output = X @ state["classifier.weight"].T + state["classifier.bias"]

      pred = F.log_softmax(output, dim=1)
      accuracy += pred.argmax(1).eq(y).sum().item() / output.shape[0]

  accuracy = accuracy / len(dataloader)
  print(f"Accuracy: {accuracy*100:.2f}%")
