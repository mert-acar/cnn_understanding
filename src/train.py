import os
import torch
from time import time
from tqdm import tqdm
from yaml import full_load
import matplotlib.pyplot as plt
import torch.nn.functional as F
from shutil import rmtree, copyfile

from model import create_model
from dataset import create_dataloader


def group_lasso_penalty(model: torch.nn.Module):
  penalty = 0
  for name, param in model.named_parameters():
    if 'features' in name:
      # if 'conv' in name and 'weight' in name:
      penalty += torch.norm(torch.norm(param.view(param.shape[0], -1), p=2, dim=1), p=1)
  return penalty


def main(config_path: str):
  with open(config_path, "r") as f:
    config = full_load(f)

  # Create the checkpoint output path
  if os.path.exists(config["output_path"]):
    c = input(
      f"Output path {config['output_path']} is not empty! Do you want to delete the folder [y / n]: "
    )
    if "y" == c.lower():
      rmtree(config["output_path"], ignore_errors=True)
    else:
      print("Exit!")
      raise SystemExit
  os.makedirs(os.path.join(config["output_path"], "checkpoints"))
  copyfile("config.yaml", os.path.join(config["output_path"], "ExperimentSummary.yaml"))

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  print(f"[INFO] Running on {device}")

  dataloaders = {
    "train": create_dataloader(split="train", **config),
    "test": create_dataloader(split="test", **config),
  }

  model = create_model(**config["model"]).to(device)
  optimizer = torch.optim.Adam(
    model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=config["scheduler_factor"], patience=config["scheduler_patience"]
  )
  criterion = torch.nn.CrossEntropyLoss()

  group_lasso_coef = config["group_lasso_coef"]

  tick = time()
  best_epoch = -1
  best_error = 999999
  metrics = {'Loss': {'train': [], 'test': []}, 'Accuracy': {'train': [], 'test': []}}
  for epoch in range(config["num_epochs"]):
    print("-" * 20)
    print(f"Epoch {epoch + 1} / {config['num_epochs']}")
    for phase in ["train", "test"]:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_error = 0
      running_accuracy = 0
      pbar = tqdm(dataloaders[phase], total=len(dataloaders[phase]), ncols=94)
      with torch.set_grad_enabled(phase == "train"):
        for data, target in pbar:
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)
          pred = F.log_softmax(output, dim=1)
          acc = pred.argmax(1).eq(target).sum().item() / data.shape[0]

          loss = criterion(output, target)
          if phase == "train":
            if group_lasso_coef is not None:
              loss += group_lasso_coef * group_lasso_penalty(model)
            loss.backward()
            optimizer.step()

          running_error += loss.item()
          running_accuracy += acc
          pbar.set_description(f"{loss.item():.5f} | {acc * 100:.3f}%")

      running_error = running_error / len(dataloaders[phase])
      running_accuracy = running_accuracy / len(dataloaders[phase])
      print(f"Loss: {running_error:.5f} | Accuracy: {running_accuracy * 100:.3f}%")
      metrics["Loss"][phase].append(running_error)
      metrics["Accuracy"][phase].append(running_accuracy)
      if phase == "test":
        scheduler.step(running_error)
        if running_error < best_error:
          best_error = running_error
          best_epoch = epoch

    ckpt_path = os.path.join(config["output_path"], "checkpoints", f"checkpoint_{epoch + 1}.pt")
    print(f"+ Saving the model to {ckpt_path}...")
    torch.save(model.state_dict(), ckpt_path)

    # If no validation improvement has been recorded for "early_stop" number of epochs
    # stop the training.
    if epoch - best_epoch >= config["early_stop"]:
      print(f"No improvements in {config['early_stop']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")

  fig, axs = plt.subplots(1, len(metrics), tight_layout=True, figsize=(10, 5))
  epochs = list(range(1, epoch + 2))
  for i, (metric, arr) in enumerate(metrics.items()):
    for phase, val in arr.items():
      axs[i].plot(epochs, val, label=phase)
    axs[i].set_xlabel("Epochs")
    axs[i].set_ylabel(metric)
    axs[i].legend()
    axs[i].grid(True)
  fig.suptitle("Model Performance Across Epochs")
  plt.savefig(os.path.join(config["output_path"], "performance_curves.png"), bbox_inches="tight")


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
