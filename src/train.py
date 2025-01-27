import os
import torch
from time import time
from tqdm import tqdm
from yaml import full_load
from shutil import copyfile

from loss import CompositeLoss
from model import create_model
from dataset import get_dataloader
from visualize import plot_performance_curves
from utils import get_metric_scores, get_device, create_dir


def group_lasso_penalty(model: torch.nn.Module) -> torch.Tensor:
  penalty = 0
  for name, param in model.named_parameters():
    if 'features' in name:
      penalty += torch.norm(torch.norm(param.view(param.shape[0], -1), p=2, dim=1), p=1)
  return penalty


def main(config_path: str):
  with open(config_path, "r") as f:
    config = full_load(f)

  # Create the checkpoint output path
  create_dir(config["output_path"])
  ckpt_path = os.path.join(config["output_path"], f"best_state.pt")
  copyfile("config.yaml", os.path.join(config["output_path"], "ExperimentSummary.yaml"))

  device = get_device()
  print(f"[INFO] Running on {device}")

  dataloaders = {
    "train": get_dataloader(split="train", **config),
    "test": get_dataloader(split="test", **config),
  }

  model = create_model(**config["model"]).to(device)
  optimizer = torch.optim.AdamW(
    model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
  )
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=config["scheduler_factor"], patience=config["scheduler_patience"]
  )

  criterion = CompositeLoss(config["criterion_args"])

  tick = time()
  best_epoch = -1
  best_error = 999999
  phases = ["test", "train"]

  metric_list = ["loss"]
  if config["metric_list"]:
    metric_list += config["metric_list"]
  metrics = {metric.lower(): {phase: [] for phase in phases} for metric in metric_list}
  for epoch in range(config["num_epochs"]):
    print("-" * 20)
    print(f"Epoch {epoch + 1} / {config['num_epochs']}")
    for phase in phases:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_metrics = {metric.lower(): 0 for metric in metric_list}
      with torch.set_grad_enabled(phase == "train"):
        for data, target in tqdm(dataloaders[phase], total=len(dataloaders[phase]), ncols=94):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)

          loss = criterion(output, target)
          running_metrics["loss"] += loss.item()

          if phase == "train":
            loss.backward()
            optimizer.step()

          metric_scores = get_metric_scores(metric_list[1:], output, target)
          for key, score in metric_scores.items():
            running_metrics[key.lower()] += score

      for key, score in running_metrics.items():
        score /= len(dataloaders[phase])
        print(f"{key}: {score:.3f}", end=" | ")
        metrics[key][phase].append(score)
      print()

      if phase == "test":
        scheduler.step(running_metrics["loss"])
        if running_metrics["loss"] < best_error:
          best_error = running_metrics["loss"]
          best_epoch = epoch
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
  plot_performance_curves(metrics, config["output_path"])


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
