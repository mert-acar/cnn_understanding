import os
import torch
from time import time
from tqdm import tqdm
from shutil import copyfile
from yaml import full_load, dump

from loss import CompositeLoss
from models import create_model
from dataset import get_dataloader
from utils import get_device, create_dir
from visualize import plot_performance_curves
from metrics import MetricCalculator


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
  scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **config["scheduler_args"])

  criterion = CompositeLoss(config["criterion_args"])

  tick = time()
  best_epoch = -1
  phases = ["test", "train"]

  # Initialize metrics tracking
  metrics = {}
  metric_names = config["metrics"].copy()
  for metric_name in metric_names:
    metrics[metric_name] = {phase: [] for phase in phases}

  metric_calculator = MetricCalculator(metric_names)
  metrics["loss"] = {phase: [] for phase in phases}
  best_metrics = {metric: float('inf') if metric == "loss" else 0 for metric in metrics.keys()}
  for epoch in range(config["num_epochs"]):
    print("-" * 20)
    print(f"Epoch {epoch + 1} / {config['num_epochs']}")
    for phase in phases:
      if phase == "train":
        model.train()
      else:
        model.eval()
      running_metrics = {metric: 0.0 for metric in metrics.keys()}

      with torch.set_grad_enabled(phase == "train"):
        for data, target in tqdm(dataloaders[phase], total=len(dataloaders[phase]), ncols=94):
          data, target = data.to(device), target.to(device)
          optimizer.zero_grad()
          output = model(data)

          # Calculate loss
          loss = criterion(output, target)
          running_metrics["loss"] += loss.item()

          if phase == "train":
            loss.backward()
            optimizer.step()

          # Calculate all metrics at once
          batch_metrics = metric_calculator.calculate_metrics(output, target)
          for metric_name, value in batch_metrics.items():
            running_metrics[metric_name] += value

      # Average metrics over epoch
      num_batches = len(dataloaders[phase])
      for metric_name, value in running_metrics.items():
        running_metrics[metric_name] = value / num_batches
        print(f"{metric_name}: {running_metrics[metric_name]:.3f}", end=" | ")
        metrics[metric_name][phase].append(running_metrics[metric_name])
      print()

      if phase == "test":
        scheduler.step(running_metrics["loss"])
        # Update best metrics and save model if loss improved
        if running_metrics["loss"] < best_metrics["loss"]:
          for metric_name, value in running_metrics.items():
            best_metrics[metric_name] = value
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

  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)
  config.update({"best_" + k: v for k, v in best_metrics.items()})
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
