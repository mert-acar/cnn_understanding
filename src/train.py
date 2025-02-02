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
  phases = ["test", "train"]

  metric_list = ["loss"]
  if config["measure_accuracy"]:
    metric_list.append("accuracy")
  metrics = {metric.lower(): {phase: [] for phase in phases} for metric in metric_list}
  best_metrics = {metric.lower(): 0 for metric in metric_list}
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

          if config["measure_accuracy"]:
            pred = torch.nn.functional.log_softmax(output, dim=1)
            acc = pred.argmax(1).eq(target).sum().item() / output.shape[0]
            running_metrics["accuracy"] += acc

      for key, score in running_metrics.items():
        score /= len(dataloaders[phase])
        print(f"{key}: {score:.3f}", end=" | ")
        metrics[key][phase].append(score)
      print()

      if phase == "test":
        scheduler.step(running_metrics["loss"])
        if running_metrics["loss"] < best_metrics["loss"]:
          for key, score in running_metrics.items():
            best_metrics[key] = score
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
  config.update(best_metrics)
  with open(os.path.join(config["output_path"], "ExperimentSummary.yaml"), "w") as f:
    dump(config, f)


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
