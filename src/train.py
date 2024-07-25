import os
import torch
import torch.nn as nn
from time import time
from tqdm import tqdm
from shutil import rmtree
from yaml import full_load
from scipy.io import savemat
import torch.nn.functional as F
from collections import defaultdict
from torch.optim.lr_scheduler import StepLR

from utils import create_dataloader, create_model


def hook_gen(key):
  def hook_fn(model, input, output):
    if not model.training:
      activations[key].append(output.detach().cpu())

  return hook_fn


def group_lasso_penalty(model):
  penalty = 0
  for module in model.modules():
    if isinstance(module, nn.Conv2d):
      # Apply penalty only on convolutional layers
      weights = module.weight
      penalty += torch.sum(torch.sqrt(torch.sum(weights**2, dim=(1, 2, 3))))
  return penalty


if __name__ == "__main__":
  activations = defaultdict(list)
  hook_targets = ["conv1"]
  with open("config.yaml", "r") as f:
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
  os.mkdir(config["output_path"])

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  print(f"[INFO] Running on {device}")

  dataloaders = {
    "train": create_dataloader(split="train", **config),
    "test": create_dataloader(split="test", **config),
  }

  model = create_model(**config).to(device)
  layer_list = dict([*model.named_modules()])
  for target in hook_targets:
    if target not in layer_list:
      print(f"[INFO] Layer {target} does not exits, skipping...")
      continue
    target_layer = layer_list[target]
    target_layer.register_forward_hook(hook_gen(target))
    print(f"[INFO] Hooking: {target_layer}")

  optimizer = torch.optim.Adam(
    model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"]
  )
  criterion = torch.nn.CrossEntropyLoss()
  scheduler = StepLR(
    optimizer, step_size=config["scheduler_step_size"], gamma=config["scheduler_gamma"]
  )

  # Add group lasso coefficient to config or set a default value
  group_lasso_coef = config.get("group_lasso_coef", 0.01)

  tick = time()
  best_epoch = -1
  best_error = 999999
  labels = []
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
          loss = criterion(output, target)

          # Add group lasso penalty only during training
          if phase == "train":
            group_lasso = group_lasso_penalty(model)
            loss += group_lasso_coef * group_lasso

          pred = F.log_softmax(output, dim=1)
          acc = pred.argmax(1).eq(target).sum().item() / data.shape[0]

          running_error += loss.item()
          running_accuracy += acc
          pbar.set_description(f"{loss.item():.5f} | {acc * 100:.3f}%")
          if phase == "train":
            loss.backward()
            optimizer.step()
          else:
            labels.extend(target.tolist())

      running_error = running_error / len(dataloaders[phase])
      running_accuracy = running_accuracy / len(dataloaders[phase])
      print(f"Loss: {running_error:.5f} | Accuracy: {running_accuracy * 100:.3f}%")
      if phase == "test":
        scheduler.step()
        if running_error < best_error:
          best_error = running_error
          best_epoch = epoch
          ckpt_path = os.path.join(config["output_path"], "checkpoint.pt")
          print(f"+ Saving the model to {ckpt_path}...")
          torch.save(model.state_dict(), ckpt_path)

    for key in activations:
      activations[key] = torch.cat(activations[key], 0).numpy()
    activations["labels"] = labels
    savemat(os.path.join(config["output_path"], f"act_epoch_{epoch + 1}.mat"), activations)
    labels = []
    activations = defaultdict(list)

    # If no validation improvement has been recorded for "early_stop" number of epochs
    # stop the training.
    if epoch - best_epoch >= config["early_stop"]:
      print(f"No improvements in {config['early_stop']} epochs, stop!")
      break

  total_time = time() - tick
  m, s = divmod(total_time, 60)
  h, m = divmod(m, 60)
  print(f"Training took {int(h):d} hours {int(m):d} minutes {s:.2f} seconds.")

