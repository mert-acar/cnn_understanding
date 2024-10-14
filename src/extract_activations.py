import os
import torch
from tqdm import tqdm
from yaml import full_load
from scipy.io import savemat
from torchvision import models
from collections import defaultdict
from utils import create_dataloader
from model import ConvNet

activations = defaultdict(list)


def hook_gen(key):
  def hook_fn(model, input, output):
    activations[key].append(output.detach().cpu())

  return hook_fn


def extract(model, dataloader, device):
  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  labels = []
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      _ = model(data)
      labels.extend(target.tolist())
  for key in activations:
    activations[key] = torch.cat(activations[key], 0).numpy()
  activations["labels"] = labels
  return activations


def main(experiment_path, checkpoint_num=3, *hook_targets):
  assert len(hook_targets) > 0, "Provide at least one layer name to hook into"

  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  print(f"[INFO] Running on {device}")

  dataloader = create_dataloader(split="test", **config)

  if "model_config" in config:
    model = ConvNet(config["model_config"]).to(device)
  else:
    model = models.resnet18(weights="DEFAULT")
    model.conv1 = torch.nn.Conv2d(1, 64, 7, 2, 3, bias=False)
    model.fc = torch.nn.Linear(512, 10, bias=True)
    model = model.to(device)

  state = torch.load(
    os.path.join(experiment_path, "checkpoints", f"checkpoint_{checkpoint_num}.pt"),
    map_location=device,
    weights_only=True
  )
  model.load_state_dict(state)
  model.eval()

  layer_list = dict([*model.named_modules()])
  for target in hook_targets:
    if target not in layer_list:
      print(f"[INFO] Layer {target} does not exits, skipping...")
      continue
    target_layer = layer_list[target]
    target_layer.register_forward_hook(hook_gen(target))
    print(f"[INFO] Hooking: {target_layer}")

  act = extract(model, dataloader, device)
  for key in act:
    if key == "labels":
      continue
    out_path = os.path.join(config["output_path"], "activations", key)
    if not os.path.exists(out_path):
      os.makedirs(out_path)
    savemat(
      os.path.join(out_path, f"act_epoch_{checkpoint_num}.mat"), {
        "activations": act[key],
        "labels": act["labels"]
      }
    )


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
