import os
import torch
from tqdm import tqdm
from yaml import full_load
from scipy.io import savemat
import torch.nn.functional as F
from utils import create_dataloader, create_model

def hook_gen(key):
  def hook_fn(model, input, output):
    activations[key].append(output.detach().cpu())
  return hook_fn

def test(experiment_path, config_path="config.yaml"):
  with open(config_path, "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")

  dataloader = create_dataloader(split="test", **config)

  model = create_model(**config).to(device)
  weights = torch.load(os.path.join(experiment_path, "checkpoint.pt"), map_location=device)
  model.load_state_dict(weights)
  model.eval()
  layer_list = dict([*model.named_modules()])
  for target in hook_targets:
    if target not in layer_list:
      print(f"[INFO] Layer {target} does not exits, skipping...")
      continue
    target_layer = layer_list[target]
    target_layer.register_forward_hook(hook_gen(target))
    print(f"[INFO] Hooking: {target_layer}")

  running_accuracy = 0
  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  labels = []
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = F.log_softmax(output, dim=1)
      acc = pred.argmax(1).eq(target).sum().item() / data.shape[0]
      running_accuracy += acc
      labels.extend(target.tolist())

  running_accuracy = running_accuracy / len(dataloader)
  print(f"Accuracy: {running_accuracy * 100:.3f}%")
  for key in activations:
    activations[key] = torch.cat(activations[key], 0).numpy()
  activations["labels"] = labels
  savemat(os.path.join(experiment_path, "test_activations.mat"), activations)


if __name__ == "__main__":
  from fire import Fire
  from collections import defaultdict
  activations = defaultdict(list)
  hook_targets = ["conv1"]
  Fire(test)
