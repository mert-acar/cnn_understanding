import os
import torch
from tqdm import tqdm
from yaml import full_load
from scipy.io import savemat
import torch.nn.functional as F
from collections import defaultdict
from utils import create_dataloader, create_model


def hook_gen(key):
  def hook_fn(model, input, output):
    activations[key].append(output.detach())

  return hook_fn


def extract(experiment_path, *hook_targets):
  with open("config.yaml", "r") as f:
    config = full_load(f)

  assert len(hook_targets) > 0, "At least one hook target must be specified"

  out_path = f"../data/act_{os.path.basename(experiment_path)}.mat"

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")

  dataloader = create_dataloader(split="test", batch_size=1)

  model = create_model(**config).to(device)
  weights = torch.load(os.path.join(experiment_path, "checkpoint.pt"), map_location=device)
  model.load_state_dict(weights)
  model.eval()

  # Hook target
  layer_list = dict([*model.named_modules()])
  for target in hook_targets:
    if target not in layer_list:
      print(f"[INFO] Layer {target} does not exits, skipping...")
      continue
    target_layer = layer_list[target]
    target_layer.register_forward_hook(hook_gen(target))
    print(f"[INFO] Hooking: {target_layer}")

  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      acc = F.log_softmax(model(data), dim=1).argmax(1).eq(target).sum().item()
      pbar.set_description(f"Accuracy: {acc * 100:.2f}%")

  for key in activations:
    activations[key] = torch.cat(activations[key], 0).cpu().numpy()

  savemat(out_path, activations)
  print(f"Activations are saved to {out_path}")


if __name__ == "__main__":
  from fire import Fire

  activations = defaultdict(list)
  Fire(extract)
