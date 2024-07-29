import os
import torch
from tqdm import tqdm
from yaml import full_load
from scipy.io import savemat
from collections import defaultdict
from utils import create_model, create_dataloader

activations = defaultdict(list)


def hook_gen(key):
  def hook_fn(model, input, output):
    activations[key].append(output.detach().cpu())

  return hook_fn


def extract(experiment_path, checkpoint_num=3, *hook_targets):
  assert len(hook_targets) > 0, "Provide at least one layer name to hook into"

  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  print(f"[INFO] Running on {device}")

  dataloader = create_dataloader(split="test", **config)

  model = create_model(**config).to(device)
  state = torch.load(
    os.path.join(experiment_path, f"checkpoint_{checkpoint_num}.pt"), map_location=device
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

  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  labels = []
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      _ = model(data)
      labels.extend(target.tolist())

  out_path = os.path.join(config["output_path"], f"act_epoch_{checkpoint_num}.mat")
  for key in activations:
    activations[key] = torch.cat(activations[key], 0).numpy()
  activations["labels"] = labels
  savemat(out_path, activations)
  print(f"Activations are saved to: {out_path}")


if __name__ == "__main__":
  from fire import Fire
  Fire(extract)
