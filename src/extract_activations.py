import torch
from tqdm import tqdm
from typing import Callable
from scipy.io import savemat
import torch.nn.functional as F
from collections import defaultdict
from torch.utils.data import DataLoader

from model import load_model
from train import create_dataloader


def get_patches(activations: torch.Tensor, window_size: int, stride: int, padding: int = 0) -> torch.Tensor:
  if padding != 0:
    activations = F.pad(activations, [padding] * 4)
  return F.unfold(activations, kernel_size=(window_size, window_size), stride=stride)


def forward_pass(model: torch.nn.Module, dataloader: DataLoader):
  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      _ = model(data)


def hook_gen(key: str, layer=None) -> Callable:
  if layer is None:

    def hook_fn_act(_, input, output):
      if key in hooked_activations:
        hooked_activations[key] = torch.cat((hooked_activations[key], output.detach().cpu()), 0)
      else:
        hooked_activations[key] = output.detach().cpu()

    return hook_fn_act
  else:
    if isinstance(layer, torch.nn.Conv2d):
      k, s, p = int(layer.kernel_size[0]), int(layer.stride[0]), int(layer.padding[0])

      def hook_fn_patch(_, input, output):
        inp_patches = get_patches(input[0], k, s, p).transpose(1, 2)
        out_patches = output.reshape(output.shape[0], output.shape[1], -1).transpose(1, 2)

        if key + "_input" in hooked_activations:
          hooked_activations[key + "_input"] = torch.cat(
            (hooked_activations[key + "_input"], inp_patches.detach().cpu()), 0
          )
        else:
          hooked_activations[key + "_input"] = inp_patches.detach().cpu()

        if key + "_output" in hooked_activations:
          hooked_activations[key + "_output"] = torch.cat(
            (hooked_activations[key + "_output"], out_patches.detach().cpu()), 0
          )
        else:
          hooked_activations[key + "_output"] = out_patches.detach().cpu()

      return hook_fn_patch
    else:
      raise NotImplementedError(f"Hook function for {layer} is not implemented")


if __name__ == "__main__":
  import os
  from yaml import full_load
  from collections import defaultdict

  experiment_path = "../logs/customnet_run2/"
  hook_targets = [f"features.{i}" for i in range(0, 9, 2)]
  patches = True
  ckpts = 33

  hooked_activations = defaultdict(torch.Tensor)

  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[INFO] Running on {device}")

  dataloader = create_dataloader(split="test", **config)
  model = load_model(experiment_path, ckpts).to(device)
  model.eval()

  layer_list = dict([*model.named_modules()])
  for target in hook_targets:
    if target not in layer_list:
      print(f"[INFO] Layer {target} does not exits, skipping...")
      continue
    target_layer = layer_list[target]
    if patches:
      target_layer.register_forward_hook(hook_gen(target, target_layer))
    else:
      target_layer.register_forward_hook(hook_gen(target, None))
    print(f"[INFO] Hooking {target}: {target_layer}")

  forward_pass(model, dataloader)

  out_path = os.path.join(config["output_path"], "activations")
  os.makedirs(out_path, exist_ok=True)

  # for key in hooked_activations:
  #   hooked_activations[key] = torch.cat(hooked_activations[key], 0)

  savemat(
    os.path.join(out_path, f"{'patches' if patches else 'act'}_epoch_{ckpts}.mat"),
    hooked_activations
  )
