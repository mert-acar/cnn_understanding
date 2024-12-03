import torch
from tqdm import tqdm
from scipy.io import savemat
import torch.nn.functional as F
from typing import Union, Callable
from collections import defaultdict
from torch.utils.data import DataLoader

from dataset import create_dataloader
from model import load_library_model, HOOK_TARGETS

hooked_activations = defaultdict(list)


def get_patches(
  activations: torch.Tensor, window_size: int, stride: int, padding: int = 0
) -> torch.Tensor:
  if padding != 0:
    activations = F.pad(activations, [padding] * 4)
  return F.unfold(activations, kernel_size=(window_size, window_size), stride=stride)


def forward_pass(model: torch.nn.Module, dataloader: DataLoader):
  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  print(f"[INFO] Running on {device}")
  model = model.to(device)
  model.eval()

  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      _ = model(data)


def hook_gen(key: str, layer: Union[None, torch.nn.Module] = None) -> Callable:
  if layer is None:

    def hook_fn(model, input, output):
      hooked_activations[key].append(output.detach().cpu())

    return hook_fn
  else:
    if isinstance(layer, torch.nn.Conv2d):
      k, s, p = int(layer.kernel_size[0]), int(layer.stride[0]), int(layer.padding[0])

      def hook_fn(model, input, output):
        inp_patches = get_patches(input[0], k, s, p).transpose(1, 2)
        out_patches = output.reshape(output.shape[0], output.shape[1], -1).transpose(1, 2)
        hooked_activations[key + "_input"].append(inp_patches.detach().cpu())
        hooked_activations[key + "_output"].append(out_patches.detach().cpu())

      return hook_fn
    else:
      raise NotImplementedError(f"Hook function for {layer} is not implemented")


def hook_layers(model: torch.nn.Module, targets: list[str]):
  layer_list = dict([*model.named_modules()])
  for target in targets:
    if target not in layer_list:
      print(f"[INFO] Layer {target} does not exits, skipping...")
      continue
    target_layer = layer_list[target]
    target_layer.register_forward_hook(hook_gen(target, None))
    print(f"[INFO] Hooking {target}: {target_layer}")


if __name__ == "__main__":
  import os

  model_name = "densenet121"
  weights = "MNIST"
  dataloader = create_dataloader(weights.lower(), "test")
  model = load_library_model(model_name, weights, checkpoint_number=1)

  hook_targets = HOOK_TARGETS[model_name]
  experiment_path = f"../logs/{model_name}_{weights}/"
  out_path = os.path.join(experiment_path, "activations")

  hook_layers(model, hook_targets)
  forward_pass(model, dataloader)
  for key in hooked_activations:
    hooked_activations[key] = torch.cat(hooked_activations[key], 0)
    print(key, "→", hooked_activations[key].shape)
  savemat(out_path, hooked_activations)
