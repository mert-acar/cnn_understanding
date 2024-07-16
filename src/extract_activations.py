import os
import torch
import pickle as p
from tqdm import tqdm
from yaml import full_load
from utils import create_dataloader, create_model


def hook_fn(model, input, output):
  activations.append(output.detach())


if __name__ == "__main__":
  activations = []
  experiment_path = "../logs/resnet18_run_1"
  out_path = "../data/class_activations_relu.pkl"

  with open("config.yaml", "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")

  dataloader = create_dataloader(split="test", batch_size=1)

  model = create_model(**config).to(device)
  weights = torch.load(os.path.join(experiment_path, "checkpoint.pt"), map_location=device)
  model.load_state_dict(weights)
  model.eval()

  # Hook target
  model.relu.register_forward_hook(hook_fn)
  # model.conv1.register_forward_hook(hook_fn)

  class_activations = {i: [] for i in range(10)}
  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      _ = model(data)
      class_activations[target.item()].append(activations[0].detach().cpu().numpy())
      activations.clear()

  with open(out_path, "wb") as f:
    p.dump(class_activations, f, protocol=p.HIGHEST_PROTOCOL)
  print(f"Activations are saved to {out_path}")
