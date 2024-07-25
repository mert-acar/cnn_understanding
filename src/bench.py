import os
import torch
from yaml import full_load
from visualize import vis2d
from utils import create_model, create_dataloader


def hook_fn(model, input, output):
  activations.append(output.detach().cpu())


if __name__ == "__main__":
  with open("config.yaml", "r") as f:
    config = full_load(f)

  activations = []
  device = torch.device("cuda" if torch.cuda.is_available() else "mps")
  model = create_model(**config).to(device)
  weights = torch.load(os.path.join("../logs/resnet18_run2", "checkpoint.pt"), map_location=device)
  model.load_state_dict(weights)
  model.eval()
  
  model.layer1[0].conv1.register_forward_hook(hook_fn)

  dataloader = create_dataloader(split="test")
  data, target = next(iter(dataloader))
  data, target = data.to(device), target.to(device)

  with torch.inference_mode():
    output = model(data)
    pred = torch.nn.functional.log_softmax(output, dim=1).argmax(1)
    print(f"Prediction: {pred.item()} | Target: {target.item()}")

  activations = activations.pop().squeeze()
  vis2d(activations)
