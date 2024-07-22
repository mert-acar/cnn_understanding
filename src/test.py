import os
import torch
from tqdm import tqdm
from yaml import full_load
import torch.nn.functional as F
from utils import create_dataloader, create_model


def test(experiment_path, config_path="config.yaml"):
  with open(config_path, "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")

  dataloader = create_dataloader(split="test", **config)

  model = create_model(**config).to(device)
  weights = torch.load(os.path.join(experiment_path, "checkpoint.pt"), map_location=device)
  model.load_state_dict(weights)
  model.eval()

  running_accuracy = 0
  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = F.log_softmax(output, dim=1)
      acc = pred.argmax(1).eq(target).sum().item() / data.shape[0]
      running_accuracy += acc
  running_accuracy = running_accuracy / len(dataloader)
  print(f"Accuracy: {running_accuracy * 100:.3f}%")


if __name__ == "__main__":
  from fire import Fire
  Fire(test)
