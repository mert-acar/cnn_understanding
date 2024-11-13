import os
import torch
from tqdm import tqdm
from yaml import full_load
from model import load_model
import torch.nn.functional as F
from train import create_dataloader
from torch.utils.data import DataLoader


def main(experiment_path: str, checkpoint_num: int = 1):
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")

  dataloader = create_dataloader(split="test", **config)

  model = load_model(experiment_path, checkpoint_num).to(device)
  model.eval()
  criterion = torch.nn.CrossEntropyLoss()
  results = test(model, dataloader, criterion, device)
  print(f"Accuracy: {results['accuracy'] * 100:.3f}% | Loss: {results['loss']:.3f}")


def test(
  model: torch.nn.Module, dataloader: DataLoader, criterion: torch.nn.Module, device: torch.device
) -> dict[str, float]:
  running_accuracy = 0
  running_error = 0
  pbar = tqdm(dataloader, total=len(dataloader), ncols=94)
  with torch.inference_mode():
    for data, target in pbar:
      data, target = data.to(device), target.to(device)
      output = model(data)
      pred = F.log_softmax(output, dim=1)
      acc = pred.argmax(1).eq(target).sum().item() / data.shape[0]
      running_accuracy += acc
      loss = criterion(output, target)
      running_error += loss.item()
  running_accuracy = running_accuracy / len(dataloader)
  running_error = running_error / len(dataloader)
  return {"loss": running_error, "accuracy": running_accuracy}


if __name__ == "__main__":
  from fire import Fire
  Fire(main)
