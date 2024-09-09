import os
import torch
from tqdm import tqdm
from yaml import full_load
import torch.nn.functional as F
from utils import create_dataloader, create_model


def main(experiment_path, checkpoint_num=1):
  with open(os.path.join(experiment_path, "ExperimentSummary.yaml"), "r") as f:
    config = full_load(f)

  device = torch.device("cuda" if torch.cuda.is_available() else "mps")

  dataloader = create_dataloader(split="test", **config)

  model = create_model(**config).to(device)
  weights = torch.load(
    os.path.join(experiment_path, f"checkpoint_{checkpoint_num}.pt"), map_location=device
  )
  model.load_state_dict(weights)
  model.eval()
  criterion = torch.nn.CrossEntropyLoss()
  results = test(model, dataloader, criterion, device)
  print(f"Accuracy: {results['accuracy'] * 100:.3f}% | Loss: {results['loss']:.3f}")


def test(model, dataloader, criterion, device):
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
