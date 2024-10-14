import os
import torch
from tqdm import tqdm
from model import load_model
from scipy.io import savemat
import torch.nn.functional as F
from utils import create_dataloader


def calculate_out_size(input_shape, kernel_size, stride, padding=0):
  h, w = map(lambda x: ((x - kernel_size + 2 * padding) // stride) + 1, input_shape)
  return h, w


def extract_patches(activations, window_size, stride, padding=0):
  if padding != 0:
    activations = F.pad(activations, [padding] * 4)
  return F.unfold(activations, kernel_size=(window_size, window_size), stride=stride)


# data = loadmat("../data/test_data.mat")
# idx = 100
# x = torch.from_numpy(data["images"][idx:idx + 1]).to(device)
# y = data["labels"][0, idx]

if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/"
  out_path = os.path.join(experiment_path, "new_act")
  os.makedirs(out_path, exist_ok=True)

  device = torch.device("cpu")
  epoch = 33

  model = load_model(experiment_path, epoch).to(device)
  model.eval()
  dataloader = create_dataloader("../data/", batch_size=4, num_workers=4, split="test")


  for idx in [0, 2, 4, 6, 8]:
    prev = model.features[0:idx]
    module = model.features[idx]
    data = {"input": [], "output": [], "labels": []}
    for x, y in tqdm(dataloader):
      with torch.inference_mode():
        x = x.to(device)
        if prev:
          x = prev(x)
        k, s, p = module.kernel_size[0], module.stride[0], module.padding[0]
        inp_patches = extract_patches(x, k, s, p).transpose(1, 2)
        data["input"].append(inp_patches)
        x = module(x)
        out_patches = x.reshape(x.shape[0], x.shape[1], -1).transpose(1, 2)
        data["output"].append(out_patches)
        data["labels"].append(y)

    for key in data:
      data[key] = torch.cat(data[key], 0)
      print(data[key].shape)

    savemat(os.path.join(out_path, f"features.{idx}_epoch_{epoch}.mat"), data)
