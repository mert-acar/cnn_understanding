from tqdm import tqdm
from scipy.io import loadmat, savemat

if __name__ == "__main__":
  experiment_path = "../logs/resnet18_run1/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [
    "layer1.0", "layer1.1", "layer2.0", "layer2.1", "layer3.0", "layer3.1", "layer4.0", "layer4.1"
  ]

  for layer in tqdm(layers):
    for epoch in range(1, 35):
      fname = experiment_path.format(layer=layer, epoch=epoch)
      data = loadmat(fname)
      savemat(fname, {"activations": data[layer], "labels": data["labels"]})
