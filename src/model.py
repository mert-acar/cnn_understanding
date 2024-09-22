import torch.nn as nn

def conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0):
  return [
    nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
    nn.ReLU(inplace=True)
  ]

class ConvNet(nn.Module):
  def __init__(self, config):
    super(ConvNet, self).__init__()
    layers = []
    for c in config["features"]:
      if c[0] == "C":
        layers = layers + conv2d(*c[1:])
      elif c[0] == "M":
        layers.append(nn.MaxPool2d(*c[1:]))
      elif c[0] == "A":
        layers.append(nn.AvgPool2d(*c[1:]))
      else:
        raise NotImplemented

    self.features = nn.Sequential(*layers)
    self.pool = None
    if "pool" in config:
      if config["pool"][0] == "A":
        self.pool = nn.AdaptiveAvgPool2d(config["pool"][1])
      elif config["pool"][0] == "M":
        self.pool = nn.AdaptiveMaxPool2d(config["pool"][1])
      else:
        raise NotImplemented
    self.flat = nn.Flatten()
    self.classifier = nn.Linear(*config["classifier"])

  def forward(self, x):
    x = self.features(x)
    if self.pool is not None:
      x = self.pool(x)
    x = self.flat(x)
    return self.classifier(x)


if __name__ == "__main__":
  import torch
  from yaml import full_load
  with open("config.yaml", "r") as f:
    config = full_load(f)
  device = torch.device("cuda")
  model = ConvNet(config["model_config"]).to(device)
  print(model)
  # inp = torch.randn(1, 1, 28, 28).to(device)
  # out = model.features(inp)
