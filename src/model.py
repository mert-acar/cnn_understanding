import torch
import torch.nn as nn


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.layers = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=2, padding=2, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, bias=False),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, bias=False),
      nn.ReLU(inplace=True),
      nn.AdaptiveAvgPool2d(output_size=(1, 1)),
      nn.Flatten(),
      nn.Linear(in_features=512, out_features=256, bias=True),
      nn.Linear(in_features=256, out_features=10, bias=True),
    )

  def forward(self, x):
    return self.layers(x)


if __name__ == "__main__":
  device = torch.device("mps")
  model = ConvNet().to(device)
  inp = torch.randn(1, 1, 28, 28).to(device)
  out = model(inp)
  print(out.shape)
