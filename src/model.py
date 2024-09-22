import torch
import torch.nn as nn


class ConvNet(nn.Module):
  def __init__(self):
    super(ConvNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

      nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

      nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

      nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
      nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
      nn.ReLU(inplace=True),
    )
    self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    self.flat = nn.Flatten()
    self.classifier = nn.Linear(512, 10, bias=True)

  def forward(self, x):
    x = self.features(x)
    x = self.flat(self.avgpool(x))
    return self.classifier(x)


if __name__ == "__main__":
  device = torch.device("cuda")
  model = ConvNet().to(device)
  inp = torch.randn(1, 1, 28, 28).to(device)
  out = model(inp)
  print(out.shape)
