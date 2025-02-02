import torch
import torch.nn as nn
import torch.nn.functional as F


class SAM(nn.Module):
  def __init__(self):
    super(SAM, self).__init__()
    self.conv = nn.Conv2d(
      in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, dilation=1, bias=False
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    max = torch.max(x, 1)[0].unsqueeze(1)
    avg = torch.mean(x, 1).unsqueeze(1)
    concat = torch.cat((max, avg), dim=1)
    return F.sigmoid(self.conv(concat)) * x


class CAM(nn.Module):
  def __init__(self, channels: int, r: int = 4):
    super(CAM, self).__init__()
    self.linear = nn.Sequential(
      nn.Linear(in_features=channels, out_features=channels // r, bias=True), nn.ReLU(inplace=True),
      nn.Linear(in_features=channels // r, out_features=channels, bias=True)
    )

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    max = F.adaptive_max_pool2d(x, output_size=1)
    avg = F.adaptive_avg_pool2d(x, output_size=1)
    b, c, _, _ = x.size()
    linear_max = self.linear(max.view(b, c)).view(b, c, 1, 1)
    linear_avg = self.linear(avg.view(b, c)).view(b, c, 1, 1)
    return F.sigmoid(linear_max + linear_avg) * x


class CBAM(nn.Module):
  def __init__(self, channels: int, r: int = 4):
    super(CBAM, self).__init__()
    self.sam = SAM()
    self.cam = CAM(channels, r)

  def forward(self, x):
    return self.sam(self.cam(x))
