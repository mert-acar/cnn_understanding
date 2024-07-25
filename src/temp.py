import torch
from utils import create_model, create_dataloader
import matplotlib.pyplot as plt

device = torch.device("mps")
model = create_model("resnet18").to(device)
state = torch.load("../logs/resnet18_run1/checkpoint.pt", map_location=device)
model.load_state_dict(state)
model.eval()

dataloader = create_dataloader(split="test", num_workers=0)
data, target = next(iter(dataloader))
data = data.to(device)

output = model.conv1(data).squeeze().detach().cpu().numpy()
_, axs = plt.subplots(8, 8)
for i, ax in enumerate(axs.ravel()):
  ax.imshow(output[i], cmap='gray')
  ax.axis(False)
plt.show()
