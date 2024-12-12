import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F


class MultinomialLogisticRegression(torch.nn.Module):
  def __init__(self, input_dim, num_classes):
    super().__init__()
    self.n_classes = num_classes
    self.f = torch.nn.Linear(input_dim, num_classes, bias=True)

  def forward(self, x):
    return F.softmax(self.f(x), dim=1)

  def predict(self, x):
    with torch.inference_mode():
      p = self.forward(x)
    return torch.argmax(p, dim=1)


def lr_loss(pred):
  k = pred.shape[-1]
  ri = torch.sqrt(torch.sum(pred, dim=0))
  num = pred / ri
  q = num / torch.sum(num, dim=0)
  loss = -1 * torch.mean(q * torch.log(pred)) / k
  return loss


def train_mlr(x, k, dev="mps", lr=0.5, n_epochs=200, patience=5):
  device = torch.device(dev)
  model = MultinomialLogisticRegression(x.shape[1], k).to(dev)
  model.train()
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  x = torch.from_numpy(x.astype(np.float32)).to(device)
  
  best_loss = float('inf')
  patience_counter = 0
  
  # for epoch in tqdm(range(n_epochs)):
  for epoch in range(n_epochs):
    optimizer.zero_grad()
    pred = model(x)
    loss = lr_loss(pred)
    print(f"epoch: {epoch} | loss: {loss.item() * 1e6:.4f}")
    loss.backward()
    optimizer.step()
    
    # Early stopping logic
    current_loss = loss.item()
    if current_loss < best_loss:
      best_loss = current_loss
      patience_counter = 0
    else:
      patience_counter += 1
      
    if patience_counter >= patience:
      print(f'Early stopping triggered at epoch {epoch}')
      break
      
  model.eval()
  return model.predict(x).cpu().numpy()
