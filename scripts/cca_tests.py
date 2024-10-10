import cca
import numpy as np
import seaborn as sns
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils import read_to_cluster

if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  layers = ["pool", "features.8", "features.6", "features.4", "features.2", "features.0"]
  scores = np.zeros((len(layers), len(layers)))
  epoch = 1
  for i, layer in enumerate(layers):
    p = experiment_path.format(layer=layer, epoch=epoch)
    X, _ = read_to_cluster(p, svd_dim=None, threshold=0.98, norm=False)
    X = X.T
    for j, layer2 in enumerate(layers):
      p2 = experiment_path.format(layer=layer2, epoch=33)
      Y, _ = read_to_cluster(p2, svd_dim=None, threshold=0.98, norm=False)
      Y = Y.T
      results = cca.get_cca_similarity(X, Y, epsilon=1e-10, verbose=False)
      s = np.mean(results["cca_coef1"])
      scores[i, j] = s
      print(f"{layer} [{epoch}] x {layer2} [33] | {X.shape} x {Y.shape} | {s:.3f}")

  plt.figure(figsize=(8, 8))
  sns.heatmap(scores, annot=True, fmt=".2f", cmap='viridis', xticklabels=layers, yticklabels=layers)
  plt.xlabel('Trained Layers')
  plt.ylabel('Untrained Layers')
  plt.show()
