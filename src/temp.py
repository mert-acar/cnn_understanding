import os
import pickle as p
import numpy as np
import matplotlib as mpl
from model import HOOK_TARGETS
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from utils import load_MNIST_labels

if __name__ == "__main__":
  labels = load_MNIST_labels()

  model_name = "efficientnetb3"
  weights = "MNIST"
  layers = HOOK_TARGETS[model_name]

  titles = [
    "Raw Data | {layer} | {ev:.2f}%", "Vanilla | {layer} | {ev:.2f}%", "CI | {layer} | {ev:.2f}%"
  ]
  colors = np.array(mpl.color_sequences['tab10'])

  for layer in layers:
    experiments = [
      "../data/MNIST/test_data.p",
      os.path.join(
        f"../logs/{model_name}_{weights}/", "activations", f'{layer.replace(".", "_")}_act.p'
      ),
      os.path.join(
        f"../logs/{model_name}_{weights}_CIL/", "activations", f'{layer.replace(".", "_")}_act.p'
      ),
    ]
    fig = plt.figure(figsize=(18, 7), tight_layout=True)
    for i, exp_path in enumerate(experiments):
      with open(exp_path, "rb") as f:
        x = p.load(f)

      x = x.reshape(x.shape[0], -1)
      pca = PCA(n_components=3, whiten=True)  # SHAPE [10000, 2]
      x = pca.fit_transform(x)

      ax = fig.add_subplot(101 + (10 * len(experiments)) + i, projection='3d')
      ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=colors[labels], alpha=0.3)
      for label in np.unique(labels):
        ax.scatter([], [], [], color=colors[label], label=str(label))

      ax.grid(True)
      ax.legend(title="Classes")
      ax.set_xlabel("PC1")
      ax.set_ylabel("PC2")
      ax.set_zlabel("PC3")
      ax.set_title(titles[i].format(layer=layer, ev=sum(pca.explained_variance_ratio_)))

    # plt.tight_layout()
    plt.show()
    plt.clf()
    plt.close("all")
