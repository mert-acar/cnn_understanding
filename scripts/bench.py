import pickle as p
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
  experiment_path = "../logs/customnet_run2/activations/{layer}/act_epoch_{epoch}.mat"
  layers = [f"features.{i}" for i in range(0, 9, 2)] + ["pool"]
  output_path = "./data/simple_flat_e{epoch}.p"
  epochs = [33, 17, 1]

  # _, axs = plt.subplots(1, 3, tight_layout=True, figsize=(15, 5))
  for epoch in epochs:
    with open(output_path.format(epoch=epoch), "rb") as f:
      cluster_data = p.load(f)
    l = []
    for layer in layers:
      X, y = cluster_data[layer]["data"]
      cluster_labels = cluster_data[layer]["cluster_labels"]
      l.append(len(set(cluster_labels)))
    plt.plot(layers, l, label=f"Epoch {epoch}")
  plt.grid(True)
  plt.legend()
  plt.xlabel("Layers")
  plt.title("Num Clusters")
  plt.savefig(f"./data/flat_n_clusters.png", bbox_inches='tight')
  plt.clf()
  plt.close()
