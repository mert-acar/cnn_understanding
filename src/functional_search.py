import os
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

if __name__ == "__main__":
  # exp_path = "../logs/resnet18_IMAGENET/"
  exp_path = "../logs/densenet121_IMAGENET/"
  # exp_path = "../logs/efficientnetb2_IMAGENET/"
  # exp_path = "../logs/customnet_IMAGENET/"
  # exp_path = "../logs/resnet18_MNIST/"
  # exp_path = "../logs/densenet121_MNIST/"
  # exp_path = "../logs/efficientnetb2_MNIST/"
  # exp_path = "../logs/customnet_MNIST/"

  out_path = os.path.join(exp_path, "figures")
  os.makedirs(out_path, exist_ok=True)

  data_path = "clusters/spectral_custom_metric_scores.csv"
  df = pd.read_csv(os.path.join(exp_path, data_path))

  vars = df["layer"].unique()
  clusters = df["n_clusters"].unique()
  colors = mpl.color_sequences['tab10']
  scores = ["silhouette", "homogeneity", "completeness", "bcss", "wcss", "chi"]
  mins, maxes = [], []
  for score in scores:
    mins.append(df[score].min())
    maxes.append(df[score].max())

  h, w = 2, 3
  best_idx = None
  best_vals = {}
  for var in vars:
    _, axs = plt.subplots(h, w, tight_layout=True)
    best_vals[var] = {}
    data = df[df["layer"] == var]
    for i in range(h):
      for j in range(w):
        flat_idx = (i * w) + j
        score = scores[flat_idx]
        # best_idx = np.argmax(data[score])
        best_idx = np.argmax(data["silhouette"])
        best_val = data[score].tolist()[best_idx]
        best_vals[var][score] = (best_val, clusters[best_idx])
        axs[i, j].plot(clusters, data[score].tolist(), color=colors[flat_idx], label=score)
        axs[i, j].plot([clusters[best_idx]], [best_val], "x", label='best')
        axs[i, j].set_xlabel("Number of Epochs")
        axs[i, j].set_xticks(clusters)
        axs[i, j].set_ylabel("Score")
        axs[i, j].set_ylim(mins[flat_idx] * 0.9, maxes[flat_idx] * 1.1)
        axs[i, j].grid()
        axs[i, j].legend()
        axs[i, j].set_title(score)
    plt.suptitle(var)
    # plt.show()
    plt.savefig(os.path.join(out_path, f"{var.replace('.','_')}.png"), bbox_inches="tight")
    plt.clf()
    plt.close("all")

  _, axs = plt.subplots(h, w, tight_layout=True)
  for i in range(h):
    for j in range(w):
      flat_idx = (i * w) + j
      score = scores[flat_idx]
      data = [v[score][0] for v in best_vals.values()]
      axs[i, j].plot(vars, data, color=colors[flat_idx])
      for k, (y, l) in enumerate(zip(data, [v[score][1] for v in best_vals.values()])):
        axs[i, j].text(k, y, f"N={l}", ha='center', va='bottom')
      axs[i, j].set_xlabel("Layers")
      axs[i, j].set_xticks(vars)
      axs[i, j].set_ylabel("Score")
      axs[i, j].set_ylim(mins[flat_idx] * 0.9, maxes[flat_idx] * 1.1)
      axs[i, j].grid()
      axs[i, j].set_title(score)
  plt.suptitle("Best Scores")
  # plt.show()
  plt.savefig(os.path.join(out_path, f"best.png"), bbox_inches="tight")
  plt.clf()
  plt.close("all")
