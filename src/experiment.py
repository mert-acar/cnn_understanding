import os
import pickle as p
import pandas as pd
import numpy as np
from yaml import full_load
from scipy.io import loadmat
from utils import load_labels
import matplotlib.pyplot as plt
from collections import defaultdict

if __name__ == "__main__":
  exp_dir = "../logs/customnet_run2/"
  epoch = 33

  labels = load_labels()

  with open(os.path.join(exp_dir, "clusters", f"patches_epoch_{epoch}_v2.p"), "rb") as f:
    clusters = p.load(f)

  in_data = defaultdict(list)
  out_data = defaultdict(list)
  for i in range(0, 9, 2):
    for arr, s in zip([in_data, out_data], ["input", "output"]):
      arr["var"].append(f"features.{i}")
      arr["homogeneity"].append(clusters[f"features.{i}_" + s]["scores"]["homogeneity"])
      arr["completeness"].append(clusters[f"features.{i}_" + s]["scores"]["completeness"])
      arr["dist_thresh"].append(clusters[f"features.{i}_" + s]["params"]["distance_threshold"])
      # arr["d"].append(clusters[f"features.{i}_" + s]["params"]["mean_l2_dist"])
      # arr["k"].append(
      #   clusters[f"features.{i}_" + s]["params"]["distance_threshold"] /
      #   clusters[f"features.{i}_" + s]["params"]["mean_l2_dist"]
      # )
      arr["num_clusters"].append(clusters[f"features.{i}_" + s]["scores"]["num_clusters"])
      arr["CHI"].append(clusters[f"features.{i}_" + s]["scores"]["calinski_harabasz_score"])
      arr["silhouette"].append(clusters[f"features.{i}_" + s]["scores"]["silhouette"])

  print(pd.DataFrame(in_data))
  print(pd.DataFrame(out_data))
