#!/home/macar/miniconda3/envs/cnn/bin/python3

import os
import pandas as pd
from yaml import full_load
from collections import defaultdict

exp_sum = "ExperimentSummary.yaml"
scores = defaultdict(list)

for root, dirs, files in os.walk("logs/"):
  if exp_sum in files:
    with open(os.path.join(root, exp_sum), "r") as f:
      config = full_load(f)
      scores["Experiment"].append(root.split("/")[-1])
      if "best_accuracy" in config:
        scores["accuracy"].append(config["best_accuracy"])
      else:
        scores["accuracy"].append("-")

      if "best_completeness_score" in config:
        scores["completeness"].append(config["best_completeness_score"])
      else:
        scores["completeness"].append("-")

      if "best_homogeneity_score" in config:
        scores["homogeneity"].append(config["best_homogeneity_score"])
      else:
        scores["homogeneity"].append("-")

df = pd.DataFrame(scores)
print(df)
