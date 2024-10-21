import re
import os
import math
import numpy as np
from glob import glob
from typing import List
from kneed import KneeLocator


def normalize_cols(x: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
  norm = np.linalg.norm(x, axis=axis, keepdims=True)
  return x / (norm + eps)


# Function to extract epoch number from the key
def extract_epoch(key: str) -> int:
  match = re.search(r'epoch_(\d+)', key)
  return int(match.group(1)) if match else 0


def get_filenames(experiment_path: str, reverse: bool = True, ext: str = "mat") -> List[str]:
  return sorted(glob(os.path.join(experiment_path, f"*.{ext}")), key=extract_epoch, reverse=reverse)


def find_non_zero_idx(data: np.ndarray, beta: float = 0.95) -> np.ndarray:
  _, C, H, W = data.shape
  stat = np.reshape(np.abs(data).mean(0), (C, H * W)).mean(-1)
  y = np.sort(stat)[::-1]
  x = list(range(len(y)))
  kn = KneeLocator(x, y, curve='convex', direction='decreasing').knee
  non_zero_idx = stat >= y[kn] * beta
  return non_zero_idx


def closest_factors(n):
  root = int(math.isqrt(n))  # integer square root
  for i in range(root, 0, -1):
    if n % i == 0:  # If i is a factor of n
      return (i, n // i)  # Return the pair of factors
