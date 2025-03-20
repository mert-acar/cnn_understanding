import matplotlib.pyplot as plt

from utils import closest_factors

from typing import Dict, List, Optional


def plot_scores(scores: Dict[str, List[float]], out_path: Optional[str] = None):
  r, c = closest_factors(len(scores))
  fig, axs = plt.subplots(r, c, tight_layout=True, figsize=(5*c, 5*r), squeeze=False)
  t = list(range(1, len(scores[list(scores.keys())[0]]) + 1))
  for i, key in enumerate(scores):
    ax = axs[i // c, i % c]
    ax.plot(t, scores[key])
    ax.axis(True)
    ax.grid(True)
    ax.set_xlabel("Steps")
    ax.set_ylabel(key)
    ax.set_title(f"{key} vs Steps")
  if out_path is not None:
    plt.savefig(out_path)
  else:
    plt.show()
  plt.close(fig)
