import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def compute_cluster_characteristics(activations, cluster_labels):
  characteristics = {}
  for label in set(cluster_labels):
    if label != -1:  # Ignore noise points
      cluster_points = activations[cluster_labels == label]
      characteristics[label] = {
        'size': len(cluster_points),
        'mean': np.mean(cluster_points, axis=0),
        'std': np.std(cluster_points, axis=0),
        'max': np.max(cluster_points, axis=0),
        'min': np.min(cluster_points, axis=0)
      }
  return characteristics


def compare_characteristics_across_layers(all_characteristics):
  characteristics_to_compare = ['size', 'mean', 'std', 'max', 'min']
  results = {}

  for char in characteristics_to_compare:
    if char == 'size':
      data = [[c['size'] for c in layer_chars.values()]
              for layer_chars in all_characteristics.values()]
    else:
      data = [[np.mean(c[char]) for c in layer_chars.values()]
              for layer_chars in all_characteristics.values()]

    h_statistic, p_value = stats.kruskal(*data)
    results[char] = {'h_statistic': h_statistic, 'p_value': p_value}

  return results


def plot_results(results):
  characteristics = list(results.keys())
  p_values = [results[char]['p_value'] for char in characteristics]

  plt.figure(figsize=(10, 6))
  plt.bar(characteristics, p_values)
  plt.axhline(y=0.05, color='r', linestyle='--')
  plt.ylabel('p-value')
  plt.title('Kruskal-Wallis Test Results for Cluster Characteristics Across Layers')
  plt.yscale('log')
  plt.show()
