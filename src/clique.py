import numpy as np
from typing import List, Tuple


class CLIQUEClustering:
  def __init__(self, xi: int, tau: int):
    self.xi = xi  # number of intervals in each dimension
    self.tau = tau  # density threshold

  def fit(self, X: np.ndarray) -> List[List[Tuple[int, ...]]]:
    self.X = X
    self.n_samples, self.n_features = X.shape

    # Step 1: Partition the data space
    self.intervals = [
      np.linspace(X[:, i].min(), X[:, i].max(), self.xi + 1) for i in range(self.n_features)
    ]

    # Step 2: Identify dense units
    self.dense_units = self._find_dense_units()

    # Step 3: Generate clusters
    return self._generate_clusters()

  def _find_dense_units(self) -> List[Tuple[int, ...]]:
    dense_units = []
    for unit in self._generate_units():
      if self._is_dense(unit):
        dense_units.append(unit)
    return dense_units

  def _generate_units(self):
    return np.ndindex(tuple([self.xi] * self.n_features))

  def _is_dense(self, unit: Tuple[int, ...]) -> bool:
    count = 0
    for point in self.X:
      if self._point_in_unit(point, unit):
        count += 1
      if count >= self.tau:
        return True
    return False

  def _point_in_unit(self, point: np.ndarray, unit: Tuple[int, ...]) -> bool:
    for i, u in enumerate(unit):
      if not (self.intervals[i][u] <= point[i] < self.intervals[i][u + 1]):
        return False
    return True

  def _generate_clusters(self) -> List[List[Tuple[int, ...]]]:
    clusters = []
    visited = set()

    for unit in self.dense_units:
      if unit not in visited:
        cluster = self._expand_cluster(unit, visited)
        clusters.append(cluster)

    return clusters

  def _expand_cluster(self, unit: Tuple[int, ...], visited: set) -> List[Tuple[int, ...]]:
    cluster = [unit]
    visited.add(unit)

    neighbors = self._get_neighbors(unit)
    for neighbor in neighbors:
      if neighbor not in visited and neighbor in self.dense_units:
        cluster.extend(self._expand_cluster(neighbor, visited))

    return cluster

  def _get_neighbors(self, unit: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    neighbors = []
    for i in range(self.n_features):
      for offset in [-1, 1]:
        neighbor = list(unit)
        neighbor[i] += offset
        if 0 <= neighbor[i] < self.xi:
          neighbors.append(tuple(neighbor))
    return neighbors
