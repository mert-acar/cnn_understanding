import numpy as np
from sklearn.manifold import TSNE
from scipy.spatial import Delaunay
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from sklearn.cluster import DBSCAN


def projection_based_clustering(data, eps=0.5, min_samples=5):
  # Step 1: t-SNE projection
  tsne = TSNE(n_components=2, random_state=42)
  projected_data = tsne.fit_transform(data)

  # Step 2: Calculate Delaunay graph
  tri = Delaunay(projected_data)

  # Step 3: Weight edges with high-dimensional distances
  edges = set()
  for simplex in tri.simplices:
    for i in range(3):
      for j in range(i + 1, 3):
        edges.add(tuple(sorted([simplex[i], simplex[j]])))

  weights = []
  for edge in edges:
    weight = np.linalg.norm(data[edge[0]] - data[edge[1]])
    weights.append(weight)

  # Step 4: Compute shortest paths using Dijkstra's algorithm
  n = len(data)
  graph = csr_matrix((weights, zip(*edges)), shape=(n, n))
  distances, _ = dijkstra(graph, return_predecessors=True)

  # Step 5: Clustering process
  clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
  labels = clustering.fit_predict(distances)

  return labels, projected_data


if __name__ == "__main__":
  # Generate some random high-dimensional data
  np.random.seed(42)
  data = np.random.randn(100, 10)

  # Perform projection-based clustering
  labels, projected_data = projection_based_clustering(data)

  print(f"Number of clusters: {len(set(labels)) - (1 if -1 in labels else 0)}")
