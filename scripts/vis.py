import numpy as np
import pandas as pd
import plotly.graph_objects as go


def create_multilayer_sankey(cluster_labels_layers):
  num_layers = len(cluster_labels_layers)
  labels = []
  sources = []
  targets = []
  values = []
  current_label_idx = 0
  label_indices = []
  for layer_idx in range(num_layers):
    unique_clusters = np.unique(cluster_labels_layers[layer_idx])
    layer_labels = [f"L{layer_idx+1}_{cluster}" for cluster in unique_clusters]
    labels.extend(layer_labels)
    label_indices.append({
      cluster: current_label_idx + i
      for i, cluster in enumerate(unique_clusters)
    })
    current_label_idx += len(unique_clusters)

  for layer_idx in range(num_layers - 1):
    clusters_A = cluster_labels_layers[layer_idx]
    clusters_B = cluster_labels_layers[layer_idx + 1]
    contingency_table = pd.crosstab(clusters_A, clusters_B)
    label_idx_A = label_indices[layer_idx]
    label_idx_B = label_indices[layer_idx + 1]
    for cluster_A in contingency_table.index:
      for cluster_B in contingency_table.columns:
        count = contingency_table.loc[cluster_A, cluster_B]
        if count > 0:
          sources.append(label_idx_A[cluster_A])
          targets.append(label_idx_B[cluster_B])
          values.append(count)

  fig = go.Figure(
    go.Sankey(
      node=dict(pad=15, thickness=20, line=dict(color="black", width=0.5), label=labels),
      link=dict(source=sources, target=targets, value=values)
    )
  )
  fig.update_layout(title_text="Cluster Transitions Across Layers", font_size=10)
  fig.show()
