import numpy as np
from tqdm import tqdm
from numpy.linalg import svd
from scipy.spatial.distance import cdist


def transform(facet_planes, plane_centers, points):
  num_points = points.shape[0]
  num_facets = facet_planes.shape[0]

  chosen_facets = np.zeros(num_points, dtype=int)
  all_residues = np.zeros((num_points, num_facets), dtype=float)

  for num_facet in range(num_facets):
    current_facet = facet_planes[num_facet, :, :]
    current_center = plane_centers[num_facet, :]
    current_points = points - current_center
    all_residues[:, num_facet] = get_residue(current_points, current_facet)

  chosen_facets = np.argmin(all_residues, axis=1)

  alphas = np.matmul(
    (points - plane_centers[chosen_facets])[:, np.newaxis, :],
    np.transpose(facet_planes[chosen_facets], (0, 2, 1))
  )

  transformed_points = plane_centers[chosen_facets] + np.squeeze(
    np.matmul(alphas, facet_planes[chosen_facets]), axis=1
  )

  return transformed_points, chosen_facets


def get_residue(points, plane):
  points_projected = np.matmul(points, plane.T)
  residue = np.linalg.norm(points - np.matmul(points_projected, plane), axis=1)
  return residue


def get_facet_planes(embeddings, facet_dim=3, residue_threshold=0.01):
  num_data = embeddings.shape[0]
  dist = cdist(embeddings, embeddings)
  topk_index = np.argsort(dist, axis=1)

  used_in_facet = np.ones(num_data) * -1
  residues = np.zeros(num_data)
  outlier_points = []
  facet_planes = []
  plane_centers = []

  facet_num = 0
  for i in tqdm(range(num_data)):
    if (used_in_facet[i] != -1):
      continue

    examples_to_select = []
    for j in range(num_data):
      if (used_in_facet[topk_index[i, j]] != -1):
        continue
      else:
        indices = examples_to_select + [j]
        current_points = embeddings[topk_index[i, indices], :]

      origin = current_points.mean(axis=0)
      current_points_centered = current_points - origin
      try:
        _, _, vh = svd(current_points_centered, full_matrices=False)
      except np.linalg.LinAlgError:
        try:
          _, _, vh = svd(
            current_points_centered + 1e-5 * np.random.randn(*current_points_centered.shape),
            full_matrices=False
          )
        except np.linalg.LinAlgError:
          print("unfixable")
          continue

      vh = vh[:facet_dim, :]
      orig_residue = get_residue(current_points_centered[0, :][np.newaxis, :], vh)
      if (orig_residue > residue_threshold):
        continue
      else:
        examples_to_select = examples_to_select + [j]

    if (len(examples_to_select) >= facet_dim):
      current_points = embeddings[topk_index[i, examples_to_select], :]
      origin = current_points[0, :]
      plane_centers.append(origin)
      current_points_centered = current_points - origin
      _, _, vh = svd(current_points_centered, full_matrices=False)
      vh = vh[:facet_dim, :]

      point_plane = np.zeros((facet_dim, embeddings.shape[1]))
      point_plane[:, :] = vh
      facet_planes.append(point_plane)

      used_in_facet[topk_index[i, examples_to_select]] = facet_num
      facet_num += 1

      temp1 = get_residue(current_points_centered, vh)
      residues[topk_index[i, examples_to_select]] = temp1
    else:
      outlier_points.append(embeddings[i, :])

  facet_planes = np.stack(facet_planes)
  plane_centers = np.stack(plane_centers)
  return facet_planes, plane_centers


def get_facet_proj(embeddings, facet_dim=3, residue_threshold=0.01):
  facet_planes, plane_centers = get_facet_planes(embeddings, facet_dim, residue_threshold)
  transformed, chosen_facets = transform(facet_planes, plane_centers, embeddings)
  return facet_planes, plane_centers, transformed, chosen_facets
