import numpy as np
from tqdm import tqdm
from numpy.linalg import svd
from scipy.spatial.distance import cdist

def similarity(x, y, y_facet, na=4, nb=0.5):
  diff = y - x
  basis_matrix = y_facet[0]
  BTB = np.dot(basis_matrix, basis_matrix.T)
  BTB_inv = np.linalg.pinv(BTB)
  projection_matrix = np.dot(basis_matrix.T, np.dot(BTB_inv, basis_matrix))
  subspace_component = np.linalg.norm(np.dot(projection_matrix, diff))
  normal_component = np.linalg.norm(diff - subspace_component)
  alpha_sim = 1 / (1 + ((normal_component / 2) ** na))
  beta_sim = 1 / (1 + (subspace_component ** nb))
  return alpha_sim * beta_sim

def pw_similarity(x, x_facet, y, y_facet, na=4, nb=0.5):
  return (similarity(x, y, y_facet, na, nb) + similarity(y, x, x_facet, na, nb)) / 2

def find_facets(planes, points):
  residues = np.zeros((len(points), len(planes)))
  for i in tqdm(range(len(planes)), desc="Assigning points to facets"):
    residues[:, i] = get_residue(points - planes[i][1], planes[i][0])
  return np.argmin(residues, axis=1)

def transform(planes, points):
  facet_planes = np.array([x[0] for x in planes])
  plane_centers = np.array([x[1] for x in planes])

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


def get_facet_planes(embeddings, facet_dim=3, residue_threshold=0.1):
  num_data = embeddings.shape[0]

  dist = cdist(embeddings, embeddings)
  topk_index = np.argsort(dist, axis=1)

  used_in_facet = np.ones(num_data) * -1
  residues = np.zeros(num_data)
  facet_planes = []
  plane_centers = []

  facet_num = 0
  pbar = tqdm(range(num_data), desc=f"{(used_in_facet != -1).sum()}/{num_data}")
  for i in pbar:
    if (used_in_facet[i] != -1):
      continue

    examples_to_select = []
    mask = used_in_facet[topk_index[i]] == -1
    candidates = topk_index[i][mask]
    for j in candidates:
      if len(examples_to_select) == 0:
        examples_to_select.append(j)
        continue

      indices = examples_to_select + [j]
      current_points = embeddings[topk_index[i, indices]]

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
      if (orig_residue <= residue_threshold):
        examples_to_select.append(j)

    if (len(examples_to_select) >= facet_dim):
      selected_idx = topk_index[i, examples_to_select]
      current_points = embeddings[selected_idx]
      origin = current_points[0]
      plane_centers.append(origin)
      current_points_centered = current_points - origin
      _, _, vh = svd(current_points_centered, full_matrices=False)
      vh = vh[:facet_dim, :]
      facet_planes.append(vh)

      used_in_facet[selected_idx] = facet_num
      facet_num += 1
      residues[selected_idx] = get_residue(current_points_centered, vh)
    pbar.set_description(f"{(used_in_facet != -1).sum()}/{num_data}")

  facet_planes = np.stack(facet_planes)
  plane_centers = np.stack(plane_centers)
  planes = list(zip(facet_planes, plane_centers))
  return planes
