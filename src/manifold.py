import torch
from tqdm import tqdm
import torch.nn.functional as F


def transform(facet_planes, plane_centers, points):
  num_points = points.shape[0]
  num_facets = facet_planes.shape[0]

  chosen_facets = torch.zeros(num_points, dtype=torch.int)
  all_residues = torch.zeros(num_points, num_facets, dtype=torch.float)

  for num_facet in range(num_facets):
    current_facet = facet_planes[num_facet, :, :]
    current_center = plane_centers[num_facet, :]
    current_points = points - current_center
    all_residues[:, num_facet] = get_residue(current_points, current_facet)

  _, chosen_facets = torch.min(all_residues, dim=1)

  alphas = torch.matmul(
    (points - plane_centers[chosen_facets]).unsqueeze(1),
    facet_planes[chosen_facets].transpose(1, 2)
  )
  transformed_points = plane_centers[chosen_facets] + torch.bmm(
    alphas, facet_planes[chosen_facets]
  ).squeeze()
  return transformed_points, chosen_facets


def get_residue(points, plane):
  points_projected = torch.matmul(points, plane.T)
  residue = torch.linalg.vector_norm(points - torch.matmul(points_projected, plane), dim=1)
  return residue


def get_facet_planes(embeddings, facet_dim=3, residue_threshold=0.01):
  num_data = embeddings.shape[0]
  dist = torch.cdist(embeddings, embeddings)
  topk_index = torch.topk(dist, num_data, largest=False, dim=1)[1]

  used_in_facet = torch.ones(num_data, dtype=torch.float) * -1
  residues = torch.zeros(num_data, dtype=torch.float)
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
        current_points = embeddings[topk_index[i, examples_to_select + [j]], :]

      origin = current_points.mean(dim=0)
      current_points_centered = current_points - origin
      try:
        _, _, vh = torch.linalg.svd(current_points_centered, full_matrices=False)
      except torch._C._LinAlgError:
        try:
          _, _, vh = torch.linalg.svd(
            current_points_centered + 1e-5 * torch.randn_like(current_points_centered),
            full_matrices=False
          )
        except torch._C._LinAlgError:
          print("unfixable")
          continue

      vh = vh[:facet_dim, :]
      orig_residue = get_residue(current_points_centered[0, :].unsqueeze(0), vh)
      if (orig_residue > residue_threshold):
        continue
      else:
        examples_to_select = examples_to_select + [j]

    if (len(examples_to_select) >= facet_dim):
      current_points = embeddings[topk_index[i, examples_to_select], :]
      origin = current_points[0, :]
      plane_centers.append(origin)
      current_points_centered = current_points - origin
      _, _, vh = torch.linalg.svd(current_points_centered, full_matrices=False)
      vh = vh[:facet_dim, :]

      point_plane = torch.zeros(facet_dim, embeddings.shape[1], dtype=torch.float)
      point_plane[:, :] = vh
      facet_planes.append(point_plane)

      used_in_facet[topk_index[i, examples_to_select]] = facet_num
      facet_num += 1

      temp1 = get_residue(current_points_centered, vh)
      residues[topk_index[i, examples_to_select]] = temp1
    else:
      outlier_points.append(embeddings[i, :])

  facet_planes = torch.stack(facet_planes)
  plane_centers = torch.stack(plane_centers)
  return facet_planes, plane_centers


def get_facet_proj(embeddings, facet_dim=3, residue_threshold=0.01):
  if not isinstance(embeddings, torch.Tensor):
    embeddings = torch.from_numpy(embeddings)
  embeddings = F.normalize(embeddings)
  facet_planes, plane_centers = get_facet_planes(embeddings, facet_dim, residue_threshold)
  transformed, chosen_facets = transform(facet_planes, plane_centers, embeddings)
  return facet_planes.numpy(), plane_centers.numpy(), transformed.numpy(), chosen_facets.numpy()
