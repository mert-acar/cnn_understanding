import torch
from tqdm import tqdm
import torch.nn.functional as F


def get_residue(points, plane):
  points_projected = torch.matmul(points, plane.T)
  residue = torch.linalg.vector_norm(points - torch.matmul(points_projected, plane), dim=1)
  return residue


def get_facet_planes(embeddings, facet_dim=3, residue_threshold=0.01):
  num_data = embeddings.shape[0]
  facet_planes = []
  plane_centers = []
  dist = torch.cdist(embeddings, embeddings)
  topk_index = torch.argsort(dist, dim=1)
  used_in_facet = torch.ones(num_data, dtype=torch.float) * -1
  residues = torch.zeros(num_data, dtype=torch.float)
  outlier_points = []

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
      _, _, vh = torch.linalg.svd(current_points_centered, full_matrices=False)
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

      plane_centers.append(origin)

      used_in_facet[topk_index[i, examples_to_select]] = facet_num
      facet_num += 1

      temp1 = get_residue(current_points_centered, vh)
      residues[topk_index[i, examples_to_select]] = temp1
    else:
      outlier_points.append(embeddings[i, :])

  facet_planes = torch.stack(facet_planes)
  plane_centers = torch.stack(plane_centers)
  return facet_planes, plane_centers, residues, used_in_facet, outlier_points


if __name__ == "__main__":
  import os
  from glob import glob
  from scipy.io import loadmat
  from kneed import KneeLocator
  from analysis import extract_epoch

  experiment_path = "../logs/resnet18_run1/"
  layers = ["layer1.1.conv2", "layer2.1.conv1", "layer3.1.conv1", "layer4.1.conv2"]

  non_zero_idx = None
  for layer in layers:
    filenames = list(
      reversed(sorted(glob(os.path.join(experiment_path, layer, "*.mat")), key=extract_epoch))
    )
    for fname in filenames:
      print(f"Working on {layer} -> {os.path.basename(fname)}")
      data = torch.from_numpy(loadmat(fname)[layer])

      if non_zero_idx is None:
        N, C, H, W = data.shape
        stat = data.abs().mean(0).reshape(C, H * W).mean(-1)
        y, _ = torch.sort(stat, descending=True)
        x = list(range(len(y)))
        kn = KneeLocator(x, y.numpy(), curve='convex', direction='decreasing').knee
        non_zero_idx = stat >= y[kn] * 0.95

      data = data[:, non_zero_idx].reshape(data.shape[0], -1)
      data = F.normalize(data)

      facet_planes, plane_centers, residues, used_in_facet, outlier_points = get_facet_planes(data)

      save_dict = {
        "facet_basis": facet_planes,
        "facet_centers": plane_centers,
        "fit_errors": residues,
        "point_facet_map": used_in_facet,
        "outliers": outlier_points
      }

      torch.save(
        save_dict,
        os.path.join(experiment_path, layer,
                     os.path.basename(fname).split(".")[0] + "_facets.pth")
      )
