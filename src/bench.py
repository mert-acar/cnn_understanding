import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if __name__ == "__main__":
  data_path = "../data/act_resnet18_run1.mat"
  activations = loadmat(data_path, variable_names=["conv1"])["conv1"]
  labels = loadmat("/Users/smol/Desktop/phd/uiuc/cnn_understanding/data/labels.mat")["labels"][0]
  non_zero_idx = [
    np.abs(activations[labels == 0]).mean(0)[j].sum() > 0 for j in range(activations.shape[1])
  ]
  activations = activations[:, non_zero_idx]
  activations = activations.reshape(activations.shape[0], -1)

  scaler = StandardScaler()
  scaled_data = scaler.fit_transform(activations)

  # Initialize PCA with enough components to cover all variance
  pca = PCA()
  pca.fit(scaled_data)

  # Determine the number of components needed to explain the threshold variance
  cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
  plt.plot(cumulative_explained_variance)
  plt.xlabel("Principal Components")
  plt.ylabel("Total Variance")
  plt.grid()
  plt.show()
