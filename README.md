# CNN Understanding
UIUC Project

## Hookable Layer Names
```
conv1 bn1 relu maxpool layer1.0.conv1 layer1.0.bn1 layer1.0.relu layer1.0.conv2 layer1.0.bn2 layer1.1.bn1 layer1.1.relu layer1.1.conv2 layer1.1.bn2 layer2.0.conv1 layer2.0.bn1 layer2.0.relu layer2.0.conv2 layer2.0.bn2 layer2.0.downsample.0 layer2.0.downsample.1 layer2.1.conv1 layer2.1.bn1 layer2.1.relu layer2.1.conv2 layer2.1.bn2 layer3.0.conv1 layer3.0.bn1 layer3.0.relu layer3.0.conv2 layer3.0.bn2 layer3.0.downsample.0 layer3.0.downsample.1 layer3.1.conv1 layer3.1.bn1 layer3.1.relu layer3.1.conv2 layer3.1.bn2 layer4.0.conv1 layer4.0.bn1 layer4.0.relu layer4.0.conv2 layer4.0.bn2 layer4.0.downsample.0 layer4.0.downsample.1 layer4.1.conv1 layer4.1.bn1 layer4.1.relu layer4.1.conv2 layer4.1.bn2 avgpool fc
```


### Sample commands:

```bash
python3 scripts/dimensionality_reduction.py pca ../data/act_resnet18_run1.mat relu
python3 scripts/dimensionality_reduction.py svd ../data/act_resnet18_run1.mat conv1
```

## Notes

#### Dimensionality Reduction -> Clustering Approach
    We aim to see the effects of training on the purity/homogeneity of the activation clusters. The main hypothesis is that training increases the purity of the early layer activations due to alignment of the subspaces across labels. In order to test this approach we train a classification CNN (resnet18) on the MNIST dataset. During training we hook into the first convolutional layer (`conv1`) to get the activations of shape [N, 64, 14, 14]. The steps go as follows:

    1. Filter noisy features
    ```python
        flat_activations = activations.reshape[activations.shape[0], -1]
        non_zero_idx = [
          np.abs(activations[labels == 0]).mean(0)[j].sum() > act_threshold for j in range(activations.shape[1])
        ]
    ```

    2. Run dimensionality reduction (SVD):
    ```python
      # Center the data
      mean = np.mean(activations, axis=0)
      u, s, vt = np.linalg.svd(activations - mean, full_matrices=False)
      s2 = s**2
      energies = np.cumsum(s2) / np.sum(s2)
      k = np.argmax(energies > threshold) + 1
      u_k = u[:, :k]
      s_k = s[:k]
      vt_k = vt[:k, :]
      recon = u_k @ np.diag(s_k)
    ```

    3. Run clustering:
    ```python
        clusters = HDBSCAN(min_cluster_size=50).fit(recon).labels_
    ```

    4. Visualize using tSNE
    

#### TODO:

    -[ ] Recording training activations has the following issue:
        The training is done in batches and the optimizer is tasked to optimize the network parameters in between each batch. So for the MNIST dataset, which has 60.000 training sample, a training run with `batch_size = 32`, the network is optimized $60000 // 32 = 1875$ times. Therefore, the first recorded batch and the last batch does not have the same scale, domain or a coherent basis for a subspace. 
        Instead we can "sample" the network at the end of each epoch with the test set activations which all have the same domain (due to the network being frozen during test time).
