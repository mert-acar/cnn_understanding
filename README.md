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
