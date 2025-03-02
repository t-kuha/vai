# Wide ResNet

- Base model: [PyTorch Hub](https://pytorch.org/hub/pytorch_vision_wide_resnet/)

## model performance

- Top-1 accuracy ([%])

| config     | float  | qauntized |
|:----------:|-------:|----------:|
| ResNet 50  | 81.606 |    79.918 |
| ResNet 101 | 82.514 |    80.73  |

## how to deploy model

```shell
# check model's compatibility with Vitis AI:
# model name: "wide_resnet50" or "wide_resnet101"
$ python run.py <model name> inspect

# test float model performance:
$ python run.py <model name> float

# quantize model (calibration):
$ python run.py <model name> calib

# deployment:
$ python run.py <model name> deploy
```
