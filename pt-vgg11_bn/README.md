# VGG11 for ImageNet 1K from TorchVision pretrained model 

- Base model: [VGG11_BN](https://pytorch.org/vision/stable/models/generated/torchvision.models.vgg11_bn.html#torchvision.models.vgg11_bn)

## model performance

- on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |        70.38 |
| quantized |        69.39 |

## how to deploy model

```shell
# test float model performance:
$ python run.py float

# quantize model (calibration):
$ python run.py calib

# deployment:
$ python run.py deploy
$ xcompiler -i quantize_result/VGG_int.xmodel -o vgg11_bn.xmodel -f <DPU config finger print value>
```
