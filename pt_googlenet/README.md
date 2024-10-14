# GoogLeNet for ImageNet 1K from TorchVision pretrained model 

- Base model: [GOOGLENET](https://pytorch.org/vision/main/models/generated/torchvision.models.googlenet.html)

## model performance

- Top-1 on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |       69.778 |
| quantized |       60.932 |

## how to deploy model

```shell
# test float model performance:
$ python run.py float

# quantize model (calibration):
$ python run.py calib

# deployment:
$ python run.py deploy
$ xcompiler -i quantize_result/GoogLeNet_int.xmodel -o googlenet.xmodel -f <DPU config finger print value>
```
