# VGG11 for ImageNet 1K from TorchVision pretrained model 

## model performance

- on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |        70.38 |
| quantized |        TODO  |

## how to deploy model

```shell
# test float model performance:
$ python run.py float

# quantize model (calibration):
$ python run.py calib

# deployment:
$ python run.py deploy
$ xcompiler -i quantize_result/VGG_int.xmodel -o vgg11.xmodel -f <DPU config finger print value>
```
