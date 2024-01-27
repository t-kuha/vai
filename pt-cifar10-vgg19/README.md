# VGG19 for CIFAR10 on PyTorch

## model performance

- on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |        93.84 |
| quantized |        93.75 |

## how to deploy model

- get model definition

```shell
$ git clone https://github.com/kuangliu/pytorch-cifar.git
$ cp pytorch-cifar/models/vgg.py .
```

- test float model performance:

```shell
$ python run.py float
```

- quantize model (calibration):

```shell
$ python run.py calib
```

- deployment:

```shell
$ python run.py deploy
$ xcompiler -i quantize_result/VGG_int.xmodel -o test.xmodel -f <DPU config finger print value>
```
