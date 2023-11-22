# TF2 ResNet50: deploy quantized model

- download TF2 ResNet50 model from Vitis AI model zoo

- deploy

```shell
$ unzip tf2_resnet50_3.5.zip
$ vai_c_tensorflow2 -m tf2_resnet50_3.5/quantized/quantized.h5 -a <arch.json>
```
