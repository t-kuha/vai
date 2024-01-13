# TF2 ResNet50: deploying quantized model from AMD/Xilinx

- download TF2 ResNet50 model from Vitis AI model zoo:

```shell
$ wget https://www.xilinx.com/bin/public/openDownload?filename=tf2_resnet50_3.5.zip
```

- generate .xmodel:
  - output: ``deploy.xmodel``

```shell
$ unzip tf2_resnet50_3.5.zip
$ vai_c_tensorflow2 -m tf2_resnet50_3.5/quantized/quantized.h5 -a <arch.json>
```
