# TF1 ResNet50: deploying quantized model from AMD/Xilinx

- download TF1 ResNet50 model from Vitis AI model zoo:

```shell
$ wget https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1-50_0.38_3.5.zip \
-o tf_resnetv1-50_0.38_3.5.zip
```

- generate .xmodel:
  - output: ``deploy.xmodel``

```shell
$ unzip tf_resnetv1-50_0.38_3.5.zip
$ vai_c_tensorflow -f tf_resnetv1-50_0.38_3.5/quantized/quantized_4.3B_91542.pb -a <arch.json>
```
