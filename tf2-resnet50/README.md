# TF2 ResNet50: deploying quantized model from AMD/Xilinx

- download TF2 ResNet50 model from Vitis AI model zoo:

```shell
$ wget https://www.xilinx.com/bin/public/openDownload?filename=tf2_resnet50_3.5.zip -O tf2_resnet50_3.5.zip
$ unzip tf2_resnet50_3.5.zip
```

- generate .xmodel:
  - output: ``deploy.xmodel``

```shell
$ vai_c_tensorflow2 -m tf2_resnet50_3.5/quantized/quantized.h5 -a <arch.json>
```

***

## example

```shell-sessin
$ vai_c_tensorflow2 -m tf2_resnet50_3.5/quantized/quantized.h5 -a ../arch.json 
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] Namespace(batchsize=1, inputs_shape=None, layout='NHWC', model_files=['tf2_resnet50_3.5/quantized/quantized.h5'], model_type='tensorflow2', named_inputs_shape=None, out_filename='/tmp/deploy_0x101000016010405_org.xmodel', proto=None)
[INFO] tensorflow2 model: /home/imagingtechnerd/work/vai/tf2-resnet50/tf2_resnet50_3.5/quantized/quantized.h5
[INFO] keras version: 2.4.0
[INFO] Tensorflow Keras model type: functional
[INFO] parse raw model     :100%|████████████████████████████████████████████████████████████████████████████████████████████| 126/126 [00:00<00:00, 21033.28it/s]             
[INFO] infer shape (NHWC)  :100%|████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 952.36it/s]               
[INFO] perform level-0 opt :100%|████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00, 28.14it/s]                    
[INFO] perform level-1 opt :100%|████████████████████████████████████████████████████████████████████████████████████████████| 2/2 [00:00<00:00, 115.72it/s]                   
[INFO] infer shape (NHWC)  :100%|████████████████████████████████████████████████████████████████████████████████████████████| 202/202 [00:00<00:00, 1671.84it/s]              
[INFO] generate xmodel     :100%|████████████████████████████████████████████████████████████████████████████████████████████| 202/202 [00:00<00:00, 380.45it/s]               
[INFO] dump xmodel: /tmp/deploy_0x101000016010405_org.xmodel
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
[UNILOG][INFO] Graph name: resnet50, with op num: 416
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/work/vai/tf2-resnet50/./meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/work/vai/tf2-resnet50/.//deploy.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 351a68f866a715ba6ce529a3655a5965, and has been saved to "/home/imagingtechnerd/work/vai/tf2-resnet50/./md5sum.txt"
```
