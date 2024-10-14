# TF1 ResNet50: deploying quantized model from AMD/Xilinx

- download TF1 ResNet50 model from Vitis AI model zoo:

```shell
$ wget https://www.xilinx.com/bin/public/openDownload?filename=tf_resnetv1-50_0.38_3.5.zip \
-O tf_resnetv1-50_0.38_3.5.zip
$ unzip tf_resnetv1-50_0.38_3.5.zip
```

- generate .xmodel:
  - output: ``deploy.xmodel``

```shell
$ vai_c_tensorflow -f tf_resnetv1-50_0.38_3.5/quantized/quantized_4.3B_91542.pb -a <arch.json>
```

***

## example

```shell-session
$ vai_c_tensorflow -f tf_resnetv1-50_0.38_3.5/quantized/quantized_4.3B_91542.pb -a ../arch/arch.json 
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] Namespace(batchsize=1, inputs_shape=None, layout='NHWC', model_files=['tf_resnetv1-50_0.38_3.5/quantized/quantized_4.3B_91542.pb'], model_type='tensorflow', named_inputs_shape=None, out_filename='/tmp/deploy_0x101000016010405_org.xmodel', proto=None)
[INFO] tensorflow model: /home/imagingtechnerd/work/vai/tf1-resnet50/tf_resnetv1-50_0.38_3.5/quantized/quantized_4.3B_91542.pb
[INFO] parse raw model     :100%|████████████████████████████████████████████████████████████████████████████████████████████| 270/270 [00:00<00:00, 18720.86it/s]             
[INFO] infer shape (NHWC)  :100%|████████████████████████████████████████████████████████████████████████████████████████████| 331/331 [00:00<00:00, 2392.87it/s]              
[INFO] perform level-0 opt :100%|████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:00<00:00, 247.45it/s]                   
[INFO] perform level-1 opt :100%|████████████████████████████████████████████████████████████████████████████████████████████| 8/8 [00:00<00:00, 292.77it/s]                   
[INFO] generate xmodel     :100%|████████████████████████████████████████████████████████████████████████████████████████████| 219/219 [00:00<00:00, 1387.58it/s]              
[INFO] dump xmodel: /tmp/deploy_0x101000016010405_org.xmodel
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
[UNILOG][INFO] Graph name: quantized_4.3B_91542, with op num: 435
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/work/vai/tf1-resnet50/./meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/work/vai/tf1-resnet50/.//deploy.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 635a7ac74171877a1471de5a2644e64c, and has been saved to "/home/imagingtechnerd/work/vai/tf1-resnet50/./md5sum.txt"
```
