# MNIST SimpleNet

- Based on [this repository](https://github.com/Coderx7/SimpleNet.git)

- This model may not work on Zynq-7000 device

## Model Performance

| Model     | Phase | Accuracy [%] |
|:---------:|:-----:|-------------:|
| float     | Train |        99.94 |
| float     | Test  |        99.75 |
| quantized | Train |        99.73 |
| quantized | Test  |        99.73 |

***

## Quantizing model

```shell-session
# Run quantization & test the quantized model
$ vai_q_caffe quantize \
--model float/lenet_train_test.prototxt \
--weights float/99.75_lenet_iter_16800.caffemodel \
-output_dir quantized \
-calib_iter 10 -keep_fixed_neuron \
-auto_test -test_iter 100
```

## Deploying model

- Remove "loss" layer (the last one in the prototxt) from ``quantize_results/deploy.prototxt``

```shell-session
$ vai_c_caffe -p quantized/deploy.prototxt \
-c quantized/deploy.caffemodel \
-a ../arch/<platform name>.json -o _deploy -n simplenet

**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] Namespace(inputs_shape=None, layout='NCHW', model_files=['quantize_results/deploy.caffemodel'], model_type='caffe', out_filename='_deploy/simplenet_org.xmodel', proto='quantize_results/deploy.prototxt')
[INFO] caffe model: quantize_results/deploy.caffemodel
[INFO] caffe model: quantize_results/deploy.prototxt
[INFO] parse raw model     :100%|███████████| 34/34 [00:00<00:00, 37.79it/s]                  
[INFO] infer shape (NCHW)  :100%|███████████| 36/36 [00:00<00:00, 60157.35it/s]               
[INFO] infer shape (NHWC)  :100%|███████████| 36/36 [00:00<00:00, 75159.26it/s]               
[INFO] generate xmodel     :100%|███████████| 36/36 [00:00<00:00, 867.43it/s]                 
[INFO] generate xmodel: /media/t-kuha/HDD_500GB/vai/_cf_mnist_simplenet/_deploy/simplenet_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/t-kuha/log/xcompiler-20210116-150401-60974"
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_CUSTOMIZED
[UNILOG][INFO] Graph name: deploy, with op num: 94
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 2, DPU subgraph number 0
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/media/t-kuha/HDD_500GB/vai/_cf_mnist_simplenet/_deploy/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/media/t-kuha/HDD_500GB/vai/_cf_mnist_simplenet/_deploy/simplenet.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 1583d5adcfc6afb13fc9efd349be0956, and been saved to "/media/t-kuha/HDD_500GB/vai/_cf_mnist_simplenet/_deploy/md5sum.txt"
```

***

## Starting from scratch

- Create MNIST LMDB

- Generate calibration dataset

```shell-session
$ python ../tf-mnist-lenet/code/create_calib_datasets.py
$ for i in $(ls -1 _dataset/calib/); do echo "${i} 0" >> _calib.txt; done
```

- Clone SimpleNet repo

```shell-session
$ git clone https://github.com/Coderx7/SimpleNet.git
$ cp -R SimpleNet/SimpNet_V1/Benchmarks\ Results\ with\ Models/MNIST/Normal\ Arch_NoDrpOut_99.75/99.75/99.75_lenet_iter_16800.caffemodel float/
```

- Testing float model

```shell-session
$ vai_q_caffe test \
--model float/lenet_train_test.prototxt \
--weights float/99.75_lenet_iter_16800.caffemodel \
-test_iter 100

...

I0116 14:01:56.764746 54366 net_test.cpp:405] Test Results: 
I0116 14:01:56.764806 54366 net_test.cpp:406] Loss: 0.0140012
I0116 14:01:56.764865 54366 net_test.cpp:421] accuracy = 0.9975
I0116 14:01:56.764933 54366 net_test.cpp:421] loss = 0.0140012 (* 1 = 0.0140012 loss)
I0116 14:01:56.764997 54366 net_test.cpp:450] Test Done!
```
