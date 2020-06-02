# CIFAR-10 ResNet

## Performance (after quantization)

| Model     | Phase | Accuracy [%] |
|:---------:|:-----:|-------------:|
| float     | Train |        99.92 |
| float     | Test  |        93.14 |
| quantized | Train |        99.62 |
| quantized | Test  |        92.33 |

***

## Quantization

- Get model info

```shell-session
$ vai_q_tensorflow inspect --input_frozen_graph float/frozen.pb
```

- Run quantization

```shell-session
$ vai_q_tensorflow quantize \
--input_frozen_graph float/frozen.pb \
--input_nodes 'input_1' \
--output_nodes 'fc10/Softmax' \
--input_shapes ?,32,32,3 \
--input_fn input_fn_cifar10.calib_input \
--output_dir quantized
```

***

## Deploying quantized model

- In the example below, deploy model will be generated in ``_deploy``

```shell-session
$ vai_c_tensorflow \
-f quantized/deploy_model.pb \
-a ../arch/ultra96v2/arch.json -o _deploy -n cifar10-resnet -q
```

- Output:

```shell-session
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[VAI_C][Warning] layer [fc10_Softmax] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

Kernel topology "cifar10resnet_kernel_graph.jpg" for network "cifar10resnet"
kernel list info for network "cifar10resnet"
                               Kernel ID : Name
                                       0 : cifar10resnet_0
                                       1 : cifar10resnet_1

                             Kernel Name : cifar10resnet_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.10MB
                              Param Size : 0.81MB
                           Workload MACs : 252.02MOPS
                         IO Memory Space : 0.07MB
                              Mean Value : 0, 0, 0, 
                      Total Tensor Count : 61
                Boundary Input Tensor(s)   (H*W*C)
                            input_1:0(0) : 32*32*3

               Boundary Output Tensor(s)   (H*W*C)
                        fc10_MatMul:0(0) : 1*1*10

                        Total Node Count : 60
                           Input Node(s)   (H*W*C)
                         conv1_Conv2D(0) : 32*32*3

                          Output Node(s)   (H*W*C)
                          fc10_MatMul(0) : 1*1*10




                             Kernel Name : cifar10resnet_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                Boundary Input Tensor(s)   (H*W*C)
                       fc10_Softmax:0(0) : 1*1*10

               Boundary Output Tensor(s)   (H*W*C)
                       fc10_Softmax:0(0) : 1*1*10

                           Input Node(s)   (H*W*C)
                            fc10_Softmax : 1*1*10

                          Output Node(s)   (H*W*C)
                            fc10_Softmax : 1*1*10
```
