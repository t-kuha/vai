# CifarNet

- Modifiled to replace LRN with BatchNorm

## Model Performance

| Model     | Phase | Accuracy [%] |
|:---------:|:-----:|-------------:|
| float     | Train |        93.91 |
| float     | Test  |        85.00 |
| quantized | Train |        93.77 |
| quantized | Test  |        84.87 |

- Reference: Performance of original (LRN) version:

| Model     | Phase | Accuracy [%] |
|:---------:|:-----:|-------------:|
| float     | Train |        93.16 |
| float     | Test  |        85.11 |
| quantized | Train |        93.08 |
| quantized | Test  |        85.23 |

***

## Quantization

- Get model info

```shell-session
$ vai_q_tensorflow inspect --input_frozen_graph float/frozen.pb

Op types used: 15 Const, 11 Identity, 5 BiasAdd, 4 Relu, 3 MatMul, 3 Reshape, 2 Conv2D, 2 LRN, 2 MaxPool, 2 Shape, 1 Pack, 1 Placeholder, 1 Softmax, 1 StridedSlice

Found 1 possible inputs: (name=input, type=float(1), shape=[?,32,32,3]) 
Found 1 possible outputs: (name=CifarNet/Predictions/Reshape_1, op=Reshape) 
```

- Run quantization

```shell-session
$ vai_q_tensorflow quantize \
--input_frozen_graph float/frozen.pb \
--input_nodes 'input' \
--output_nodes 'CifarNet/Predictions/Reshape_1' \
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
-a ../arch/ultra96v2/arch.json -o _deploy -n cifarnet -q
```

- Output:

```shell-session
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[VAI_C][Warning] layer [CifarNet_Predictions_Softmax] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

Kernel topology "cifarnet_kernel_graph.jpg" for network "cifarnet"
kernel list info for network "cifarnet"
                               Kernel ID : Name
                                       0 : cifarnet_0
                                       1 : cifarnet_1

                             Kernel Name : cifarnet_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 0.01MB
                              Param Size : 1.68MB
                           Workload MACs : 65.56MOPS
                         IO Memory Space : 0.02MB
                              Mean Value : 0, 0, 0, 
                      Total Tensor Count : 6
                Boundary Input Tensor(s)   (H*W*C)
                              input:0(0) : 32*32*3

               Boundary Output Tensor(s)   (H*W*C)
             CifarNet_logits_MatMul:0(0) : 1*1*10

                        Total Node Count : 5
                           Input Node(s)   (H*W*C)
                CifarNet_conv1_Conv2D(0) : 32*32*3

                          Output Node(s)   (H*W*C)
               CifarNet_logits_MatMul(0) : 1*1*10




                             Kernel Name : cifarnet_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                Boundary Input Tensor(s)   (H*W*C)
       CifarNet_Predictions_Softmax:0(0) : 1*1*10

               Boundary Output Tensor(s)   (H*W*C)
       CifarNet_Predictions_Softmax:0(0) : 1*1*10

                           Input Node(s)   (H*W*C)
            CifarNet_Predictions_Softmax : 1*1*10

                          Output Node(s)   (H*W*C)
            CifarNet_Predictions_Softmax : 1*1*10
```
