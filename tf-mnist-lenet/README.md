# MNIST LeNet

## Performance (after quantization)

| Model     | Phase | Accuracy [%] |
|:---------:|:-----:|-------------:|
| float     | Train |        99.82 |
| float     | Test  |        99.18 |
| quantized | Train |        99.82 |
| quantized | Test  |        99.15 |

***

## Deploying quantized model

- In the example below, deploy model will be generated in ``_deploy``

```shell-session
$ vai_c_tensorflow \
-f quantized/deploy_model.pb \
-a ../arch/ultra96v2/arch.json -o _deploy -n lenet -q
```

- Output:

```shell-session
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
arch.json
[VAI_C][Warning] layer [dense_1_Softmax] (type: Softmax) is not supported in DPU, deploy it in CPU instead.

Kernel topology "lenet_kernel_graph.jpg" for network "lenet"
kernel list info for network "lenet"
                               Kernel ID : Name
                                       0 : lenet_0
                                       1 : lenet_1

                             Kernel Name : lenet_0
--------------------------------------------------------------------------------
                             Kernel Type : DPUKernel
                               Code Size : 6.54KB
                              Param Size : 1.14MB
                           Workload MACs : 23.98MOPS
                         IO Memory Space : 0.03MB
                              Mean Value : 0, 0, 0, 
                      Total Tensor Count : 5
                Boundary Input Tensor(s)   (H*W*C)
                       conv2d_input:0(0) : 28*28*1

               Boundary Output Tensor(s)   (H*W*C)
                     dense_1_MatMul:0(0) : 1*1*10

                        Total Node Count : 4
                           Input Node(s)   (H*W*C)
                        conv2d_Conv2D(0) : 28*28*1

                          Output Node(s)   (H*W*C)
                       dense_1_MatMul(0) : 1*1*10




                             Kernel Name : lenet_1
--------------------------------------------------------------------------------
                             Kernel Type : CPUKernel
                Boundary Input Tensor(s)   (H*W*C)
                    dense_1_Softmax:0(0) : 1*1*10

               Boundary Output Tensor(s)   (H*W*C)
                    dense_1_Softmax:0(0) : 1*1*10

                           Input Node(s)   (H*W*C)
                         dense_1_Softmax : 1*1*10

                          Output Node(s)   (H*W*C)
                         dense_1_Softmax : 1*1*10
```
