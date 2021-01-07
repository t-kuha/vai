# MNIST LeNet

## Model Performance

| Model     | Phase | Accuracy [%] |
|:---------:|:-----:|-------------:|
| float     | Train |        99.82 |
| float     | Test  |        99.18 |
| quantized | Train |        99.82 |
| quantized | Test  |        99.15 |

***

## Deploying quantized model

- In the examples below, deploy model will be generated in ``_deploy``

### Vitis-AI v1.3

```shell-session
$ vai_c_tensorflow \
-f quantized/quantize_eval_model.pb \
-a ../arch/arch.json -o _deploy -n lenet
```

- Output:

```shell-session
**************************************************
* VITIS_AI Compilation - Xilinx Inc.
**************************************************
[INFO] Namespace(inputs_shape=None, layout='NHWC', model_files=['quantized/quantize_eval_model.pb'], model_type='tensorflow', out_filename='_deploy/lenet_org.xmodel', proto=None)
[INFO] tensorflow model: quantized/quantize_eval_model.pb
[INFO] parse raw model     :100%|████████████████████████████████████████████████| 21/21 [00:00<00:00, 19269.39it/s]               
[INFO] infer shape (NHWC)  :100%|████████████████████████████████████████████████| 26/26 [00:00<00:00, 31554.37it/s]               
[INFO] infer shape (NHWC)  :100%|████████████████████████████████████████████████| 23/23 [00:00<00:00, 37580.44it/s]               
[INFO] generate xmodel     :100%|████████████████████████████████████████████████| 23/23 [00:00<00:00, 309.57it/s]                 
[INFO] generate xmodel: /media/t-kuha/HDD_500GB/vai/tf-mnist-lenet/_deploy/lenet_org.xmodel
[UNILOG][INFO] The compiler log will be dumped at "/tmp/t-kuha/log/xcompiler-20210107-221120-4598"
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: function
[UNILOG][INFO] Target architecture: DPUCZDX8G_CUSTOMIZED
[UNILOG][INFO] Graph name: quantize_eval_model, with op num: 35
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/media/t-kuha/HDD_500GB/vai/tf-mnist-lenet/_deploy/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/media/t-kuha/HDD_500GB/vai/tf-mnist-lenet/_deploy/lenet.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 87a240b3b78a9a455b736c1ee219a7e8, and been saved to "/media/t-kuha/HDD_500GB/vai/tf-mnist-lenet/_deploy/md5sum.txt"
```

### Vitis-AI v1.2 and before

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
