# AlexNet for ImageNet 1K from TorchVision pretrained model 

- Base model: [AlexNet](https://pytorch.org/vision/main/models/generated/torchvision.models.alexnet.html)

## model performance

- Top-1 on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |       56.556 |
| quantized |       55.324 |

## how to deploy model

```shell
# test float model performance:
$ python run.py float

# quantize model (calibration):
$ python run.py calib

# deployment:
$ python run.py deploy
$ xcompiler -i quantize_result/AlexNet_int.xmodel -o alexnet.xmodel -f <DPU config finger print value>
```

```shell-session
$ xcompiler -i quantize_result/AlexNet_int.xmodel -o alexnet.xmodel -f 0x101000036010407
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B4096_0101000036010407
[UNILOG][INFO] Graph name: AlexNet, with op num: 71
[UNILOG][INFO] Begin to compile...
[UNILOG][WARNING] xir::Op{name = AlexNet__AlexNet_AdaptiveAvgPool2d_avgpool__976_fix, type = pool-fix}'s input and output is unchanged, so it will be removed.
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/work/vai/pt_alexnet/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/work/vai/pt_alexnet/alexnet.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 99705a59cd50aff1c2818bc86cceae5c, and has been saved to "/home/imagingtechnerd/work/vai/pt_alexnet/md5sum.txt"
```

***

## inspection results

<details>
  <summary>click to expand</summary>

```shell
$ python run.py inspect

[VAIQ_NOTE]: Loading NNDCT kernels...

[VAIQ_NOTE]: Inspector is on.

[VAIQ_NOTE]: =>Start to inspect model...

[VAIQ_NOTE]: =>Quant Module is in 'cpu'.

[VAIQ_NOTE]: =>Parsing AlexNet...

[VAIQ_NOTE]: Start to trace and freeze model...

[VAIQ_NOTE]: The input model nndct_st_AlexNet_ed is torch.nn.Module.

[VAIQ_NOTE]: Finish tracing.

[VAIQ_NOTE]: Processing ops...
██████████████████████████████████████████████████| 24/24 [00:00<00:00, 2583.63it/s, OpInfo: name = return_0, type = Return]                                            

[VAIQ_NOTE]: =>Doing weights equalization...

[VAIQ_NOTE]: =>Quantizable module is generated.(quantize_result/AlexNet.py)

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[classifier]/Linear[4]/ret.17, op type:nndct_dense, output shape: [1, 4096]
node name:AlexNet::AlexNet/Sequential[classifier]/ReLU[5]/993, op type:nndct_relu, output shape: [1, 4096]


WARNING: Logging before InitGoogleLogging() is written to STDERR
I20241123 08:01:18.497710  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:18.497754  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:18.497769  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:18.497843  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_dense_nndct_relu_YBC2GLVXNMeW0zfk, with op num: 9
I20241123 08:01:18.497857  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:19.209702  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:19.209761  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[classifier]/Linear[1]/ret.15, op type:nndct_dense, output shape: [1, 4096]
node name:AlexNet::AlexNet/Sequential[classifier]/ReLU[2]/987, op type:nndct_relu, output shape: [1, 4096]


I20241123 08:01:19.585420  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:19.585469  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:19.585479  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:19.585557  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_dense_nndct_relu_RHvp4JUAB92u5Qkb, with op num: 9
I20241123 08:01:19.585572  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:21.879609  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:21.879717  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[features]/Conv2d[8]/ret.9, op type:nndct_conv2d, output shape: [1, 13, 13, 256]
node name:AlexNet::AlexNet/Sequential[features]/ReLU[9]/922, op type:nndct_relu, output shape: [1, 13, 13, 256]


I20241123 08:01:21.892158  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:21.892216  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:21.892225  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:21.892330  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_conv2d_nndct_relu_MZom9AnTN7DUq8iK, with op num: 9
I20241123 08:01:21.892346  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:21.932348  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:21.932466  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[features]/Conv2d[6]/ret.7, op type:nndct_conv2d, output shape: [1, 13, 13, 384]
node name:AlexNet::AlexNet/Sequential[features]/ReLU[7]/901, op type:nndct_relu, output shape: [1, 13, 13, 384]


I20241123 08:01:21.941517  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:21.941622  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:21.941637  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:21.941849  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_conv2d_nndct_relu_ibQzmJPXo4Hg2Vlc, with op num: 9
I20241123 08:01:21.941883  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:21.977181  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:21.977259  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[features]/Conv2d[3]/ret.5, op type:nndct_conv2d, output shape: [1, 27, 27, 192]
node name:AlexNet::AlexNet/Sequential[features]/ReLU[4]/864, op type:nndct_relu, output shape: [1, 27, 27, 192]


I20241123 08:01:21.983644  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:21.983748  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:21.983768  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:21.983911  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_conv2d_nndct_relu_wkeiP3asf4DMdXNx, with op num: 9
I20241123 08:01:21.983942  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.004830  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.004911  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[features]/Conv2d[0]/ret.3, op type:nndct_conv2d, output shape: [1, 55, 55, 64]
node name:AlexNet::AlexNet/Sequential[features]/ReLU[1]/827, op type:nndct_relu, output shape: [1, 55, 55, 64]


I20241123 08:01:22.009294  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.009323  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.009331  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.009411  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_conv2d_nndct_relu_62DAhmpbnqwVYiWH, with op num: 9
I20241123 08:01:22.009438  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.015830  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.015928  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_18:
node name:AlexNet::AlexNet/Sequential[features]/Conv2d[10]/ret.11, op type:nndct_conv2d, output shape: [1, 13, 13, 256]
node name:AlexNet::AlexNet/Sequential[features]/ReLU[11]/943, op type:nndct_relu, output shape: [1, 13, 13, 256]


I20241123 08:01:22.021852  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.021911  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.021919  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.022022  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_conv2d_nndct_relu_LTwh1kl2VWmYdUOR, with op num: 9
I20241123 08:01:22.022043  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.055644  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.055749  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for reshape_fix_1:
node name:AlexNet::AlexNet/ret.13, op type:nndct_reshape, output shape: [1, 9216]


I20241123 08:01:22.063532  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.063575  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.063583  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.063656  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_reshape_9KetQmdru7lxskob, with op num: 7
I20241123 08:01:22.063671  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.066850  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.066897  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for pool_fix_4:
node name:AlexNet::AlexNet/AdaptiveAvgPool2d[avgpool]/976, op type:nndct_avgpool, output shape: [1, 6, 6, 256]


I20241123 08:01:22.070545  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.070582  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.070590  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.070655  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_avgpool_2cO5BPtD7r0X1hMk, with op num: 6
I20241123 08:01:22.070669  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
W20241123 08:01:22.074234  3034 RedundantOpReductionPass.cpp:663] [UNILOG][WARNING] xir::Op{name = AlexNet__AlexNet_AdaptiveAvgPool2d_avgpool__976_fix, type = pool-fix}'s input and output is unchanged, so it will be removed.
I20241123 08:01:22.074463  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 2, DPU subgraph number 0
I20241123 08:01:22.074501  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for pool_fix_5:
node name:AlexNet::AlexNet/Sequential[features]/MaxPool2d[2]/842, op type:nndct_maxpool, output shape: [1, 27, 27, 64]


I20241123 08:01:22.078164  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.078193  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.078202  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.078267  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_maxpool_qp0sr1yC5zRgcF29, with op num: 4
I20241123 08:01:22.078284  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.082058  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.082101  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for pool_fix_5:
node name:AlexNet::AlexNet/Sequential[features]/MaxPool2d[5]/879, op type:nndct_maxpool, output shape: [1, 13, 13, 192]


I20241123 08:01:22.084249  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.084281  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.084300  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.084411  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_maxpool_li0sKV6gbMjnAvoL, with op num: 4
I20241123 08:01:22.084439  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.087914  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.088009  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for pool_fix_5:
node name:AlexNet::AlexNet/Sequential[features]/MaxPool2d[12]/958, op type:nndct_maxpool, output shape: [1, 6, 6, 256]


I20241123 08:01:22.089807  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.089833  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.089840  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.089895  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_maxpool_W8Bu69RmgQCfUTpz, with op num: 4
I20241123 08:01:22.089908  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.093461  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.093505  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: Find subgraph for convlike_fix_20:
node name:AlexNet::AlexNet/Sequential[classifier]/Linear[6]/ret, op type:nndct_dense, output shape: [1, 1000]


I20241123 08:01:22.186538  3034 compile_pass_manager.cpp:352] [UNILOG][INFO] Compile mode: dpu
I20241123 08:01:22.186605  3034 compile_pass_manager.cpp:353] [UNILOG][INFO] Debug mode: null
I20241123 08:01:22.186614  3034 compile_pass_manager.cpp:357] [UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
I20241123 08:01:22.186751  3034 compile_pass_manager.cpp:465] [UNILOG][INFO] Graph name: nndct_dense_iRBqTe9ZOCsVwoJt, with op num: 8
I20241123 08:01:22.186786  3034 compile_pass_manager.cpp:478] [UNILOG][INFO] Begin to compile...
I20241123 08:01:22.482924  3034 compile_pass_manager.cpp:489] [UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
I20241123 08:01:22.483076  3034 compile_pass_manager.cpp:504] [UNILOG][INFO] Compile done.

[VAIQ_NOTE]: All the operators are assigned to the DPU(see more details in 'quantize_result/inspect_0x101000016010405.txt')

[VAIQ_NOTE]: =>Finish inspecting.
```
</details>
