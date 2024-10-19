# GoogLeNet for ImageNet 1K from TorchVision pretrained model 

- Base model: [GOOGLENET](https://pytorch.org/vision/main/models/generated/torchvision.models.googlenet.html)

## model performance

- Top-1 on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |       69.778 |
| quantized |       60.932 |

## how to deploy model

```shell
# test float model performance:
$ python run.py float

# quantize model (calibration):
$ python run.py calib

# deployment:
$ python run.py deploy
$ xcompiler -i quantize_result/GoogLeNet_int.xmodel -o googlenet.xmodel -f <DPU config finger print value>
```

```shell-session
$ xcompiler -i quantize_result/GoogLeNet_int.xmodel -o googlenet.xmodel -f 0x101000016010405
[UNILOG][WARNING] The operator named GoogLeNet__GoogLeNet_ret_19, type: nndct_select, is not defined in XIR. XIR creates the definition of this operator automatically. You should specify the shape and the data_type of the output tensor of this operation by set_attr("shape", std::vector<int>) and set_attr("data_type", std::string)
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
[UNILOG][INFO] Graph name: GoogLeNet, with op num: 473
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 6, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/work/vai/pt_googlenet/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/work/vai/pt_googlenet/googlenet.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 40a6ce2a62d1f5f8c3831800cc31917d, and has been saved to "/home/imagingtechnerd/work/vai/pt_googlenet/md5sum.txt"
```
