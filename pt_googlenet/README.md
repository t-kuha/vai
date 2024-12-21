# GoogLeNet for ImageNet 1K from TorchVision pretrained model 

- Base model: [GOOGLENET](https://pytorch.org/vision/main/models/generated/torchvision.models.googlenet.html)

## model performance

- Top-1 on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |       69.778 |
| quantized |       61.200 |

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
$ xcompiler -i quantize_result/GoogLeNet_int.xmodel -o googlenet.xmodel -f 0x101000036010407
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B4096_0101000036010407
[UNILOG][INFO] Graph name: GoogLeNet, with op num: 423
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/vai/pt_googlenet/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/vai/pt_googlenet/googlenet.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is f1cef0075b4159d71afdccd824889a47, and has been saved to "/home/imagingtechnerd/vai/pt_googlenet/md5sum.txt"
```
