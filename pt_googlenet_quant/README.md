# Quantized version of GoogLeNet (from TorchVision)

- Base model: [Quantized GoogLeNet](https://pytorch.org/vision/main/models/googlenet_quant.html)

## model performance

- Top-1 on CPU

| config    | accuracy [%] |
|:---------:|-------------:|
| float     |       62.458 |
| quantized |       61.226 |

## how to deploy model

```shell
# check if model compatibility
$ python run.py inspect

# test float model performance:
$ python run.py float

# quantize model (calibration):
$ python run.py calib

# deployment:
$ python run.py deploy
$ xcompiler -i quantize_result/QuantizableGoogLeNet_int.xmodel -o googlenetq.xmodel -f <DPU config finger print value>
```

```shell-session
$ xcompiler -i quantize_result/GoogLeNet_int.xmodel -o googlenet.xmodel -f 0x101000016010405
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B4096_0101000036010407
[UNILOG][INFO] Graph name: QuantizableGoogLeNet, with op num: 423
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/vai/pt_googlenet_quant/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/vai/pt_googlenet_quant/googlenetq.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 8b549308bcea5ff04e6e63d61635e02b, and has been saved to "/home/imagingtechnerd/vai/pt_googlenet_quant/md5sum.txt"```
