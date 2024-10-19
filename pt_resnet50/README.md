# ResNet50 (PyTorch version)

- Download one of the following package & unzip it
    - https://www.xilinx.com/bin/public/openDownload?filename=pt_resnet50_0.3_3.5.zip
    - https://www.xilinx.com/bin/public/openDownload?filename=pt_resnet50_0.4_3.5.zip
    - https://www.xilinx.com/bin/public/openDownload?filename=pt_resnet50_0.5_3.5.zip
    - https://www.xilinx.com/bin/public/openDownload?filename=pt_resnet50_0.6_3.5.zip
    - https://www.xilinx.com/bin/public/openDownload?filename=pt_resnet50_0.7_3.5.zip
    - https://www.xilinx.com/bin/public/openDownload?filename=pt_resnet50_3.5.zip


# deployment

```shell-session
$ xcompiler -i <unzipped folder>/quantized/ResNet_0_int.xmodel -o resnet50.xmodel -f <DPU config finger print value>
```

- Example output

```shell-session
$ xcompiler -i pt_resnet50_0.5_3.5/quantized/ResNet_0_int.xmodel -o resnet50.xmodel -f 0x101000016010405
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
[UNILOG][INFO] Graph name: ResNet_0, with op num: 417
[UNILOG][INFO] Begin to compile...
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/work/vai/pt_resnet50/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/work/vai/pt_resnet50/resnet50.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is 87220a1cf2e4018b8ff846fb3a7496dd, and has been saved to "/home/imagingtechnerd/work/vai/pt_resnet50/md5sum.txt"
```
