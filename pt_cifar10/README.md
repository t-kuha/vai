# CIFAR10 based on Kaggle model 

- Base model: [Trained models for CIFAR-10 dataset](https://www.kaggle.com/datasets/firuzjuraev/trained-models-for-cifar10-dataset)

## model performance

- Top-1 accuracy

| config    | acc. (float) [%] | acc. (qaunt.) [%] |
|:---------:|-------------:|-------------:|
| DensNet   |        95.10 |        91.95 |
| Inception |        94.77 |        94.65 |
| ResNet50  |        95.30 |        95.25 |
| VGG19     |        93.18 |        92.89 |
| Xception  |        93.49 |        94.34 |

## how to deploy model

- Download model data (`archive.zip`) from the link above & unzip it

```shell
# check model's compatibility with Vitis AI:
$ python run.py <model name> inspact

# test float model performance:
$ python run.py <model name> float

# quantize model (calibration):
$ python run.py <model name> calib

# deployment:
$ python run.py <model name> deploy
[UNILOG][INFO] Compile mode: dpu
[UNILOG][INFO] Debug mode: null
[UNILOG][INFO] Target architecture: DPUCZDX8G_ISA1_B2304_0101000016010405
[UNILOG][INFO] Graph name: VGG, with op num: 138
[UNILOG][INFO] Begin to compile...
[UNILOG][WARNING] xir::Op{name = VGG__VGG_Sequential_features__AvgPool2d_53__3061_fix, type = pool-fix}'s input and output is unchanged, so it will be removed.
[UNILOG][INFO] Total device subgraph number 3, DPU subgraph number 1
[UNILOG][INFO] Compile done.
[UNILOG][INFO] The meta json is saved to "/home/imagingtechnerd/vai/pt_cifar10_vgg19/meta.json"
[UNILOG][INFO] The compiled xmodel is saved to "/home/imagingtechnerd/vai/pt_cifar10_vgg19/vgg19_cifar10.xmodel"
[UNILOG][INFO] The compiled xmodel's md5sum is d2ca26cfc4559443b5c425062b8ec5ef, and has been saved to "/home/imagingtechnerd/vai/pt_cifar10_vgg19/md5sum.txt"
```
