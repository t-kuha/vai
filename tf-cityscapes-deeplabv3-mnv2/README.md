# Deeplab v2 Mobilenet v2 for Cityscapes Dataset

## Get the original model

```shell-session
$ wget http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz
$ tar xf deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz
```

## Quantization

```shell-session
$ vai_q_tensorflow quantize \
--input_frozen_graph deeplabv3_mnv2_cityscapes_train/frozen_inference_graph.pb \
--input_nodes sub_7 --output_nodes ResizeBilinear_2 --input_shapes 1,1025,2049,3 \
--input_fn module.calib_input \
--calib_iter 100 \
--gpu 0
```

# Deploy

```shell-session
$ vai_c_tensorflow \
-f quantized/deploy_model.pb \
-a ../arch/ultra96v2/arch.json -o _deploy -n deeplab -q
```

- Output:

```shell-session

```
