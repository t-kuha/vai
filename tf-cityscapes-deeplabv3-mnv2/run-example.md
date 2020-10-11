# How to run TensorFlow DeepLab example

- TensorFlow Version: 1.15.3

## Prerequisite

```shell-session
$ export WORKDIR=$(pwd)
```

## Get pre-trained model

```shell-session
$ wget http://download.tensorflow.org/models/deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz
$ tar xf deeplabv3_mnv2_cityscapes_train_2018_02_05.tar.gz
$ echo 'model_checkpoint_path: "model.ckpt"' > ${WORKDIR}/deeplabv3_mnv2_cityscapes_train/checkpoint
$ echo 'all_model_checkpoint_paths: "model.ckpt"' >> ${WORKDIR}/deeplabv3_mnv2_cityscapes_train/checkpoint
```

## Get dataset

```shell-session
$ unzip -q ${WORKDIR}/leftImg8bit_trainvaltest.zip
$ unzip -qo ${WORKDIR}/gtFine_trainvaltest.zip
```

## Get sources

```shell-session
$ git clone https://github.com/tensorflow/models.git -b v1.13.0
$ git clone https://github.com/mcordts/cityscapesScripts.git
```

## Prepare TFRecord

```shell-session
$ mkdir -p ${WORKDIR}/models/research/deeplab/datasets/cityscapes/tfrecord
$ mv ${WORKDIR}/leftImg8bit/ ${WORKDIR}/models/research/deeplab/datasets/cityscapes
$ mv ${WORKDIR}/gtFine/ ${WORKDIR}/models/research/deeplab/datasets/cityscapes
$ mv ${WORKDIR}/cityscapesScripts/cityscapesscripts/ ${WORKDIR}/models/research/deeplab/datasets/cityscapes
$ cd ${WORKDIR}/models/research/deeplab/datasets/

$ PYTHONPATH=${WORKDIR}/models/research/deeplab/datasets/cityscapes \
python cityscapes/cityscapesscripts/preparation/createTrainIdLabelImgs.py

$ python build_cityscapes_data.py \
--cityscapes_root=./cityscapes \
--output_dir=cityscapes/tfrecord \
```

## Run inference

```shell-session
$ cd ${WORKDIR}/models/research/
$ PYTHONPATH=${PYTHONPATH}:$(pwd):$(pwd)/slim python deeplab/eval.py \
--logtostderr --eval_split="val" \
--model_variant="mobilenet_v2" \
--eval_crop_size=1025 --eval_crop_size=2049 \
--dataset="cityscapes" \
--eval_logdir=_eval_log \
--max_number_of_iterations=1 \
--dataset_dir=${WORKDIR}/models/research/deeplab/datasets/cityscapes/tfrecord \
--checkpoint_dir=${WORKDIR}/deeplabv3_mnv2_cityscapes_train
```

## Result

- mean IOU: 0.706700146
