# Installing Vitis AI WITHOUT using Docker

- Vitis AI version: v1.2.1

## Get sources

```shell-session
# Vitis AI repo
$ git clone https://github.com/Xilinx/Vitis-AI.git -b v1.2.1

# Compiler etc.
$ wget -O conda-channel.tar.gz https://www.xilinx.com/bin/public/openDownload?filename=conda-channel_1.2.tar.gz
$ tar xf conda-channel.tar.gz

$ export SCRATCH=$(pwd)/Vitis-AI/docker/
$ export {MY_CONDA_CHANNEL}=file://$(pwd)/conda-channel/linux-64/
```

## Install

- CPU-only version

```shell-session
# Caffe
$ conda create -n vitis-ai-caffe \
python=3.6 caffe_decent \
--file ${SCRATCH}/conda_requirements.txt \
-c ${MY_CONDA_CHANNEL} -c defaults -c conda-forge/label/gcc7 && \
conda activate vitis-ai-caffe && pip install -r ${SCRATCH}/pip_requirements.txt && \
conda deactivate

# Tensorflow
$ conda create -n vitis-ai-tensorflow \
python=3.6 vai_q_tensorflow keras \
--file ${SCRATCH}/conda_requirements.txt \
-c ${MY_CONDA_CHANNEL} -c defaults -c conda-forge/label/gcc7 && \
conda activate vitis-ai-tensorflow && pip install -r ${SCRATCH}/pip_requirements.txt && \
conda deactivate
```
