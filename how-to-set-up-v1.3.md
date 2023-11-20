# Installing Vitis AI WITHOUT using Docker

- Vitis AI version: v1.3
- Only for edge (Zynq) environment

## Get sources

```shell-session
# Create workspace directory
$ mkdir vai && cd vai

# Install miniconda (anaconda)
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ /bin/bash ./Miniconda3-latest-Linux-x86_64.sh -b -p <conda installation path>

# Clone Vitis AI repo
$ git clone https://github.com/Xilinx/Vitis-AI.git -b v1.3
# Change for installation w/o docker
$ sed -i 's|/scratch|'"$(pwd)"'|' Vitis-AI/setup/docker/docker/cpu_conda/vitis-ai-*.yml
$ sed -i 's|/scratch|'"$(pwd)"'|' Vitis-AI/setup/docker/docker/gpu_conda/vitis-ai-*.yml

# Compiler etc.
$ wget https://www.xilinx.com/bin/public/openDownload?filename=conda-channel_1.3.411-01.tar.gz \
-O conda-channel.tar.gz
$ tar xf conda-channel.tar.gz

$ mkdir scratch
$ cp Vitis-AI/setup/docker/docker/cpu_conda/*.yml scratch/
$ cp Vitis-AI/setup/docker/docker/gpu_conda/*.yml scratch/
$ cp Vitis-AI/setup/docker/docker/pip_requirements.txt scratch/
$ cp Vitis-AI/setup/docker/docker/pip_requirements_neptune.txt scratch/
```

## Install

- CPU-only version

```shell-session
$ . <conda installation path>/etc/profile.d/conda.sh

$ conda env create -f scratch/vitis-ai-pytorch.yml \
&& conda activate vitis-ai-pytorch \
&& pip install -r scratch/pip_requirements.txt

$ conda env create -f scratch/vitis-ai-caffe.yml \
&& conda activate vitis-ai-caffe \
&& pip install -r scratch/pip_requirements.txt

$ conda env create -f scratch/vitis-ai-tensorflow.yml \
&& conda activate vitis-ai-tensorflow \
&& pip install -r scratch/pip_requirements.txt

$ conda env create -f scratch/vitis-ai-tensorflow2.yml \
&& conda activate vitis-ai-tensorflow2 \
&& pip install -r scratch/pip_requirements.txt \
&& pip install --ignore-installed tensorflow==2.3.0
```

- GPU version

```shell-session
$ . <conda installation path>/etc/profile.d/conda.sh

$ conda env create -f scratch/vitis-ai-optimizer_darknet.yml
$ conda env create -f scratch/vitis-ai-optimizer_caffe.yml
$ conda env create -f scratch/vitis-ai-optimizer_tensorflow.yml

$ conda env create -f scratch/vitis-ai-lstm.yml \
&& conda activate vitis-ai-lstm \
&& pip install -r scratch/pip_requirements.txt
        
$ conda env create -f scratch/vitis-ai-pytorch.yml \
&& conda activate vitis-ai-pytorch \
&& pip install -r scratch/pip_requirements.txt

$ conda env create -f scratch/vitis-ai-caffe.yml \
&& conda activate vitis-ai-caffe \
&& pip install -r scratch/pip_requirements.txt

$ conda env create -f scratch/vitis-ai-tensorflow.yml \
&& conda activate vitis-ai-tensorflow \
&& pip install -r scratch/pip_requirements.txt

$ conda env create -f scratch/vitis-ai-tensorflow2.yml \
&& conda activate vitis-ai-tensorflow2 \
&& pip install -r scratch/pip_requirements.txt \
&& pip install --ignore-installed tensorflow==2.3.0
```

## Optional - free up disk space

```shell-session
$ conda clean --all
$ cd .. && rm -r vai
```
