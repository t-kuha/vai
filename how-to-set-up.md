# Setting up Vitis AI without using Docker

- Ubuntu: 22.04.5 LTS
- Vitis AI version: v3.5

***

## Pre-requisite

- Install Miniconda

```shell
$ wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
$ chmod +x ./Miniconda3-latest-Linux-x86_64.sh
$ ./Miniconda3-latest-Linux-x86_64.sh

$ conda create -n vai-pt
$ conda create -n vai-tf2
$ conda create -n vai-tf1
```

- Get Vitis AI conda packages

```shell
$ wget -O conda-channel.tar.gz \
https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-3.5.0.tar.gz
$ tar xf conda-channel.tar.gz
$ export VAI_CONDA_CHANNEL="file://$(pwd)/conda-channel"
$ conda config --env --append channels ${VAI_CONDA_CHANNEL}
```

- Donwload Vitis-AI repo

```shell
$ git clone https://github.com/Xilinx/Vitis-AI.git
$ pushd Vitis-AI
$ git checkout 1eed93cde2ff3ed841ddb03dff0714b4e2d34f5b
$ popd
```

## Pytorch

```shell
$ conda activate vai-pt

# install necessary packages
$ conda install python=3.8.18  # 3.8.18 only?
$ conda install numpy=1.24.2 tqdm pyyaml cmake ninja
$ pip install torch==2.0.0+cpu torchvision==0.15.1+cpu --index-url https://download.pytorch.org/whl/cpu
# glog: v0.6.0 is necessary for xcompiler
$ conda install -c conda-forge glog=0.6
$ conda install xcompiler

# need to build pytorch_nndct from source for PyTorch >= 2.0 
$ pushd Vitis-AI/src/vai_quantizer/vai_q_pytorch/pytorch_binding
$ python setup.py sdist bdist_wheel -d _dist
$ pip install _dist/pytorch_nndct-3.5.0-cp38-cp38-linux_x86_64.whl
$ popd

# clean-up
$ conda clean --all
$ pip cache purge
```

## TensorFlow 2

```shell
$ conda activate vai-tf2

# install necessary packages
$ conda install python==3.8.18  # Python v3.8.18
# glog: v0.6.0 is necessary for xcompiler
$ conda install -c conda-forge glog=0.6
# other necessary packages
$ conda install tqdm
$ pip install numpy==1.22.1
# tensorflow along with libprotobuf 3.21.12
$ pip install protobuf==3.20.3
$ pip install tensorflow==2.12 keras==2.12

# clean-up
$ conda clean --all
$ pip cache purge
```

## TensorFlow 1

```shell
$ conda activate vai-tf1

# install necessary packages
$ conda install python==3.6.*  # Python v3.6.13
$ conda install tqdm libprotobuf=3.8.*
$ conda install -c conda-forge glog=0.6
$ pip install numpy==1.18.*
$ pip install tensorflow==1.15.*
$ conda install xcompiler xnnc
$ conda install vaic

# clean-up
$ conda clean --all
$ pip cache purge
```

***

## How to build packages from source (optional)

```shell
$ pushd Vitis-AI/src/vai_runtime
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# unilog
$ pushd unilog
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}
$ popd

# xir
$ pushd xir
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH} --build-python
$ popd

# target-factory
$ pushd target_factory/
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}
$ popd

# xnnc
$ pushd ../vai_quantizer/xnnc4xir/
$ chmod +x ./pip_pkg.sh
$ ./pip_pkg.sh
$ popd
```
