# Vitis AI without Docker

- Vitis AI version: v3.5

***

## Requisite

- install Miniconda

```shell
$ wget 

$ conda create -n vitisai-pytorch
$ conda create -n vitisai-tf1
$ conda create -n vitisai-tf2
```

- get 

```shell
$ wget -O conda-channel.tar.gz \
https://www.xilinx.com/bin/public/openDownload?filename=conda-channel-3.5.0.tar.gz
$ tar xf conda-channel.tar.gz
```


## Pytorch

```shell
$ conda activate vitisai-pytorch

# Python v3.8.18
$ conda install python==3.8.*
$ conda install libprotobuf=3.21.* protobuf=4.21.*
$ pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
# for runtime & quantizer
# glog: v0.6.0 is necessary for xcompiler
$ conda install -c conda-forge glog
$ conda install cmake ninja libboost pybind11
# other necessary packages
$ conda install tqdm pyyaml
$ pip install graphviz

# clean-up
$ conda clean --all
$ pip cache purge
```

```shell
$ cd Vitis-AI/src/vai_runtime
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# unilog
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}
# xir
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH} --build-python
# target-factory
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}

# xnnc
$ chmod +x ./pip_pkg.sh
$ ./pip_pkg.sh

# vai_q_pytorch / pytorch_binding

```

## TensorFlow 2

```shell
$ conda activate vitisai-tf2

# Python v3.8.18
$ conda install python==3.8.*
# tensorflow along with libprotobuf 3.21.12
$ conda install -c conda-forge tensorflow==2.12.1
# for runtime & quantizer
# glog: v0.6.0 is necessary for xcompiler
$ conda install -c conda-forge glog
$ conda install cmake ninja libboost pybind11 swig
# other necessary packages
$ conda install tqdm
$ pip install graphviz

# pip install graphviz
$ conda clean --all
$ pip cache purge
```

```shell
$ cd Vitis-AI/src/vai_runtime
$ export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

# unilog
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}
# xir
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH} --build-python
# target-facttory
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}

# xnnc
$ chmod +x ./pip_pkg.sh
$ ./pip_pkg.sh

# quantizer (need >= 16GB RAM)
$ ./build.sh --build_with_cpu --conda --type=release --clean
$ pip install pkgs/vai_q_tensorflow2-3.5.0-py2.py3-none-any.whl
```

## TensorFlow 1

```shell
$ conda activate vitisai-tf1

# Python v3.6.13
$ conda install python==3.6.*
# tensorflow along with libprotobuf 3.21.12
# $ conda install -c conda-forge tensorflow==1.15.*
$ pip install tensorflow==1.15.*
$ conda install libprotobuf=3.8.*
$ conda install cmake ninja swig libboost pybind11
# other necessary packages
$ conda install tqdm
$ pip install graphviz

# unilog
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}
# xir
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH} --build-python
# target-facttory
$ ./cmake.sh --type=release --install-prefix=${CMAKE_PREFIX_PATH}

# vai_q_tensorflow1.x
$ ./build.sh --build_with_cpu --conda -type=release
```