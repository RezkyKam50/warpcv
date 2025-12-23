#!/bin/bash

source /etc/os-release

if [[ "$ID" == "arch" ]]; then
    export CUDA_HOME=/opt/cuda
    export CUDA_PATH=/opt/cuda
    export PATH=/opt/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/opt/cuda/lib64:${LD_LIBRARY_PATH:-}

    export CUDNN_HOME=/usr
    export CUDNN_INCLUDE_DIR=/usr/include
    export CUDNN_LIBRARY=/usr/lib/libcudnn.so

    export TENSORRT_HOME=/usr
    export TRT_INCLUDE_DIR=/usr/include
    export TRT_LIBRARY_DIR=/usr/lib

elif [[ "$ID" == "fedora" ]]; then
    export CUDA_HOME=/usr/local/cuda
    export CUDA_PATH=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

    export CUDNN_HOME=/usr
    export CUDNN_INCLUDE_DIR=/usr/include
    export CUDNN_LIBRARY=/usr/lib64/libcudnn.so

    export TENSORRT_HOME=/usr
    export TRT_INCLUDE_DIR=/usr/include
    export TRT_LIBRARY_DIR=/usr/lib64

elif [[ "$ID" == "rhel" || "$ID" == "centos" ]]; then
    export CUDA_HOME=/usr/local/cuda
    export CUDA_PATH=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

    export CUDNN_HOME=/usr
    export CUDNN_INCLUDE_DIR=/usr/include
    export CUDNN_LIBRARY=/usr/lib64/libcudnn.so

    export TENSORRT_HOME=/usr
    export TRT_INCLUDE_DIR=/usr/include
    export TRT_LIBRARY_DIR=/usr/lib64

elif [[ "$ID" == "ubuntu" || "$ID" == "debian" ]]; then
    export CUDA_HOME=/usr/local/cuda
    export CUDA_PATH=/usr/local/cuda
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

    export CUDNN_HOME=/usr
    export CUDNN_INCLUDE_DIR=/usr/include
    export CUDNN_LIBRARY=/usr/lib/x86_64-linux-gnu/libcudnn.so

    export TENSORRT_HOME=/usr
    export TRT_INCLUDE_DIR=/usr/include
    export TRT_LIBRARY_DIR=/usr/lib/x86_64-linux-gnu

else
    echo "Unsupported or unknown OS: $ID"
fi
