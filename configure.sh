#!/bin/bash

CUDA_V=V13.0.88

echo "checking uv installation ..."
if command -v uv &>/dev/null; then
    echo "VERIFIED: uv is installed: $(uv --version)"
    ((verified_count++))
else
    echo "Cannot Install WarpCV: uv is NOT installed"
    exit 1
fi
echo
echo "checking CUDA ${CUDA_V} ..."
if command -v nvcc &>/dev/null; then
    cuda_version=$(nvcc --version | grep "release" | awk '{print $6}' | sed 's/,//')
    if [[ "$cuda_version" == $CUDA_V ]]; then
        echo "VERIFIED: CUDA $cuda_version detected"
        ((verified_count++))
    else
        echo "WARN: CUDA detected but version is $cuda_version (expected ${CUDA_V}) which may lead to compatibility issues"
    fi
else
    echo "WARN: CUDA (nvcc) not found"
    exit 1
fi

echo "Installing dependencies ..."
./scripts/installDependencies.sh

echo "Building CuPy ..."
./scripts/buildCupy.sh