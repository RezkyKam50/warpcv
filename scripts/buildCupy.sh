#!/bin/bash

source ./scripts/getCompiler.sh
source ./scripts/getCuda.sh
source .venv/bin/activate

cd ./3rdparty/cupy
echo "removing previous build artifacts ..."
sudo rm -rf build
echo "removing previous cupy-egg-info ..."
sudo rm -rf cupy-egg-info
uv pip install -e . --no-build-isolation --verbose --force-reinstall