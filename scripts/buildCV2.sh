#!/bin/bash
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH

PY_ABI="3.13"
export PATH="/usr/bin:$PATH"
source ./scripts/getCompiler.sh
source ./scripts/getCuda.sh

source .venv/bin/activate
 
rm -rf build_opencv
mkdir -p build_opencv
cd build_opencv

C_FLAGS="-O3 -march=native -mtune=native"
CXX_FLAGS="-O3 -march=native -mtune=native"

CUDA_FLAGS="\
-O3 \
--use_fast_math"

cmake ../3rdparty/opencv \
  -G Ninja \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D CMAKE_BUILD_TYPE=Release \
  -D PYTHON3_EXECUTABLE=$(which python3) \
  -D PYTHON_LIBRARIES=/usr/local/lib/libpython${PY_ABI}.so \
  -D PYTHON3_INCLUDE_DIR=/usr/local/include/python${PY_ABI} \
  -D PYTHON3_LIBRARY=/usr/local/lib/libpython${PY_ABI}.so \
  -D PYTHON3_PACKAGES_PATH=$(python3 -c "import site; print(site.getsitepackages()[0])") \
  -D BUILD_SHARED_LIBS=ON \
  -D BUILD_opencv_python3=ON \
  -D BUILD_opencv_python_bindings_generator=ON \
  -D ENABLE_CCACHE=OFF \
  -D WITH_CUDA=ON \
  -D CMAKE_C_FLAGS="$C_FLAGS" \
  -D CMAKE_CXX_FLAGS="$CXX_FLAGS" \
  -D CMAKE_CUDA_FLAGS="$CUDA_FLAGS" \
  -D ENABLE_CCACHE=ON \
  -D OPENCV_EXTRA_MODULES_PATH=../3rdparty/opencv_contrib/modules \
  -D CMAKE_INSTALL_PREFIX=$VIRTUAL_ENV 

ninja -j$(nproc)
ninja install

export PYTHONPATH=$PYTHONPATH:$HOME/WarpCV/build_opencv/lib/python3