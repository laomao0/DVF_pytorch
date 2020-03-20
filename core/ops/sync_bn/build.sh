#!/usr/bin/env bash
echo "Compiling sync_bn_kernel kernels..."
if [ -f src/cuda/sync_bn_kernel.o ]; then
    rm src/cuda/sync_bn_kernel.o
fi
if [ -d _ext ]; then
    rm -rf _ext
fi

TORCH=$(python -c "import os; import torch; print(os.path.dirname(torch.__file__))")
echo ${TORCH}
echo ${TORCH}/include/TH


cd src/cuda
nvcc -c -o sync_bn_kernel.o sync_bn_kernel.cu \
     -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -arch=sm_75 -I ${TORCH}/include/TH -I ${TORCH}/include/THC -I ${TORCH}/include

# cd ../../
# python3 build.py
