#!/bin/bash
set -e

# Set up Python environment
export UV_PYTHON_DIR="/home/admin-grant-jr/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$UV_PYTHON_DIR/lib:$LD_LIBRARY_PATH"
export PYTHONHOME="$UV_PYTHON_DIR"
export PYTHONPATH="$(pwd)/.venv/lib/python3.10/site-packages:$(pwd)"

echo "Building server..."
cd server
PYO3_PYTHON=/home/admin-grant-jr/github/index-tts/.venv/bin/python cargo build --release
cd ..

echo "Starting server with optimizations..."
export TORCH_CUDA_ARCH_LIST="7.5"
export PATH="$(pwd)/.venv/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TARS_DEVICE="cuda:0"
export TARS_FP16=1
export TARS_TORCH_COMPILE=1
export TARS_ACCEL=1
export TARS_DEEPSPEED=1
export TARS_WARMUP=1

./server/target/release/server
