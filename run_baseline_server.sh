#!/bin/bash
set -e

# Set up Python environment
export UV_PYTHON_DIR="/home/admin-grant-jr/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$UV_PYTHON_DIR/lib:$LD_LIBRARY_PATH"
export PYTHONHOME="$UV_PYTHON_DIR"
export PYTHONPATH="$(pwd)/.venv/lib/python3.10/site-packages:$(pwd)"

echo "Starting server WITHOUT torch.compile..."
export TARS_DEVICE="cuda:0"
export TARS_FP16=1
export TARS_TORCH_COMPILE=0
export TARS_ACCEL=0
export TARS_WARMUP=1

./server/target/release/server
