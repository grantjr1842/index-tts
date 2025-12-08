#!/bin/bash
set -e

# Set up Python environment
# Use the venv's bin directory for ninja and other tools
export PATH="$(pwd)/.venv/bin:$PATH"

# Point to the uv-managed Python for shared libraries
export UV_PYTHON_DIR="/home/admin-grant-jr/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$UV_PYTHON_DIR/lib:$LD_LIBRARY_PATH"

# Set PYTHONPATH to include our packages
export PYTHONPATH="$(pwd)/.venv/lib/python3.10/site-packages:$(pwd)"

# NOTE: Do NOT set PYTHONHOME - it causes stdlib version conflicts with pyo3

echo "Building server..."
cd server
PYO3_PYTHON=/home/admin-grant-jr/github/index-tts/.venv/bin/python cargo build --release
cd ..

echo "Starting server with optimizations..."
export TARS_DEVICE="cuda:0"
export TARS_FP16=1
export TARS_TORCH_COMPILE=1
export TARS_ACCEL=0
# Disable DeepSpeed for 8GB GPUs - it adds ~100MB workspace overhead that causes OOM
# For GPUs with 12GB+ VRAM, set TARS_DEEPSPEED=1
export TARS_DEEPSPEED=0
export TARS_WARMUP=1

./server/target/release/server
