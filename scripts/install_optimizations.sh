#!/usr/bin/env bash

set -e

# Navigate to project root
cd "$(dirname "$0")/.."

echo "Installing optimization dependencies (Deepspeed, Flash Attention)..."
echo "This may take a while as it compiles from source."

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "uv not found. Please install uv first."
    exit 1
fi

# Auto-detect GPU architecture to fix build warnings
echo "Detecting GPU architecture..."
GPU_CAP=""

# Try nvidia-smi first (doesn't require python/torch installed)
if command -v nvidia-smi &> /dev/null; then
    GPU_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -n 1 | tr -d ' ')
    if [ -n "$GPU_CAP" ]; then
        echo "Detected GPU compute capability (via nvidia-smi): $GPU_CAP"
    fi
fi

# Fallback to python if nvidia-smi failed or returned empty
if [ -z "$GPU_CAP" ]; then
    if command -v python3 &> /dev/null; then
        GPU_CAP=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "")
    elif command -v, python &> /dev/null; then
        GPU_CAP=$(python -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "")
    fi
    if [ -n "$GPU_CAP" ]; then
        echo "Detected GPU compute capability (via python): $GPU_CAP"
    fi
fi

if [ -n "$GPU_CAP" ]; then
    export TORCH_CUDA_ARCH_LIST="$GPU_CAP"
    echo "Set TORCH_CUDA_ARCH_LIST=$GPU_CAP"
    
    # Check for Turing (7.5) to enable experimental FlashAttention build
    if [ "$GPU_CAP" == "7.5" ]; then
            echo "Turing GPU detected. Enabling experimental FlashAttention build (Force Source Build)."
            export FLASH_ATTENTION_FORCE_BUILD="TRUE"
            # Pass flag to uv sync
            UV_FLAGS="--no-binary-package flash-attn"
    fi
else
    echo "Could not detect GPU capabilities (nvidia-smi failed and torch cuda not available?). Using default."
fi

echo "Syncing dependencies with extras: deepspeed, flash-attn"
uv sync --extra deepspeed --extra flash-attn $UV_FLAGS --verbose

echo "Optimization dependencies installed successfully."
