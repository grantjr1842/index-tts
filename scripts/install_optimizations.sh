#!/bin/bash
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
if command -v python3 &> /dev/null; then
    # Get capability as "X.Y"
    GPU_CAP=$(python3 -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "")
    
    if [ -n "$GPU_CAP" ]; then
        echo "Detected GPU compute capability: $GPU_CAP"
        export TORCH_CUDA_ARCH_LIST="$GPU_CAP"
        echo "Set TORCH_CUDA_ARCH_LIST=$GPU_CAP"
    else
        echo "Could not detect GPU capabilities (torch cuda not available?). Using default."
    fi
elif command -v python &> /dev/null; then
    GPU_CAP=$(python -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'{cap[0]}.{cap[1]}')" 2>/dev/null || echo "")
    if [ -n "$GPU_CAP" ]; then
        echo "Detected GPU compute capability: $GPU_CAP"
        export TORCH_CUDA_ARCH_LIST="$GPU_CAP"
        echo "Set TORCH_CUDA_ARCH_LIST=$GPU_CAP"
    fi
fi

echo "Syncing dependencies with extras: deepspeed, flash-attn"
uv sync --extra deepspeed --extra flash-attn

echo "Optimization dependencies installed successfully."
