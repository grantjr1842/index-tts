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

echo "Syncing dependencies with extras: deepspeed, flash-attn"
uv sync --extra deepspeed --extra flash-attn

echo "Optimization dependencies installed successfully."
