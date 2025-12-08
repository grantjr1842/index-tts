#!/usr/bin/env python3
"""Debug script to check device placement after INT8 quantization."""

import torch
import sys
sys.path.insert(0, "/home/admin-grant-jr/github/index-tts")

from transformers import Wav2Vec2BertModel
from indextts.utils.vram_utils import quantize_model_int8

print("Loading wav2vec2-bert model...")
model = Wav2Vec2BertModel.from_pretrained(
    "facebook/w2v-bert-2.0",
    attn_implementation="eager",
)

# Move to GPU first
model = model.to("cuda:0")
model.eval()

print("\nBefore quantization - checking device of all parameters:")
devices = set()
for name, param in model.named_parameters():
    devices.add(str(param.device))
    if len(devices) > 1:
        print(f"  {name}: {param.device}")
print(f"  Unique devices: {devices}")

print("\nApplying INT8 quantization...")
quantized = quantize_model_int8(model)

print("\nAfter quantization - checking device of all parameters:")
devices = set()
cuda_params = []
for name, param in quantized.named_parameters():
    devices.add(str(param.device))
    if 'cuda' in str(param.device):
        cuda_params.append(name)
print(f"  Unique devices: {devices}")
if cuda_params:
    print(f"  Still on CUDA: {len(cuda_params)} params")
    for p in cuda_params[:5]:
        print(f"    - {p}")
    if len(cuda_params) > 5:
        print(f"    ... and {len(cuda_params)-5} more")

print("\nAfter quantization - checking device of all buffers:")
buffer_devices = set()
for name, buf in quantized.named_buffers():
    buffer_devices.add(str(buf.device))
print(f"  Unique buffer devices: {buffer_devices}")

print("\nTest inference:")
try:
    dummy_input = torch.randn(1, 100, 160).cpu()  # CPU input
    dummy_mask = torch.ones(1, 100).cpu()
    output = quantized(input_features=dummy_input, attention_mask=dummy_mask)
    print("  ✅ CPU inference successful!")
except Exception as e:
    print(f"  ❌ CPU inference failed: {e}")
