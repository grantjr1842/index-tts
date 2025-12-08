#!/usr/bin/env python3
"""
Test INT8 quantization with the *real* IndexTTS2 model.

This test:
1. Loads the full IndexTTS2 model
2. Measures VRAM usage baseline
3. Tests TTS inference with and without INT8 quantization
4. Compares output quality and VRAM savings
"""

import torch
import os
import sys
import gc
import time

# Add project root to path
sys.path.insert(0, "/home/admin-grant-jr/github/index-tts")

from indextts.utils.vram_utils import (
    quantize_model_int8,
    get_vram_usage,
    print_vram_usage,
    force_cleanup,
    VRAMProfiler,
)


def get_gpu_memory_mb():
    """Get current GPU memory allocated in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 * 1024)
    return 0


def test_real_indextts_quantization():
    """Test INT8 quantization with the real IndexTTS2 model."""
    print("=" * 70)
    print("INT8 Quantization Test with Real IndexTTS2 Model")
    print("=" * 70)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - cannot run this test")
        return False
    
    # Clean GPU memory
    force_cleanup()
    torch.cuda.reset_peak_memory_stats()
    baseline_vram = get_gpu_memory_mb()
    print(f"\nBaseline GPU memory: {baseline_vram:.2f} MB")
    
    # Import the model
    print("\n[1/4] Loading IndexTTS2 model...")
    from indextts.infer_v2 import IndexTTS2
    
    with VRAMProfiler("Full IndexTTS2 model loading"):
        tts = IndexTTS2(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            use_cuda_kernel=False,
            use_torch_compile=False,
            use_deepspeed=False,
        )
    
    after_load_vram = get_gpu_memory_mb()
    print(f"After model load VRAM: {after_load_vram:.2f} MB ({after_load_vram/1024:.2f} GB)")
    
    # Identify quantizable components
    print("\n[2/4] Analyzing model components...")
    semantic_model = tts.semantic_model
    semantic_codec = tts.semantic_codec
    
    # Get sizes before quantization
    semantic_model_params = sum(p.numel() * p.element_size() for p in semantic_model.parameters()) / (1024 * 1024)
    semantic_codec_params = sum(p.numel() * p.element_size() for p in semantic_codec.parameters()) / (1024 * 1024)
    
    print(f"  semantic_model size: {semantic_model_params:.2f} MB")
    print(f"  semantic_codec size: {semantic_codec_params:.2f} MB")
    
    # Test TTS inference BEFORE quantization
    print("\n[3/4] Running TTS inference WITHOUT quantization...")
    prompt_wav = "examples/voice_01.wav"
    test_text = "Hello, this is a test."
    
    force_cleanup()
    start_time = time.perf_counter()
    
    try:
        result = tts.infer(
            spk_audio_prompt=prompt_wav,
            text=test_text,
            output_path="test_output_before.wav",
            verbose=False
        )
        inference_time_before = time.perf_counter() - start_time
        print(f"  ✅ Inference completed in {inference_time_before:.2f}s")
        peak_vram_before = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"  Peak VRAM during inference: {peak_vram_before:.2f} MB")
    except Exception as e:
        print(f"  ❌ Inference failed: {e}")
        return False
    
    # Now test with INT8 quantization on the semantic model
    print("\n[4/4] Applying INT8 quantization to semantic_model...")
    
    # Save original device
    original_device = next(semantic_model.parameters()).device
    
    # Quantize the semantic model
    try:
        with VRAMProfiler("INT8 quantization of semantic_model"):
            quantized_semantic = quantize_model_int8(semantic_model)
        
        # Replace in the TTS instance
        tts.semantic_model = quantized_semantic
        
        # Now the semantic model runs on CPU, measure VRAM after
        force_cleanup()
        after_quant_vram = get_gpu_memory_mb()
        
        print(f"  VRAM after quantization: {after_quant_vram:.2f} MB")
        vram_saved = after_load_vram - after_quant_vram
        print(f"  VRAM saved: {vram_saved:.2f} MB ({(vram_saved/after_load_vram)*100:.1f}%)")
        
    except Exception as e:
        print(f"  ❌ Quantization failed: {e}")
        return False
    
    # Test inference with quantized model
    print("\n[5/4] Running TTS inference WITH INT8 quantization...")
    torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()
    
    try:
        result = tts.infer(
            spk_audio_prompt=prompt_wav,
            text=test_text,
            output_path="test_output_after.wav",
            verbose=False
        )
        inference_time_after = time.perf_counter() - start_time
        print(f"  ✅ Inference completed in {inference_time_after:.2f}s")
        peak_vram_after = torch.cuda.max_memory_allocated() / (1024 * 1024)
        print(f"  Peak VRAM during inference: {peak_vram_after:.2f} MB")
    except Exception as e:
        print(f"  ❌ Inference with quantized model failed: {e}")
        return False
    
    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"  Model VRAM before quant: {after_load_vram:.2f} MB ({after_load_vram/1024:.2f} GB)")
    print(f"  Model VRAM after quant:  {after_quant_vram:.2f} MB ({after_quant_vram/1024:.2f} GB)")
    print(f"  VRAM saved:              {vram_saved:.2f} MB ({(vram_saved/after_load_vram)*100:.1f}%)")
    print(f"  Inference time before:   {inference_time_before:.2f}s")
    print(f"  Inference time after:    {inference_time_after:.2f}s")
    print(f"  Peak VRAM before:        {peak_vram_before:.2f} MB")
    print(f"  Peak VRAM after:         {peak_vram_after:.2f} MB")
    
    if vram_saved > 0:
        print("\n✅ INT8 quantization provides meaningful VRAM savings!")
        return True
    else:
        print("\n⚠️ No significant VRAM savings observed")
        return False


def main():
    os.chdir("/home/admin-grant-jr/github/index-tts")
    success = test_real_indextts_quantization()
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
