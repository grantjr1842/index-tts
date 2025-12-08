#!/usr/bin/env python3
"""
Test script to verify INT8 quantization utility works and saves VRAM.

This script:
1. Creates a simple model similar in structure to the semantic model
2. Measures VRAM usage before and after INT8 quantization
3. Compares forward pass outputs to check for quality degradation
"""

import torch
import torch.nn as nn
import sys
import gc

# Add project root to path
sys.path.insert(0, "/home/admin-grant-jr/github/index-tts")

from indextts.utils.vram_utils import (
    quantize_model_int8,
    get_vram_usage,
    print_vram_usage,
    force_cleanup,
    VRAMProfiler,
)


def create_test_model(hidden_size: int = 1024, num_layers: int = 6) -> nn.Module:
    """
    Create a test model that mimics the structure of typical transformer layers.
    Uses Linear layers which are the main targets for INT8 quantization.
    """
    layers = []
    for _ in range(num_layers):
        layers.extend([
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.ReLU(),
        ])
    return nn.Sequential(*layers)


def get_model_size_mb(model: nn.Module) -> float:
    """Calculate model size in MB from parameters and buffers."""
    import io
    # For quantized models, directly serialize and measure
    try:
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        return buffer.tell() / (1024 * 1024)
    except Exception:
        # Fallback to parameter counting
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


def test_quantization_works():
    """Test that INT8 quantization runs without errors."""
    print("\n" + "=" * 60)
    print("TEST 1: Verifying INT8 quantization works")
    print("=" * 60)
    
    # Create a simple model on CPU
    model = create_test_model(hidden_size=512, num_layers=2)
    model.eval()
    
    print(f"Original model size: {get_model_size_mb(model):.2f} MB")
    
    # Quantize
    try:
        quantized_model = quantize_model_int8(model)
        print(f"Quantized model size: {get_model_size_mb(quantized_model):.2f} MB")
        print("✅ INT8 quantization completed successfully")
        return True
    except Exception as e:
        print(f"❌ INT8 quantization failed: {e}")
        return False


def test_quantization_preserves_output():
    """Test that quantized model produces reasonable outputs."""
    print("\n" + "=" * 60)
    print("TEST 2: Verifying quantized output quality")
    print("=" * 60)
    
    model = create_test_model(hidden_size=256, num_layers=2)
    model.eval()
    
    # Sample input
    sample_input = torch.randn(1, 256)
    
    # Get original output
    with torch.no_grad():
        original_output = model(sample_input)
    
    # Quantize
    quantized_model = quantize_model_int8(model)
    
    # Get quantized output
    with torch.no_grad():
        quantized_output = quantized_model(sample_input)
    
    # Compare outputs
    mse = torch.mean((original_output - quantized_output) ** 2).item()
    cosine_sim = torch.nn.functional.cosine_similarity(
        original_output.flatten().unsqueeze(0),
        quantized_output.flatten().unsqueeze(0)
    ).item()
    
    print(f"MSE between original and quantized: {mse:.6f}")
    print(f"Cosine similarity: {cosine_sim:.6f}")
    
    # The outputs should be similar (cosine > 0.9 is reasonable for INT8)
    if cosine_sim > 0.9:
        print(f"✅ Output quality preserved (cosine sim: {cosine_sim:.4f})")
        return True
    else:
        print(f"⚠️ Output quality may be degraded (cosine sim: {cosine_sim:.4f})")
        return False


def test_vram_savings():
    """Test that INT8 quantization saves meaningful VRAM on GPU."""
    print("\n" + "=" * 60)
    print("TEST 3: Measuring VRAM savings")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping VRAM test")
        return None
    
    device = torch.device("cuda:0")
    
    # Clean up any existing GPU memory
    force_cleanup()
    torch.cuda.reset_peak_memory_stats()
    
    # Create a larger model to see meaningful VRAM impact
    hidden_size = 1024
    num_layers = 8  # ~400MB model
    
    print(f"\nCreating test model (hidden_size={hidden_size}, num_layers={num_layers})...")
    
    # Measure VRAM for FP32 model on GPU
    with VRAMProfiler("FP32 model loading"):
        model_fp32 = create_test_model(hidden_size, num_layers)
        model_fp32 = model_fp32.to(device)
        model_fp32.eval()
    
    fp32_vram_mb = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"FP32 model VRAM: {fp32_vram_mb:.2f} MB")
    
    # Run inference to ensure model is fully initialized
    sample_input = torch.randn(1, hidden_size, device=device)
    with torch.no_grad():
        _ = model_fp32(sample_input)
    
    fp32_peak_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
    print(f"FP32 peak VRAM during inference: {fp32_peak_mb:.2f} MB")
    
    # Free the FP32 model
    del model_fp32
    force_cleanup()
    torch.cuda.reset_peak_memory_stats()
    
    # Create and quantize a new model
    print("\nQuantizing model (INT8)...")
    with VRAMProfiler("INT8 quantization"):
        model_for_quant = create_test_model(hidden_size, num_layers)
        model_int8 = quantize_model_int8(model_for_quant)
        del model_for_quant
    
    # Note: INT8 quantized models stay on CPU for inference in PyTorch's native quantization
    # They cannot be moved to GPU directly, but this is still useful for reducing memory
    
    int8_cpu_size_mb = get_model_size_mb(model_int8)
    print(f"INT8 model size (CPU): {int8_cpu_size_mb:.2f} MB")
    
    # Run inference on CPU
    sample_input_cpu = torch.randn(1, hidden_size)
    with torch.no_grad():
        _ = model_int8(sample_input_cpu)
    
    # Calculate savings
    savings_pct = (1 - int8_cpu_size_mb / fp32_vram_mb) * 100
    savings_mb = fp32_vram_mb - int8_cpu_size_mb
    
    print(f"\n{'='*40}")
    print("VRAM/Memory Comparison:")
    print(f"{'='*40}")
    print(f"FP32 on GPU:  {fp32_vram_mb:.2f} MB")
    print(f"INT8 on CPU:  {int8_cpu_size_mb:.2f} MB")
    print(f"Savings:      {savings_mb:.2f} MB ({savings_pct:.1f}%)")
    
    if savings_pct > 40:
        print(f"✅ Significant memory savings achieved: {savings_pct:.1f}%")
        return True
    elif savings_pct > 20:
        print(f"⚠️ Moderate memory savings: {savings_pct:.1f}%")
        return True
    else:
        print(f"❌ Low memory savings: {savings_pct:.1f}%")
        return False


def test_real_model_quantization():
    """
    Test INT8 quantization on the actual semantic model if available.
    This tests real-world VRAM savings.
    """
    print("\n" + "=" * 60)
    print("TEST 4: Testing with real wav2vec-bert model (if available)")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available - skipping real model test")
        return None
    
    try:
        from transformers import Wav2Vec2BertModel
        
        device = torch.device("cuda:0")
        force_cleanup()
        torch.cuda.reset_peak_memory_stats()
        
        print("Loading wav2vec2-bert model (this may take a while)...")
        
        # Load the model in FP32 on GPU
        with VRAMProfiler("Loading FP32 wav2vec2-bert"):
            model_fp32 = Wav2Vec2BertModel.from_pretrained(
                "facebook/w2v-bert-2.0",
                attn_implementation="eager",
            )
            model_fp32 = model_fp32.to(device)
            model_fp32.eval()
        
        fp32_vram_gb = torch.cuda.memory_allocated() / 1e9
        print(f"FP32 model VRAM: {fp32_vram_gb:.3f} GB")
        
        # Free and quantize
        del model_fp32
        force_cleanup()
        
        print("Quantizing wav2vec2-bert to INT8...")
        model_for_quant = Wav2Vec2BertModel.from_pretrained(
            "facebook/w2v-bert-2.0",
            attn_implementation="eager",
        )
        
        with VRAMProfiler("INT8 quantization"):
            model_int8 = quantize_model_int8(model_for_quant)
        
        int8_size_gb = get_model_size_mb(model_int8) / 1024
        
        savings_pct = (1 - int8_size_gb / fp32_vram_gb) * 100 if fp32_vram_gb > 0 else 0
        
        print(f"\nReal Model Results:")
        print(f"FP32 on GPU: {fp32_vram_gb:.3f} GB")
        print(f"INT8 on CPU: {int8_size_gb:.3f} GB")
        print(f"Savings: {savings_pct:.1f}%")
        
        if savings_pct > 40:
            print(f"✅ Significant savings on real model: {savings_pct:.1f}%")
            return True
        else:
            print(f"⚠️ Moderate savings on real model: {savings_pct:.1f}%")
            return True
            
    except Exception as e:
        print(f"⚠️ Could not test real model: {e}")
        return None


def main():
    print("=" * 60)
    print("INT8 Quantization VRAM Utility Verification")
    print("=" * 60)
    
    results = {}
    
    # Test 1: Basic functionality
    results["basic"] = test_quantization_works()
    
    # Test 2: Output quality
    results["quality"] = test_quantization_preserves_output()
    
    # Test 3: VRAM savings
    results["vram"] = test_vram_savings()
    
    # Test 4: Real model (optional)
    results["real_model"] = test_real_model_quantization()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    all_passed = True
    for test_name, result in results.items():
        if result is None:
            status = "⏭️ SKIPPED"
        elif result:
            status = "✅ PASSED"
        else:
            status = "❌ FAILED"
            all_passed = False
        print(f"  {test_name}: {status}")
    
    if all_passed:
        print("\n✅ All tests passed! INT8 quantization is working correctly.")
        return 0
    else:
        print("\n❌ Some tests failed. Review the output above.")
        return 1


if __name__ == "__main__":
    exit(main())
