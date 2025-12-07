
import time
import torch
import sys
import os

# Add repo root to path
sys.path.append(os.getcwd())

from indextts.infer_v2 import IndexTTS2

def benchmark():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize basic config
    # Assuming default paths exist as per file listing
    try:
        model = IndexTTS2(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            device=device,
            use_fp16=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    text = "This is a test sentence for benchmarking the inference speed of Index TTS two. " * 5
    spk_prompt = "interstellar-tars-01-resemble-denoised.wav"
    
    if not os.path.exists(spk_prompt):
        print(f"Speaker prompt {spk_prompt} not found.")
        return

    print("Warming up...")
    # Warmup
    try:
        model.infer(spk_prompt, "Warmup", None, return_audio=True)
    except Exception as e:
        print(f"Warmup failed: {e}")
        # Continue anyway, might allow debugging
    
    print("Starting benchmark...")
    start_time = time.time()
    n_runs = 3
    
    for i in range(n_runs):
        iter_start = time.time()
        res = model.infer(
            spk_audio_prompt=spk_prompt,
            text=text,
            output_path=None,
            return_audio=True,
            return_numpy=False # Keep as tensor to measure raw internal speed
        )
        print(f"Run {i+1}: {time.time() - iter_start:.4f}s")
        
    total_time = time.time() - start_time
    print(f"Average time: {total_time / n_runs:.4f}s")

if __name__ == "__main__":
    benchmark()
