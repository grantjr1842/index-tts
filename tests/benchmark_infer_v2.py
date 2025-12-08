import time
import torch
import sys
import os
import json
import argparse
import numpy as np

# Add repo root to path
sys.path.append(os.getcwd())

from indextts.infer_v2 import IndexTTS2

def benchmark(args):
    device = args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Torch compile enabled: {args.compile}")
    
    # Initialize basic config
    try:
        model = IndexTTS2(
            cfg_path="checkpoints/config.yaml",
            model_dir="checkpoints",
            device=device,
            use_fp16=True,
            use_torch_compile=args.compile
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Load cases
    cases = []
    if os.path.exists(args.cases):
        with open(args.cases, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                     try:
                        cases.append(json.loads(line))
                     except json.JSONDecodeError as e:
                         print(f"Skipping invalid json line: {e}")
    else:
        print(f"Cases file {args.cases} not found.")
        return

    print(f"Loaded {len(cases)} test cases.")

    # Warmup
    print("Warming up...")
    try:
        # Use first case audio if available, else look for sample
        warmup_audio = cases[0].get("prompt_audio", "tests/sample_prompt.wav")
        # heuristic to find file if in tests/
        if not os.path.exists(warmup_audio) and os.path.exists(os.path.join("tests", warmup_audio)):
            warmup_audio = os.path.join("tests", warmup_audio)
            
        if os.path.exists(warmup_audio):
            # Short warmup text
            model.infer(warmup_audio, "Warmup.", None, return_audio=True)
        else:
            print("No audio found for warmup, skipping.")
    except Exception as e:
        print(f"Warmup failed: {e}")

    print("Starting benchmark...")
    
    results = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        
    for i, case in enumerate(cases):
        text = case.get("text")
        prompt_audio = case.get("prompt_audio")
        infer_mode = case.get("infer_mode", 0) # Use infer mode if present, though lib might not expose it easily in infer() args? 
        # checking infer_v2.py infer signature: infer(self, spk_audio_prompt, text, output_path, return_audio=False, return_numpy=False, stream=False)
        # It doesn't seem to take infer_mode directly in infer() unless it's handled internally or I missed it.
        # But cases.jsonl has it. Maybe it serves as a config override? 
        # I'll ignore it for now or check if I need to pass it.
        
        # Path resolution
        if prompt_audio and not os.path.exists(prompt_audio):
             if os.path.exists(os.path.join("tests", prompt_audio)):
                  prompt_audio = os.path.join("tests", prompt_audio)
        
        if not prompt_audio or not os.path.exists(prompt_audio):
            print(f"Skipping case {i}: Audio prompt not found ({prompt_audio})")
            continue

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            
        iter_start = time.time()
        try:
            # We assume return_audio=True returns (sample_rate, audio_data)
            # audio_data might be numpy or tensor depending on return_numpy
            res = model.infer(
                spk_audio_prompt=prompt_audio,
                text=text,
                output_path=None,
                return_audio=True,
                return_numpy=True
            )
            gen_time = time.time() - iter_start
            
            # res can be an InferenceResult object or a tuple depending on version/config
            if hasattr(res, 'sampling_rate') and hasattr(res, 'audio'):
                 sr = res.sampling_rate
                 audio = res.audio
            else:
                 sr, audio = res
                 
            if audio is None:
                 print(f"Case {i}: No audio returned.")
                 continue
                 
            # Audio shape handling (channels, samples) or just (samples)
            if hasattr(audio, 'shape'):
                audio_len_samples = audio.shape[-1] 
            else:
                audio_len_samples = len(audio)

            audio_dur = audio_len_samples / sr
            
            rtf = audio_dur / gen_time if gen_time > 0 else 0
            
            vram_peak = 0
            if torch.cuda.is_available():
                vram_peak = torch.cuda.max_memory_allocated() / (1024**3)
            
            print(f"Case {i}: TextLen={len(text)}, GenTime={gen_time:.4f}s, AudioDur={audio_dur:.4f}s, RTF={rtf:.4f}, VRAM={vram_peak:.2f}GB")
            
            results.append({
                "case_index": i,
                "text": text[:50] + "..." if len(text) > 50 else text,
                "text_len": len(text),
                "gen_time": gen_time,
                "audio_dur": audio_dur,
                "rtf": rtf,
                "vram_peak": vram_peak,
                "error": None
            })
            
        except Exception as e:
            print(f"Case {i} failed: {e}")
            results.append({
                "case_index": i,
                "error": str(e)
            })

    if args.output:
        try:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2)
            print(f"Results saved to {args.output}")
        except Exception as e:
            print(f"Failed to save results: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark IndexTTS2 inference")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile optimization")
    parser.add_argument("--cases", type=str, default="tests/cases.jsonl", help="Path to test cases jsonl")
    parser.add_argument("--output", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cuda/cpu)")
    
    args = parser.parse_args()
    benchmark(args)
