#!/usr/bin/env python
"""Generate synthetic TARS voice clips using IndexTTS2.

This script uses IndexTTS2 voice cloning to generate synthetic speech
clips from a reference audio file and text prompts.

Usage:
    uv run tools/generate_tars_synthetic_clips.py \
        --reference interstellar-tars-01-resemble-denoised.wav \
        --input data/tars_synthesis_input.jsonl \
        --output-dir data/tars_synthetic_clips \
        --use-fp16
"""

import argparse
import json
import os
import time
from pathlib import Path

# Suppress torch inductor warnings for older GPUs
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "no"

import logging
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic TARS voice clips using IndexTTS2"
    )
    parser.add_argument(
        "--reference", "-r",
        type=Path,
        default=Path("interstellar-tars-01-resemble-denoised.wav"),
        help="Reference audio file for voice cloning"
    )
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=Path("data/tars_synthesis_input.jsonl"),
        help="Input JSONL file with text prompts"
    )
    parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default=Path("data/tars_synthetic_clips"),
        help="Output directory for generated clips"
    )
    parser.add_argument(
        "--cfg-path",
        type=Path,
        default=Path("checkpoints/config.yaml"),
        help="Path to IndexTTS2 config"
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Path to IndexTTS2 model directory"
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use FP16 inference for lower VRAM"
    )
    parser.add_argument(
        "--use-cuda-kernel",
        action="store_true",
        help="Use CUDA kernel acceleration"
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Use DeepSpeed acceleration"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Maximum number of samples to generate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run on (cuda/cpu)"
    )
    args = parser.parse_args()

    # Validate inputs
    if not args.reference.exists():
        print(f"ERROR: Reference audio not found: {args.reference}")
        return 1
    
    if not args.input.exists():
        print(f"ERROR: Input JSONL not found: {args.input}")
        return 1
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load text prompts
    print(f"Loading prompts from {args.input}...")
    prompts = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line))
    
    if args.max_samples:
        prompts = prompts[:args.max_samples]
    
    print(f"Loaded {len(prompts)} prompts")
    
    # Initialize IndexTTS2
    print(f"\nInitializing IndexTTS2...")
    print(f"  Config: {args.cfg_path}")
    print(f"  Model dir: {args.model_dir}")
    print(f"  FP16: {args.use_fp16}")
    print(f"  Device: {args.device}")
    
    from indextts.infer_v2 import IndexTTS2
    
    tts = IndexTTS2(
        cfg_path=str(args.cfg_path),
        model_dir=str(args.model_dir),
        use_fp16=args.use_fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
    )
    
    print(f"\nReference audio: {args.reference}")
    print(f"Output directory: {args.output_dir}")
    print(f"\n{'='*60}")
    print("Starting synthesis...")
    print(f"{'='*60}\n")
    
    # Generate clips
    manifest = []
    total_duration = 0.0
    start_time = time.time()
    
    for idx, prompt in enumerate(prompts, 1):
        sample_id = prompt["id"]
        text = prompt["text"]
        output_path = args.output_dir / f"{sample_id}.wav"
        
        print(f"[{idx}/{len(prompts)}] Generating: {text[:50]}...")
        
        sample_start = time.time()
        
        try:
            # Use infer() with voice cloning from reference
            result = tts.infer(
                spk_audio_prompt=str(args.reference),
                text=text,
                output_path=str(output_path),
                use_random=False,  # Deterministic for consistent voice
                verbose=False,
            )
            
            sample_duration = time.time() - sample_start
            
            # Get audio duration from file
            import soundfile as sf
            audio_info = sf.info(output_path)
            audio_duration = audio_info.duration
            
            manifest.append({
                "id": sample_id,
                "text": text,
                "path": str(output_path.name),
                "duration": audio_duration,
                "generation_time": sample_duration,
            })
            
            total_duration += audio_duration
            print(f"         -> {output_path.name} ({audio_duration:.2f}s, gen: {sample_duration:.1f}s)")
            
        except Exception as e:
            print(f"         -> ERROR: {e}")
            manifest.append({
                "id": sample_id,
                "text": text,
                "error": str(e),
            })
    
    elapsed = time.time() - start_time
    
    # Save manifest
    manifest_path = args.output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump({
            "reference_audio": str(args.reference),
            "total_clips": len([m for m in manifest if "path" in m]),
            "total_duration": total_duration,
            "total_generation_time": elapsed,
            "clips": manifest,
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Synthesis complete!")
    print(f"{'='*60}")
    print(f"  Generated: {len([m for m in manifest if 'path' in m])}/{len(prompts)} clips")
    print(f"  Total audio: {total_duration:.1f}s")
    print(f"  Total time: {elapsed:.1f}s")
    print(f"  RTF: {elapsed/total_duration:.2f}x" if total_duration > 0 else "")
    print(f"  Manifest: {manifest_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
