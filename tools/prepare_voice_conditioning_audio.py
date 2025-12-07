#!/usr/bin/env python
"""Prepare audio clips for voice conditioning.

This script selects the best synthetic clips, concatenates them,
and prepares combined audio for Moshi TTS voice conditioning.

Usage:
    uv run tools/prepare_voice_conditioning_audio.py \
        --input-dir data/tars_synthetic_clips \
        --output data/tars_persona_combined.wav \
        --target-duration 12.0
"""

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf


def calculate_audio_quality_score(audio: np.ndarray, sample_rate: int) -> dict:
    """Calculate simple audio quality metrics.
    
    Returns a dict with:
    - peak: Peak amplitude (should be close to but not exceed 1.0)
    - rms: RMS level (higher = louder, more presence)
    - crest: Crest factor (peak/rms, lower = more compressed)
    - silence_ratio: Ratio of near-silent samples
    """
    peak = np.abs(audio).max()
    rms = np.sqrt(np.mean(audio ** 2))
    crest = peak / rms if rms > 0 else 0
    
    # Silence threshold at -60dB
    silence_threshold = 10 ** (-60 / 20)
    silence_ratio = np.mean(np.abs(audio) < silence_threshold)
    
    # Higher score = better quality
    # Prefer: high RMS, low silence ratio, moderate crest factor
    score = rms * 10 - silence_ratio * 5 + (1 / (crest + 1)) * 2
    
    return {
        "peak": float(peak),
        "rms": float(rms),
        "crest": float(crest),
        "silence_ratio": float(silence_ratio),
        "score": float(score),
    }


def normalize_audio(audio: np.ndarray, peak_db: float = -1.0) -> np.ndarray:
    """Normalize audio to target peak level."""
    peak_linear = 10 ** (peak_db / 20)
    current_peak = np.abs(audio).max()
    if current_peak > 0:
        audio = audio * (peak_linear / current_peak)
    return audio


def main():
    parser = argparse.ArgumentParser(
        description="Prepare audio clips for voice conditioning"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        required=True,
        help="Input directory with synthetic clips"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("data/tars_persona_combined.wav"),
        help="Output combined audio file"
    )
    parser.add_argument(
        "--target-duration",
        type=float,
        default=12.0,
        help="Target duration in seconds (default: 12.0)"
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=10.0,
        help="Minimum acceptable duration (default: 10.0)"
    )
    parser.add_argument(
        "--max-duration",
        type=float,
        default=15.0,
        help="Maximum duration to include (default: 15.0)"
    )
    parser.add_argument(
        "--gap-duration",
        type=float,
        default=0.3,
        help="Gap between clips in seconds (default: 0.3)"
    )
    parser.add_argument(
        "--target-sample-rate",
        type=int,
        default=24000,
        help="Target sample rate for output (default: 24000 for Mimi)"
    )
    parser.add_argument(
        "--normalize-peak-db",
        type=float,
        default=-1.0,
        help="Normalize to this peak level in dB (default: -1.0)"
    )
    args = parser.parse_args()

    # Load manifest
    manifest_path = args.input_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"ERROR: Manifest not found: {manifest_path}")
        return 1
    
    with open(manifest_path) as f:
        manifest = json.load(f)
    
    clips = [c for c in manifest["clips"] if "path" in c]
    print(f"Found {len(clips)} clips in manifest")
    
    # Score each clip
    print("\nScoring clips for quality...")
    scored_clips = []
    for clip in clips:
        clip_path = args.input_dir / clip["path"]
        if not clip_path.exists():
            print(f"  WARNING: Clip not found: {clip_path}")
            continue
        
        audio, sr = sf.read(clip_path)
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)  # Convert to mono
        
        quality = calculate_audio_quality_score(audio, sr)
        scored_clips.append({
            **clip,
            "audio": audio,
            "sample_rate": sr,
            "quality": quality,
        })
    
    # Sort by quality score (descending)
    scored_clips.sort(key=lambda c: c["quality"]["score"], reverse=True)
    
    print(f"\nTop 5 clips by quality:")
    for i, clip in enumerate(scored_clips[:5], 1):
        q = clip["quality"]
        print(f"  {i}. {clip['id']}: score={q['score']:.3f}, rms={q['rms']:.4f}, dur={clip['duration']:.2f}s")
    
    # Select clips up to target duration
    print(f"\nSelecting clips for {args.target_duration:.1f}s target...")
    selected = []
    total_duration = 0.0
    
    for clip in scored_clips:
        clip_duration = clip["duration"]
        gap = args.gap_duration if selected else 0
        
        if total_duration + clip_duration + gap > args.max_duration:
            continue
        
        selected.append(clip)
        total_duration += clip_duration + gap
        
        if total_duration >= args.target_duration:
            break
    
    if total_duration < args.min_duration:
        print(f"WARNING: Only {total_duration:.1f}s of audio selected (min: {args.min_duration:.1f}s)")
    
    print(f"\nSelected {len(selected)} clips ({total_duration:.1f}s total)")
    
    # Resample and concatenate
    print(f"\nConcatenating at {args.target_sample_rate}Hz...")
    
    import torchaudio
    import torch
    
    combined_audio = []
    gap_samples = int(args.gap_duration * args.target_sample_rate)
    gap = np.zeros(gap_samples, dtype=np.float32)
    
    for i, clip in enumerate(selected):
        audio = clip["audio"]
        sr = clip["sample_rate"]
        
        # Resample if needed
        if sr != args.target_sample_rate:
            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            audio_tensor = torchaudio.functional.resample(audio_tensor, sr, args.target_sample_rate)
            audio = audio_tensor.squeeze(0).numpy()
        
        audio = audio.astype(np.float32)
        
        if i > 0:
            combined_audio.append(gap)
        combined_audio.append(audio)
        
        print(f"  + {clip['id']}: {len(audio)/args.target_sample_rate:.2f}s")
    
    # Combine and normalize
    combined = np.concatenate(combined_audio)
    combined = normalize_audio(combined, peak_db=args.normalize_peak_db)
    
    final_duration = len(combined) / args.target_sample_rate
    print(f"\nFinal combined audio: {final_duration:.2f}s")
    print(f"  Peak: {np.abs(combined).max():.4f}")
    print(f"  RMS: {np.sqrt(np.mean(combined**2)):.4f}")
    
    # Save output
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(args.output, combined, args.target_sample_rate)
    print(f"\nSaved to: {args.output}")
    
    # Save selection manifest
    selection_manifest = {
        "source_manifest": str(manifest_path),
        "selected_clips": [{"id": c["id"], "duration": c["duration"], "quality_score": c["quality"]["score"]} for c in selected],
        "total_duration": final_duration,
        "sample_rate": args.target_sample_rate,
        "output_path": str(args.output),
    }
    selection_path = args.output.with_suffix(".json")
    with open(selection_path, "w") as f:
        json.dump(selection_manifest, f, indent=2)
    print(f"Selection manifest: {selection_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())
