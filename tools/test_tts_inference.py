
import argparse
import torch
import soundfile as sf
import os
import logging
from pathlib import Path
from moshi.models import loaders
from moshi.models.tts import TTSModel, DEFAULT_DSM_TTS_VOICE_REPO

# Suppress Torch Inductor warnings (RTX 20xx compatibility)
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "no"
logging.getLogger("torch._inductor").setLevel(logging.ERROR)


def main():
    parser = argparse.ArgumentParser(
        description="Test TTS inference with voice conditioning"
    )
    parser.add_argument("--repo", type=str, default="kyutai/tts-1.6b-en_fr",
                        help="HuggingFace repo for the TTS model")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run on (cuda/cpu)")
    parser.add_argument("--voice-cond", type=str, default="data/tars_voice_conditioning_tts.safetensors",
                        help="Path to voice conditioning safetensors file")
    parser.add_argument("--text", type=str, default="This is TARS. I am ready to assist you.",
                        help="Text to synthesize")
    parser.add_argument("--out", type=Path, default="tars_test_output.wav",
                        help="Output audio file path")
    
    # Generation parameters
    parser.add_argument("--cfg-coef", type=float, default=2.0,
                        help="CFG coefficient (1.0-4.0, higher=stricter voice conditioning)")
    parser.add_argument("--temp", type=float, default=0.6,
                        help="Temperature (0.1-1.0, lower=more deterministic)")
    parser.add_argument("--n-q", type=int, default=32,
                        help="Number of codebooks (8-32, higher=better quality but slower)")
    parser.add_argument("--padding-bonus", type=float, default=0.0,
                        help="Additive bonus for padding logits (positive=slower speech)")
    parser.add_argument("--padding-between", type=int, default=1,
                        help="Padding between words for better articulation (0-3)")
    parser.add_argument("--max-gen-length", type=int, default=30000,
                        help="Maximum generation steps (frames at 12.5 Hz)")
    parser.add_argument("--initial-padding", type=int, default=2,
                        help="Initial padding steps to prevent first word cutoff")
    parser.add_argument("--max-padding", type=int, default=8,
                        help="Maximum consecutive padding tokens allowed")
    parser.add_argument("--final-padding", type=int, default=4,
                        help="Extra steps after last word")
    
    # Data type
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"],
                        help="Model data type")
    
    args = parser.parse_args()

    # Map dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading Moshi TTS model from {args.repo}...")
    print(f"  Device: {args.device}, dtype: {args.dtype}")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.repo)
    
    # Initialize TTSModel with all tuning parameters
    tts_model = TTSModel.from_checkpoint_info(
        checkpoint_info,
        voice_repo=DEFAULT_DSM_TTS_VOICE_REPO,
        device=args.device,
        dtype=dtype,
        temp=args.temp,
        initial_padding=args.initial_padding,
        max_padding=args.max_padding,
    )
    
    # Set additional parameters
    tts_model.n_q = args.n_q
    tts_model.padding_bonus = args.padding_bonus
    tts_model.max_gen_length = args.max_gen_length
    tts_model.final_padding = args.final_padding
    
    print("\nGeneration Parameters:")
    print(f"  cfg_coef: {args.cfg_coef}")
    print(f"  temp: {args.temp}")
    print(f"  n_q: {args.n_q}")
    print(f"  padding_bonus: {args.padding_bonus}")
    print(f"  padding_between: {args.padding_between}")
    print(f"  initial_padding: {args.initial_padding}")
    print(f"  max_padding: {args.max_padding}")
    print(f"  final_padding: {args.final_padding}")
    
    print(f"\nGenerating audio for text: '{args.text}'")
    print(f"Using voice conditioning: {args.voice_cond}")

    # Check if voice conditioning file exists
    if not Path(args.voice_cond).exists():
        print(f"ERROR: Voice conditioning file not found: {args.voice_cond}")
        return

    # Use simple_generate for straightforward generation
    # voice argument can be a local path ending in .safetensors
    generated_wavs = tts_model.simple_generate(
        text=args.text,
        voice=args.voice_cond,
        cfg_coef=args.cfg_coef,
        show_progress=True,
    )

    if not generated_wavs:
        print("No audio generated.")
        return

    # simple_generate returns a list of 1D float32 tensors
    audio_tensor = generated_wavs[0]
    
    # Convert to numpy for soundfile
    audio_np = audio_tensor.cpu().numpy()
    
    print(f"\nGenerated audio shape: {audio_np.shape}")
    print(f"Audio duration: {len(audio_np) / tts_model.mimi.sample_rate:.2f}s")
    print(f"Audio range: min={audio_np.min():.4f}, max={audio_np.max():.4f}")
    
    # Save to file
    sf.write(args.out, audio_np, int(tts_model.mimi.sample_rate))
    print(f"Saved audio to {args.out}")


if __name__ == "__main__":
    main()
