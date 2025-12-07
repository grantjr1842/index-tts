
import argparse
import torch
import torchaudio
import warnings

# Suppress specific TorchAudio warnings
warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*", category=UserWarning)

from pathlib import Path
from moshi.models import loaders
from safetensors.torch import save_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio_file", type=Path, help="Input audio file")
    parser.add_argument("--out", type=Path, default="voice_conditioning.safetensors", help="Output file")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--repo", type=str, default="kyutai/tts-1.6b-en_fr")
    args = parser.parse_args()

    print(f"Loading Mimi model from {args.repo}...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.repo)
    mimi = loaders.get_mimi(checkpoint_info.mimi_weights, device=args.device)
    mimi.eval()
    
    print(f"Processing {args.audio_file}...")
    try:
        # Prefer TorchCodec path to avoid deprecated torchaudio.load warnings.
        wav, sr = torchaudio.load_with_torchcodec(str(args.audio_file), channels_first=True)
    except Exception:
        wav, sr = torchaudio.load(args.audio_file)
    
    # Resample if needed
    if sr != mimi.sample_rate:
        print(f"Resampling from {sr} to {mimi.sample_rate}...")
        wav = torchaudio.functional.resample(wav, sr, mimi.sample_rate)
    
    # Convert to mono if needed
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    
    # Add batch dim
    wav = wav.unsqueeze(0).to(args.device)
    
    # Calculate available VRAM and adjust duration
    if args.device == "cuda" and torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(args.device).total_memory
        # Heuristic: 8GB VRAM -> ~10s safe buffer for 1.6B model + execution
        # Scaling factor: 10s / 8GB ~= 1.25 s/GB? 
        # Actually proper estimation is complex, but let's assume 10s is safe for 8GB (2070).
        # We can try to use more if we have more.
        # Let's be conservative: 1.0s per GB of VRAM.
        safe_duration = (vram / (1024**3)) * 1.2 # e.g. 8GB * 1.2 = 9.6s, 24GB -> 28.8s
        print(f"Auto-calculated safe duration: {safe_duration:.2f}s based on {vram/1024**3:.1f}GB VRAM")
        target_duration = safe_duration
    else:
        target_duration = 10.0 # Default fallback

    target_samples = int(target_duration * mimi.sample_rate)
    if wav.shape[-1] > target_samples:
        print(f"Truncating to {target_duration:.2f}s...")
        wav = wav[..., :target_samples]
    elif wav.shape[-1] < target_samples:
        # Pad with silence
        missing = target_samples - wav.shape[-1]
        wav = torch.nn.functional.pad(wav, (0, missing))

    # Encode to latent (unquantized)
    with torch.no_grad():
        emb = mimi.encode_to_latent(wav, quantize=False)
    
    print(f"Encoded shape: {emb.shape}")
    
    # Save as safetensors
    tensors = {
        "speaker_wavs": emb.cpu()
    }
    save_file(tensors, args.out)
    print(f"Saved {list(tensors.keys())} to {args.out}")

if __name__ == "__main__":
    main()
