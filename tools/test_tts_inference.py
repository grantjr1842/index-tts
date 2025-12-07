
import argparse
import torch
import torchaudio
import numpy as np
import os
import logging
import warnings

# Suppress TorchAudio warnings
warnings.filterwarnings("ignore", message=".*load_with_torchcodec.*", category=UserWarning)
warnings.filterwarnings("ignore", message=".*StreamingMediaDecoder.*", category=UserWarning)

# Suppress Torch Inductor warnings (RTX 20xx compatibility)
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "no"
logging.getLogger("torch._inductor").setLevel(logging.ERROR)

from pathlib import Path
from moshi.models import loaders
from moshi.models.lm import LMGen
from safetensors.torch import load_file

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=str, default="kyutai/tts-1.6b-en_fr")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--voice-cond", type=Path, default="data/tars_voice_conditioning_tts.safetensors")
    parser.add_argument("--text", type=str, default="This is TARS. I am ready to assist you.")
    parser.add_argument("--out", type=Path, default="tars_test_output.wav")
    args = parser.parse_args()

    print(f"Loading Moshi TTS model from {args.repo}...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(args.repo)
    
    # Load Moshi (LM)
    moshi = loaders.get_moshi_lm(
        checkpoint_info.moshi_weights,
        lm_kwargs=checkpoint_info.lm_config,
        device=args.device,
        dtype=torch.float16
    )
    moshi.eval()
    
    # Load Mimi (Codec)
    mimi = loaders.get_mimi(checkpoint_info.mimi_weights, device=args.device)
    mimi.eval()

    print("Loading voice conditioning...")
    # Load conditioning tensor (speaker_wavs)
    cond_tensors = load_file(args.voice_cond, device=args.device)
    # Ensure it's in the dict format expected by LMGen/ConditionProvider
    # ConditionTensors is dict[str, ConditionType]
    # ConditionType is (tensor, mask)
    
    # The saved tensor is just the raw tensor. We need to wrap it.
    # The mask should be all True (ones) since we padded appropriately during creation.
    speaker_wavs = cond_tensors["speaker_wavs"]
    # Check shape [B, D=512, T] -> [B, T, D=512]
    if speaker_wavs.shape[1] == 512: 
        speaker_wavs = speaker_wavs.transpose(1, 2)
        
    mask = torch.ones((speaker_wavs.shape[0], speaker_wavs.shape[1]), dtype=torch.bool, device=args.device)
    
    from moshi.conditioners.base import ConditionType, TensorCondition
    
    # Create valid TensorCondition input for the conditioner
    wav_input = TensorCondition(speaker_wavs, mask)
    
    final_cond_tensors = {}
    
    # Use the model's conditioner to project the tensor (512 -> 2048)
    if "speaker_wavs" in moshi.condition_provider.conditioners:
        print("Projecting speaker_wavs...")
        wav_cond = moshi.condition_provider.conditioners["speaker_wavs"]
        # Forward pass projects the tensor
        final_cond_tensors["speaker_wavs"] = wav_cond(wav_input)
    else:
        print("WARNING: speaker_wavs conditioner not found in provider!")
        final_cond_tensors["speaker_wavs"] = ConditionType(speaker_wavs, mask)

    # Manually generate 'cfg' and 'control' conditions which are required by the fuser
    if "cfg" in moshi.condition_provider.conditioners:
        print("Adding default cfg=1.0")
        cfg_cond = moshi.condition_provider.conditioners["cfg"]
        # prepare expects list of inputs. For LUTConditioner, it's list of strings.
        # It handles batching. We have batch size 1.
        cfg_prepared = cfg_cond.prepare(["1.0"]) 
        # forward returns ConditionType(tensor, mask)
        final_cond_tensors["cfg"] = cfg_cond(cfg_prepared)

    if "control" in moshi.condition_provider.conditioners:
        print("Adding default control=ok")
        control_cond = moshi.condition_provider.conditioners["control"]
        control_prepared = control_cond.prepare(["ok"])
        final_cond_tensors["control"] = control_cond(control_prepared)

    print("Generating audio...")
    # Prepare text tokens
    tokenizer = checkpoint_info.get_text_tokenizer()
    text_tokens = tokenizer.encode(args.text)
    # Add start/end tokens? Moshi usually handles raw text tokenization.
    # But LMGen expects codes.
    
    # LMGen step() takes input_tokens [B, K, 1].
    # We need to run the generation loop.
    
    # Wait, LMGen is for streaming generation step-by-step.
    # I can use a simpler loop.
    
    # Prepare initial prompt
    # Text token needs to be seeded.
    # Moshi generation usually starts with a specific start token.
    
    # Let's use the provided LMGen class which handles delays etc.
    lm_gen = LMGen(
        moshi,
        temp=0.8,
        temp_text=0.8,
        condition_tensors=final_cond_tensors
    )
    
    # We need to feed text tokens one by one? 
    # Or is there a high level generate?
    # No, LMGen is low level streaming.
    
    # Let's iterate.
    # We feed text tokens into the text stream (index 0).
    # We collect audio tokens from audio streams (indices 1..8).
    
    # Moshi 1.6B TTS might have different codebook setup.
    # Config says "text_card": 8000.
    # "n_q": 32 (32 audio codebooks? that's a lot).
    # "dep_q": 32.
    
    # Let's assume standard streaming:
    # We feed text tokens.
    
    device = args.device
    
    # Pad text tokens with padding id if needed?
    # Actually, for TTS, we feed text and it generates audio.
    # We can feed the whole text sequence?
    # LMGen.step() takes [B, K, 1].
    
    # Codes tensor: [B, K, T]
    # K = num_codebooks = n_q (32) + 1 (text) = 33.
    
    # We construct the input sequence for the text.
    # Audio codebooks should be masked/ungenerated.
    
    codes = []
    
    # Start token?
    # moshi.text_initial_token_id
    
    # Feed text
    all_text_tokens = [moshi.text_initial_token_id] + text_tokens + [moshi.existing_text_end_padding_id]
    
    audio_tokens_list = []
    
    with lm_gen.streaming(batch_size=1):
        for i, text_token in enumerate(all_text_tokens):
            # Construct input step
            # Text is known. Audio is ungenerated.
            
            # Input to step() is what we *observed* at the previous step to predict current?
            # Or is it the current input to generate next?
            # StreamingTransformer predicts next token.
            
            # LMGen.step handles the cache and delays.
            # input_tokens: [B, K, 1]
            
            # We place the text token in channel 0.
            # We place audio tokens... wait.
            # In text-to-speech, we provide text and generate audio.
            
            # step() returns the *generated* tokens for this step (delayed).
            
            # Input tensor layout:
            # dim 0: Batch
            # dim 1: Codebooks (0=text, 1..32=audio)
            # dim 2: Time (1)
            
            input_step = torch.full((1, moshi.num_codebooks, 1), moshi.ungenerated_token_id, dtype=torch.long, device=device)
            input_step[0, 0, 0] = text_token
            
            # For audio, we feed what we generated in previous steps?
            # LMGen.step() internally handles feeding back generated tokens into cache?
            # "The user should provide the true tokens for the modality they stream"
            # Here we stream text.
            # The audio tokens are generated by the model.
            
            # But LMGen.step() docstring says:
            # "Input codes of shape [B, K, T]... returns None or tensor."
            
            # If I look at LMGen source again:
            # It writes input_tokens into cache.
            # It runs the model.
            # It samples new tokens.
            # It writes *generated* tokens into cache.
            
            # So I should pass the text token I want to FORCE.
            # And I should pass ungenerated_token_id for audio so it generates it.
            
            out = lm_gen.step(input_step)
            if out is not None:
                audio_tokens_list.append(out)

    # We might need to continue generating after text ends to flush audio buffers
    # (due to delays and audio being longer than text usually).
    # How long? Until EOS? Or fixed duration?
    
    # tts-1.6b "audio_delay": 1.28
    
    print("Finished text, continuing generation...")
    # Continue generating for some time 
    extra_steps = 500 # Arbitrary
    with lm_gen.streaming(batch_size=1): # Re-entering context? No, should stay in.
        pass # My loop structure is wrong, I exited context.
        
    # Correct loop:
    all_codes = []
    
    with lm_gen.streaming(batch_size=1):
         # Feed text
        for text_token in all_text_tokens:
            input_step = torch.full((1, moshi.num_codebooks, 1), moshi.ungenerated_token_id, dtype=torch.long, device=device)
            input_step[0, 0, 0] = text_token
            out = lm_gen.step(input_step)
            if out is not None:
                all_codes.append(out)
        
        # Continue generating audio
        # Feed padding text
        padding_token = moshi.existing_text_padding_id
        
        for _ in range(1500): # Generate ~30s of audio tokens (50Hz frame rate -> 1500 = 30s)
            input_step = torch.full((1, moshi.num_codebooks, 1), moshi.ungenerated_token_id, dtype=torch.long, device=device)
            input_step[0, 0, 0] = padding_token
            out = lm_gen.step(input_step)
            if out is not None:
                all_codes.append(out)

    if len(all_codes) == 0:
        print("No codes generated!")
        return

    # Concatenate codes
    codes = torch.cat(all_codes, dim=2) # [B, K, T]
    
    # Extract audio codes (remove text channel 0)
    audio_codes = codes[:, 1:, :] # [B, 32, T]
    
    print(f"Generated audio codes shape: {audio_codes.shape}")
    
    # Decode with Mimi
    with torch.no_grad():
        wav_out = mimi.decode(audio_codes)
    
    print(f"Decoded wav shape: {wav_out.shape}")
    
    # Save to file using torchaudio
    torchaudio.save(args.out, wav_out.squeeze(0).cpu(), mimi.sample_rate)
    print(f"Saved audio to {args.out}")

if __name__ == "__main__":
    main()
