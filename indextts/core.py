"""Core inference orchestration logic for IndexTTS."""
import os
import time
from typing import Any, Optional, cast

import gradio as gr

from indextts.types import IndexTTS2Client, InferFn, NormalizeEmoVecFn


def generate_speech(
    tts: IndexTTS2Client,
    emo_control_method: int,
    prompt: Optional[str],
    text: str,
    emo_ref_path: Optional[str],
    emo_weight: float,
    vec1: float,
    vec2: float,
    vec3: float,
    vec4: float,
    vec5: float,
    vec6: float,
    vec7: float,
    vec8: float,
    emo_text: Optional[str],
    emo_random: bool,
    max_text_tokens_per_segment: int = 120,
    do_sample: bool = True,
    top_p: float = 0.8,
    top_k: float = 30.0,
    temperature: float = 0.8,
    length_penalty: float = 0.0,
    num_beams: int = 3,
    repetition_penalty: float = 10.0,
    max_mel_tokens: int = 1500,
    output_path: Optional[str] = None,
    verbose: bool = False,
) -> str:
    """
    Generate speech using IndexTTS2.
    
    Args:
        tts: IndexTTS2 client instance
        emo_control_method: Emotion control method (0=speaker, 1=reference audio, 2=vectors, 3=text)
        prompt: Speaker audio prompt path
        text: Text to synthesize
        emo_ref_path: Emotion reference audio path
        emo_weight: Emotion weight/alpha
        vec1-vec8: Emotion vector components
        emo_text: Emotion description text
        emo_random: Whether to use random emotion sampling
        max_text_tokens_per_segment: Maximum text tokens per segment
        do_sample: Whether to use sampling
        top_p: Top-p sampling parameter
        top_k: Top-k sampling parameter
        temperature: Temperature for sampling
        length_penalty: Length penalty
        num_beams: Number of beams for beam search
        repetition_penalty: Repetition penalty
        max_mel_tokens: Maximum mel tokens to generate
        output_path: Output file path (auto-generated if None)
        verbose: Enable verbose logging
        
    Returns:
        Path to the generated audio file
    """
    if not output_path:
        output_path = os.path.join("outputs", f"spk_{int(time.time())}.wav")

    kwargs: dict[str, float | int | bool | None] = {
        "do_sample": bool(do_sample),
        "top_p": float(top_p),
        "top_k": int(top_k) if int(top_k) > 0 else None,
        "temperature": float(temperature),
        "length_penalty": float(length_penalty),
        "num_beams": int(num_beams),
        "repetition_penalty": float(repetition_penalty),
        "max_mel_tokens": int(max_mel_tokens),
    }
    
    if not isinstance(emo_control_method, int):
        emo_control_method = int(
            getattr(emo_control_method, "value", emo_control_method))
    
    if emo_control_method == 0:  # emotion from speaker
        emo_ref_path = None  # remove external reference audio
    
    vec: Optional[list[float]] = None
    if emo_control_method == 2:  # emotion from custom vectors
        raw_vec: list[float] = [vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8]
        normalize_vec: NormalizeEmoVecFn = cast(
            NormalizeEmoVecFn, tts.normalize_emo_vec
        )
        vec = normalize_vec(raw_vec, apply_bias=True)

    if emo_text == "":
        # erase empty emotion descriptions; `infer()` will then automatically use the main prompt
        emo_text = None

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    
    infer_fn: InferFn = cast(InferFn, tts.infer)
    output: Any = infer_fn(
        spk_audio_prompt=prompt,
        text=text,
        output_path=output_path,
        emo_audio_prompt=emo_ref_path,
        emo_alpha=emo_weight,
        emo_vector=vec,
        use_emo_text=(emo_control_method == 3),
        emo_text=emo_text,
        use_random=emo_random,
        verbose=verbose,
        max_text_tokens_per_segment=int(max_text_tokens_per_segment),
        **kwargs,
    )
    
    return output
