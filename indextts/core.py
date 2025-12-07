"""Core inference orchestration logic for IndexTTS."""
import hashlib
import json
import os
import shutil
from typing import Any, Optional, cast

import gradio as gr

from indextts.types import IndexTTS2Client, InferFn, NormalizeEmoVecFn

CACHE_DIR = os.path.join("outputs", "cache")


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
    os.makedirs(CACHE_DIR, exist_ok=True)

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

    cache_payload: dict[str, Any] = {
        "emo_control_method": emo_control_method,
        "prompt": prompt,
        "text": text,
        "emo_ref_path": emo_ref_path,
        "emo_weight": emo_weight,
        "vec": vec,
        "emo_text": emo_text,
        "emo_random": bool(emo_random),
        "max_text_tokens_per_segment": int(max_text_tokens_per_segment),
        "generation": kwargs,
        "model_version": getattr(tts, "model_version", None),
    }
    cache_key = hashlib.sha256(
        json.dumps(cache_payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")

    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        print(f"Cache hit for key {cache_key}, returning cached audio at {cache_path}")
        if output_path and os.path.abspath(output_path) != os.path.abspath(cache_path):
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            shutil.copy2(cache_path, output_path)
            return output_path
        return cache_path

    generation_path = cache_path
    if output_path and os.path.abspath(output_path) == os.path.abspath(cache_path):
        generation_path = output_path
    elif output_path:
        # still generate into cache, then copy to user-provided location
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    else:
        output_path = cache_path

    print(f"Emo control mode:{emo_control_method},weight:{emo_weight},vec:{vec}")
    
    infer_fn: InferFn = cast(InferFn, tts.infer)
    output: Any = infer_fn(
        spk_audio_prompt=prompt,
        text=text,
        output_path=generation_path,
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

    if output_path and os.path.abspath(output_path) != os.path.abspath(generation_path):
        if os.path.isfile(generation_path):
            shutil.copy2(generation_path, output_path)
            return output_path
    return output or generation_path
