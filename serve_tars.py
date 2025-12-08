import io
import json
import os
import hashlib
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict, Tuple, Iterable

import numpy as np
import torch
import uvicorn
import soundfile as sf
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# IndexTTS imports
from indextts.logging import setup_logging
print("Setting up logging...")
setup_logging()
print("Importing IndexTTS2...")
from indextts.infer_v2 import IndexTTS2
print("IndexTTS2 imported.")

CACHE_DIR = os.path.join("outputs", "cache")
TARS_REFERENCE_AUDIO = os.path.abspath(
    os.getenv("TARS_REFERENCE_AUDIO", "interstellar-tars-01-resemble-denoised.wav")
)
CHECKPOINT_DIR = os.getenv("TARS_CHECKPOINT_DIR", "checkpoints")
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, os.getenv("TARS_CONFIG_FILE", "config.yaml"))

from contextlib import asynccontextmanager

class Settings:
    """
    Lightweight config surface driven by environment variables.
    Defaults bias toward GPU + fp16 when available.
    """

    max_concurrency: int = int(os.getenv("TARS_MAX_CONCURRENCY", "2"))
    device: Optional[str] = os.getenv("TARS_DEVICE")
    use_fp16: Optional[bool] = None
    use_torch_compile: bool = os.getenv("TARS_TORCH_COMPILE", "1") == "1"
    use_cuda_kernel: Optional[bool] = True
    use_int8: bool = os.getenv("TARS_INT8", "1") == "1"
    enable_streaming: bool = os.getenv("TARS_ENABLE_STREAMING", "1") == "1"
    warmup_on_start: bool = os.getenv("TARS_WARMUP", "1") == "1"

    def __init__(self) -> None:
        device = self.device
        if device is None:
            if torch.cuda.is_available():
                device = "cuda:0"
            elif hasattr(torch, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"
        self.device = device
        if self.use_fp16 is None:
            self.use_fp16 = device != "cpu"
        if self.use_cuda_kernel is None:
            self.use_cuda_kernel = device.startswith("cuda")


settings = Settings()

# Global resources
tts_model = None  # type: ignore[var-annotated]
executor: ThreadPoolExecutor | None = None
concurrency_sem: asyncio.Semaphore | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model, executor, concurrency_sem
    print("Loading IndexTTS2 model...")
    if not os.path.exists(TARS_REFERENCE_AUDIO):
        print(f"WARNING: Reference audio not found at {TARS_REFERENCE_AUDIO}")

    os.makedirs(CACHE_DIR, exist_ok=True)

    print(f"Using device: {settings.device}")
    print(f"INT8 quantization: {settings.use_int8}")

    tts_model = IndexTTS2(
        cfg_path=CONFIG_PATH,
        model_dir=CHECKPOINT_DIR,
        use_fp16=bool(settings.use_fp16),
        device=settings.device,
        use_cuda_kernel=settings.use_cuda_kernel,
        use_torch_compile=settings.use_torch_compile,
    )
    executor = ThreadPoolExecutor(max_workers=max(settings.max_concurrency, 1))
    concurrency_sem = asyncio.Semaphore(settings.max_concurrency)
    print("Model loaded successfully!")
    yield
    print("Shutting down TARS server...")
    if executor:
        executor.shutdown(wait=True)

# Initialize FastAPI app with lifespan
app = FastAPI(title="TARS Voice Server", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    speed: float = 1.0
    max_text_tokens_per_segment: int = 120


def _hash_payload(payload: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def _maybe_cached(cache_payload: Dict[str, Any]) -> Optional[Tuple[str, bytes]]:
    cache_key = _hash_payload(cache_payload)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")
    if os.path.isfile(cache_path) and os.path.getsize(cache_path) > 0:
        with open(cache_path, "rb") as f:
            return cache_path, f.read()
    return None


def _save_cache(cache_payload: Dict[str, Any], data: bytes) -> str:
    cache_key = _hash_payload(cache_payload)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.wav")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        f.write(data)
    return cache_path


def _blocking_infer(request: TTSRequest) -> Tuple[bytes, int, Dict[str, Any]]:
    """
    Run infer in a worker thread. Returns (wav_bytes, sample_rate, cache_payload).
    """
    if tts_model is None:
        raise RuntimeError("Model not loaded")
    cache_payload = {
        "text": request.text,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k,
        "speed": request.speed,
        "max_text_tokens_per_segment": request.max_text_tokens_per_segment,
        "ref": TARS_REFERENCE_AUDIO,
    }
    cached = _maybe_cached(cache_payload)
    if cached:
        path, data = cached
        print(f"Cache hit: {path}")
        return data, 0, cache_payload

    audio_result = tts_model.infer(
        spk_audio_prompt=TARS_REFERENCE_AUDIO,
        text=request.text,
        output_path=None,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        return_audio=True,
        return_numpy=True,
        max_text_tokens_per_segment=request.max_text_tokens_per_segment,
    )

    result = audio_result
    audio_data = result.audio.astype(np.int16)
    sample_rate = result.sampling_rate
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sample_rate, format="WAV")
    data = buffer.getvalue()
    _save_cache(cache_payload, data)
    return data, sample_rate, cache_payload


async def _with_concurrency_limit(coro):
    if concurrency_sem is None:
        return await coro
    
    if concurrency_sem.locked():
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Server is busy, please retry.",
        )
    
    await concurrency_sem.acquire()
    try:
        return await coro
    finally:
        concurrency_sem.release()


@app.post("/tts")
async def generate_speech(request: TTSRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not os.path.exists(TARS_REFERENCE_AUDIO):
        raise HTTPException(status_code=500, detail="Reference audio file missing on server")

    async def run():
        loop = asyncio.get_running_loop()
        try:
            data, _, _ = await loop.run_in_executor(executor, _blocking_infer, request)  # type: ignore[arg-type]
            return Response(content=data, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as e:  # pragma: no cover - defensive
            raise HTTPException(status_code=500, detail=str(e))

    return await _with_concurrency_limit(run())


def _streaming_generator(request: TTSRequest) -> Iterable[bytes]:
    if tts_model is None:
        raise RuntimeError("Model not loaded")
    gen = tts_model.infer(
        spk_audio_prompt=TARS_REFERENCE_AUDIO,
        text=request.text,
        output_path=None,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        stream_return=True,
    )
    for chunk in gen:
        wav = chunk.squeeze(0).cpu().numpy()
        buffer = io.BytesIO()
        sf.write(buffer, wav, 24000, format="WAV")
        yield buffer.getvalue()


@app.post("/tts/stream")
async def generate_speech_stream(request: TTSRequest):
    if not settings.enable_streaming:
        raise HTTPException(status_code=404, detail="Streaming disabled")
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    if not os.path.exists(TARS_REFERENCE_AUDIO):
        raise HTTPException(status_code=500, detail="Reference audio file missing on server")

    async def run_stream():
        loop = asyncio.get_running_loop()
        try:
            gen = await loop.run_in_executor(executor, _streaming_generator, request)  # type: ignore[arg-type]
            return StreamingResponse(gen, media_type="audio/wav")
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    return await _with_concurrency_limit(run_stream())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    ready = tts_model is not None and os.path.exists(TARS_REFERENCE_AUDIO)
    return {"ready": ready, "ref_audio": TARS_REFERENCE_AUDIO, "device": settings.device}

if __name__ == "__main__":
    print("Starting Uvicorn...")
    uvicorn.run(app, host="0.0.0.0", port=8009)
