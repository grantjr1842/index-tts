import io
import json
import os
import hashlib
import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Any, Dict, Tuple, Iterable

# === Torch Inductor config (MUST be set before importing torch) ===
# Enable torch.compile cache to persist compiled graphs across restarts
TORCH_COMPILE_CACHE_DIR = os.path.abspath(os.path.join("checkpoints", "torch_compile_cache"))
os.makedirs(TORCH_COMPILE_CACHE_DIR, exist_ok=True)
os.environ["TORCHINDUCTOR_CACHE_DIR"] = TORCH_COMPILE_CACHE_DIR
# Enable FX graph cache for torch.compile
os.environ["TORCHINDUCTOR_FX_GRAPH_CACHE"] = "1"
# Disable max_autotune_gemm which requires more SMs than available on smaller GPUs
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "0"
# Ensure cache dir exists
if not os.path.exists(TORCH_COMPILE_CACHE_DIR):
    os.makedirs(TORCH_COMPILE_CACHE_DIR, exist_ok=True)
    print(f"Created torch compile cache directory: {TORCH_COMPILE_CACHE_DIR}")
else:
    print(f"Using existing torch compile cache directory: {TORCH_COMPILE_CACHE_DIR}")

import numpy as np
import torch
import torch._inductor.config
import uvicorn
import soundfile as sf
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

# Also set via config API for completeness (env var should suffice but this ensures it)
torch._inductor.config.max_autotune_gemm = False

# IndexTTS imports
from indextts.logging import (
    setup_logging, print_stage, print_section_header, print_section_footer,
    print_section_item, print_config_section, print_request_start,
    print_request_complete, GracefulShutdown, COLORS
)
print_stage("Setting up logging", "progress")
setup_logging()
print_stage("Importing IndexTTS2", "progress")
from indextts.infer_v2 import IndexTTS2
print_stage("IndexTTS2 imported", "complete")

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
    use_int8: bool = os.getenv("TARS_INT8", "0") == "1"  # Default OFF to keep semantic model on GPU
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
    
    # Display configuration section
    print()
    print_config_section("TARS Server Configuration", {
        "Device": settings.device,
        "FP16": settings.use_fp16,
        "INT8": settings.use_int8,
        "Torch Compile": settings.use_torch_compile,
        "Compile Cache": TORCH_COMPILE_CACHE_DIR if settings.use_torch_compile else "N/A",
        "CUDA Kernel": settings.use_cuda_kernel,
        "Streaming": settings.enable_streaming,
        "Warmup": settings.warmup_on_start,
        "Max Concurrency": settings.max_concurrency,
        "Reference Audio": os.path.basename(TARS_REFERENCE_AUDIO),
        "Checkpoint Dir": CHECKPOINT_DIR,
        "Cache Dir": CACHE_DIR,
    })
    print()
    
    if not os.path.exists(TARS_REFERENCE_AUDIO):
        print_stage(f"Reference audio not found at {TARS_REFERENCE_AUDIO}", "failed")

    os.makedirs(CACHE_DIR, exist_ok=True)

    # Model loading section
    print_section_header("Model Loading")
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
    print_section_footer()
    
    # Warmup section - triggers torch.compile and caches the compiled graph
    # Warmup section - triggers torch.compile and caches the compiled graph
    if settings.warmup_on_start and settings.use_torch_compile:
        print()
        print_section_header("Warmup (torch.compile)", 50)
        
        # Check if cache is already populated
        cache_exists = os.path.exists(TORCH_COMPILE_CACHE_DIR) and len(os.listdir(TORCH_COMPILE_CACHE_DIR)) > 0
        
        should_invalidate = False
        if cache_exists:
            # Check modification times
            cache_mtime = os.path.getmtime(TORCH_COMPILE_CACHE_DIR)
            
            # Dependencies to check against cache
            deps = [os.path.abspath(__file__)]
            if os.path.exists(CONFIG_PATH):
                deps.append(os.path.abspath(CONFIG_PATH))
                
            for dep in deps:
                if os.path.getmtime(dep) > cache_mtime:
                    print_section_item("Invalidating", f"{os.path.basename(dep)} changed")
                    should_invalidate = True
                    break
        
        if cache_exists and not should_invalidate:
            print_section_item("Status", "Cache found. Skipping warmup inference.")
            print_section_item("Cache", f"Found in {TORCH_COMPILE_CACHE_DIR}")
            print_section_item("Result", f"{COLORS['green']}Skipped{COLORS['reset']}")
        else:
            if should_invalidate:
                import shutil
                print_section_item("Status", "Invalidating stale cache...")
                try:
                    shutil.rmtree(TORCH_COMPILE_CACHE_DIR)
                    os.makedirs(TORCH_COMPILE_CACHE_DIR, exist_ok=True)
                except Exception as e:
                     print_section_item("Warning", f"Failed to clear cache: {e}")

            print_section_item("Status", "Running warmup inference...")
            warmup_start = time.perf_counter()
            try:
                # Run a minimal inference to trigger torch.compile
                _ = tts_model.infer(
                    spk_audio_prompt=TARS_REFERENCE_AUDIO,
                    text="Hi",  # Minimal text for fast warmup
                    output_path=None,
                    return_audio=True,
                    return_numpy=True,
                )
                warmup_time = time.perf_counter() - warmup_start
                print_section_item("Duration", f"{warmup_time:.2f}s")
                print_section_item("Cache", f"Saved to {TORCH_COMPILE_CACHE_DIR}")
                print_section_item("Result", f"{COLORS['green']}Success{COLORS['reset']}")
            except Exception as e:
                print_section_item("Result", f"{COLORS['red']}Failed: {e}{COLORS['reset']}")
        print_section_footer(50)
    
    # Server ready section
    print()
    print_section_header("Server Ready", 40)
    print_section_item("Port", "8009")
    print_section_item("Endpoints", "/tts, /tts/stream, /healthz, /readyz")
    if torch.cuda.is_available():
        print_section_item("VRAM", f"{torch.cuda.memory_allocated()/1e9:.2f}GB")
    print_section_footer(40)
    print()
    
    yield
    
    # Shutdown handled by GracefulShutdown or here
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
    
    request_id = str(uuid.uuid4())[:8]
    print_request_start(request_id, request.text)
    start_time = time.perf_counter()
    
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
        # Estimate audio length from file size (16-bit mono @ 22050Hz)
        audio_length = len(data) / (22050 * 2)
        duration = time.perf_counter() - start_time
        print_request_complete(request_id, duration, audio_length, duration / audio_length if audio_length > 0 else 0, cached=True)
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
    
    duration = time.perf_counter() - start_time
    audio_length = result.duration_sec if result.duration_sec else len(audio_data) / sample_rate
    rtf = result.rtf if result.rtf else (duration / audio_length if audio_length > 0 else 0)
    print_request_complete(request_id, duration, audio_length, rtf)
    
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


def _streaming_producer(request: TTSRequest, queue: asyncio.Queue, loop: asyncio.AbstractEventLoop) -> None:
    """
    Producer function that runs in a thread pool and pushes audio chunks to the queue.
    This enables true streaming - chunks are sent to the client as soon as they're generated.
    """
    if tts_model is None:
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)
        return
    
    request_id = str(uuid.uuid4())[:8]
    print_request_start(request_id, request.text + " (stream)")
    start_time = time.perf_counter()

    try:
        gen = tts_model.infer(
            spk_audio_prompt=TARS_REFERENCE_AUDIO,
            text=request.text,
            output_path=None,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            stream_return=True,
        )
        total_chunks = 0
        total_bytes = 0
        for chunk in gen:
            # chunk is (1, T) float32 but ALREADY SCALED to [-32767, 32767] by IndexTTS2
            wav = chunk.squeeze(0).cpu().numpy()
            # Just cast to int16, DO NOT multiply by 32767 again
            wav_int16 = wav.astype(np.int16)
            data = wav_int16.tobytes()
            total_chunks += 1
            total_bytes += len(data)
            # Put chunk in queue - this is thread-safe
            asyncio.run_coroutine_threadsafe(queue.put(data), loop)
        
        duration = time.perf_counter() - start_time
        print_request_complete(request_id, duration, 0, 0, message_extra=f"chunks={total_chunks} bytes={total_bytes}")
    except Exception as e:
        # Put error in queue so consumer can handle it
        asyncio.run_coroutine_threadsafe(queue.put(e), loop)
    finally:
        # Signal end of stream
        asyncio.run_coroutine_threadsafe(queue.put(None), loop)


async def _streaming_consumer(queue: asyncio.Queue):
    """
    Async generator that yields chunks from the queue as they arrive.
    This enables true real-time streaming to the client.
    """
    while True:
        chunk = await queue.get()
        if chunk is None:
            # End of stream
            break
        if isinstance(chunk, Exception):
            # Producer encountered an error
            raise chunk
        yield chunk


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
        queue: asyncio.Queue = asyncio.Queue()
        
        # Start producer in thread pool - it will push chunks to queue as they're generated
        if executor is not None:
            executor.submit(_streaming_producer, request, queue, loop)
        else:
            raise HTTPException(status_code=503, detail="Server not ready")
        
        # Return streaming response with async consumer
        # Media type is audio/pcm since we're sending raw S16LE PCM data, not WAV
        return StreamingResponse(
            _streaming_consumer(queue),
            media_type="audio/pcm",
            headers={
                "X-Audio-Sample-Rate": "22050",
                "X-Audio-Channels": "1",
                "X-Audio-Format": "s16le",
            }
        )

    return await _with_concurrency_limit(run_stream())


@app.get("/healthz")
async def healthz():
    return {"status": "ok"}


@app.get("/readyz")
async def readyz():
    ready = tts_model is not None and os.path.exists(TARS_REFERENCE_AUDIO)
    return {"ready": ready, "ref_audio": TARS_REFERENCE_AUDIO, "device": settings.device}

def _cleanup():
    """Cleanup function for graceful shutdown."""
    global executor
    if executor:
        executor.shutdown(wait=False)


if __name__ == "__main__":
    print()
    print_section_header("TARS Voice Server", 50)
    print_section_item("Version", "2.0")
    print_section_item("Port", "8009")
    print_section_footer(50)
    print()
    
    with GracefulShutdown(cleanup=_cleanup):
        uvicorn.run(app, host="0.0.0.0", port=8009)
