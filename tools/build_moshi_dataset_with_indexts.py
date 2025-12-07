#!/usr/bin/env python
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import logging
import multiprocessing as mp
import os
import queue
import sys
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any, Literal, Optional, Protocol, Union, cast, TypedDict

import numpy as np
import numpy.typing as npt
import librosa
import soundfile as sf  # type: ignore[import-untyped]
import torch

try:
    from indextts.infer_v2 import IndexTTS2
except ImportError:
    # This script might be run where indextts is not installed/importable
    # but we handle that dynamically below or via mock_inference
    IndexTTS2 = None  # type: ignore

AudioArray = npt.NDArray[np.float32]
Float64Array = npt.NDArray[np.float64]


class SoundFileModule(Protocol):
    def write(
        self,
        file: str | os.PathLike[str] | int,
        data: npt.ArrayLike,
        samplerate: int,
        subtype: str | None = None,
        endian: str | None = None,
        format: str | None = None,
        closefd: bool = True,
        compression_level: int | None = None,
        bitrate_mode: str | None = None,
    ) -> None:
        ...

    def read(
        self,
        file: str | os.PathLike[str] | int,
        frames: int = -1,
        start: int = 0,
        stop: int | None = None,
        dtype: str = "float64",
        always_2d: bool = False,
        fill_value: float | None = None,
        out: npt.NDArray[Any] | None = None,
        samplerate: int | None = None,
        channels: int | None = None,
        format: str | None = None,
        subtype: str | None = None,
        endian: str | None = None,
        closefd: bool = True,
    ) -> tuple[Float64Array, int]:
        ...


sf: SoundFileModule = cast(SoundFileModule, sf)


class InferenceResultLike(Protocol):
    audio: npt.ArrayLike
    sampling_rate: int
    duration_sec: float | None


class InferFn(Protocol):
    def __call__(
        self,
        *,
        spk_audio_prompt: str | None,
        text: str,
        output_path: str | os.PathLike[str] | None,
        emo_audio_prompt: str | None = None,
        emo_alpha: float = 1.0,
        emo_vector: list[float] | None = None,
        use_emo_text: bool = False,
        emo_text: str | None = None,
        use_random: bool = False,
        interval_silence: int = 200,
        verbose: bool = False,
        max_text_tokens_per_segment: int = 120,
        stream_return: bool = False,
        more_segment_before: int = 0,
        return_audio: bool = False,
        return_numpy: bool = False,
        **generation_kwargs: Any,
    ) -> InferenceResultLike | None:
        ...


class IndexTTS2Inferable(Protocol):
    infer: InferFn


class ManifestEntry(TypedDict):
    path: str
    duration: float


# Use spawn to avoid torch/fork CUDA deserialization issues when running
# multi-process workers.
try:
    mp.set_start_method("spawn", force=True)
except RuntimeError:
    pass


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = REPO_ROOT / "interstellar-tars-01-resemble-denoised.wav"


@dataclass
class WorkerConfig:
    cfg_path: str
    model_dir: str
    use_fp16: bool
    use_cuda_kernel: bool
    use_deepspeed: bool
    device: Optional[str]
    use_accel: bool
    use_torch_compile: bool
    mock_inference: bool = False
    mock_sample_rate: int = 22050
    deterministic: bool = False
    no_sampling: bool = False
    seed: Optional[int] = None


@dataclass
class SynthesisTask:
    sample_id: str
    role: Literal["user", "assistant"]
    text: str
    spk_audio_prompt: Optional[str] = None
    emo_audio_prompt: Optional[str] = None
    emo_vector: Optional[list[float]] = None
    emo_alpha: float = 1.0


@dataclass
class SynthesisResult:
    sample_id: str
    role: Literal["user", "assistant"]
    audio: AudioArray
    sampling_rate: int
    duration: float


WorkerMessage = Union[
    tuple[Literal["ok"], SynthesisResult],
    tuple[Literal["error", "fatal"], dict[str, Any]],
    tuple[Literal["worker_exit"], int],
]


def _mock_audio_from_text(text: str, sample_rate: int = 22050) -> tuple[AudioArray, int, float]:
    """Generate deterministic synthetic audio for benchmarking."""

    digest = hashlib.sha256(text.encode("utf-8", errors="ignore")).digest()
    seed = int.from_bytes(digest[:4], "little")
    rng = np.random.default_rng(seed)
    approx_duration = min(18.0, max(0.35, 0.045 * max(1, len(text.strip()))))
    num_samples = max(1, int(approx_duration * sample_rate))
    audio_arr = rng.standard_normal(num_samples).astype(np.float32) * 0.02
    audio_arr = _normalize_audio(audio_arr)
    return audio_arr, sample_rate, num_samples / sample_rate


def _normalize_audio(audio: npt.ArrayLike) -> AudioArray:
    """Convert IndexTTS2 int-style output to float32 -1..1 for soundfile."""

    audio_arr: AudioArray = np.asarray(audio, dtype=np.float32)
    peak = float(np.max(np.abs(audio_arr))) if audio_arr.size else 0.0
    if peak > 1.1:
        audio_arr = audio_arr / 32767.0
    return audio_arr


def _pad_audio(audio: AudioArray, target_len: int) -> AudioArray:
    if audio.shape[0] >= target_len:
        return audio
    padded: AudioArray = np.zeros(target_len, dtype=audio.dtype)
    padded[: audio.shape[0]] = audio
    return padded


def _write_stereo(left: AudioArray, right: AudioArray, sample_rate: int, output_path: Path) -> float:
    max_len = max(left.shape[0], right.shape[0])
    stereo = np.stack([
        _pad_audio(left, max_len),
        _pad_audio(right, max_len),
    ], axis=1)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), stereo, sample_rate)
    return max_len / sample_rate if sample_rate else 0.0


# ============================================================================
# Common Pipeline Helper Functions (Phase 1 Refactoring)
# ============================================================================

def _common_flush_manifest(manifest_buffer: list[ManifestEntry], index_f: IO[str]) -> None:
    """Common manifest flushing logic used by all pipeline implementations."""
    if not manifest_buffer:
        return
    for entry in manifest_buffer:
        json.dump(entry, index_f)
        index_f.write("\n")
    index_f.flush()
    manifest_buffer.clear()


def _common_finalize_sample(
    sample_id: str,
    outstanding: dict[str, dict[str, SynthesisResult]],
    stereo_dir: Path,
    manifest_buffer: list[ManifestEntry],
    manifest_buffer_size: int,
    index_f: IO[str],
    total_samples_ref: list[int],
    total_duration_ref: list[float],
) -> Optional[tuple[int, float]]:
    """Common sample finalization logic used by all pipeline implementations."""
    pair = outstanding.pop(sample_id)
    user_res = pair.get("user")
    asst_res = pair.get("assistant")
    if user_res is None or asst_res is None:
        return None
    
    if user_res.sampling_rate != asst_res.sampling_rate:
        raise ValueError(
            f"Sample rate mismatch for {sample_id}: {user_res.sampling_rate} vs {asst_res.sampling_rate}"
        )
    
    stereo_rel_path = Path("data_stereo") / f"{sample_id}.wav"
    stereo_abs_path = stereo_dir / f"{sample_id}.wav"
    duration_sec = _write_stereo(
        left=asst_res.audio,
        right=user_res.audio,
        sample_rate=asst_res.sampling_rate,
        output_path=stereo_abs_path,
    )
    
    manifest_buffer.append(
        {
            "path": str(stereo_rel_path).replace(os.sep, "/"),
            "duration": float(duration_sec),
        }
    )
    
    if len(manifest_buffer) >= manifest_buffer_size:
        _common_flush_manifest(manifest_buffer, index_f)
    
    total_samples_ref[0] += 1
    total_duration_ref[0] += duration_sec
    print(
        f"[{total_samples_ref[0]}] wrote {stereo_rel_path} (duration={duration_sec:.2f}s, "
        f"cumulative={total_duration_ref[0]:.2f}s)"
    )
    
    return total_samples_ref[0], total_duration_ref[0]


def _common_enqueue_task(
    task: SynthesisTask,
    task_queue: Union[mp.Queue, queue.Queue],
    stop_event: threading.Event,
    tasks_enqueued_ref: list[int],
) -> bool:
    """Common task enqueueing logic used by all pipeline implementations."""
    while not stop_event.is_set():
        try:
            task_queue.put(task, timeout=0.5)
            tasks_enqueued_ref[0] += 1
            return True
        except queue.Full:
            if stop_event.is_set():
                return False
            continue
    return False


# ============================================================================
# Unified Worker Interface (Phase 2 Refactoring)
# ============================================================================

@dataclass
class WorkerSetup:
    """Unified worker configuration for both process and thread-based pipelines."""
    worker_cfg: WorkerConfig
    worker_count: int
    planner_buffer: int
    manifest_buffer_size: int
    max_gpu_concurrency: Optional[int] = None
    
    def __post_init__(self) -> None:
        self.planner_buffer = max(1, int(self.planner_buffer))
        self.manifest_buffer_size = max(1, int(self.manifest_buffer_size))


class UnifiedWorkerManager:
    """Unified worker management for both process and thread-based pipelines."""
    
    def __init__(self, setup: WorkerSetup, use_processes: bool = True):
        self.setup = setup
        self.use_processes = use_processes
        self.stop_event = threading.Event()
        self.planner_done = threading.Event()
        self.planner_error: queue.Queue[BaseException] = queue.Queue(maxsize=1)
        
        if use_processes:
            self._setup_process_workers()
        else:
            self._setup_thread_workers()
    
    def _setup_process_workers(self) -> None:
        """Setup process-based workers."""
        ctx = mp.get_context("spawn")
        self.task_queue: mp.Queue[Optional[SynthesisTask]] = ctx.Queue(
            maxsize=self.setup.planner_buffer)
        self.result_queue: mp.Queue[WorkerMessage] = ctx.Queue(
            maxsize=self.setup.planner_buffer * max(2, self.setup.worker_count))
        
        self.processes: list[BaseProcess] = []
        for worker_id in range(self.setup.worker_count):
            proc = ctx.Process(
                target=_synthesis_worker,
                args=(worker_id, self.task_queue, self.result_queue, self.setup.worker_cfg),
                daemon=True,
            )
            proc.start()
            self.processes.append(proc)
    
    def _setup_thread_workers(self) -> None:
        """Setup thread-based workers."""
        self.task_queue: queue.Queue[Optional[SynthesisTask]] = queue.Queue(
            maxsize=self.setup.planner_buffer)
        self.result_queue: queue.Queue[WorkerMessage] = queue.Queue(
            maxsize=max(self.setup.planner_buffer * max(2, self.setup.worker_count),
                       self.setup.manifest_buffer_size))
        
        try:
            self.shared_tts = _create_tts(self.setup.worker_cfg)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize IndexTTS2 for threaded workers: {exc}")
        
        # Setup GPU concurrency controls
        if self.setup.max_gpu_concurrency is not None:
            gate_size = max(1, int(self.setup.max_gpu_concurrency))
            self.gpu_gate: Optional[threading.Semaphore] = None
            if not self.setup.worker_cfg.mock_inference:
                self.gpu_gate = threading.Semaphore(gate_size)
            self.inference_lock: Optional[threading.Lock] = None
            if not self.setup.worker_cfg.mock_inference and gate_size <= 1:
                self.inference_lock = threading.Lock()
        else:
            self.gpu_gate = None
            self.inference_lock = None
    
    def get_task_queue(self) -> Union[mp.Queue, queue.Queue]:
        """Get the appropriate task queue type."""
        return self.task_queue
    
    def get_result_queue(self) -> Union[mp.Queue, queue.Queue]:
        """Get the appropriate result queue type."""
        return self.result_queue
    
    def cleanup(self) -> None:
        """Cleanup worker resources."""
        self.stop_event.set()
        if hasattr(self, 'processes'):
            for proc in self.processes:
                if proc.is_alive():
                    proc.terminate()
    
    def wait_for_planner_completion(self) -> None:
        """Wait for planner to complete and check for errors."""
        if hasattr(self, 'planner_thread'):
            self.planner_thread.join()
        
        if not self.planner_error.empty():
            self.stop_event.set()
            raise self.planner_error.get()


# ============================================================================
# Unified Configuration Management (Phase 3 Refactoring)
# ============================================================================

@dataclass
class PipelineConfig:
    """Unified pipeline configuration."""
    input_jsonl: Path
    index_path: Path
    stereo_dir: Path
    tmp_dir: Path
    user_spk_prompt: Optional[str]
    assistant_prompt: Optional[str]
    keep_temp: bool
    max_samples: Optional[int]
    
    def validate(self) -> None:
        """Validate pipeline configuration."""
        if not self.input_jsonl.exists():
            raise FileNotFoundError(f"Input file not found: {self.input_jsonl}")
        if self.max_samples is not None and self.max_samples <= 0:
            raise ValueError("max_samples must be positive or None")


def _create_deterministic_seeds(base_seed: int) -> dict[str, int]:
    """Create deterministic seeds for all random processes."""
    return {
        "numpy_seed": base_seed,
        "torch_seed": base_seed + 1,
        "random_seed": base_seed + 2,
        "worker_seed_offset": base_seed + 1000,
    }


def _setup_deterministic_environment(seeds: dict[str, int]) -> None:
    """Setup deterministic environment for reproducible results."""
    import random
    
    # Set seeds
    np.random.seed(seeds["numpy_seed"])
    
    try:
        import torch
        torch.manual_seed(seeds["torch_seed"])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seeds["torch_seed"])
    except ImportError:
        pass  # torch not available
    
    random.seed(seeds["random_seed"])


def _validate_worker_config(cfg: WorkerConfig) -> None:
    """Validate worker configuration."""
    if cfg.mock_inference:
        return
    
    if not Path(cfg.cfg_path).exists():
        raise FileNotFoundError(f"Config file not found: {cfg.cfg_path}")
    if not Path(cfg.model_dir).exists():
        raise FileNotFoundError(f"Model directory not found: {cfg.model_dir}")
    
    # Validate device
    if cfg.device is not None:
        valid_devices = ["cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        if cfg.device not in valid_devices and not cfg.device.startswith("cuda:"):
            raise ValueError(f"Invalid device: {cfg.device}")


# ============================================================================
# Common Pipeline Execution Logic (Phase 2-3 Refactoring)
# ============================================================================

class CommonPipelineLogic:
    """Common pipeline execution logic shared across all implementations."""
    
    def __init__(
        self,
        config: PipelineConfig,
        worker_setup: WorkerSetup,
        manager: UnifiedWorkerManager,
    ):
        self.config = config
        self.worker_setup = worker_setup
        self.manager = manager
        
        # Common state
        self.total_samples = 0
        self.total_duration = 0.0
        self.outstanding: dict[str, dict[str, Any]] = {}
        self.manifest_buffer: list[ManifestEntry] = []
        self.start_time = time.perf_counter()
        self.tasks_enqueued = 0
        self.samples_planned = 0
        
        # Progress tracking
        self.samples_completed = 0
        self.samples_total = 0
        self.last_progress_update = time.perf_counter()
        self.last_status_update = time.perf_counter()
        
        # References for common helper functions
        self.total_samples_ref = [self.total_samples]
        self.total_duration_ref = [self.total_duration]
        self.tasks_enqueued_ref = [self.tasks_enqueued]
    
    def create_common_functions(self):
        """Create common helper functions with proper closure references."""
        
        def flush_manifest(index_f: IO[str]) -> None:
            _common_flush_manifest(self.manifest_buffer, index_f)

        def finalize_sample(sample_id: str, index_f: IO[str]) -> None:
            result = _common_finalize_sample(
                sample_id=sample_id,
                outstanding=self.outstanding,
                stereo_dir=self.config.stereo_dir,
                manifest_buffer=self.manifest_buffer,
                manifest_buffer_size=self.worker_setup.manifest_buffer_size,
                index_f=index_f,
                total_samples_ref=self.total_samples_ref,
                total_duration_ref=self.total_duration_ref,
            )
            if result:
                self.total_samples, self.total_duration = result
                # Update progress tracking
                self.samples_completed += 1
                # Get duration from the result or calculate from total
                duration = self.total_duration_ref[0] - (self.total_duration if self.total_samples > 1 else 0)
                # Print progress with the new tracking
                self.print_progress(sample_id, duration)

        def enqueue_task(task: SynthesisTask) -> bool:
            return _common_enqueue_task(
                task, self.manager.get_task_queue(), self.manager.stop_event, self.tasks_enqueued_ref)

        return flush_manifest, finalize_sample, enqueue_task
    
    def print_progress(self, sample_id: str, duration: float, force: bool = False) -> None:
        """Print progress update with rate limiting."""
        now = time.perf_counter()
        # Update every second or when forced
        if not force and (now - self.last_progress_update) < 1.0:
            return
        
        self.last_progress_update = now
        elapsed = now - self.start_time
        
        # Calculate rate (samples per minute)
        rate = (self.samples_completed / elapsed * 60) if elapsed > 0 else 0
        
        # Calculate ETA
        remaining = self.samples_total - self.samples_completed
        eta_seconds = (remaining / rate * 60) if rate > 0 else 0
        eta_min = int(eta_seconds / 60)
        
        # Format output
        progress_pct = int(self.samples_completed / self.samples_total * 100) if self.samples_total > 0 else 0
        
        print(
            f"[{self.samples_completed}/{self.samples_total}] ✓ {sample_id} "
            f"({duration:.2f}s) | Total: {self.total_duration:.2f}s | "
            f"Rate: {rate:.1f} samp/min | ETA: {eta_min}min"
        )
    
    def print_status_update(self) -> None:
        """Print periodic status update."""
        now = time.perf_counter()
        # Status update every 10 seconds
        if (now - self.last_status_update) < 10.0:
            return
        
        self.last_status_update = now
        elapsed = now - self.start_time
        progress_pct = int(self.samples_completed / self.samples_total * 100) if self.samples_total > 0 else 0
        
        elapsed_min = int(elapsed / 60)
        elapsed_sec = int(elapsed % 60)
        
        remaining = self.samples_total - self.samples_completed
        rate = (self.samples_completed / elapsed * 60) if elapsed > 0 else 0
        eta_seconds = (remaining / rate * 60) if rate > 0 else 0
        eta_min = int(eta_seconds / 60)
        eta_sec = int(eta_seconds % 60)
        
        print(
            f"[Progress] {self.samples_completed}/{self.samples_total} complete ({progress_pct}%) | "
            f"Elapsed: {elapsed_min}m{elapsed_sec}s | ETA: {eta_min}m{eta_sec}s"
        )
    
    def get_final_stats(self) -> tuple[int, float]:
        """Get final processing statistics."""
        elapsed = time.perf_counter() - self.start_time
        if self.outstanding:
            missing_ids = ", ".join(list(self.outstanding.keys())[:5])
            raise RuntimeError(
                f"Pipeline finished with incomplete samples: {missing_ids}"
            )
        print(
            f"\n[Complete] Pipeline finished in {elapsed:.2f}s — "
            f"samples: {self.total_samples}, total duration: {self.total_duration:.2f}s"
        )
        return self.total_samples, self.total_duration


def _maybe_write_mono(audio: AudioArray, sample_rate: int, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), audio, sample_rate)


def _create_tts(cfg: WorkerConfig) -> Optional[IndexTTS2Inferable]:
    if cfg.mock_inference:
        return None
    return cast(IndexTTS2Inferable, IndexTTS2(
        cfg_path=cfg.cfg_path,
        model_dir=cfg.model_dir,
        use_fp16=cfg.use_fp16,
        use_cuda_kernel=cfg.use_cuda_kernel,
        use_deepspeed=cfg.use_deepspeed,
        device=cfg.device,
        use_accel=cfg.use_accel,
        use_torch_compile=cfg.use_torch_compile,
    ))


def _run_synthesis_task(
    *,
    task: SynthesisTask,
    cfg: WorkerConfig,
    tts: Optional[IndexTTS2Inferable],
    inference_lock: Optional[threading.Lock] = None,
    gpu_semaphore: Optional[threading.Semaphore] = None,
) -> SynthesisResult:
    """Shared inference helper for both process and thread workers."""

    if cfg.mock_inference:
        audio, sampling_rate, duration = _mock_audio_from_text(
            task.text, sample_rate=cfg.mock_sample_rate
        )
        return SynthesisResult(
            sample_id=task.sample_id,
            role=task.role,
            audio=audio,
            sampling_rate=sampling_rate,
            duration=float(duration),
        )

    if tts is None:
        raise ValueError(
            "IndexTTS2 instance is required when mock_inference is False.")

    guard = gpu_semaphore or contextlib.nullcontext()
    with guard:
        if inference_lock is not None:
            inference_lock.acquire()
        try:
            # Setup per-sample deterministic seeding if requested
            generation_kwargs = {}
            if cfg.deterministic or cfg.seed is not None:
                # Derive a deterministic per-sample seed from sample_id and role
                import hashlib
                seed_input = f"{task.sample_id}_{task.role}_{cfg.seed or 42}"
                per_sample_seed = int(hashlib.sha256(seed_input.encode()).hexdigest(), 16) % (2**32)
                
                import random
                import torch
                random.seed(per_sample_seed)
                np.random.seed(per_sample_seed) 
                torch.manual_seed(per_sample_seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(per_sample_seed)
            
            # Apply no-sampling settings if requested
            if cfg.no_sampling:
                generation_kwargs.update({
                    'do_sample': False,
                    'top_p': 1.0,
                    'top_k': 0,
                    'temperature': 1.0,
                    'num_beams': 1,
                    'repetition_penalty': 1.0,
                })
            
            infer_fn: InferFn = cast(InferFn, tts.infer)
            inference = infer_fn(
                spk_audio_prompt=task.spk_audio_prompt,
                text=task.text,
                output_path=None,
                emo_audio_prompt=task.emo_audio_prompt,
                emo_vector=task.emo_vector,
                emo_alpha=task.emo_alpha,
                return_audio=True,
                return_numpy=True,
                verbose=False,
                **generation_kwargs
            )
        finally:
            if inference_lock is not None:
                inference_lock.release()

    if inference is None:
        raise RuntimeError("IndexTTS2 returned no audio output")

    audio_source: npt.ArrayLike = inference.audio
    audio = _normalize_audio(np.asarray(
        audio_source, dtype=np.float32).reshape(-1))
    duration = (
        float(inference.duration_sec)
        if inference.duration_sec is not None
        else audio.shape[-1] / inference.sampling_rate
    )
    sampling_rate = int(inference.sampling_rate)
    return SynthesisResult(
        sample_id=task.sample_id,
        role=task.role,
        audio=audio,
        sampling_rate=sampling_rate,
        duration=float(duration),
    )


def _synthesis_worker(
    worker_id: int,
    task_queue: mp.Queue[Optional[SynthesisTask]],
    result_queue: mp.Queue[WorkerMessage],
    cfg: WorkerConfig,
) -> None:
    """Worker loop that keeps an IndexTTS2 instance hot and returns in-memory audio."""

    try:
        tts = _create_tts(cfg)
    except Exception as exc:  # pragma: no cover - worker bootstrap failures are surfaced upstream
        result_queue.put(
            ("fatal", {"worker_id": worker_id, "message": str(exc)}))
        return

    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            res = _run_synthesis_task(
                task=task, cfg=cfg, tts=tts, inference_lock=None, gpu_semaphore=None
            )
            result_queue.put(("ok", res))
        except Exception as exc:  # pragma: no cover - surfaced to main process
            result_queue.put((
                "error",
                {
                    "worker_id": worker_id,
                    "sample_id": task.sample_id,
                    "role": task.role,
                    "message": str(exc),
                },
            ))

    result_queue.put(("worker_exit", worker_id))


def _synthesis_worker_threaded(
    worker_id: int,
    task_queue: queue.Queue[Optional[SynthesisTask]],
    result_queue: queue.Queue[WorkerMessage],
    cfg: WorkerConfig,
    tts: Optional[IndexTTS2Inferable],
    gpu_semaphore: Optional[threading.Semaphore] = None,
    inference_lock: Optional[threading.Lock] = None,
) -> None:
    """Threaded worker loop that reuses a shared IndexTTS2 instance."""

    while True:
        task = task_queue.get()
        if task is None:
            break
        try:
            res = _run_synthesis_task(
                task=task,
                cfg=cfg,
                tts=tts,
                inference_lock=inference_lock,
                gpu_semaphore=gpu_semaphore,
            )
            # Construct task_id as expected by the consumer loop
            task_id = f"{task.sample_id}_{task.role}"
            result_queue.put(("result", (task_id, res)))
        except Exception as exc:
            result_queue.put((
                "error",
                {
                    "worker_id": worker_id,
                    "sample_id": task.sample_id,
                    "role": task.role,
                    "message": str(exc),
                },
            ))

    result_queue.put(("worker_exit", worker_id))


def _build_dataset_parallel(
    *,
    input_jsonl: Path,
    index_path: Path,
    stereo_dir: Path,
    tmp_dir: Path,
    user_spk_prompt: Optional[str],
    assistant_prompt: Optional[str],
    keep_temp: bool,
    max_samples: Optional[int],
    worker_cfg: WorkerConfig,
    worker_count: int,
    planner_buffer: int,
    manifest_buffer_size: int,
) -> tuple[int, float]:
    """Refactored parallel pipeline using unified components."""
    
    # Create unified configuration
    pipeline_config = PipelineConfig(
        input_jsonl=input_jsonl,
        index_path=index_path,
        stereo_dir=stereo_dir,
        tmp_dir=tmp_dir,
        user_spk_prompt=user_spk_prompt,
        assistant_prompt=assistant_prompt,
        keep_temp=keep_temp,
        max_samples=max_samples,
    )
    pipeline_config.validate()
    
    worker_setup = WorkerSetup(
        worker_cfg=worker_cfg,
        worker_count=worker_count,
        planner_buffer=planner_buffer,
        manifest_buffer_size=manifest_buffer_size,
    )
    
    # Validate worker configuration
    _validate_worker_config(worker_cfg)
    
    # Setup deterministic environment if needed
    if worker_cfg.deterministic or worker_cfg.seed is not None:
        base_seed = worker_cfg.seed or 42
        seeds = _create_deterministic_seeds(base_seed)
        _setup_deterministic_environment(seeds)
    
    try:
        # Create unified worker manager
        manager = UnifiedWorkerManager(worker_setup, use_processes=True)
        
        # Create common pipeline logic
        pipeline_logic = CommonPipelineLogic(pipeline_config, worker_setup, manager)
        flush_manifest, finalize_sample, enqueue_task = pipeline_logic.create_common_functions()
        
        # Open index file for writing
        with open(index_path, "w", encoding="utf-8") as index_f:
            json.dump({"version": "v1", "type": "indexed_dataset"}, index_f)
            index_f.write("\n")
            
            # Start planner thread
            def planner_loop():
                nonlocal pipeline_logic
                try:
                    with open(pipeline_config.input_jsonl, "r", encoding="utf-8") as f_in:
                        for line_idx, line in enumerate(f_in):
                            if manager.stop_event.is_set():
                                break
                            if max_samples is not None and pipeline_logic.samples_planned >= max_samples:
                                break
                            
                            data = json.loads(line.strip())
                            sample_id = data.get("sample_id", data.get("id", f"sample_{line_idx}"))
                            
                            # Plan user synthesis
                            user_task = SynthesisTask(
                                sample_id=sample_id,
                                role="user",
                                text=data.get("user_text", data.get("user", "")),
                                spk_audio_prompt=pipeline_config.user_spk_prompt,
                            )
                            if enqueue_task(user_task):
                                if sample_id not in pipeline_logic.outstanding:
                                    pipeline_logic.outstanding[sample_id] = {}
                                pipeline_logic.samples_planned += 1
                            
                            # Plan assistant synthesis
                            asst_task = SynthesisTask(
                                sample_id=sample_id,
                                role="assistant", 
                                text=data.get("assistant_text", data.get("assistant", "")),
                                spk_audio_prompt=pipeline_config.assistant_prompt,
                            )
                            if enqueue_task(asst_task):
                                if sample_id not in pipeline_logic.outstanding:
                                    pipeline_logic.outstanding[sample_id] = {}
                                pipeline_logic.samples_planned += 1
                except Exception as exc:
                    manager.planner_error.put(exc)
                finally:
                    manager.planner_done.set()
                    # Signal completion to workers
                    for _ in range(worker_count):
                        manager.get_task_queue().put(None)
            
            # Start and run planner
            import threading
            planner_thread = threading.Thread(target=planner_loop, daemon=True)
            planner_thread.start()
            manager.planner_thread = planner_thread
            
            # Process results
            worker_exits = 0
            while worker_exits < worker_count:
                try:
                    msg_type, msg_data = manager.get_result_queue().get(timeout=0.5)
                    if msg_type == "result":
                        task_id, result = msg_data
                        sample_id, role = task_id.rsplit("_", 1)
                        if sample_id in pipeline_logic.outstanding:
                            pipeline_logic.outstanding[sample_id][role] = result
                            if len(pipeline_logic.outstanding[sample_id]) == 2:
                                finalize_sample(sample_id, index_f)
                    elif msg_type == "error":
                        print(f"Worker error: {msg_data}", file=sys.stderr)
                        manager.stop_event.set()
                        raise RuntimeError(f"Worker error: {msg_data}")
                    elif msg_type == "worker_exit":
                        worker_exits += 1
                    else:
                        manager.stop_event.set()
                        raise RuntimeError(f"Unknown worker message type: {msg_type}")
                except queue.Empty:
                    if manager.stop_event.is_set():
                        break
                    continue
            
            # CRITICAL FIX: Drain any remaining results from queue after all workers exit
            # Workers may have put results before their exit messages, so we need to process
            # any remaining results to prevent "incomplete samples" errors
            print(f"[Cleanup] All workers exited, draining remaining results from queue...")
            drained_count = 0
            while True:
                try:
                    msg_type, msg_data = manager.get_result_queue().get(timeout=0.1)
                    if msg_type == "result":
                        task_id, result = msg_data
                        sample_id, role = task_id.rsplit("_", 1)
                        if sample_id in pipeline_logic.outstanding:
                            pipeline_logic.outstanding[sample_id][role] = result
                            if len(pipeline_logic.outstanding[sample_id]) == 2:
                                finalize_sample(sample_id, index_f)
                            drained_count += 1
                    # Ignore other message types during drain
                except queue.Empty:
                    break
            
            if drained_count > 0:
                print(f"[Cleanup] Processed {drained_count} remaining results from queue")
            
            flush_manifest(index_f)
        
        # Wait for completion and get final stats
        manager.wait_for_planner_completion()
        return pipeline_logic.get_final_stats()
        
    except Exception:
        manager.cleanup()
        raise


def _build_dataset_parallel_threaded(
    *,
    input_jsonl: Path,
    index_path: Path,
    stereo_dir: Path,
    tmp_dir: Path,
    user_spk_prompt: Optional[str],
    assistant_prompt: Optional[str],
    keep_temp: bool,
    max_samples: Optional[int],
    worker_cfg: WorkerConfig,
    worker_count: int,
    planner_buffer: int,
    manifest_buffer_size: int,
    max_gpu_concurrency: int,
) -> tuple[int, float]:
    """Refactored threaded pipeline using unified components."""
    
    # Create unified configuration
    pipeline_config = PipelineConfig(
        input_jsonl=input_jsonl,
        index_path=index_path,
        stereo_dir=stereo_dir,
        tmp_dir=tmp_dir,
        user_spk_prompt=user_spk_prompt,
        assistant_prompt=assistant_prompt,
        keep_temp=keep_temp,
        max_samples=max_samples,
    )
    pipeline_config.validate()
    
    worker_setup = WorkerSetup(
        worker_cfg=worker_cfg,
        worker_count=worker_count,
        planner_buffer=planner_buffer,
        manifest_buffer_size=manifest_buffer_size,
        max_gpu_concurrency=max_gpu_concurrency,
    )
    
    # Validate worker configuration
    _validate_worker_config(worker_cfg)
    
    # Setup deterministic environment if needed
    if worker_cfg.deterministic or worker_cfg.seed is not None:
        base_seed = worker_cfg.seed or 42
        seeds = _create_deterministic_seeds(base_seed)
        _setup_deterministic_environment(seeds)
    
    try:
        # Create unified worker manager (threaded)
        manager = UnifiedWorkerManager(worker_setup, use_processes=False)
        
        # Create common pipeline logic
        pipeline_logic = CommonPipelineLogic(pipeline_config, worker_setup, manager)
        flush_manifest, finalize_sample, enqueue_task = pipeline_logic.create_common_functions()
        
        # Open index file for writing
        with open(index_path, "w", encoding="utf-8") as index_f:
            json.dump({"version": "v1", "type": "indexed_dataset"}, index_f)
            index_f.write("\n")
            
            # Start planner thread (using shared TTS for threaded workers)
            def planner_loop():
                nonlocal pipeline_logic
                try:
                    with open(pipeline_config.input_jsonl, "r", encoding="utf-8") as f_in:
                        for line_idx, line in enumerate(f_in):
                            if manager.stop_event.is_set():
                                break
                            if max_samples is not None and pipeline_logic.samples_planned >= max_samples:
                                break
                            
                            data = json.loads(line.strip())
                            sample_id = data.get("sample_id", data.get("id", f"sample_{line_idx}"))
                            
                            # Plan user synthesis
                            user_task = SynthesisTask(
                                sample_id=sample_id,
                                role="user",
                                text=data.get("user_text", data.get("user", "")),
                                spk_audio_prompt=pipeline_config.user_spk_prompt,
                            )
                            if enqueue_task(user_task):
                                if sample_id not in pipeline_logic.outstanding:
                                    pipeline_logic.outstanding[sample_id] = {}
                                pipeline_logic.samples_planned += 1
                            
                            # Plan assistant synthesis
                            asst_task = SynthesisTask(
                                sample_id=sample_id,
                                role="assistant", 
                                text=data.get("assistant_text", data.get("assistant", "")),
                                spk_audio_prompt=pipeline_config.assistant_prompt,
                            )
                            if enqueue_task(asst_task):
                                if sample_id not in pipeline_logic.outstanding:
                                    pipeline_logic.outstanding[sample_id] = {}
                                pipeline_logic.samples_planned += 1
                except Exception as exc:
                    manager.planner_error.put(exc)
                finally:
                    manager.planner_done.set()
                    # Signal completion to workers
                    for _ in range(worker_count):
                        manager.get_task_queue().put(None)
            
            # Start and run planner
            import threading
            planner_thread = threading.Thread(target=planner_loop, daemon=True)
            planner_thread.start()
            manager.planner_thread = planner_thread
            
            # Start worker threads (for threaded implementation)
            worker_threads = []
            for worker_id in range(worker_count):
                thread = threading.Thread(
                    target=_synthesis_worker_threaded,
                    args=(worker_id, manager.get_task_queue(), manager.get_result_queue(), 
                          worker_cfg, manager.shared_tts, manager.gpu_gate, manager.inference_lock),
                    daemon=True,
                )
                thread.start()
                worker_threads.append(thread)
            
            # Process results
            worker_exits = 0
            while worker_exits < worker_count:
                try:
                    msg_type, msg_data = manager.get_result_queue().get(timeout=0.5)
                    if msg_type == "result":
                        task_id, result = msg_data
                        sample_id, role = task_id.rsplit("_", 1)
                        if sample_id in pipeline_logic.outstanding:
                            pipeline_logic.outstanding[sample_id][role] = result
                            if len(pipeline_logic.outstanding[sample_id]) == 2:
                                finalize_sample(sample_id, index_f)
                    elif msg_type == "error":
                        print(f"Worker error: {msg_data}", file=sys.stderr)
                        manager.stop_event.set()
                        raise RuntimeError(f"Worker error: {msg_data}")
                    elif msg_type == "worker_exit":
                        worker_exits += 1
                    else:
                        manager.stop_event.set()
                        raise RuntimeError(f"Unknown worker message type: {msg_type}")
                except queue.Empty:
                    if manager.stop_event.is_set():
                        break
                    continue
            
            # CRITICAL FIX: Drain any remaining results from queue after all workers exit
            print(f"[Cleanup] All workers exited, draining remaining results from queue...")
            drained_count = 0
            while True:
                try:
                    msg_type, msg_data = manager.get_result_queue().get(timeout=0.1)
                    if msg_type == "result":
                        task_id, result = msg_data
                        sample_id, role = task_id.rsplit("_", 1)
                        if sample_id in pipeline_logic.outstanding:
                            pipeline_logic.outstanding[sample_id][role] = result
                            if len(pipeline_logic.outstanding[sample_id]) == 2:
                                finalize_sample(sample_id, index_f)
                            drained_count += 1
                    # Ignore other message types during drain
                except queue.Empty:
                    break
            
            if drained_count > 0:
                print(f"[Cleanup] Processed {drained_count} remaining results from queue")
            
            flush_manifest(index_f)
        
        # Wait for completion and get final stats
        manager.wait_for_planner_completion()
        return pipeline_logic.get_final_stats()
        
    except Exception:
        manager.cleanup()
        raise


def synthesize_line(
    tts: Optional[IndexTTS2Inferable],
    text: str,
    output_path: Path,
    spk_audio_prompt: Optional[str] = None,
    emo_audio_prompt: Optional[str] = None,
    emo_vector: Optional[list[float]] = None,
    emo_alpha: float = 1.0,
    *,
    mock_inference: bool = False,
    mock_sample_rate: int = 22050,
) -> tuple[Path, float, int]:
    """Synthesize a single utterance and return (path, duration_sec, sample_rate)."""

    if not text.strip():
        raise ValueError(
            "Text is empty; dataset rows must provide non-empty text fields.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if mock_inference:
        audio, sr, duration_sec = _mock_audio_from_text(
            text, sample_rate=mock_sample_rate)
        _maybe_write_mono(audio, sr, output_path)
        return output_path, float(duration_sec), int(sr)
    if tts is None:
        raise ValueError(
            "IndexTTS2 instance is required when mock_inference is False.")

    try:
        infer_fn: InferFn = cast(InferFn, tts.infer)
        inference = infer_fn(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=None,
            emo_audio_prompt=emo_audio_prompt,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            return_audio=True,
            return_numpy=True,
            verbose=False,
        )
        if inference is None:
            raise RuntimeError("IndexTTS2 returned no audio output")
        audio_source: npt.ArrayLike = inference.audio
        audio = _normalize_audio(np.asarray(
            audio_source, dtype=np.float32).reshape(-1))
        sr = int(inference.sampling_rate)
        duration_sec = (
            float(
                inference.duration_sec) if inference.duration_sec is not None else audio.shape[0] / sr
        )
        _maybe_write_mono(audio, sr, output_path)
        return output_path, float(duration_sec), int(sr)
    except Exception as exc:  # pragma: no cover - direct passthrough to surface inference failures
        raise RuntimeError(
            f"IndexTTS2 inference failed for text snippet '{text[:32]}...'") from exc


def make_stereo(left_wav: Path, right_wav: Path, stereo_out: Path) -> tuple[float, int]:
    """Load two mono WAVs, pad to a shared length, and emit a stereo WAV."""
    
    left_raw, sr_left = cast(tuple[Float64Array, int], sf.read(str(left_wav)))
    right_raw, sr_right = cast(tuple[Float64Array, int], sf.read(str(right_wav)))

    left: AudioArray = np.asarray(left_raw, dtype=np.float32)
    right: AudioArray = np.asarray(right_raw, dtype=np.float32)

    if left.ndim == 2:
        left = left.mean(axis=1)
    if right.ndim == 2:
        right = right.mean(axis=1)

    if sr_left != sr_right:
        raise ValueError(f"Sample rate mismatch: {sr_left} vs {sr_right}")

    # Use the existing _write_stereo function to avoid duplication
    duration_sec = _write_stereo(left, right, sr_left, stereo_out)
    return float(duration_sec), int(sr_left)


def _build_dataset_legacy(
    *,
    input_jsonl: Path,
    index_path: Path,
    stereo_dir: Path,
    tmp_dir: Path,
    user_spk_prompt: Optional[str],
    assistant_prompt: Optional[str],
    max_samples: Optional[int],
    keep_temp: bool,
    worker_cfg: WorkerConfig,
) -> tuple[int, float]:
    tts: Optional[IndexTTS2Inferable] = None
    if not worker_cfg.mock_inference:
        tts = cast(
            IndexTTS2Inferable,
            IndexTTS2(
                cfg_path=worker_cfg.cfg_path,
                model_dir=worker_cfg.model_dir,
                use_fp16=worker_cfg.use_fp16,
                use_cuda_kernel=worker_cfg.use_cuda_kernel,
                use_deepspeed=worker_cfg.use_deepspeed,
                device=worker_cfg.device,
                use_accel=worker_cfg.use_accel,
                use_torch_compile=worker_cfg.use_torch_compile,
            ),
        )

    total_duration = 0.0
    total_samples = 0

    with open(index_path, "w", encoding="utf-8") as index_f, open(
        input_jsonl, "r", encoding="utf-8"
    ) as f_in:
        for line_idx, line in enumerate(f_in):
            if max_samples is not None and total_samples >= max_samples:
                break

            line = line.strip()
            if not line:
                continue

            try:
                sample = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON on line {line_idx + 1}: {exc}") from exc

            for required_key in ("user_text", "assistant_text"):
                if required_key not in sample:
                    raise KeyError(
                        f"Missing '{required_key}' field on line {line_idx + 1}."
                    )

            sample_id = sample.get("id", f"sample_{line_idx:06d}")
            user_text = sample["user_text"]
            assistant_text = sample["assistant_text"]

            tmp_user = tmp_dir / f"{sample_id}_user.wav"
            tmp_asst = tmp_dir / f"{sample_id}_assistant.wav"

            tmp_user = tmp_dir / f"{sample_id}_user.wav"
            tmp_asst = tmp_dir / f"{sample_id}_assistant.wav"

            synthesize_line(
                tts=tts,
                text=user_text,
                output_path=tmp_user,
                spk_audio_prompt=user_spk_prompt,
                mock_inference=worker_cfg.mock_inference,
                mock_sample_rate=worker_cfg.mock_sample_rate,
            )

            synthesize_line(
                tts=tts,
                text=assistant_text,
                output_path=tmp_asst,
                spk_audio_prompt=assistant_prompt,
                mock_inference=worker_cfg.mock_inference,
                mock_sample_rate=worker_cfg.mock_sample_rate,
            )

            stereo_rel_path = Path("data_stereo") / f"{sample_id}.wav"
            stereo_abs_path = stereo_dir / f"{sample_id}.wav"

            duration_sec, _ = make_stereo(
                left_wav=tmp_asst,
                right_wav=tmp_user,
                stereo_out=stereo_abs_path,
            )

            json.dump(
                {
                    "path": str(stereo_rel_path).replace(os.sep, "/"),
                    "duration": float(duration_sec),
                },
                index_f,
            )
            index_f.write("\n")
            index_f.flush()

            total_samples += 1
            total_duration += duration_sec
            print(
                f"[{total_samples}] wrote {stereo_rel_path} (duration={duration_sec:.2f}s, "
                f"cumulative={total_duration:.2f}s)"
            )

            if not keep_temp:
                tmp_user.unlink(missing_ok=True)
                tmp_asst.unlink(missing_ok=True)

    return total_samples, total_duration


def build_dataset(
    input_jsonl: Path,
    output_root: Path,
    cfg_path: Path,
    model_dir: Path,
    user_spk_prompt: Optional[str],
    assistant_spk_prompt: Optional[str],
    dataset_name: str = "mycooldataset",
    use_fp16: bool = False,
    use_cuda_kernel: bool = False,
    use_deepspeed: bool = False,
    max_samples: Optional[int] = None,
    keep_temp: bool = False,
    device: Optional[str] = None,
    use_accel: bool = False,
    use_torch_compile: bool = False,
    worker_count: int = 0,
    planner_buffer: int = 8,
    manifest_buffer_size: int = 32,
    worker_backend: Literal["auto", "process", "thread"] = "auto",
    max_gpu_concurrency: int = 1,
    legacy_io: bool = False,
    mock_inference: bool = False,
    seed: Optional[int] = None,
    deterministic: bool = False,
    no_sampling: bool = False,
):
    """
    Build a moshi-finetune compatible dataset using IndexTTS2.
    """
    output_root = Path(output_root)
    stereo_dir = output_root / "data_stereo"
    index_path = output_root / f"{dataset_name}.jsonl"

    stereo_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_root / "tmp_mono"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # Setup deterministic mode if requested
    if deterministic or seed is not None:
        import random
        import torch
        import numpy as np
        
        global_seed = seed if seed is not None else 42
        print(f"[deterministic] Setting global seed to {global_seed}")
        
        random.seed(global_seed)
        np.random.seed(global_seed)
        torch.manual_seed(global_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(global_seed)
        
        if deterministic:
            print("[deterministic] Enabling torch deterministic algorithms")
            torch.use_deterministic_algorithms(True)
            if torch.cuda.is_available():
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                # Disable TF32 for determinism
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False

    if mock_inference:
        print(
            "[mock] Using synthetic audio for benchmarking; IndexTTS2 will not be loaded.")

    if user_spk_prompt and not Path(user_spk_prompt).expanduser().exists():
        raise FileNotFoundError(
            f"User speaker prompt '{user_spk_prompt}' was not found."
        )
    assistant_prompt = assistant_spk_prompt or user_spk_prompt
    if assistant_prompt and not Path(assistant_prompt).expanduser().exists():
        raise FileNotFoundError(
            f"Assistant speaker prompt '{assistant_prompt}' was not found."
        )

    worker_cfg = WorkerConfig(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
        device=device,
        use_accel=use_accel,
        use_torch_compile=use_torch_compile,
        mock_inference=mock_inference,
        deterministic=deterministic,
        no_sampling=no_sampling,
        seed=seed,
    )

    max_gpu_concurrency = max(1, int(max_gpu_concurrency or 1))
    
    # Force single GPU concurrency for deterministic parity testing
    if deterministic and max_gpu_concurrency > 1:
        print(f"[deterministic] Forcing max_gpu_concurrency=1 (was {max_gpu_concurrency}) for parity testing")
        max_gpu_concurrency = 1
    
    backend_choice = (worker_backend or "auto").lower()
    device_lower = (device or "").lower()
    use_thread_backend = False
    if backend_choice == "thread":
        use_thread_backend = True
    elif backend_choice == "process":
        use_thread_backend = False
    else:
        use_thread_backend = (
            worker_count
            and worker_count > 1
            and not mock_inference
            and device_lower.startswith("cuda")
        )

    effective_planner_buffer = planner_buffer
    effective_manifest_buffer = manifest_buffer_size
    if use_thread_backend and device_lower.startswith("cuda"):
        effective_planner_buffer = max(
            2, min(planner_buffer, worker_count * 2))
        effective_manifest_buffer = max(
            1, min(manifest_buffer_size, effective_planner_buffer * 4)
        )
        if (
            effective_planner_buffer != planner_buffer
            or effective_manifest_buffer != manifest_buffer_size
        ):
            print(
                f"[threaded] tuning buffers for shared-GPU mode "
                f"(planner={effective_planner_buffer}, manifest={effective_manifest_buffer})"
            )

    use_parallel = not legacy_io and worker_count and worker_count > 0
    if use_parallel:
        backend_label = "thread" if use_thread_backend else "process"
        print(
            f"Using {backend_label} backend with worker-count={worker_count} on device={device or 'default'}"
        )
        if use_thread_backend:
            total_samples, total_duration = _build_dataset_parallel_threaded(
                input_jsonl=input_jsonl,
                index_path=index_path,
                stereo_dir=stereo_dir,
                tmp_dir=tmp_dir,
                user_spk_prompt=user_spk_prompt,
                assistant_prompt=assistant_prompt,
                keep_temp=keep_temp,
                max_samples=max_samples,
                worker_cfg=worker_cfg,
                worker_count=worker_count,
                planner_buffer=effective_planner_buffer,
                manifest_buffer_size=effective_manifest_buffer,
                max_gpu_concurrency=max_gpu_concurrency,
            )
        else:
            total_samples, total_duration = _build_dataset_parallel(
                input_jsonl=input_jsonl,
                index_path=index_path,
                stereo_dir=stereo_dir,
                tmp_dir=tmp_dir,
                user_spk_prompt=user_spk_prompt,
                assistant_prompt=assistant_prompt,
                keep_temp=keep_temp,
                max_samples=max_samples,
                worker_cfg=worker_cfg,
                worker_count=worker_count,
                planner_buffer=effective_planner_buffer,
                manifest_buffer_size=effective_manifest_buffer,
            )
    else:
        total_samples, total_duration = _build_dataset_legacy(
            input_jsonl=input_jsonl,
            index_path=index_path,
            stereo_dir=stereo_dir,
            tmp_dir=tmp_dir,
            user_spk_prompt=user_spk_prompt,
            assistant_prompt=assistant_prompt,
            max_samples=max_samples,
            keep_temp=keep_temp,
            worker_cfg=worker_cfg,
        )

    avg_dur = (total_duration / total_samples) if total_samples else 0.0
    print(
        f"\nDone. JSONL index written to: {index_path}\n"
        f"Summary -> samples: {total_samples}, total duration: {total_duration:.2f}s, avg: {avg_dur:.2f}s"
    )
    print(
        "You can now run (from moshi-finetune repo):\n"
        f"  python annotate.py {index_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Build moshi-finetune dataset using IndexTTS2."
    )
    parser.add_argument(
        "--input-jsonl",
        type=Path,
        required=True,
        help="Input JSONL with fields: id (optional), user_text, assistant_text.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Output directory, e.g. data/mycooldataset_root",
    )
    parser.add_argument(
        "--cfg-path",
        type=Path,
        required=True,
        help="Path to IndexTTS2 config.yaml (e.g. checkpoints/config.yaml).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        required=True,
        help="Directory with IndexTTS2 checkpoints (e.g. checkpoints).",
    )
    parser.add_argument(
        "--user-spk-prompt",
        type=str,
        default=str(DEFAULT_PROMPT),
        help=(
            "Reference audio for user voice (defaults to interstellar-tars-01-resemble-denoised.wav)."
        ),
    )
    parser.add_argument(
        "--assistant-spk-prompt",
        type=str,
        default=None,
        help="Reference audio for assistant/moshi voice "
             "(defaults to user-spk-prompt if omitted).",
    )
    parser.add_argument(
        "--dataset-name",
        type=str,
        default="mycooldataset",
        help="Base name for the JSONL index file.",
    )
    parser.add_argument(
        "--use-fp16",
        action="store_true",
        help="Use FP16 inference for IndexTTS2.",
    )
    parser.add_argument(
        "--use-cuda-kernel",
        action="store_true",
        help="Use CUDA kernel optimizations for IndexTTS2.",
    )
    parser.add_argument(
        "--use-deepspeed",
        action="store_true",
        help="Use DeepSpeed for IndexTTS2.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional limit on number of samples to process (handy for smoke tests).",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep intermediate mono WAVs for inspection instead of deleting them.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to bind each IndexTTS2 worker to (e.g. cuda:0, cpu).",
    )
    parser.add_argument(
        "--use-accel",
        action="store_true",
        help="Enable IndexTTS2 accelerator engine if available.",
    )
    parser.add_argument(
        "--use-torch-compile",
        action="store_true",
        help="Wrap IndexTTS2 model forward calls in torch.compile for extra speed.",
    )
    parser.add_argument(
        "--worker-count",
        type=int,
        default=0,
        help="Number of parallel synthesis workers to spawn (0 keeps legacy single worker).",
    )
    parser.add_argument(
        "--worker-backend",
        type=str,
        choices=["auto", "process", "thread"],
        default="auto",
        help=(
            "Parallel backend to use. 'auto' will use thread workers on CUDA to share a "
            "single model copy and reduce VRAM pressure; 'process' keeps the existing "
            "multiprocessing pipeline."
        ),
    )
    parser.add_argument(
        "--max-gpu-concurrency",
        type=int,
        default=1,
        help=(
            "Max in-flight GPU calls when using the threaded backend (helps prevent OOM on "
            "single-GPU setups)."
        ),
    )
    parser.add_argument(
        "--planner-buffer",
        type=int,
        default=8,
        help="Max tasks queued ahead of workers when using the parallel pipeline.",
    )
    parser.add_argument(
        "--manifest-buffer-size",
        type=int,
        default=32,
        help="How many manifest rows to buffer before flushing to disk.",
    )
    parser.add_argument(
        "--legacy-io",
        action="store_true",
        help="Force the legacy temp-file pipeline even if worker-count > 0.",
    )
    parser.add_argument(
        "--mock-inference",
        action="store_true",
        help="Skip IndexTTS2 and emit synthetic audio for benchmarking throughput only.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Global random seed for reproducible results (used for parity testing).",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode: set global seeds, torch deterministic algorithms, and disable TF32.",
    )
    parser.add_argument(
        "--no-sampling",
        action="store_true",
        help="Disable stochastic sampling in IndexTTS2 (sets do_sample=False, top_p=1.0, etc.) for deterministic output.",
    )

    args = parser.parse_args()

    build_dataset(
        input_jsonl=args.input_jsonl,
        output_root=args.output_root,
        cfg_path=args.cfg_path,
        model_dir=args.model_dir,
        user_spk_prompt=args.user_spk_prompt,
        assistant_spk_prompt=args.assistant_spk_prompt,
        dataset_name=args.dataset_name,
        use_fp16=args.use_fp16,
        use_cuda_kernel=args.use_cuda_kernel,
        use_deepspeed=args.use_deepspeed,
        max_samples=args.max_samples,
        keep_temp=args.keep_temp,
        device=args.device,
        use_accel=args.use_accel,
        use_torch_compile=args.use_torch_compile,
        worker_count=args.worker_count,
        worker_backend=args.worker_backend,
        max_gpu_concurrency=args.max_gpu_concurrency,
        planner_buffer=args.planner_buffer,
        manifest_buffer_size=args.manifest_buffer_size,
        legacy_io=args.legacy_io,
        mock_inference=args.mock_inference,
        seed=args.seed,
        deterministic=args.deterministic,
        no_sampling=args.no_sampling,
    )


if __name__ == "__main__":
    main()
