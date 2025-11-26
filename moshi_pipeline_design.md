---
description: Optimized Moshi dataset builder pipeline and IndexTTS2 hook design
last_updated: 2025-11-24
related_issue: https://github.com/grantjr1842/index-tts/issues/3
---

# Moshi Dataset Builder Optimization Plan

## 1. Problem Statement
- `tools/build_moshi_dataset_with_indexts.py` currently performs *strictly sequential* inference + file I/O work, so the GPU sits idle while mono WAVs are re-read for stereo packing and JSONL flushing happens one record at a time.
- Every utterance re-parses speaker prompts and re-opens temporary files, leading to redundant disk churn and reduced throughput when targeting large Moshi finetune datasets.
- Existing CLI toggles only expose FP16/CUDA/DeepSpeed, leaving no way to configure worker pools, device binding, or acceleration features introduced inside `IndexTTS2` (e.g., accelerator engine, torch.compile, etc.).

**Success metrics**
1. ≥2× throughput versus the current single-thread baseline on `examples/moshi_sample.jsonl` (same hardware, documented commands/logs).
2. No regression in CPU-only mode or in the `--keep-temp` legacy path.
3. Logs/CLI feedback clearly communicate worker progress and queue depth for large jobs.

## 2. Current Bottlenecks
1. **Disk-bound mono/stereo loop** – `synthesize_line` always hits disk (`sf.read`) after `IndexTTS2.infer` already wrote the file. We lose the in-memory tensor returned by IndexTTS2 and recompute durations from scratch.
2. **Serial JSONL writes** – metadata is flushed per record, preventing buffered writes and preventing parallel work.
3. **Single-threaded pipeline** – we must wait for both user + assistant generations before stereo assembly; there is no overlap between JSON parsing, inference, or file emission.
4. **Prompt cache resets** – even though `IndexTTS2` caches prompts internally, the builder does not expose CLI switches to keep a long-lived instance warmed on the desired device.

## 3. Target Architecture

```
┌────────┐    ┌────────────────┐    ┌──────────────────────┐    ┌──────────────────────┐
│Reader  │───▶│ Task Planner   │───▶│ Synth Workers (GPU)  │───▶│ Stereo + Manifest    │
└────────┘    └────────────────┘    └──────────────────────┘    └──────────────────────┘
```

### 3.1 Reader & Planner
- Stream the JSONL file once, validating schema + expanding each record into *two* synthesis tasks (user + assistant) tagged with `sample_id`, role, and prompt selection.
- Provide `--max-samples` short-circuiting before enqueuing tasks to avoid partial sample pairs.
- Support back-pressure via a bounded queue (`--planner-buffer N`) so we do not load the entire dataset into memory.

### 3.2 Synthesis Workers
- Maintain *one* `IndexTTS2` instance per worker (start in the worker process/thread) to keep prompt/style caches hot.
- Pull tasks from the planner queue and call a new memory-returning API (see §4) to obtain `(wav_tensor, sample_rate, duration)` without touching the filesystem.
- Emit a `SynthesisResult` dataclass containing role, sample_id, duration, SR, and the raw mono tensor. Send results to the stereo stage via another queue.
- CLI knobs: `--worker-count`, `--device`, `--use-accel`, `--use-torch-compile`, `--seed`, etc., passed through to `IndexTTS2`.

### 3.3 Stereo Assembly & Manifest Writer
- Accumulate both roles per `sample_id` in-memory; once both arrive, pad + interleave to stereo using NumPy and immediately write the final WAV (single disk write per sample).
- Buffer manifest entries in memory (e.g., 32–64 entries) before flushing to JSONL to reduce fsync pressure; still auto-flush on shutdown.
- Gate temporary mono dumps behind `--keep-temp`, writing them only if explicitly requested for debugging.

### 3.4 Instrumentation & Observability
- Structured logging (JSON or prefixed text) indicating queue depth, worker throughput, average latency.
- Optional `--progress` flag to print a tqdm-style bar or periodic stats for headless batch runs.

## 4. Required IndexTTS2 Hooks
1. **Memory-returning inference** – extend `IndexTTS2.infer`/`infer_generator` with `return_audio=True` (default False). When enabled, skip `torchaudio.save` and instead return `(sampling_rate, np.ndarray|torch.Tensor)` plus duration, allowing callers to manage persistence.
2. **Metadata struct** – optionally expose a lightweight `InferenceResult` namedtuple so callers can read `duration_sec`, `wav_tensor`, and `rtf` without parsing stdout.
3. **Stream callback** – allow passing a callback that receives each streamed chunk before concatenation; useful if we later want true streaming writes.
4. **Cache introspection** – expose helper(s) to clear or warm caches (speaker/emotion) per worker so the builder can reset between different prompt files deterministically.

Implementation note: these hooks should live behind keyword arguments so the existing CLI/web UI behavior is unchanged. The builder can then call `IndexTTS2.infer(..., return_audio=True)` and avoid all redundant disk I/O.

## 5. Task Breakdown (mirrors Issue #3 checkboxes)
1. **Audio reuse + duration metadata**
   - Introduce memory-returning inference hook.
   - Update `synthesize_line` logic to consume the in-memory tensor, optionally spilling to disk only when `--keep-temp` is set.
   - Duration derived from tensor length divided by SR.
2. **Buffered manifest + optional mono dumps**
   - Implement stereo assembler that pads + stacks results purely in-memory.
   - Add `--buffer-size` for JSONL flush interval.
3. **CLI flag expansion**
   - Wire through device + acceleration toggles, worker count, queue size, random seed, log level.
   - Document defaults + interactions with `IndexTTS2.__init__` knobs (`use_accel`, `use_torch_compile`, etc.).
4. **Parallel pipeline**
   - Add multiprocessing (preferred) or `concurrent.futures.ProcessPoolExecutor` worker pool with graceful shutdown and signal handling.
   - Include mock/inference-only benchmarking mode (e.g., generate white noise) so throughput math can be validated without full checkpoints.
5. **Docs + Benchmarks**
   - Update `README.md` and/or dedicated doc section with usage, tuning suggestions, and benchmark table referencing the ≥2× goal.
   - Attach benchmark logs/summary comment to Issue #3.
6. **Regression validation**
   - Provide automated or documented steps showing legacy single-worker path matches current outputs (byte-wise or checksum) on `examples/moshi_sample.jsonl`.

## 6. Risks & Mitigations
- **GPU VRAM pressure with multiple workers** – mitigate via queue back-pressure and CLI warnings; document that worker count >1 duplicates model weights.
- **File descriptor exhaustion** – ensure planner closes JSONL input file promptly and use context managers for any optional temp writes.
- **Windows/macos spawn quirks** – gate multiprocessing behind `if __name__ == "__main__"` and offer `--worker-count 1` fallback for unsupported environments.

## 7. Validation Plan
- Benchmark commands for both baseline and optimized pipelines recorded in `moshi_pipeline_design.md` once implemented (append table with dataset size, duration, throughput, GPU).
- Unit-style regression: run builder twice (old path vs new `--legacy-io`) and compare SHA256 of outputs.
- Manual QA: listen to random stereo pairs to confirm channel ordering + silence padding.

## 8. Next Steps
1. Implement IndexTTS2 return-audio hook(s) to unlock in-memory pipeline work.
2. Land the queuing/worker refactor + CLI surface area.
3. Document + benchmark, then hand off to /task-implementer for execution per Issue #3.
