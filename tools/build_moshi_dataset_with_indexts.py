
#!/usr/bin/env python
import argparse
import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from indextts.infer_v2 import IndexTTS2


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROMPT = REPO_ROOT / "interstellar-tars-01-resemble-denoised.wav"


def synthesize_line(
    tts: IndexTTS2,
    text: str,
    output_path: Path,
    spk_audio_prompt: Optional[str] = None,
    emo_audio_prompt: Optional[str] = None,
    emo_vector: Optional[list] = None,
    emo_alpha: float = 1.0,
) -> tuple[Path, float, int]:
    """Synthesize a single utterance and return (path, duration_sec, sample_rate)."""

    if not text.strip():
        raise ValueError("Text is empty; dataset rows must provide non-empty text fields.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        tts.infer(
            spk_audio_prompt=spk_audio_prompt,
            text=text,
            output_path=str(output_path),
            emo_audio_prompt=emo_audio_prompt,
            emo_vector=emo_vector,
            emo_alpha=emo_alpha,
            verbose=False,
        )
    except Exception as exc:  # pragma: no cover - direct passthrough to surface inference failures
        raise RuntimeError(f"IndexTTS2 inference failed for text snippet '{text[:32]}...'") from exc

    audio, sr = sf.read(str(output_path))
    if audio.ndim == 2:  # enforce mono for downstream stereo packing
        audio = audio.mean(axis=1)
    duration_sec = audio.shape[0] / sr
    return output_path, float(duration_sec), int(sr)


def make_stereo(left_wav: Path, right_wav: Path, stereo_out: Path) -> tuple[float, int]:
    """Load two mono WAVs, pad to a shared length, and emit a stereo WAV."""

    left, sr_left = sf.read(str(left_wav))
    right, sr_right = sf.read(str(right_wav))

    if left.ndim == 2:
        left = left.mean(axis=1)
    if right.ndim == 2:
        right = right.mean(axis=1)

    if sr_left != sr_right:
        raise ValueError(f"Sample rate mismatch: {sr_left} vs {sr_right}")

    sr = sr_left
    max_len = max(len(left), len(right))

    def pad(x, target_len):
        if len(x) == target_len:
            return x
        out = np.zeros(target_len, dtype=x.dtype)
        out[: len(x)] = x
        return out

    left_p = pad(left, max_len)
    right_p = pad(right, max_len)

    stereo = np.stack([left_p, right_p], axis=1)
    stereo_out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(stereo_out), stereo, sr)

    duration_sec = max_len / sr
    return float(duration_sec), int(sr)


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

    if user_spk_prompt and not Path(user_spk_prompt).expanduser().exists():
        raise FileNotFoundError(
            f"User speaker prompt '{user_spk_prompt}' was not found."
        )
    assistant_prompt = assistant_spk_prompt or user_spk_prompt
    if assistant_prompt and not Path(assistant_prompt).expanduser().exists():
        raise FileNotFoundError(
            f"Assistant speaker prompt '{assistant_prompt}' was not found."
        )

    tts = IndexTTS2(
        cfg_path=str(cfg_path),
        model_dir=str(model_dir),
        use_fp16=use_fp16,
        use_cuda_kernel=use_cuda_kernel,
        use_deepspeed=use_deepspeed,
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
                raise ValueError(f"Invalid JSON on line {line_idx + 1}: {exc}") from exc

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

            synthesize_line(
                tts=tts,
                text=user_text,
                output_path=tmp_user,
                spk_audio_prompt=user_spk_prompt,
            )

            synthesize_line(
                tts=tts,
                text=assistant_text,
                output_path=tmp_asst,
                spk_audio_prompt=assistant_prompt,
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
                f"[{total_samples}] wrote {stereo_rel_path} (duration={duration_sec:.2f}s, cumulative={total_duration:.2f}s)"
            )

            if not keep_temp:
                tmp_user.unlink(missing_ok=True)
                tmp_asst.unlink(missing_ok=True)

    avg_dur = (total_duration / total_samples) if total_samples else 0.0
    print(
        "\nDone. JSONL index written to: {index_path}\n"
        "Summary -> samples: {samples}, total duration: {total:.2f}s, avg: {avg:.2f}s"
        .format(index_path=index_path, samples=total_samples, total=total_duration, avg=avg_dur)
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
    )


if __name__ == "__main__":
    main()
