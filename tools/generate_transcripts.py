#!/usr/bin/env python3
"""Generate JSON transcripts for moshi-finetune from source text data.

Since we already have ground truth text from the input JSONL, we can create 
simplified transcripts without running Whisper. The transcripts are structured
to match what moshi-finetune expects.
"""
import json
import os
from pathlib import Path

# Paths
source_jsonl = Path("/home/admin-grant-jr/github/index-tts/data/interstellar_moshi.jsonl")
dataset_dir = Path("/home/admin-grant-jr/github/index-tts/data/tars_moshi_dataset")
stereo_dir = dataset_dir / "data_stereo"

# Load manifest to get durations
manifest_path = dataset_dir / "tars_dataset.jsonl"
durations = {}
with open(manifest_path) as f:
    for line in f:
        entry = json.loads(line)
        if "path" in entry:
            # Extract sample ID from path like "data_stereo/tars_000.wav"
            basename = Path(entry["path"]).stem  # e.g., "tars_000"
            durations[basename] = entry["duration"]

# Load source text
source_data = {}
with open(source_jsonl) as f:
    for line in f:
        entry = json.loads(line)
        source_data[entry["id"]] = entry

# Generate JSON transcripts for each WAV
for wav_file in sorted(stereo_dir.glob("*.wav")):
    sample_id = wav_file.stem  # e.g., "tars_000"
    
    if sample_id not in source_data:
        print(f"Warning: No source data for {sample_id}")
        continue
    
    src = source_data[sample_id]
    duration = durations.get(sample_id, 5.0)  # Default 5s if not found
    
    # Create transcript structure matching moshi-finetune format
    # Left channel (0) = assistant/moshi, Right channel (1) = user
    transcript = {
        "audio_path": str(wav_file.relative_to(dataset_dir)),
        "segments": [
            {
                "channel": 1,  # Right = user
                "text": src["user_text"],
                "start": 0.0,
                "end": duration / 2  # Assume first half is user
            },
            {
                "channel": 0,  # Left = assistant
                "text": src["assistant_text"],
                "start": duration / 2,
                "end": duration
            }
        ]
    }
    
    # Write JSON file next to WAV
    json_path = wav_file.with_suffix(".json")
    with open(json_path, "w") as f:
        json.dump(transcript, f, indent=2)
    
    print(f"Created {json_path.name}")

print(f"\nDone! Generated {len(list(stereo_dir.glob('*.json')))} JSON transcripts")
