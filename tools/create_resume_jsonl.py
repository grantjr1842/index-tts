import json
import os

INPUT_FILE = "data/interstellar_moshi.jsonl"
OUTPUT_FILE = "data/interstellar_moshi_resume.jsonl"
MANIFEST_FILE = "data/interstellar_moshi_build_legacy/mycooldataset.jsonl"

def main():
    # Read existing manifest to find completed IDs
    completed_ids = set()
    if os.path.exists(MANIFEST_FILE):
        with open(MANIFEST_FILE, "r") as f:
            for line in f:
                if line.strip():
                    entry = json.loads(line)
                    # path is like "data_stereo/tars_000.wav"
                    filename = os.path.basename(entry["path"])
                    id_ = os.path.splitext(filename)[0]
                    completed_ids.add(id_)
    
    print(f"Found {len(completed_ids)} completed samples.")

    # Filter input file
    resume_entries = []
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                if entry["id"] not in completed_ids:
                    resume_entries.append(entry)
    
    print(f"Writing {len(resume_entries)} samples to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w") as f:
        for entry in resume_entries:
            f.write(json.dumps(entry) + "\n")

if __name__ == "__main__":
    main()
