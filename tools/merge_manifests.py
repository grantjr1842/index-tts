import json
import os
import glob

OUTPUT_DIR = "data/interstellar_moshi_build_legacy"
FINAL_MANIFEST = os.path.join(OUTPUT_DIR, "manifest.jsonl")

def main():
    parts = glob.glob(os.path.join(OUTPUT_DIR, "mycooldataset*.jsonl"))
    print(f"Found {len(parts)} manifest parts: {parts}")
    
    all_entries = []
    seen_paths = set()
    
    for part in sorted(parts):
        print(f"Reading {part}...")
        with open(part, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        entry = json.loads(line)
                        path = entry.get("path")
                        if path and path not in seen_paths:
                            full_path = os.path.join(OUTPUT_DIR, path)
                            if os.path.exists(full_path):
                                all_entries.append(entry)
                                seen_paths.add(path)
                            else:
                                print(f"Skipping missing file: {path}")
                    except json.JSONDecodeError:
                        print(f"Skipping invalid line in {part}")

    print(f"Total unique entries: {len(all_entries)}")
    
    # Sort by path for tidiness
    all_entries.sort(key=lambda x: x["path"])
    
    print(f"Writing to {FINAL_MANIFEST}...")
    with open(FINAL_MANIFEST, "w") as f:
        for entry in all_entries:
            f.write(json.dumps(entry) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
