import json
import os
from collections import defaultdict
import argparse

INPUT_FILE = "data/interstellar_moshi.jsonl"
BUILD_DIR = "data/interstellar_moshi_build_legacy/data_stereo"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="Actually delete files")
    args = parser.parse_args()

    content_map = defaultdict(list)
    
    # Read source file
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    print(f"Reading {INPUT_FILE}...")
    with open(INPUT_FILE, "r") as f:
        for line in f:
            if line.strip():
                entry = json.loads(line)
                key = (entry.get("user_text", ""), entry.get("assistant_text", ""))
                content_map[key].append(entry["id"])

    # Identify duplicates
    duplicates_to_remove = []
    
    for key, ids in content_map.items():
        if len(ids) > 1:
            remove = ids[1:]
            duplicates_to_remove.extend(remove)
            
    print(f"Total duplicates to remove: {len(duplicates_to_remove)}")
    
    files_to_delete = []
    for id_ in duplicates_to_remove:
        wav_path = os.path.join(BUILD_DIR, f"{id_}.wav")
        if os.path.exists(wav_path):
            files_to_delete.append(wav_path)
            
    print(f"Found {len(files_to_delete)} .wav files to delete.")
    
    if args.delete:
        print("Deleting files...")
        for f in files_to_delete:
            try:
                os.remove(f)
                print(f"Deleted {f}")
            except Exception as e:
                print(f"Error deleting {f}: {e}")
        print("Done.")
    else:
        print("Dry run. Use --delete to confirm.")

if __name__ == "__main__":
    main()
