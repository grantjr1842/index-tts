
import os
import sys
import numpy as np
import soundfile as sf


def analyze_audio(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.wav')])
    if not files:
        print("No wav files found.")
        return

    print(f"Analyzing {len(files)} files in {folder}...\n")
    
    results = []
    
    for f in files:
        path = os.path.join(folder, f)
        try:
            data, samplerate = sf.read(path)
            
            # Handle multi-channel
            if len(data.shape) > 1:
                data = data.mean(axis=1)
                
            duration = len(data) / samplerate
            max_amp = np.max(np.abs(data))
            rms = np.sqrt(np.mean(data**2))
            
            is_silent = max_amp < 0.01  # Threshold for silence
            
            results.append({
                "file": f,
                "duration": duration,
                "max_amp": max_amp,
                "rms": rms,
                "is_silent": is_silent
            })
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Summary
    silent_count = sum(1 for r in results if r['is_silent'])
    avg_rms = np.mean([r['rms'] for r in results]) if results else 0
    
    print(f"Summary:")
    print(f"  Total Files: {len(results)}")
    print(f"  Silent Files: {silent_count}")
    print(f"  Avg RMS: {avg_rms:.4f}")
    
    if silent_count > 0:
        print("\nSilent files detected:")
        for r in results:
            if r['is_silent']:
                print(f"  {r['file']} (Max Amp: {r['max_amp']:.4f})")
    else:
        print("\nAll files contain audio signal.")
        
    # Print detailed stats for first 5
    print("\nSample Details (First 5):")
    for r in results[:5]:
        print(f"  {r['file']}: {r['duration']:.2f}s, RMS: {r['rms']:.4f}, Max: {r['max_amp']:.4f}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python verify_audio_content.py <folder>")
        sys.exit(1)
    
    analyze_audio(sys.argv[1])
