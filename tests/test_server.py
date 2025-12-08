import requests
import json
import time
import sys
import os
import logging
import wave
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

BASE_URL = "http://localhost:8009"
OUTPUT_DIR = os.path.join("outputs", "test_results")

def ensure_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        logger.info(f"Created output directory: {OUTPUT_DIR}")
    else:
        logger.info(f"Using output directory: {OUTPUT_DIR}")

def save_audio(data, prefix="test", is_pcm=False, sample_rate=24000):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{prefix}_{timestamp}.wav"
    filepath = os.path.join(OUTPUT_DIR, filename)
    
    try:
        if is_pcm:
            # Wrap raw PCM in a WAV container
            with wave.open(filepath, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)  # 16-bit
                wav_file.setframerate(sample_rate)
                wav_file.writeframes(data)
        else:
            # Already a WAV file
            with open(filepath, "wb") as f:
                f.write(data)
        logger.info(f"Saved audio to: {filepath} ({len(data)} bytes)")
        return filepath
    except Exception as e:
        logger.error(f"Failed to save audio: {e}")
        return None

def test_health():
    logger.info("----------------------------------------------------------------")
    logger.info("Testing /healthz...")
    try:
        resp = requests.get(f"{BASE_URL}/healthz")
        resp.raise_for_status()
        assert resp.json() == {"status": "ok"}
        logger.info("✅ /healthz passed")
    except Exception as e:
        logger.error(f"❌ /healthz failed: {e}")
        sys.exit(1)

def test_tts_post():
    logger.info("----------------------------------------------------------------")
    logger.info("Testing /tts (blocking)...")
    payload = {
        "text": "Hello, this is a test of the Python server.",
        "temperature": 0.7
    }
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/tts", json=payload)
        resp.raise_for_status()
        logger.info(f"Response status: {resp.status_code}")
        logger.info(f"Content-Type: {resp.headers.get('content-type')}")
        audio_len = len(resp.content)
        logger.info(f"Received {audio_len} bytes")
        elapsed = time.time() - start
        logger.info(f"Latency: {elapsed:.2f}s")
        
        if audio_len == 0:
            raise ValueError("Received empty audio")
            
        save_audio(resp.content, prefix="tts_blocking", is_pcm=False)
        logger.info("✅ /tts passed")
    except Exception as e:
        logger.error(f"❌ /tts failed: {e}")
        sys.exit(1)

def test_tts_stream():
    logger.info("----------------------------------------------------------------")
    logger.info("Testing /tts/stream (streaming)...")
    payload = {
        "text": "This is a longer sentence to ensure we get multiple chunks from the streaming endpoint.",
        "temperature": 0.7
    }
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/tts/stream", json=payload, stream=True)
        resp.raise_for_status()
        
        chunk_count = 0
        total_bytes = 0
        first_chunk_time = None
        full_audio = bytearray()
        
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                chunk_count += 1
                total_bytes += len(chunk)
                full_audio.extend(chunk)
                
        elapsed = time.time() - start
        ttfb = first_chunk_time - start if first_chunk_time else 0
        
        logger.info(f"Received {chunk_count} chunks, {total_bytes} bytes")
        logger.info(f"Total time: {elapsed:.2f}s")
        logger.info(f"Time to first chunk: {ttfb:.2f}s")
        
        if total_bytes == 0:
             raise ValueError("Received empty stream")

        # Streaming endpoint now returns raw PCM S16LE which should range [-32768, 32767]
        # The model in infer_v2.py defaults to 22050Hz for streaming chunks.
        save_audio(full_audio, prefix="tts_stream", is_pcm=True, sample_rate=22050)
        logger.info("✅ /tts/stream passed")
    except Exception as e:
        logger.error(f"❌ /tts/stream failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    ensure_output_dir()
    logger.info("Waiting for server to be ready...")
    # Simple retry logic for local dev comfort
    for i in range(5):
        try:
            requests.get(f"{BASE_URL}/healthz")
            break
        except requests.exceptions.ConnectionError:
            logger.warning(f"Server not ready, retrying ({i+1}/5)...")
            time.sleep(2)
    else:
        logger.error("❌ Server unreachable. Make sure 'python serve_tars.py' is running.")
        sys.exit(1)

    test_health()
    test_tts_post()
    test_tts_stream()
