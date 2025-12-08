import requests
import json
import time
import sys

BASE_URL = "http://localhost:8009"

def test_health():
    print("Testing /healthz...")
    try:
        resp = requests.get(f"{BASE_URL}/healthz")
        resp.raise_for_status()
        assert resp.json() == {"status": "ok"}
        print("✅ /healthz passed")
    except Exception as e:
        print(f"❌ /healthz failed: {e}")
        sys.exit(1)

def test_tts_post():
    print("Testing /tts (blocking)...")
    payload = {
        "text": "Hello, this is a test of the Python server.",
        "temperature": 0.7
    }
    try:
        start = time.time()
        resp = requests.post(f"{BASE_URL}/tts", json=payload)
        resp.raise_for_status()
        print(f"Response status: {resp.status_code}")
        print(f"Content-Type: {resp.headers.get('content-type')}")
        audio_len = len(resp.content)
        print(f"Received {audio_len} bytes")
        elapsed = time.time() - start
        print(f"Latency: {elapsed:.2f}s")
        
        if audio_len == 0:
            raise ValueError("Received empty audio")
            
        print("✅ /tts passed")
    except Exception as e:
        print(f"❌ /tts failed: {e}")
        sys.exit(1)

def test_tts_stream():
    print("Testing /tts/stream (streaming)...")
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
        
        for chunk in resp.iter_content(chunk_size=None):
            if chunk:
                if first_chunk_time is None:
                    first_chunk_time = time.time()
                chunk_count += 1
                total_bytes += len(chunk)
                
        elapsed = time.time() - start
        ttfb = first_chunk_time - start if first_chunk_time else 0
        
        print(f"Received {chunk_count} chunks, {total_bytes} bytes")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Time to first chunk: {ttfb:.2f}s")
        
        if total_bytes == 0:
             raise ValueError("Received empty stream")

        print("✅ /tts/stream passed")
    except Exception as e:
        print(f"❌ /tts/stream failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    print("Waiting for server to be ready...")
    # Simple retry logic for local dev comfort
    for i in range(5):
        try:
            requests.get(f"{BASE_URL}/healthz")
            break
        except requests.exceptions.ConnectionError:
            print(f"Server not ready, retrying ({i+1}/5)...")
            time.sleep(2)
    else:
        print("❌ Server unreachable. Make sure 'python serve_tars.py' is running.")
        sys.exit(1)

    test_health()
    test_tts_post()
    test_tts_stream()
