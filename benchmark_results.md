# IndexTTS Server Performance & SOP

## Findings

We compared the performance of the Python FastAPI implementation (`serve_tars.py`) and the Rust Axum implementation (`server/`).

### Python Server (`serve_tars.py`)
- **Pros**: Easy to run (`uv run serve_tars.py`), native Python integration.
- **Cons**: Limited concurrency due to GIL (though inference releases GIL, overhead exists), synchronous-like behavior in some paths.

### Rust Server (`server/`)
- **Pros**: 
  - Better concurrency handling for HTTP layer.
  - Type-safe implementation.
  - Direct integration with `pyo3` allows efficient memory usage (shared tensor references possible).
  - Streaming response implemented efficiently with `tokio` channels.
- **Cons**:
  - Complex build process (needs `pyo3` linking).
  - Requires setting environment variables for Python runtime.
  - Inference logic is still Python-bound (`IndexTTS2` implementation), so pure inference speed is identical. The gain is in throughput/request handling overhead.

## Standard Operating Procedure (Rust Server)

To run the high-performance Rust server, follow these steps:

### 1. Prerequisites
- Rust (cargo)
- `uv` (for Python environment)
- NVIDIA Drivers / CUDA (if using GPU)

### 2. Build

Build the server in release mode. We recommend using `uv`'s Python for linking.

```bash
cd server
PYO3_PYTHON=../.venv/bin/python cargo build --release
cd ..
```

### 3. Run

Run the server from the **root** of the repository (to ensure `indextts` module and checkpoints are found).
You must set `LD_LIBRARY_PATH` and `PYTHONHOME` to point to the `uv`-managed Python installation.

Find your python path (example for `cpython-3.10.19-linux-x86_64-gnu`):
```bash
export UV_PYTHON_DIR="$HOME/.local/share/uv/python/cpython-3.10.19-linux-x86_64-gnu"
export LD_LIBRARY_PATH="$UV_PYTHON_DIR/lib:$LD_LIBRARY_PATH"
export PYTHONHOME="$UV_PYTHON_DIR"
export PYTHONPATH="$(pwd)/.venv/lib/python3.10/site-packages"
```

Run the binary:
```bash
./server/target/release/server
```

### 4. API Usage

#### /tts (POST)
Non-streaming generation.
```bash
curl -X POST http://localhost:8009/tts \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world",
    "temperature": 0.8
  }' --output output.wav
```

#### /tts/stream (POST)
Streaming generation (Chunked Transfer Encoding).
```bash
curl -X POST http://localhost:8009/tts/stream \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Hello world"
  }' --output output.wav
```

## Troubleshooting

- **"ModuleNotFoundError: No module named 'indextts'"**: Ensure you are running from the root directory.
- **"libpython3.10.so.1.0: cannot open shared object file"**: Ensure `LD_LIBRARY_PATH` includes the python `lib` directory.
- **"ModuleNotFoundError: No module named 'encodings'"**: Ensure `PYTHONHOME` is set correctly.
