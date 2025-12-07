import os
import io
import torch
import uvicorn
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

# IndexTTS imports
from indextts.infer_v2 import IndexTTS2

# Constants
TARS_REFERENCE_AUDIO = os.path.abspath("interstellar-tars-01-resemble-denoised.wav")
CHECKPOINT_DIR = "checkpoints"
CONFIG_PATH = os.path.join(CHECKPOINT_DIR, "config.yaml")

from contextlib import asynccontextmanager

# Global TTS model variable
tts_model = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global tts_model
    print("Loading IndexTTS2 model...")
    # Check if reference audio exists
    if not os.path.exists(TARS_REFERENCE_AUDIO):
        print(f"WARNING: Reference audio not found at {TARS_REFERENCE_AUDIO}")
    
    # Initialize the model
    # Note: Using default settings similar to webui.py
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    tts_model = IndexTTS2(
        cfg_path=CONFIG_PATH,
        model_dir=CHECKPOINT_DIR,
        use_fp16=True if device != "cpu" else False,
        device=device
    )
    print("Model loaded successfully!")
    yield
    # Clean up if needed
    print("Shutting down TARS server...")

# Initialize FastAPI app with lifespan
app = FastAPI(title="TARS Voice Server", lifespan=lifespan)

class TTSRequest(BaseModel):
    text: str
    temperature: float = 0.8
    top_p: float = 0.8
    top_k: int = 30
    speed: float = 1.0

@app.post("/tts")
async def generate_speech(request: TTSRequest):
    if tts_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not os.path.exists(TARS_REFERENCE_AUDIO):
         raise HTTPException(status_code=500, detail="Reference audio file missing on server")

    print(f"Generating speech for: '{request.text}'")
    
    try:
        # Perform inference
        # The infer method expects the path to the reference audio
        audio_result = tts_model.infer(
            spk_audio_prompt=TARS_REFERENCE_AUDIO,
            text=request.text,
            output_path=None,  # Return raw audio data
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            return_audio=True,
            return_numpy=True
        )
        
        # infer returns a generator or result directly depending on flags, 
        # but with return_audio=True in infer_v2.py it yields an InferenceResult
        
        # The infer method returns the InferenceResult object directly when return_audio=True
        # because it does list(...)[0] internally.
        result = audio_result
        
        # result.audio is a numpy array (because we passed return_numpy=True)
        audio_data = result.audio
        sample_rate = result.sampling_rate
        
        # Convert to WAV bytes
        # Audio from IndexTTS is likely float32 or int16, soundfile handles numpy arrays well
        buffer = io.BytesIO()
        sf.write(buffer, audio_data, sample_rate, format='WAV')
        buffer.seek(0)
        
        return Response(content=buffer.read(), media_type="audio/wav")

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8009)
