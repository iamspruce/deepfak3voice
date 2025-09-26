import os
import time
import tempfile
import logging
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import traceback

import torch
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import noisereduce as nr
from huggingface_hub import snapshot_download
import psutil

# Import your VibeVoice components
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
MODEL_NAME = "microsoft/VibeVoice-1.5B"
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 24000
MAX_AUDIO_LENGTH = 300  # 5 minutes max
MAX_TEXT_LENGTH = 10000  # 10k characters max

app = FastAPI(
    title="VibeVoice Inference Server",
    description="FastAPI server for VibeVoice text-to-speech with single and multi-speaker support",
    version="1.0.0"
)

# Add CORS middleware
origins = [
    "http://localhost",
    "http://localhost:3000",
    "https://your-frontend-app.com",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, # Only allows your specified domains
    allow_credentials=True,
    allow_methods=["GET", "POST"], # Be specific about allowed methods
    allow_headers=["*"],
)

# Global variables
processor = None
model = None
device = None
model_stats = {}

class SystemInfo(BaseModel):
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_total: Optional[float]
    gpu_memory_available: Optional[float]
    cpu_count: int
    ram_total: float
    ram_available: float
    model_loaded: bool
    model_path: Optional[str]

class SingleSpeakerRequest(BaseModel):
    text: str

class MultiSpeakerRequest(BaseModel):
    text: str

class AudioStats(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    processing_time_seconds: float
    model_inference_time_seconds: float
    gpu_memory_used_mb: Optional[float]

def check_gpu_compatibility():
    """Check if compatible GPU is available and return device info."""
    global device
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_memory_available = (torch.cuda.get_device_properties(0).total_memory - 
                               torch.cuda.memory_allocated(0)) / 1024**3
        
        logger.info(f"GPU detected: {gpu_name} with {gpu_memory_total:.1f}GB total memory")
        return True, gpu_name, gpu_memory_total, gpu_memory_available
    else:
        device = torch.device("cpu")
        logger.warning("No CUDA GPU available, falling back to CPU")
        return False, None, None, None

def get_system_info() -> SystemInfo:
    """Get current system information."""
    gpu_available, gpu_name, gpu_total, gpu_available_mem = check_gpu_compatibility()
    
    return SystemInfo(
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        gpu_memory_total=gpu_total,
        gpu_memory_available=gpu_available_mem,
        cpu_count=psutil.cpu_count(),
        ram_total=psutil.virtual_memory().total / 1024**3,
        ram_available=psutil.virtual_memory().available / 1024**3,
        model_loaded=model is not None,
        model_path=str(MODELS_DIR / MODEL_NAME) if model else None
    )

def download_model():
    """Download model if not already present."""
    model_path = MODELS_DIR / MODEL_NAME.replace("/", "_")
    
    if not model_path.exists():
        logger.info(f"Downloading model {MODEL_NAME} to {model_path}")
        try:
            # REMOVED ignore_patterns to ensure weights are downloaded
            snapshot_download(
                repo_id=MODEL_NAME,
                local_dir=model_path,
            )
            logger.info(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")
    
    return model_path

def load_model():
    """Load the VibeVoice model and processor."""
    global processor, model, model_stats

    if model is not None:
        return

    start_time = time.time()
    
    try:
        # Check GPU compatibility first
        gpu_available, _, _, _ = check_gpu_compatibility()
        if not gpu_available:
            logger.warning("No GPU available - model performance may be slow")

        # IMPORTANT: Ensure your download_model function actually downloads the model weights.
        # The original code had an ignore_patterns that skipped .bin/.safetensors files.
        model_path = download_model()
        
        # Load processor
        logger.info("Loading VibeVoice processor...")
        processor = VibeVoiceProcessor.from_pretrained(str(model_path))
        
        # --- Corrected Model Loading Logic ---
        logger.info("Loading VibeVoice model for inference...")
        
        # Set up device-specific loading parameters
        if str(device) == "cuda":
            load_dtype = torch.bfloat16
            attn_impl = "flash_attention_2"
        else: # CPU or MPS
            load_dtype = torch.float32
            attn_impl = "sdpa"

        logger.info(f"Using device: {device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl}")
        
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_path),
                torch_dtype=load_dtype,
                attn_implementation=attn_impl,
            )
        except Exception as e:
            logger.warning(f"Failed to load with '{attn_impl}': {e}. Falling back to 'sdpa'.")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_path),
                torch_dtype=load_dtype,
                attn_implementation='sdpa',
            )

        model = model.to(device)
        model.eval()
        model.set_ddpm_inference_steps(num_steps=10) # Set inference steps as in the example

        load_time = time.time() - start_time
        model_stats = {
            "load_time_seconds": load_time,
            "model_path": str(model_path),
            "device": str(device),
            "parameters": f"{(sum(p.numel() for p in model.parameters()) / 1e9):.2f}B",
        }
        
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")
    
def preprocess_audio(audio_file: bytes, filename: str) -> np.ndarray:
    """
    Preprocess uploaded audio file - convert format, resample, denoise.
    
    Args:
        audio_file: Raw audio file bytes
        filename: Original filename for format detection
        
    Returns:
        numpy.ndarray: Preprocessed audio array
    """
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(suffix=Path(filename).suffix, delete=False) as tmp_file:
            tmp_file.write(audio_file)
            tmp_path = tmp_file.name
        
        # Load audio using librosa (handles multiple formats)
        audio, sr = librosa.load(tmp_path, sr=None, mono=True)
        
        # Clean up temp file
        os.unlink(tmp_path)
        
        # Resample to target sample rate if needed
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Apply noise reduction
        audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE, prop_decrease=0.8)
        
        # Normalize audio
        audio = audio / np.max(np.abs(audio))
        
        # Check duration
        duration = len(audio) / SAMPLE_RATE
        if duration > MAX_AUDIO_LENGTH:
            logger.warning(f"Audio duration {duration:.1f}s exceeds maximum {MAX_AUDIO_LENGTH}s")
            # Truncate to max length
            audio = audio[:int(MAX_AUDIO_LENGTH * SAMPLE_RATE)]
        
        logger.info(f"Preprocessed audio: {len(audio)} samples, {len(audio)/SAMPLE_RATE:.2f}s duration")
        return audio
        
    except Exception as e:
        logger.error(f"Failed to preprocess audio: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to process audio file: {str(e)}")

def generate_audio(text: str, voice_samples: Optional[List[np.ndarray]] = None) -> tuple[np.ndarray, AudioStats]:
    """
    Generate audio using VibeVoice model.
    
    Args:
        text: Input text
        voice_samples: Optional voice samples for cloning
        
    Returns:
        tuple: (generated_audio, stats)
    """
    if model is None or processor is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    start_time = time.time()
    gpu_memory_start = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else None
    
    try:
        if len(text) > MAX_TEXT_LENGTH:
            raise HTTPException(status_code=400, detail=f"Text too long. Maximum {MAX_TEXT_LENGTH} characters.")
        
        model_start = time.time()
        
        # --- Corrected Input Formatting ---
        # The processor expects batch inputs, so we wrap our single input in lists.
        inputs = processor(
            text=[text],
            voice_samples=[voice_samples],
            return_tensors="pt",
            padding=True
        )
        
        # Move inputs to the correct device
        for key, value in inputs.items():
            if torch.is_tensor(value):
                inputs[key] = value.to(device)
        
        # --- Corrected Generation Call and Output Handling ---
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=None,
                cfg_scale=1.3,  # A sensible default from the example
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
            )
            
            # The actual audio is in the `speech_outputs` attribute of the output object
            generated_audio = outputs.speech_outputs[0].cpu().numpy()
            
        model_inference_time = time.time() - model_start
        total_time = time.time() - start_time
        
        gpu_memory_used = None
        if torch.cuda.is_available() and gpu_memory_start is not None:
            gpu_memory_end = torch.cuda.memory_allocated(0) / 1024**2
            gpu_memory_used = gpu_memory_end - gpu_memory_start
        
        stats = AudioStats(
            duration_seconds=len(generated_audio) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            channels=1,
            processing_time_seconds=total_time,
            model_inference_time_seconds=model_inference_time,
            gpu_memory_used_mb=gpu_memory_used
        )
        
        logger.info(f"Generated audio: {stats.duration_seconds:.2f}s in {total_time:.2f}s")
        return generated_audio, stats
        
    except Exception as e:
        logger.error(f"Failed to generate audio: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

def audio_to_bytes(audio: np.ndarray, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
        sf.write(tmp_file.name, audio, sample_rate)
        with open(tmp_file.name, "rb") as f:
            audio_bytes = f.read()
        os.unlink(tmp_file.name)
    return audio_bytes

@app.on_event("startup")
async def startup_event():
    """Load model on startup."""
    logger.info("Starting VibeVoice server...")
    load_model()

@app.get("/", response_model=dict)
async def root():
    """Root endpoint with server info."""
    system_info = get_system_info()
    return {
        "message": "VibeVoice Inference Server",
        "version": "1.0.0",
        "system_info": system_info.dict(),
        "model_stats": model_stats,
        "endpoints": {
            "/single-speaker": "Single speaker TTS with voice cloning",
            "/multi-speaker": "Multi-speaker TTS with voice cloning",
            "/health": "Health check endpoint",
            "/system": "System information"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    system_info = get_system_info()
    return {
        "status": "healthy" if system_info.model_loaded else "model_not_loaded",
        "gpu_available": system_info.gpu_available,
        "model_loaded": system_info.model_loaded
    }

@app.get("/system", response_model=SystemInfo)
async def get_system_info_endpoint():
    """Get detailed system information."""
    return get_system_info()

@app.post("/single-speaker")
def single_speaker_tts(
    text: str = Form(...),
    voice_file: UploadFile = File(...)
):
    """
    Generate speech using single speaker with voice cloning.
    
    Args:
        text: Text to convert to speech
        voice_file: Audio file for voice cloning (WAV/MP3)
    """
    try:
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not voice_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read and preprocess audio
        audio_bytes = voice_file.read()
        voice_sample = preprocess_audio(audio_bytes, voice_file.filename)
        
        # Generate audio
        generated_audio, stats = generate_audio(text, [voice_sample])
        
        # Convert to bytes
        audio_output = audio_to_bytes(generated_audio)
        
        # Return as streaming response with stats in headers
        response = StreamingResponse(
            iter([audio_output]),
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(stats.duration_seconds),
                "X-Processing-Time": str(stats.processing_time_seconds),
                "X-Model-Time": str(stats.model_inference_time_seconds),
                "X-Sample-Rate": str(stats.sample_rate),
                "X-GPU-Memory-Used": str(stats.gpu_memory_used_mb) if stats.gpu_memory_used_mb else "N/A",
                "Content-Disposition": "attachment; filename=generated_speech.wav"
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Single speaker TTS failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/multi-speaker")
def multi_speaker_tts(
    text: str = Form(...),
    voice_files: List[UploadFile] = File(...)
):
    """
    Generate speech using multi-speaker with voice cloning.
    
    Args:
        text: Text with speaker labels (e.g., "Speaker 1: Hello\nSpeaker 2: Hi there")
        voice_files: List of audio files for voice cloning, ordered by speaker ID
    """
    try:
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not voice_files:
            raise HTTPException(status_code=400, detail="At least one voice file required")
        
        # Check text format
        if "Speaker" not in text:
            raise HTTPException(
                status_code=400, 
                detail="Text must contain speaker labels (e.g., 'Speaker 1: Hello\\nSpeaker 2: Hi')"
            )
        
        # Process voice files
        voice_samples = []
        for i, voice_file in enumerate(voice_files):
            if not voice_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                raise HTTPException(
                    status_code=400, 
                    detail=f"Unsupported audio format in file {i+1}: {voice_file.filename}"
                )
            
            audio_bytes = voice_file.read()
            voice_sample = preprocess_audio(audio_bytes, voice_file.filename)
            voice_samples.append(voice_sample)
        
        # Generate audio
        generated_audio, stats = generate_audio(text, voice_samples)
        
        # Convert to bytes
        audio_output = audio_to_bytes(generated_audio)
        
        # Return as streaming response with stats in headers
        response = StreamingResponse(
            iter([audio_output]),
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": str(stats.duration_seconds),
                "X-Processing-Time": str(stats.processing_time_seconds),
                "X-Model-Time": str(stats.model_inference_time_seconds),
                "X-Sample-Rate": str(stats.sample_rate),
                "X-GPU-Memory-Used": str(stats.gpu_memory_used_mb) if stats.gpu_memory_used_mb else "N/A",
                "X-Speakers": str(len(voice_samples)),
                "Content-Disposition": "attachment; filename=generated_multispeaker_speech.wav"
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-speaker TTS failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload-model")
async def reload_model():
    """Reload the model (useful for updates)."""
    global processor, model
    
    try:
        processor = None
        model = None
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        load_model()
        
        return {"message": "Model reloaded successfully", "stats": model_stats}
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "speak:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1  # Important: only 1 worker for GPU models
    )