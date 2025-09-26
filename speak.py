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
CFG_SCALE = 1.3  # From working implementation
DDPM_STEPS = 10  # From working implementation

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

        # Download model if needed
        model_path = download_model()

        # Load processor
        logger.info("Loading VibeVoice processor...")
        processor = VibeVoiceProcessor.from_pretrained(str(model_path))
        
        # Decide dtype & attention implementation (from working implementation)
        if str(device) == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = "flash_attention_2"
        else:  # CPU or MPS (fallback to CPU in your setup)
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"

        logger.info(f"Using device: {device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        
        # Load model with device-specific logic (from working implementation)
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_path),
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map="auto" if str(device) == "cuda" else None,
            )
            if str(device) != "cuda":
                model.to(device)
        except Exception as e:
            logger.warning(f"Failed to load with '{attn_impl_primary}': {e}. Falling back to 'sdpa'.")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_path),
                torch_dtype=load_dtype,
                attn_implementation="sdpa",
                device_map="auto" if str(device) == "cuda" else None,
            )
            if str(device) != "cuda":
                model.to(device)
        
        # Post-load config from working implementation
        model.eval()
        model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Collect model stats
        model_stats = {
            "load_time_seconds": load_time,
            "device": str(device),
            "torch_dtype": str(load_dtype),
            "attn_implementation": attn_impl_primary,
            "memory_usage_mb": torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else 0
        }
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

PROJECT_ROOT = Path(__file__).resolve().parent
def preprocess_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Preprocess uploaded audio file."""
    try:
        # Save to project root as a temp file
        suffix = Path(filename).suffix or ".wav"
        temp_path = PROJECT_ROOT / f"temp_{os.getpid()}{suffix}"  # unique name
        
        with open(temp_path, "wb") as f:
            f.write(audio_bytes)
        
        try:
            # Load audio
            audio, sr = librosa.load(temp_path, sr=None)
            
            # Resample if needed
            if sr != SAMPLE_RATE:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
            
            # Normalize
            audio = librosa.util.normalize(audio)
            
            # Noise reduction
            audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)
            
            # Trim silence (optional)
            audio, _ = librosa.effects.trim(audio, top_db=20)
            
            # Check length
            if len(audio) / SAMPLE_RATE > MAX_AUDIO_LENGTH:
                raise ValueError(
                    f"Audio too long: {len(audio)/SAMPLE_RATE:.1f}s > {MAX_AUDIO_LENGTH}s"
                )
            
            return audio
        
        finally:
            if temp_path.exists():
                os.remove(temp_path)
    
    except Exception as e:
        logger.error(f"Failed to preprocess audio: {e}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")
def audio_to_bytes(audio: np.ndarray) -> bytes:
    """Convert numpy audio array to WAV bytes."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        sf.write(temp_file.name, audio, SAMPLE_RATE)
        temp_file.seek(0)
        return temp_file.read()

def generate_audio(text: str, voice_samples: List[np.ndarray]) -> tuple[np.ndarray, AudioStats]:
    """Generate audio using VibeVoice model."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = time.time()
    
    try:
        # Validate text length
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long: {len(text)} > {MAX_TEXT_LENGTH}")
        
        # Format text as script (enhanced from working implementation's parsing)
        num_speakers = len(voice_samples)
        if num_speakers == 1:
            formatted_text = f"Speaker 1: {text.strip()}"
        else:
            # For multi-speaker: Ensure labels; simple fallback if missing
            if "Speaker" not in text:
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if lines:
                    formatted_lines = [f"Speaker {i+1}: {line}" for i, line in enumerate(lines)]
                    formatted_text = "\n".join(formatted_lines)
                else:
                    formatted_text = text
            else:
                formatted_text = text
        
        # Clean apostrophes (from working implementation)
        formatted_text = formatted_text.replace("’", "'")
        
        # Prepare inputs (batched as in working implementation)
        inputs = processor(
            text=[formatted_text],  # Wrap in list for batch
            voice_samples=voice_samples,  # List of arrays (processor handles)
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move tensors to device (from working implementation)
        target_device = device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(target_device)
        
        # Patch missing config attr (your previous fix)
        if not hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = getattr(model.config, 'language_model_num_hidden_layers', 
                                                    getattr(model.config, 'encoder_num_hidden_layers', 28))
        
        # Generate (exact kwargs from working implementation)
        inference_start = time.time()
        with torch.inference_mode():
            generated_audio = model.generate(
                **inputs,
                max_new_tokens=None,  # Key to avoid cache arg errors
                cfg_scale=CFG_SCALE,
                tokenizer=processor.tokenizer,
                generation_config={'do_sample': False},
                verbose=True,
            )
        
        inference_time = time.time() - inference_start
        
        # Process output (from working implementation)
        generated_audio = generated_audio.speech_outputs[0].cpu().to(torch.float32).numpy()
        
        # Get stats
        gpu_mem_used = torch.cuda.memory_allocated(0) / 1024**2 if torch.cuda.is_available() else None
        
        stats = AudioStats(
            duration_seconds=len(generated_audio) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            channels=1,
            processing_time_seconds=time.time() - start_time,
            model_inference_time_seconds=inference_time,
            gpu_memory_used_mb=gpu_mem_used
        )
        
        return generated_audio, stats
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

@app.on_event("startup")
def startup_event():
    load_model()

@app.get("/")
async def root():
    return {
        "message": "VibeVoice Inference Server is running",
        "version": "1.0.0",
        "model": MODEL_NAME,
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
async def single_speaker_tts(
    text: str = Form(...),
    voice_file: UploadFile = File(...)
):
    """
    Generate speech using single speaker with voice cloning.
    
    Args:
        text: Text to convert to speech
        voice_file: Audio file for voice cloning (WAV/MP3)
    """
    logger.info("Received single-speaker TTS request", text, voice_file.headers, voice_file.filename, voice_file.content_type, voice_file.size)
    try:
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not voice_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read and preprocess audio
        audio_bytes = await voice_file.read()
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
async def multi_speaker_tts(
    text: str = Form(...),
    voice_files: List[UploadFile] = File(...)
):
    """
    Generate speech using multiple speakers with voice cloning.
    
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
            
            audio_bytes = await voice_file.read()
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