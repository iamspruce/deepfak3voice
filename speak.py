import os
import threading
import time
import io
import wave
import logging
from typing import Optional, List, Dict, Any, AsyncGenerator
from pathlib import Path
import traceback
import asyncio
import requests
from datetime import datetime, timedelta
import torch
import numpy as np
import librosa
import soundfile as sf
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, WebSocket, WebSocketDisconnect
import base64
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import noisereduce as nr
from huggingface_hub import snapshot_download
import psutil
import GPUtil
from sse_starlette.sse import EventSourceResponse
import re

# Import your VibeVoice components
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Dynamic model selection
MODEL_7B = "aoi-ot/VibeVoice-Large"
MODEL_1_5B = "microsoft/VibeVoice-1.5B"
MODELS_DIR = Path("./models")
MODELS_DIR.mkdir(exist_ok=True)
SAMPLE_RATE = 24000
MAX_AUDIO_LENGTH = 300  # 5 minutes max
MAX_TEXT_LENGTH = 10000  # 10k characters max
CFG_SCALE = 1.3
DDPM_STEPS = 10
MIN_VRAM_FOR_7B = 18  

# Performance optimizations
WARMUP_ITERATIONS = 3
AUDIO_CACHE_SIZE = 100

# License verification
GUMROAD_PRODUCT_ID = "Apv6fptxqEqWgANZQAtsfQ=="
LICENSE_GRACE_PERIOD = 86400  # 24 hours in seconds
license_cache = {}  # {license_key: {valid: bool, last_check: timestamp, purchase_info: dict}}


voice_cache = {}  # {cache_key: preprocessed_voice_sample}
voice_cache_lock = threading.Lock()



app = FastAPI(
    title="VibeVoice Inference Server",
    description="FastAPI server for VibeVoice text-to-speech with streaming and dynamic model selection",
    version="2.2.0"
)

# Add CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins="*",
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Global variables
processor = None
model = None
device = None
model_stats = {}
warmup_complete = False
current_model_name = None

# Performance monitoring
performance_stats = {
    "total_requests": 0,
    "average_inference_time": 0,
    "average_preprocessing_time": 0,
    "average_postprocessing_time": 0,
    "peak_memory_usage": 0,
    "cache_hits": 0,
    "cache_misses": 0,
    "streaming_requests": 0
}

class SystemInfo(BaseModel):
    gpu_available: bool
    gpu_name: Optional[str]
    gpu_memory_total: Optional[float]
    gpu_memory_available: Optional[float]
    gpu_memory_used: Optional[float]
    gpu_utilization: Optional[float]
    gpu_temperature: Optional[float]
    cpu_count: int
    cpu_usage: float
    ram_total: float
    ram_available: float
    ram_used: float
    model_loaded: bool
    model_path: Optional[str]
    current_model: Optional[str]
    warmup_complete: bool
    performance_stats: Dict[str, Any]

class AudioStats(BaseModel):
    duration_seconds: float
    sample_rate: int
    channels: int
    processing_time_seconds: float
    model_inference_time_seconds: float
    preprocessing_time_seconds: float
    postprocessing_time_seconds: float
    gpu_memory_used_mb: Optional[float]
    gpu_memory_peak_mb: Optional[float]
    gpu_utilization: Optional[float]
    cpu_usage_during_inference: Optional[float]
    queue_time_seconds: float
    total_latency_seconds: float

class StreamingConfig(BaseModel):
    chunk_duration_ms: int = 200  # Duration of each audio chunk in milliseconds
    buffer_size: int = 4  # Number of chunks to buffer
    sample_rate: int = 24000


def get_voice_cache_key(generation_type: str, voice_ids: List[str]) -> str:
    """Generate cache key for voice samples."""
    return f"{generation_type}:{'|'.join(sorted(voice_ids))}"

def cache_voice_samples(cache_key: str, voice_samples: List[np.ndarray]):
    """Cache preprocessed voice samples."""
    with voice_cache_lock:
        voice_cache[cache_key] = voice_samples
        logger.info(f"Cached voice samples for key: {cache_key}")

def get_cached_voice_samples(cache_key: str) -> Optional[List[np.ndarray]]:
    """Get cached voice samples if available."""
    with voice_cache_lock:
        return voice_cache.get(cache_key)

def get_detailed_gpu_info():
    """Get comprehensive GPU information."""
    try:
        if torch.cuda.is_available():
            gpu_id = 0
            gpu_name = torch.cuda.get_device_name(gpu_id)
            gpu_props = torch.cuda.get_device_properties(gpu_id)
            total_memory = gpu_props.total_memory / 1024**3
            allocated_memory = torch.cuda.memory_allocated(gpu_id) / 1024**3
            cached_memory = torch.cuda.memory_reserved(gpu_id) / 1024**3
            available_memory = total_memory - allocated_memory
            
            # Get GPU utilization and temperature using GPUtil if available
            gpu_util = None
            gpu_temp = None
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    gpu_util = gpu.load * 100
                    gpu_temp = gpu.temperature
            except:
                pass
            
            return {
                "available": True,
                "name": gpu_name,
                "total_memory": total_memory,
                "allocated_memory": allocated_memory,
                "cached_memory": cached_memory,
                "available_memory": available_memory,
                "utilization": gpu_util,
                "temperature": gpu_temp
            }
    except Exception as e:
        logger.error(f"Failed to get GPU info: {e}")
    
    return {
        "available": False,
        "name": None,
        "total_memory": None,
        "allocated_memory": None,
        "cached_memory": None,
        "available_memory": None,
        "utilization": None,
        "temperature": None
    }

def select_optimal_model():
    """Select the optimal model based on GPU VRAM availability."""
    gpu_info = get_detailed_gpu_info()
    
    if gpu_info["available"] and gpu_info["total_memory"] >= MIN_VRAM_FOR_7B:
        logger.info(f"GPU has {gpu_info['total_memory']:.1f}GB VRAM, selecting 7B model")
        return MODEL_7B
    else:
        if gpu_info["available"]:
            logger.info(f"GPU has {gpu_info['total_memory']:.1f}GB VRAM (< {MIN_VRAM_FOR_7B}GB), selecting 1.5B model")
        else:
            logger.info("No GPU available, selecting 1.5B model for CPU inference")
        return MODEL_1_5B

def check_gpu_compatibility():
    """Check if compatible GPU is available and return device info."""
    global device
    
    gpu_info = get_detailed_gpu_info()
    
    if gpu_info["available"]:
        device = torch.device("cuda")
        logger.info(f"GPU detected: {gpu_info['name']} with {gpu_info['total_memory']:.1f}GB total memory")
        return True, gpu_info
    else:
        device = torch.device("cpu")
        logger.warning("No CUDA GPU available, falling back to CPU")
        return False, gpu_info

def check_flash_attn_available():
    """Check if flash-attn is installed and usable."""
    try:
        import flash_attn
        # Check if GPU supports it (requires Ampere or newer - compute capability 8.0+)
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability()
            if major >= 8:
                return True
            else:
                logger.info(f"GPU compute capability {major}.x < 8.0, flash attention not supported")
                return False
        return False
    except ImportError:
        return False

def get_optimal_attention_mode():
    """Determine best attention mode based on hardware and availability."""
    if str(device) != "cuda":
        return "sdpa"  # CPU always uses SDPA
    
    if check_flash_attn_available():
        logger.info("Flash Attention 2 available and supported")
        return "flash_attention_2"
    else:
        logger.info("Flash Attention not available, using SDPA")
        return "sdpa"

def get_system_info() -> SystemInfo:
    """Get comprehensive system information."""
    gpu_available, gpu_info = check_gpu_compatibility()
    cpu_percent = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory()
    
    return SystemInfo(
        gpu_available=gpu_available,
        gpu_name=gpu_info.get("name"),
        gpu_memory_total=gpu_info.get("total_memory"),
        gpu_memory_available=gpu_info.get("available_memory"),
        gpu_memory_used=gpu_info.get("allocated_memory"),
        gpu_utilization=gpu_info.get("utilization"),
        gpu_temperature=gpu_info.get("temperature"),
        cpu_count=psutil.cpu_count(),
        cpu_usage=cpu_percent,
        ram_total=memory.total / 1024**3,
        ram_available=memory.available / 1024**3,
        ram_used=memory.used / 1024**3,
        model_loaded=model is not None,
        model_path=str(MODELS_DIR / current_model_name.replace("/", "_")) if current_model_name else None,
        current_model=current_model_name,
        warmup_complete=warmup_complete,
        performance_stats=performance_stats
    )

def download_model(model_name: str):
    """Download model if not already present."""
    model_path = MODELS_DIR / model_name.replace("/", "_")
    
    if not model_path.exists():
        logger.info(f"Downloading model {model_name} to {model_path}")
        try:
            snapshot_download(
                repo_id=model_name,
                local_dir=model_path,
            )
            logger.info(f"Model downloaded successfully to {model_path}")
        except Exception as e:
            logger.error(f"Failed to download model: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to download model: {str(e)}")
    
    return model_path

def warmup_model():
    """Warmup model with dummy data for optimal performance."""
    global warmup_complete
    
    if not model or not processor:
        logger.warning("Model or processor not loaded, skipping warmup")
        warmup_complete = False
        return
    
    if not hasattr(model, 'generate'):
        logger.warning("Model doesn't have generate method, skipping warmup")
        warmup_complete = False
        return
    
    logger.info("Starting model warmup...")
    warmup_start = time.time()
    
    try:
        # Create dummy audio and text
        dummy_audio = np.random.randn(SAMPLE_RATE * 3).astype(np.float32)  # 3 seconds
        dummy_text = "Speaker 1: This is a warmup text."
        
        for i in range(WARMUP_ITERATIONS):
            logger.info(f"Warmup iteration {i+1}/{WARMUP_ITERATIONS}")
            
            # Process inputs
            inputs = processor(
                text=[dummy_text],
                voice_samples=[dummy_audio],
                padding=True,
                return_tensors="pt",
                return_attention_mask=True,
            )
            
            # Move to device
            for k, v in inputs.items():
                if torch.is_tensor(v):
                    inputs[k] = v.to(device)
            
            # Generate
            with torch.inference_mode():
                _ = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=CFG_SCALE,
                    tokenizer=processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=False,
                )
        
        # Clear cache after warmup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        warmup_complete = True
        warmup_time = time.time() - warmup_start
        logger.info(f"Model warmup completed in {warmup_time:.2f} seconds")
        
    except Exception as e:
        logger.error(f"Model warmup failed: {e}")
        logger.error(traceback.format_exc())
        warmup_complete = False

def load_model():
    """Load the VibeVoice model and processor with optimizations."""
    global processor, model, model_stats, current_model_name

    if model is not None:
        return

    start_time = time.time()
    
    try:
        # Check GPU compatibility first
        gpu_available, gpu_info = check_gpu_compatibility()
        if not gpu_available:
            logger.warning("No GPU available - model performance may be slow")

        # Select optimal model based on hardware
        current_model_name = select_optimal_model()
        logger.info(f"Selected model: {current_model_name}")

        # Download model if needed
        model_path = download_model(current_model_name)

        # Load processor
        logger.info("Loading VibeVoice processor...")
        processor = VibeVoiceProcessor.from_pretrained(str(model_path))
        
        # Determine optimal dtype and attention mode
        if str(device) == "cuda":
            load_dtype = torch.bfloat16
            attn_impl_primary = get_optimal_attention_mode()
        else:
            load_dtype = torch.float32
            attn_impl_primary = "sdpa"

        logger.info(f"Using device: {device}, torch_dtype: {load_dtype}, attn_implementation: {attn_impl_primary}")
        
        # Load model with optimizations
        try:
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_path),
                torch_dtype=load_dtype,
                attn_implementation=attn_impl_primary,
                device_map=None,  # Don't use device_map="auto" without quantization
                low_cpu_mem_usage=True,
            )
            # Manually move to device
            model.to(device)
            logger.info(f"Model loaded successfully with {attn_impl_primary} attention")
            
        except Exception as e:
            logger.warning(f"Failed to load with '{attn_impl_primary}': {e}. Falling back to 'eager'.")
            model = VibeVoiceForConditionalGenerationInference.from_pretrained(
                str(model_path),
                torch_dtype=load_dtype,
                attn_implementation="eager",
                device_map=None,
                low_cpu_mem_usage=True,
            )
            model.to(device)
            attn_impl_primary = "eager"
            logger.info("Model loaded successfully with eager attention (fallback)")
        
        # Model optimizations
        model.eval()
        model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
        
        load_time = time.time() - start_time
        logger.info(f"Model loaded successfully in {load_time:.2f} seconds")
        
        # Collect model stats
        model_stats = {
            "model_name": current_model_name,
            "load_time_seconds": load_time,
            "device": str(device),
            "torch_dtype": str(load_dtype),
            "attn_implementation": attn_impl_primary,
            "memory_usage_mb": gpu_info.get("allocated_memory", 0) * 1024 if gpu_info.get("allocated_memory") else 0,
        }
        
        # Perform warmup
        warmup_model()
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

def preprocess_audio(audio_bytes: bytes, filename: str) -> np.ndarray:
    """Audio preprocessing with better performance."""
    preprocess_start = time.time()
    
    try:
        # Use in-memory processing to avoid file I/O
        audio_io = io.BytesIO(audio_bytes)
        
        # Load audio directly from memory
        audio, sr = librosa.load(audio_io, sr=None)
        
        # Batch operations for efficiency
        if sr != SAMPLE_RATE:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
        
        # Normalize and denoise in one pass
        audio = librosa.util.normalize(audio)
        
        # Optional noise reduction (can be disabled for speed)
        audio = nr.reduce_noise(y=audio, sr=SAMPLE_RATE)
        
        # Trim silence with optimized parameters
        audio, _ = librosa.effects.trim(audio, top_db=20)
        
        # Validate length
        if len(audio) / SAMPLE_RATE > MAX_AUDIO_LENGTH:
            raise ValueError(f"Audio too long: {len(audio)/SAMPLE_RATE:.1f}s > {MAX_AUDIO_LENGTH}s")
        
        preprocess_time = time.time() - preprocess_start
        performance_stats["average_preprocessing_time"] = (
            performance_stats["average_preprocessing_time"] * 0.9 + preprocess_time * 0.1
        )
        
        return audio
        
    except Exception as e:
        logger.error(f"Failed to preprocess audio: {e}")
        raise HTTPException(status_code=400, detail=f"Audio preprocessing failed: {str(e)}")

def audio_to_bytes(audio: np.ndarray) -> bytes:
    """Audio conversion to bytes."""
    postprocess_start = time.time()
    
    # Handle torch tensors
    if isinstance(audio, torch.Tensor):
        if audio.dtype == torch.bfloat16:
            audio = audio.to(torch.float32).cpu().numpy()
        else:
            audio = audio.cpu().numpy()
    
    # Vectorized operations
    if audio.ndim > 1:
        audio = np.squeeze(audio)
    
    # Ensure float32 and clip in one operation
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    
    # Convert to int16 PCM efficiently
    audio_int16 = (audio * 32767).astype(np.int16)
    
    # Write to in-memory WAV
    byte_io = io.BytesIO()
    with wave.open(byte_io, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(audio_int16.tobytes())
    
    postprocess_time = time.time() - postprocess_start
    performance_stats["average_postprocessing_time"] = (
        performance_stats["average_postprocessing_time"] * 0.9 + postprocess_time * 0.1
    )
    
    return byte_io.getvalue()

def audio_chunk_to_bytes(audio_chunk: np.ndarray) -> bytes:
    """Convert audio chunk to bytes with dithering."""
    if isinstance(audio_chunk, torch.Tensor):
        if audio_chunk.ndim > 1:
            audio_chunk = audio_chunk.squeeze(0)
        if audio_chunk.dtype == torch.bfloat16:
            audio = audio_chunk.detach().cpu().to(torch.float32).numpy()
        else:
            audio = audio_chunk.detach().cpu().numpy()
    else:
        audio = audio_chunk
    
    # This squeeze might be redundant if the tensor is already squeezed, but it's safe to keep.
    if audio.ndim > 1 and audio.shape[0] == 1:
         audio = np.squeeze(audio, axis=0)

    # Remove DC offset (handle both mono and multi-channel)
    if audio.ndim > 1:
        # For multi-channel, subtract mean from each channel independently
        audio = audio - np.mean(audio, axis=1, keepdims=True)
    else:
        # For mono
        audio = audio - np.mean(audio)
    
    # Clip
    audio = np.clip(audio.astype(np.float32), -1.0, 1.0)
    
   
    dither = np.random.uniform(-1, 1, size=audio.shape) * (1 / 32767.0) # More correct dithering
    audio = audio + dither

    audio = np.clip(audio, -1.0, 1.0)
    
    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    
    return audio_int16.tobytes()

async def generate_audio_async(text: str, voice_samples: List[np.ndarray]) -> tuple[np.ndarray, AudioStats]:
    """Asynchronous audio generation with detailed stats."""
    if model is None or processor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if not warmup_complete:
        raise HTTPException(status_code=503, detail="Model is still warming up")
    
    total_start = time.time()
    queue_start = total_start
    
    try:
        # Validate text length
        if len(text) > MAX_TEXT_LENGTH:
            raise ValueError(f"Text too long: {len(text)} > {MAX_TEXT_LENGTH}")
        
        # Format text efficiently
        num_speakers = len(voice_samples)
        if num_speakers == 1:
            formatted_text = f"Speaker 1: {text.strip()}"
        else:
            if not re.search(r"^\s*Speaker\s+\d+\s*:", text, re.MULTILINE):
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if lines:
                    formatted_lines = [f"Speaker {i+1}: {line}" for i, line in enumerate(lines)]
                    formatted_text = "\n".join(formatted_lines)
                else:
                    formatted_text = text
            else:
                formatted_text = text
        
        formatted_text = formatted_text.replace("ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢", "'")
        
        # Track CPU usage before inference
        cpu_before = psutil.cpu_percent(interval=None)
        gpu_before = get_detailed_gpu_info()
        
        # Process inputs
        processing_start = time.time()
        inputs = processor(
            text=[formatted_text],
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move tensors to device efficiently
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device, non_blocking=True)
        
        queue_time = processing_start - queue_start
        
        # Patch config if needed
        if not hasattr(model.config, 'num_hidden_layers'):
            model.config.num_hidden_layers = getattr(
                model.config, 'language_model_num_hidden_layers',
                getattr(model.config, 'encoder_num_hidden_layers', 28)
            )
        
        # Generate with performance monitoring
        inference_start = time.time()
        
        with torch.inference_mode():
            if str(device) == "cuda" and torch.cuda.is_available():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    generated_audio = model.generate(
                        **inputs,
                        max_new_tokens=None,
                        cfg_scale=CFG_SCALE,
                        tokenizer=processor.tokenizer,
                        generation_config={'do_sample': False},
                        verbose=False,
                    )
            else:
                generated_audio = model.generate(
                    **inputs,
                    max_new_tokens=None,
                    cfg_scale=CFG_SCALE,
                    tokenizer=processor.tokenizer,
                    generation_config={'do_sample': False},
                    verbose=False,
                )
        
        inference_time = time.time() - inference_start
        
        # Get final stats
        cpu_after = psutil.cpu_percent(interval=None)
        gpu_after = get_detailed_gpu_info()
        
        generated_audio = generated_audio.speech_outputs[0].cpu().to(torch.float32).numpy()
        
        total_time = time.time() - total_start
        
        # Update performance stats
        performance_stats["total_requests"] += 1
        performance_stats["average_inference_time"] = (
            performance_stats["average_inference_time"] * 0.9 + inference_time * 0.1
        )
        
        if gpu_after.get("allocated_memory"):
            performance_stats["peak_memory_usage"] = max(
                performance_stats["peak_memory_usage"],
                gpu_after["allocated_memory"] * 1024
            )
        
        stats = AudioStats(
            duration_seconds=len(generated_audio) / SAMPLE_RATE,
            sample_rate=SAMPLE_RATE,
            channels=1,
            processing_time_seconds=processing_start - total_start,
            model_inference_time_seconds=inference_time,
            preprocessing_time_seconds=0,  # Set by caller
            postprocessing_time_seconds=0,  # Set by caller
            gpu_memory_used_mb=gpu_after.get("allocated_memory", 0) * 1024 if gpu_after.get("allocated_memory") else None,
            gpu_memory_peak_mb=gpu_after.get("cached_memory", 0) * 1024 if gpu_after.get("cached_memory") else None,
            gpu_utilization=gpu_after.get("utilization"),
            cpu_usage_during_inference=cpu_after - cpu_before if cpu_after > cpu_before else None,
            queue_time_seconds=queue_time,
            total_latency_seconds=total_time
        )
        
        return generated_audio, stats
        
    except Exception as e:
        logger.error(f"Audio generation failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Audio generation failed: {str(e)}")

async def generate_streaming_audio(text: str, voice_samples: List[np.ndarray]) -> AsyncGenerator[Dict[str, Any], None]:
    """Generate streaming audio with real-time chunks."""
    if model is None or processor is None:
        yield {"type": "error", "message": "Model not loaded", "timestamp": time.time()}
        return
    
    if not warmup_complete:
        yield {"type": "error", "message": "Model warming up", "timestamp": time.time()}
        return
    
    try:
        # Format text
        num_speakers = len(voice_samples)
        if num_speakers == 1:
            formatted_text = f"Speaker 1: {text.strip()}"
        else:
            if not re.search(r"^\s*Speaker\s+\d+\s*:", text, re.MULTILINE):
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                if lines:
                    formatted_lines = [f"Speaker {i+1}: {line}" for i, line in enumerate(lines)]
                    formatted_text = "\n".join(formatted_lines)
                else:
                    formatted_text = text
            else:
                formatted_text = text
        
        formatted_text = formatted_text.replace("'", "'")
        
        # Process inputs
        inputs = processor(
            text=[formatted_text],
            voice_samples=voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        # Move to device
        for k, v in inputs.items():
            if torch.is_tensor(v):
                inputs[k] = v.to(device, non_blocking=True)
        
        # CRITICAL FIX: Reinitialize scheduler steps before generation
        model.set_ddpm_inference_steps(num_steps=DDPM_STEPS)
        
        # Use AsyncAudioStreamer
        from vibevoice.modular.streamer import AsyncAudioStreamer
        audio_streamer = AsyncAudioStreamer(batch_size=1)
        
        stop_flag = threading.Event()
        
        def should_stop():
            return stop_flag.is_set()
        
        def run_generation():
            try:
                with torch.inference_mode():
                    if str(device) == "cuda" and torch.cuda.is_available():
                        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                            model.generate(
                                **inputs,
                                max_new_tokens=None,
                                cfg_scale=CFG_SCALE,
                                tokenizer=processor.tokenizer,
                                generation_config={'do_sample': False},
                                verbose=False,
                                audio_streamer=audio_streamer,
                                stop_check_fn=should_stop,
                            )
                    else:
                        model.generate(
                            **inputs,
                            max_new_tokens=None,
                            cfg_scale=CFG_SCALE,
                            tokenizer=processor.tokenizer,
                            generation_config={'do_sample': False},
                            verbose=False,
                            audio_streamer=audio_streamer,
                            stop_check_fn=should_stop,
                        )
            except Exception as e:
                logger.error(f"Generation error: {e}")
                logger.error(traceback.format_exc())
        
        generation_task = asyncio.create_task(asyncio.to_thread(run_generation))
        
        # INCREASED chunk size to reduce clicks/pops
        CHUNK_ACCUMULATION_SIZE = 4800  # 200ms at 24kHz - smoother playback
        accumulated_audio = []
        accumulated_samples = 0
        chunk_count = 0
        
        try:
            async for audio_chunk in audio_streamer.get_stream(0):
                numpy_chunk = audio_chunk.to(torch.float32).cpu().numpy()
                
                accumulated_audio.append(numpy_chunk)
                accumulated_samples += numpy_chunk.size
                
                if accumulated_samples >= CHUNK_ACCUMULATION_SIZE:
                    combined_chunk = np.concatenate(accumulated_audio)
                    
                    # Apply crossfade to reduce clicks
                    combined_chunk = apply_crossfade(combined_chunk, SAMPLE_RATE)
                    
                    chunk_count += 1
                    audio_bytes = audio_chunk_to_bytes(combined_chunk)
                    
                    chunk_data = {
                        "type": "audio_chunk",
                        "chunk_id": chunk_count,
                        "audio_data": audio_bytes.hex(),
                        "sample_rate": SAMPLE_RATE,
                        "duration_ms": len(combined_chunk) / SAMPLE_RATE * 1000,
                        "timestamp": time.time()
                    }
                    
                    yield chunk_data
                    
                    accumulated_audio = []
                    accumulated_samples = 0
            
            # Send remaining audio
            if accumulated_audio:
                combined_chunk = np.concatenate(accumulated_audio)
                combined_chunk = apply_crossfade(combined_chunk, SAMPLE_RATE)
                chunk_count += 1
                
                audio_bytes = audio_chunk_to_bytes(combined_chunk)
                
                yield {
                    "type": "audio_chunk",
                    "chunk_id": chunk_count,
                    "audio_data": audio_bytes.hex(),
                    "sample_rate": SAMPLE_RATE,
                    "duration_ms": len(combined_chunk) / SAMPLE_RATE * 1000,
                    "timestamp": time.time()
                }
        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            logger.error(traceback.format_exc())
            stop_flag.set()
            yield {"type": "error", "message": str(e), "timestamp": time.time()}
        
        finally:
            try:
                await generation_task
            except Exception as e:
                logger.error(f"Generation task error: {e}")
            
            yield {
                "type": "complete",
                "total_chunks": chunk_count,
                "timestamp": time.time()
            }
        
        performance_stats["streaming_requests"] += 1
        
    except Exception as e:
        logger.error(f"Streaming generation failed: {e}")
        logger.error(traceback.format_exc())
        yield {"type": "error", "message": str(e), "timestamp": time.time()}

# in your server code
def apply_crossfade(audio: np.ndarray, sample_rate: int, fade_ms: int = 10) -> np.ndarray:
    """Apply fade-in and fade-out to reduce clicks."""
    # Check if the audio is 2D (multi-channel)
    if audio.ndim == 1:
        num_samples = len(audio)
    else:
        num_samples = audio.shape[1] # Get length from the time axis (columns)

    fade_samples = int(sample_rate * fade_ms / 1000)
    fade_samples = min(fade_samples, num_samples // 4)

    if num_samples < fade_samples * 2:
        return audio

    # Create fade curves
    fade_in = np.linspace(0, 1, fade_samples)
    fade_out = np.linspace(1, 0, fade_samples)

    # Apply fades correctly to multi-channel audio
    if audio.ndim == 1:
        # Mono audio
        audio[:fade_samples] *= fade_in
        audio[-fade_samples:] *= fade_out
    else:
        # Multi-channel audio: apply fade to every channel
        audio[:, :fade_samples] *= fade_in
        audio[:, -fade_samples:] *= fade_out
    
    return audio

@app.on_event("startup")
async def startup_event():
    logger.info("Starting VibeVoice Inference Server v2.2...")
    load_model()

@app.get("/")
async def root():
    return {
        "message": "VibeVoice Inference Server v2.2 - Dynamic Model Selection & Streaming",
        "version": "2.2.0",
        "current_model": current_model_name,
        "model_stats": model_stats,
        "performance_stats": performance_stats,
        "warmup_complete": warmup_complete,
        "endpoints": {
            "/single-speaker": "Single speaker TTS with voice cloning",
            "/multi-speaker": "Multi-speaker TTS with voice cloning", 
            "/single-speaker-stream": "Single speaker TTS with SSE streaming",
            "/multi-speaker-stream": "Multi-speaker TTS with SSE streaming",
            "/health": "Health check endpoint",
            "/system": "Detailed system information",
            "/metrics": "Performance metrics",
            "/warmup": "Trigger model warmup"
        }
    }

@app.get("/health")
async def health_check(license_key: str = None):
    """Enhanced health check endpoint with license verification."""
    
    # Check license if provided
    license_status = None
    if license_key:
        cached_status = check_license_validity(license_key)
        
        if cached_status["needs_verification"]:
            # Perform fresh verification
            license_status = await verify_and_cache_license(license_key)
        else:
            license_status = cached_status
        
        # If license is invalid, return error
        if not license_status["valid"]:
            return {
                "status": "unauthorized",
                "error": "Invalid or expired license",
                "license_status": license_status["status"],
                "message": license_status["message"],
                "model_loaded": False,
                "warmup_complete": False
            }
    
    # Get system info
    system_info = get_system_info()
    
    response = {
        "status": "healthy" if system_info.model_loaded and warmup_complete else "warming_up" if system_info.model_loaded else "model_not_loaded",
        "gpu_available": system_info.gpu_available,
        "model_loaded": system_info.model_loaded,
        "current_model": current_model_name,
        "warmup_complete": warmup_complete,
        "gpu_memory_usage": system_info.gpu_memory_used,
        "gpu_utilization": system_info.gpu_utilization,
        "cpu_usage": system_info.cpu_usage,
        "ram_usage_percent": (system_info.ram_used / system_info.ram_total) * 100
    }
    
    # Add license status if checked
    if license_status:
        response["license_status"] = license_status["status"]
        response["license_valid"] = license_status["valid"]
        if "time_until_recheck" in license_status:
            response["license_recheck_in_hours"] = license_status["time_until_recheck"] / 3600
    
    return response

@app.post("/verify-license")
async def verify_license_endpoint(license_key: str = Form(...)):
    """
    Verify license key for initial setup.
    """
    if not license_key or not license_key.strip():
        raise HTTPException(status_code=400, detail="License key is required")
    
    result = await verify_and_cache_license(license_key)
    
    if not result["valid"]:
        raise HTTPException(
            status_code=403, 
            detail=result["message"]
        )
    
    return {
        "valid": result["valid"],
        "status": result["status"],
        "message": result["message"]
    }
    
@app.get("/system", response_model=SystemInfo)
async def get_system_info_endpoint():
    """Get comprehensive system information."""
    return get_system_info()

@app.get("/metrics")
async def get_performance_metrics():
    """Get detailed performance metrics."""
    return {
        "performance_stats": performance_stats,
        "model_stats": model_stats,
        "system_info": get_system_info().dict(),
        "warmup_complete": warmup_complete
    }

@app.post("/warmup")
async def trigger_warmup():
    """Manually trigger model warmup."""
    global warmup_complete
    warmup_complete = False
    warmup_model()
    return {"message": "Warmup completed", "warmup_complete": warmup_complete}

@app.post("/single-speaker")
async def single_speaker_tts(
    text: str = Form(...),
    voice_file: UploadFile = File(...)
):
    """Optimized single speaker TTS endpoint."""
    request_start = time.time()
    
    try:
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not voice_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
            raise HTTPException(status_code=400, detail="Unsupported audio format")
        
        # Read and preprocess audio
        audio_bytes = await voice_file.read()
        preprocess_start = time.time()
        voice_sample = preprocess_audio(audio_bytes, voice_file.filename)
        preprocess_time = time.time() - preprocess_start
        
        # Generate audio
        generated_audio, stats = await generate_audio_async(text, [voice_sample])
        
        # Update preprocessing time in stats
        stats.preprocessing_time_seconds = preprocess_time
        
        # Convert to bytes
        postprocess_start = time.time()
        audio_output = audio_to_bytes(generated_audio)
        postprocess_time = time.time() - postprocess_start
        
        # Update postprocessing time in stats
        stats.postprocessing_time_seconds = postprocess_time
        
        # Calculate total request time
        total_request_time = time.time() - request_start
        
        # Return optimized streaming response
        response = StreamingResponse(
            iter([audio_output]),
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": f"{stats.duration_seconds:.3f}",
                "X-Processing-Time": f"{stats.processing_time_seconds:.3f}",
                "X-Inference-Time": f"{stats.model_inference_time_seconds:.3f}",
                "X-Preprocessing-Time": f"{stats.preprocessing_time_seconds:.3f}",
                "X-Postprocessing-Time": f"{stats.postprocessing_time_seconds:.3f}",
                "X-Queue-Time": f"{stats.queue_time_seconds:.3f}",
                "X-Total-Latency": f"{stats.total_latency_seconds:.3f}",
                "X-Request-Time": f"{total_request_time:.3f}",
                "X-Sample-Rate": str(stats.sample_rate),
                "X-GPU-Memory-Used": f"{stats.gpu_memory_used_mb:.1f}" if stats.gpu_memory_used_mb else "N/A",
                "X-GPU-Utilization": f"{stats.gpu_utilization:.1f}%" if stats.gpu_utilization else "N/A",
                "X-CPU-Usage": f"{stats.cpu_usage_during_inference:.1f}%" if stats.cpu_usage_during_inference else "N/A",
                "X-Model-Used": current_model_name,
                "Content-Disposition": "attachment; filename=generated_speech.wav",
                "Cache-Control": "no-cache"
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
    """Optimized multi-speaker TTS endpoint."""
    request_start = time.time()
    
    try:
        # Validate inputs
        if not text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        if not voice_files:
            raise HTTPException(status_code=400, detail="At least one voice file required")
        
        if "Speaker" not in text:
            raise HTTPException(
                status_code=400,
                detail="Text must contain speaker labels (e.g., 'Speaker 1: Hello\\nSpeaker 2: Hi')"
            )
        
        # Process voice files in parallel
        preprocess_start = time.time()
        voice_samples = []
        
        async def process_voice_file(voice_file, index):
            if not voice_file.filename.lower().endswith(('.wav', '.mp3', '.m4a', '.flac')):
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported audio format in file {index+1}: {voice_file.filename}"
                )
            
            audio_bytes = await voice_file.read()
            return preprocess_audio(audio_bytes, voice_file.filename)
        
        # Process files concurrently
        tasks = [process_voice_file(vf, i) for i, vf in enumerate(voice_files)]
        voice_samples = await asyncio.gather(*tasks)
        
        preprocess_time = time.time() - preprocess_start
        
        # Generate audio
        generated_audio, stats = await generate_audio_async(text, voice_samples)
        
        # Update preprocessing time
        stats.preprocessing_time_seconds = preprocess_time
        
        # Convert to bytes
        postprocess_start = time.time()
        audio_output = audio_to_bytes(generated_audio)
        postprocess_time = time.time() - postprocess_start
        
        stats.postprocessing_time_seconds = postprocess_time
        
        # Calculate total request time
        total_request_time = time.time() - request_start
        
        response = StreamingResponse(
            iter([audio_output]),
            media_type="audio/wav",
            headers={
                "X-Audio-Duration": f"{stats.duration_seconds:.3f}",
                "X-Processing-Time": f"{stats.processing_time_seconds:.3f}",
                "X-Inference-Time": f"{stats.model_inference_time_seconds:.3f}",
                "X-Preprocessing-Time": f"{stats.preprocessing_time_seconds:.3f}",
                "X-Postprocessing-Time": f"{stats.postprocessing_time_seconds:.3f}",
                "X-Queue-Time": f"{stats.queue_time_seconds:.3f}",
                "X-Total-Latency": f"{stats.total_latency_seconds:.3f}",
                "X-Request-Time": f"{total_request_time:.3f}",
                "X-Sample-Rate": str(stats.sample_rate),
                "X-GPU-Memory-Used": f"{stats.gpu_memory_used_mb:.1f}" if stats.gpu_memory_used_mb else "N/A",
                "X-GPU-Utilization": f"{stats.gpu_utilization:.1f}%" if stats.gpu_utilization else "N/A",
                "X-CPU-Usage": f"{stats.cpu_usage_during_inference:.1f}%" if stats.cpu_usage_during_inference else "N/A",
                "X-Speakers": str(len(voice_samples)),
                "X-Model-Used": current_model_name,
                "Content-Disposition": "attachment; filename=generated_multispeaker_speech.wav",
                "Cache-Control": "no-cache"
            }
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Multi-speaker TTS failed: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/tts-stream")
async def websocket_tts_stream(websocket: WebSocket):
    """WebSocket endpoint with voice caching."""
    await websocket.accept()
    logger.info("WebSocket connection established.")

    try:
        while True:
            try:
                config_data = await websocket.receive_json()
                logger.info("Received new TTS request.")

                text = config_data.get('text', '')
                generation_type = config_data.get('type', 'single')
                voice_files_data = config_data.get('voice_files', [])

                if not text.strip():
                    await websocket.send_json({
                        "type": "error",
                        "message": "Text cannot be empty"
                    })
                    continue

                # Generate cache key
                if generation_type == 'single':
                    cache_key = get_voice_cache_key(generation_type, ["single"])
                else:
                    cache_key = get_voice_cache_key(generation_type, [f"voice_{i}" for i in range(len(voice_files_data))])

                # Try to get from cache
                voice_samples = get_cached_voice_samples(cache_key)

                if voice_samples is None:
                    # Process and cache voice files
                    logger.info("Processing and caching voice samples...")
                    voice_samples = []
                    for idx, voice_data in enumerate(voice_files_data):
                        audio_bytes = base64.b64decode(voice_data['data'])
                        voice_sample = preprocess_audio(audio_bytes, f"voice_{idx}.wav")
                        voice_samples.append(voice_sample)
                    
                    cache_voice_samples(cache_key, voice_samples)
                else:
                    logger.info("Using cached voice samples.")

                if generation_type == 'multi' and len(voice_samples) < 2:
                    await websocket.send_json({
                        "type": "error",
                        "message": "Multi-speaker requires at least 2 voices"
                    })
                    continue

                await websocket.send_json({
                    "type": "ready",
                    "message": "Starting generation."
                })

                chunk_count = 0
                async for chunk_data in generate_streaming_audio(text, voice_samples):
                    try:
                        await websocket.send_json(chunk_data)
                        if chunk_data["type"] == "audio_chunk":
                            chunk_count += 1
                        elif chunk_data["type"] in ["complete", "error"]:
                            break
                    except RuntimeError as e:
                        # Connection closed during send
                        logger.info(f"Connection closed during streaming: {e}")
                        break
                    
                
                logger.info(f"Streaming completed: {chunk_count} chunks sent.")

            except Exception as e:
                logger.error(f"Error processing TTS job: {e}")
                try:
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
                except:
                    pass
                continue

            except WebSocketDisconnect:
                logger.info("Client disconnected during job.")
                break  # Exit the while loop
            except RuntimeError as e:
                if "disconnect" in str(e).lower():
                    logger.info("Client disconnected.")
                    break  # Exit the while loop
                else:
                    logger.error(f"Runtime error: {e}")
                    break
            except Exception as e:
                logger.error(f"Error processing TTS job: {e}")
                logger.error(traceback.format_exc())
                # Try to send error, but don't fail if connection is closed
                try:
                    if websocket.client_state.name == "CONNECTED":
                        await websocket.send_json({
                            "type": "error",
                            "message": str(e)
                        })
                except:
                    pass
                break  # Exit on error

    except WebSocketDisconnect:
        logger.info("WebSocket disconnected.")
    except Exception as e:
        logger.error(f"Fatal WebSocket error: {e}")
    finally:
        logger.info("Closing WebSocket connection.")
        try:
            await websocket.close()
        except:
            pass
          
@app.post("/reload-model")
async def reload_model():
    """Reload the model with optimizations."""
    global processor, model, warmup_complete
    
    try:
        processor = None
        model = None
        warmup_complete = False
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        load_model()
        
        return {
            "message": "Model reloaded successfully", 
            "current_model": current_model_name,
            "stats": model_stats,
            "warmup_complete": warmup_complete
        }
        
    except Exception as e:
        logger.error(f"Failed to reload model: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")

async def verify_gumroad_license(license_key: str, product_id: str = GUMROAD_PRODUCT_ID) -> dict:
    """
    Verify license key with Gumroad API.
    Returns: {success: bool, valid: bool, purchase: dict, message: str}
    """
    try:
        logger.info(f"Verifying license key with Gumroad API...")
        
        response = requests.post(
            "https://api.gumroad.com/v2/licenses/verify",
            data={
                "product_id": product_id,
                "license_key": license_key,
            },
            timeout=10
        )
        
        data = response.json()
        
        if response.status_code == 200 and data.get("success"):
            return {
                "success": True,
                "valid": True,
                "purchase": data.get("purchase", {}),
                "uses": data.get("uses", 0),
                "message": "License is valid"
            }
        else:
            return {
                "success": True,
                "valid": False,
                "message": data.get("message", "Invalid license key")
            }
            
    except requests.exceptions.Timeout:
        logger.error("Gumroad API timeout")
        return {
            "success": False,
            "valid": False,
            "message": "Verification timeout - using grace period if available"
        }
    except Exception as e:
        logger.error(f"License verification failed: {e}")
        return {
            "success": False,
            "valid": False,
            "message": f"Verification error: {str(e)}"
        }

def check_license_validity(license_key: str) -> dict:
    """
    Check if license is valid with 24h grace period.
    Returns: {valid: bool, status: str, message: str, needs_verification: bool}
    """
    if not license_key or not license_key.strip():
        return {
            "valid": False,
            "status": "invalid",
            "message": "No license key provided",
            "needs_verification": True
        }
    
    # Check cache
    cached = license_cache.get(license_key)
    
    if cached:
        time_since_check = time.time() - cached['last_check']
        
        # If within grace period
        if time_since_check < LICENSE_GRACE_PERIOD:
            if cached['valid']:
                return {
                    "valid": True,
                    "status": "valid",
                    "message": "License is valid (cached)",
                    "needs_verification": False,
                    "last_check": cached['last_check'],
                    "time_until_recheck": LICENSE_GRACE_PERIOD - time_since_check
                }
            else:
                # Was invalid within grace period
                return {
                    "valid": False,
                    "status": "invalid",
                    "message": "License key is invalid",
                    "needs_verification": False
                }
        else:
            # Grace period expired, need fresh verification
            if cached['valid']:
                # Allow use during verification (grace period extension)
                return {
                    "valid": True,
                    "status": "grace_period",
                    "message": "Grace period active - verification needed",
                    "needs_verification": True
                }
            else:
                return {
                    "valid": False,
                    "status": "expired",
                    "message": "License verification expired",
                    "needs_verification": True
                }
    
    # No cache entry
    return {
        "valid": False,
        "status": "unverified",
        "message": "License not yet verified",
        "needs_verification": True
    }

async def verify_and_cache_license(license_key: str) -> dict:
    """
    Verify license with Gumroad and update cache.
    Returns: {valid: bool, status: str, message: str}
    """
    result = await verify_gumroad_license(license_key)
    
    if result["success"]:
        # Update cache
        license_cache[license_key] = {
            "valid": result["valid"],
            "last_check": time.time(),
            "purchase_info": result.get("purchase", {})
        }
        
        return {
            "valid": result["valid"],
            "status": "valid" if result["valid"] else "invalid",
            "message": result["message"]
        }
    else:
        # Verification failed (network error, etc)
        # Check if we have cached data
        cached_status = check_license_validity(license_key)
        
        if cached_status["valid"]:
            # Use grace period
            return {
                "valid": True,
                "status": "grace_period",
                "message": "Using cached validation due to verification error"
            }
        else:
            return {
                "valid": False,
                "status": "error",
                "message": result["message"]
            }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "speak:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        loop="uvloop",  # Use uvloop for better performance
        http="httptools"  # Use httptools for better HTTP parsing
    )