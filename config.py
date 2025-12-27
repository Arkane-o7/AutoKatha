"""
AutoKatha Configuration
"""
import os
from pathlib import Path

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not installed, use environment variables directly

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
TEMP_DIR = BASE_DIR / "temp"
MODELS_DIR = BASE_DIR / "models"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)
MODELS_DIR.mkdir(exist_ok=True)

# ============================================================================
# IMAGE GENERATION SETTINGS
# ============================================================================
# Default prompts for image generation (no art style transformation)
DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, watermark, text"

# ============================================================================
# COMFYUI SETTINGS (Optional - for advanced workflows)
# ============================================================================
# ComfyUI server connection
COMFYUI_HOST = "127.0.0.1"
COMFYUI_PORT = 8000  # Desktop app uses port 8000

# Path to your custom workflow JSON (exported as API format from ComfyUI)
# Set to None to use the default SDXL workflow
COMFYUI_WORKFLOW = BASE_DIR / "Anime Art.json"

# Node IDs in your workflow (for injecting prompts)
COMFYUI_PROMPT_NODE = "6"      # Positive prompt node
COMFYUI_NEGATIVE_NODE = "7"    # Negative prompt node  
COMFYUI_SEED_NODE = "3"        # KSampler node (for seed)

# ============================================================================
# SUPPORTED LANGUAGES - For TTS
# ============================================================================
LANGUAGES = {
    "en": "English",
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "kn": "Kannada",
    "ml": "Malayalam",
    "pa": "Punjabi",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "zh": "Chinese",
}

# ============================================================================
# MODEL CONFIGURATIONS
# ============================================================================

# Moondream2 - Comic Reader
MOONDREAM_MODEL = "vikhyatk/moondream2"
MOONDREAM_REVISION = "2025-06-21"  # Latest stable with caption/query API

# SDXL Lightning - Artist
SDXL_LIGHTNING_MODEL = "ByteDance/SDXL-Lightning"
SDXL_LIGHTNING_STEPS = 4  # 4-step model for speed

# Stable Video Diffusion - Animator
SVD_MODEL = "stabilityai/stable-video-diffusion-img2vid-xt"
SVD_FRAMES = 25  # Number of frames to generate
SVD_FPS = 6  # Frames per second

# Coqui TTS - Voice
TTS_MODEL = "tts_models/multilingual/multi-dataset/xtts_v2"

# Whisper - Captions
WHISPER_MODEL = "base"  # Options: tiny, base, small, medium, large

# ============================================================================
# LLM SETTINGS (for scene splitting and prompts)
# ============================================================================
# Groq API (recommended - faster & smarter)
# Get your API key at: https://console.groq.com/keys
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")  # Set via environment variable
GROQ_MODEL = "llama-3.1-70b-versatile"  # Options: llama-3.1-70b-versatile, llama-3.1-8b-instant, mixtral-8x7b-32768

# Ollama (local fallback)
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_HOST = "http://localhost:11434"

# LLM Backend: "groq" or "ollama" (auto-selects groq if API key is set)
LLM_BACKEND = "groq" if GROQ_API_KEY else "ollama"

# ============================================================================
# VIDEO SETTINGS
# ============================================================================
VIDEO_WIDTH = 1920
VIDEO_HEIGHT = 1080
VIDEO_FPS = 24
PANEL_DURATION = 5  # seconds per panel if no audio timing

# ============================================================================
# DEVICE SETTINGS
# ============================================================================
import torch

def get_device():
    """Get the best available device for PyTorch."""
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

DEVICE = get_device()
TORCH_DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32

print(f"🖥️  AutoKatha using device: {DEVICE}")
