# AutoKatha 📚➡️🎬

Transform comics and storybooks into animated videos with AI-generated narration, art style transfer, and subtitles.

## Features

- 📖 **Smart Comic Reading**: Uses Moondream2 VLM to understand comic panels and extract text
- ✍️ **AI Screenwriting**: Enhances descriptions into engaging narration using Gemma 3
- 🎨 **Art Style Transfer**: Transform panels into traditional Indian art styles (Madhubani, Kerala Mural, Mughal Miniature, etc.)
- 🎬 **AI Animation**: Stable Video Diffusion for realistic motion, or Ken Burns for speed
- 🎙️ **Multilingual Voice**: XTTS v2 with voice cloning support
- 📝 **Auto Captions**: Whisper-powered subtitle generation
- 🖥️ **Local Processing**: Everything runs on your Mac M4 Max

## Supported Art Styles

| Style | Origin | Best For |
|-------|--------|----------|
| Madhubani | Bihar | Folk tales, nature stories |
| Kerala Mural | Kerala | Mythological narratives |
| Mughal Miniature | Mughal Era | Royal/historical stories |
| Warli | Maharashtra | Tribal, simple narratives |
| Kalamkari | Andhra Pradesh | Epic stories, textile art look |
| Pattachitra | Odisha | Religious narratives |
| Tanjore | Tamil Nadu | Divine/religious content |
| Gond | Madhya Pradesh | Nature, tribal stories |
| Amar Chitra Katha | Modern | Classic Indian comics |
| Anime | Japan | Modern, dynamic stories |

## Requirements

- **Hardware**: MacBook Pro M4 Max (36GB RAM recommended)
- **Software**:
  - macOS 14+
  - Python 3.10+
  - Ollama (for LLM)
  - FFmpeg (for video processing)

## Quick Start

### 1. Clone and Setup

```bash
cd /Users/abhilaksh/Projects/AutoKatha
chmod +x setup.sh
./setup.sh
```

### 2. Start Ollama

```bash
ollama serve
```

### 3. Launch AutoKatha

```bash
source venv/bin/activate
python app.py
```

### 4. Open in Browser

Navigate to: **http://127.0.0.1:7860**

## Manual Installation

If the setup script doesn't work:

```bash
# Install system dependencies
brew install ffmpeg poppler

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Install Ollama and model
brew install ollama
ollama serve &
ollama pull gemma3:4b
```

## Usage

### Web Interface

1. **Upload Comic**: PDF or images (PNG, JPG)
2. **Set Title**: Name for your video
3. **Choose Art Style**: Select from dropdown
4. **Choose Language**: Hindi, English, Tamil, etc.
5. **Configure Options**:
   - AI Animation (SVD) for real motion
   - Style strength (0.3-0.9)
   - Voice sample for cloning (optional)
6. **Generate**: Click and wait (10-30 minutes)

### Command Line

```python
from pipeline.orchestrator import generate_video

video_path = generate_video(
    comic_path="my_comic.pdf",
    title="The Story of Rama",
    art_style="kerala_mural",
    language="hi",
    use_svd=True
)
print(f"Video saved: {video_path}")
```

### Step-by-Step Processing

For more control:

```python
from pipeline.orchestrator import AutoKatha

ak = AutoKatha()

# Create project
ak.create_project("My Story", art_style="madhubani", language="en")

# Run steps individually
ak.step_read_comic("comic.pdf")
ak.step_write_script()
ak.step_stylize_panels(strength=0.7)
ak.step_animate(use_svd=False)  # Use Ken Burns for speed
ak.step_generate_voice()
ak.step_generate_captions()
video = ak.step_compose_video()
```

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      GRADIO WEB UI                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 1. COMIC READER (Moondream2 VLM)                                │
│    └─► Extract panels, understand visuals, read text            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. SCREENWRITER (Gemma 3 via Ollama)                            │
│    └─► Transform descriptions into engaging narration           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. ARTIST (SDXL Lightning)                                      │
│    └─► Apply art style (Madhubani, Kerala, Mughal, etc.)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. ANIMATOR (Stable Video Diffusion / Ken Burns)                │
│    └─► Create motion from still images                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. VOICE (Coqui XTTS v2)                                        │
│    └─► Generate multilingual narration with voice cloning       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. CAPTIONER (Whisper)                                          │
│    └─► Generate accurate subtitles                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. COMPOSER (FFmpeg)                                            │
│    └─► Merge video, audio, subtitles into final MP4             │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Management

AutoKatha uses sequential model loading to fit within 36GB RAM:

1. Load model → Process → Unload → Clear memory
2. Only one heavy model active at a time
3. Aggressive garbage collection between stages

## Troubleshooting

### Out of Memory
- Disable AI Animation (use Ken Burns instead)
- Process fewer panels at once
- Close other applications

### Ollama Connection Error
```bash
# Check if Ollama is running
pgrep ollama

# Start Ollama
ollama serve

# Pull model if needed
ollama pull gemma3:4b
```

### FFmpeg Not Found
```bash
brew install ffmpeg
```

### Slow First Run
First run downloads ~10-15GB of models. Subsequent runs are faster.

### MPS/Metal Errors
```bash
# Update PyTorch
pip install --upgrade torch torchvision torchaudio
```

## Project Structure

```
AutoKatha/
├── app.py                  # Gradio web interface
├── config.py               # Configuration and art styles
├── requirements.txt        # Python dependencies
├── setup.sh               # Setup script
├── pipeline/
│   ├── __init__.py
│   ├── memory_manager.py   # GPU memory management
│   ├── comic_reader.py     # Moondream2 VLM
│   ├── screenwriter.py     # Ollama/Gemma script generation
│   ├── artist.py           # SDXL Lightning style transfer
│   ├── animator.py         # SVD / Ken Burns animation
│   ├── voice.py            # XTTS v2 TTS
│   ├── captioner.py        # Whisper subtitles
│   ├── composer.py         # FFmpeg video composition
│   └── orchestrator.py     # Main pipeline coordinator
├── outputs/                # Generated videos
├── temp/                   # Temporary files
└── models/                 # Cached models
```

## Supported Languages

| Code | Language |
|------|----------|
| en | English |
| hi | Hindi |
| ta | Tamil |
| te | Telugu |
| bn | Bengali |
| mr | Marathi |
| gu | Gujarati |
| kn | Kannada |
| ml | Malayalam |
| pa | Punjabi |
| es | Spanish |
| fr | French |
| de | German |
| ja | Japanese |
| zh | Chinese |

## License

MIT License - Use freely for personal projects.

## Credits

Built with:
- [Moondream2](https://github.com/vikhyatk/moondream) - Vision Language Model
- [Ollama](https://ollama.ai) - Local LLM inference
- [SDXL Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) - Fast image generation
- [Stable Video Diffusion](https://stability.ai) - Image to video
- [Coqui TTS](https://github.com/coqui-ai/TTS) - Text to speech
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Gradio](https://gradio.app) - Web interface
- [FFmpeg](https://ffmpeg.org) - Video processing

---

Made with ❤️ for Indian storytelling
