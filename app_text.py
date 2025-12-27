"""
AutoKatha - Text Story to Video
Simplified interface: paste story text → get animated video

Features:
- Multiple aspect ratios (16:9, 9:16, 1:1)
- Dynamic scene count based on text length
- Character consistency across scenes
- Enhanced image prompts with style tokens
- Multi-backend TTS (F5-TTS, XTTS, Edge-TTS)
"""
import gradio as gr
from pathlib import Path
from datetime import datetime
import json
import torch
from PIL import Image
import tempfile
import os

from pipeline.memory_manager import MemoryManager, cleanup
from pipeline.comfyui_client import ComfyUIClient
from pipeline.story_processor import (
    StoryProcessor, AspectRatio, Character, Scene, 
    StoryAnalysis, chunk_large_text
)
import config


class TextToVideo:
    """Text story to video pipeline with enhanced features."""
    
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR
        self.device = config.DEVICE
        self.comfyui_client = None
        self.use_comfyui = False
        self.diffusers_pipe = None
        self._tts = None
        
        # Story processor for intelligent scene splitting
        self.story_processor = StoryProcessor()
        
        # Current processing state
        self.current_analysis: StoryAnalysis = None
        
        # Try to connect to ComfyUI
        self._init_comfyui()
    
    def _init_comfyui(self):
        """Initialize ComfyUI client if available."""
        try:
            self.comfyui_client = ComfyUIClient(
                host=getattr(config, 'COMFYUI_HOST', '127.0.0.1'),
                port=getattr(config, 'COMFYUI_PORT', 8188)
            )
            if self.comfyui_client.is_available():
                self.use_comfyui = True
                print("✅ ComfyUI connected - using ComfyUI for image generation")
            else:
                print("⚠️ ComfyUI not running - falling back to diffusers")
        except Exception as e:
            print(f"⚠️ ComfyUI connection failed: {e} - falling back to diffusers")
    
    def get_aspect_dimensions(self, aspect_ratio: str) -> tuple:
        """Get image dimensions for aspect ratio."""
        dimensions = {
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "1:1": (1024, 1024),
            "21:9": (1024, 440),
        }
        return dimensions.get(aspect_ratio, (1024, 576))
    
    def calculate_scenes(self, text: str, mode: str = "auto", custom_count: int = 5) -> int:
        """
        Calculate number of scenes based on mode.
        
        Args:
            text: Story text
            mode: "auto", "short" (3-5), "medium" (6-10), "long" (11-20), "custom"
            custom_count: Custom scene count when mode is "custom"
        """
        if mode == "custom":
            return custom_count
        elif mode == "auto":
            return self.story_processor.calculate_scene_count(text)
        elif mode == "short":
            return min(5, max(3, len(text.split()) // 200))
        elif mode == "medium":
            return min(10, max(6, len(text.split()) // 150))
        elif mode == "long":
            return min(20, max(11, len(text.split()) // 100))
        else:
            return 5
        
    def split_into_scenes(
        self, 
        story_text: str, 
        num_scenes: int = 5,
        extract_characters: bool = True,
        style_prefix: str = "",
    ) -> list:
        """
        Enhanced scene splitting with character extraction and consistency.
        """
        # Use story processor for intelligent splitting
        analysis = self.story_processor.process_story(
            text=story_text,
            title="Story",
            num_scenes=num_scenes,
            style_prefix=style_prefix,
            extract_characters=extract_characters,
        )
        
        self.current_analysis = analysis
        
        # Convert to legacy format for compatibility
        scenes = []
        for scene in analysis.scenes:
            # Enhance prompt with character consistency
            enhanced_prompt = scene.image_prompt
            if analysis.characters:
                enhanced_prompt = self.story_processor.enhance_image_prompt(
                    base_prompt=scene.image_prompt,
                    characters=[c for c in analysis.characters if c.name in scene.characters],
                    style_tokens=style_prefix,
                )
            
            scenes.append({
                "narration": scene.narration,
                "image_prompt": enhanced_prompt,
                "characters": scene.characters,
                "setting": scene.setting,
                "mood": scene.mood,
            })
        
        return scenes
    
    def generate_image(self, prompt: str, pipe=None, aspect_ratio: str = "16:9") -> Image.Image:
        """
        Generate image using ComfyUI (preferred) or SDXL Lightning (fallback).
        
        Args:
            prompt: Image prompt
            pipe: Diffusers pipeline (for reuse)
            aspect_ratio: Target aspect ratio
        """
        width, height = self.get_aspect_dimensions(aspect_ratio)
        
        # Try ComfyUI first
        if self.use_comfyui and self.comfyui_client:
            return self._generate_with_comfyui(prompt, width, height), None
        
        # Fallback to diffusers
        return self._generate_with_diffusers(prompt, pipe, width, height)
    
    def _generate_with_comfyui(self, prompt: str, width: int = 1024, height: int = 576) -> Image.Image:
        """Generate image using ComfyUI API."""
        print(f"   🎨 Generating with ComfyUI ({width}x{height})...")
        
        # Get workflow path from config if set
        workflow_path = getattr(config, 'COMFYUI_WORKFLOW', None)
        if workflow_path:
            workflow_path = str(workflow_path)
        
        # Get node IDs from config
        prompt_node = getattr(config, 'COMFYUI_PROMPT_NODE', '6')
        negative_node = getattr(config, 'COMFYUI_NEGATIVE_NODE', '7')
        seed_node = getattr(config, 'COMFYUI_SEED_NODE', '3')
        
        # Add anime style prefix for the LoRA
        style_prefix = "masterpiece, best quality, very aesthetic, haiz_ai, no lineart, no outline, <lora:haiz_ai_illu:1>, "
        styled_prompt = style_prefix + prompt[:250]
        
        try:
            image = self.comfyui_client.generate_image(
                prompt=styled_prompt,
                workflow_path=workflow_path,
                prompt_node_id=prompt_node,
                negative_node_id=negative_node,
                seed_node_id=seed_node,
                negative_prompt=config.DEFAULT_NEGATIVE_PROMPT
            )
            return image
        except Exception as e:
            print(f"   ❌ ComfyUI error: {e}")
            print(f"   🔄 Falling back to diffusers...")
            self.use_comfyui = False
            image, _ = self._generate_with_diffusers(prompt, None)
            return image
    
    def _generate_with_diffusers(self, prompt: str, pipe=None, width: int = 1024, height: int = 576) -> tuple:
        """Generate image using SDXL Lightning via diffusers."""
        from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
        
        # Truncate prompt to avoid CLIP token limit (77 tokens ~ 300 chars)
        full_prompt = prompt[:300] if len(prompt) > 300 else prompt
        negative = config.DEFAULT_NEGATIVE_PROMPT
        
        if pipe is None and self.diffusers_pipe is None:
            print("🎨 Loading SDXL Lightning...")
            
            # Use float32 for MPS to avoid NaN issues
            dtype = torch.float32 if self.device == "mps" else torch.float16
            
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
            )
            
            # Load Lightning LoRA
            pipe.load_lora_weights(
                "ByteDance/SDXL-Lightning",
                weight_name="sdxl_lightning_4step_lora.safetensors"
            )
            pipe.fuse_lora()
            
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config
            )
            
            pipe = pipe.to(self.device)
            
            if self.device == "mps":
                pipe.enable_attention_slicing()
                # Ensure VAE is in float32 for MPS
                pipe.vae = pipe.vae.to(torch.float32)
            
            self.diffusers_pipe = pipe
        
        pipe = pipe or self.diffusers_pipe
        
        with torch.inference_mode():
            result = pipe(
                prompt=full_prompt,
                negative_prompt=negative,
                num_inference_steps=4,
                guidance_scale=2.0,
                width=width,
                height=height,
            )
            image = result.images[0]
        
        # Verify image is valid (not all black)
        import numpy as np
        img_array = np.array(image)
        if img_array.max() < 10:  # Nearly all black
            print("   ⚠️ Warning: Image appears black, regenerating...")
            # Try again with different seed
            with torch.inference_mode():
                result = pipe(
                    prompt=full_prompt,
                    negative_prompt=negative,
                    num_inference_steps=4,
                    guidance_scale=2.0,
                    width=width,
                    height=height,
                )
                image = result.images[0]
        
        return image, pipe
    
    def generate_audio(self, text: str, language: str, output_path: Path, speaker_wav: str = None) -> Path:
        """Generate TTS audio using edge-tts (simple, reliable)."""
        import edge_tts
        import asyncio
        
        # Voice mapping for languages
        voice_map = {
            "en": "en-US-AriaNeural",
            "hi": "hi-IN-SwaraNeural",
            "ta": "ta-IN-PallaviNeural",
            "te": "te-IN-ShrutiNeural",
            "bn": "bn-IN-TanishaaNeural",
            "mr": "mr-IN-AarohiNeural",
            "gu": "gu-IN-DhwaniNeural",
            "kn": "kn-IN-SapnaNeural",
            "ml": "ml-IN-SobhanaNeural",
            "pa": "pa-IN-GurpreetNeural",
            "es": "es-ES-ElviraNeural",
            "fr": "fr-FR-DeniseNeural",
            "de": "de-DE-KatjaNeural",
            "ja": "ja-JP-NanamiNeural",
            "zh": "zh-CN-XiaoxiaoNeural",
        }
        
        voice = voice_map.get(language, "en-US-AriaNeural")
        
        async def _generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
        
        asyncio.run(_generate())
        return output_path
    
    def cleanup_tts(self):
        """Cleanup TTS resources (no-op for edge-tts)."""
        pass
    
    def create_video_from_scenes(
        self,
        scenes: list,
        images: list,
        audio_files: list,
        output_path: Path,
        progress_callback=None
    ) -> Path:
        """Combine images and audio into final video."""
        from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
        
        clips = []
        
        for i, (scene, image_path, audio_path) in enumerate(zip(scenes, images, audio_files)):
            if progress_callback:
                progress_callback(i, len(scenes), f"Creating clip {i+1}/{len(scenes)}")
            
            print(f"   🎬 Clip {i+1}/{len(scenes)}...")
            
            # Get audio duration
            try:
                audio_clip = AudioFileClip(str(audio_path))
                audio_duration = audio_clip.duration
                print(f"      🔊 Audio duration: {audio_duration:.1f}s")
            except Exception as e:
                print(f"      ⚠️ Audio error: {e}")
                # Create silent audio of 5 seconds
                from moviepy.audio.AudioClip import AudioClip
                import numpy as np
                
                def make_frame(t):
                    return np.array([0.0])  # Silent
                
                audio_clip = AudioClip(make_frame, duration=5.0)
                audio_duration = 5.0
            
            # Create image clip - add 0.5s padding
            clip_duration = audio_duration + 0.5
            img_clip = ImageClip(str(image_path)).with_duration(clip_duration)
            
            # Set fps for the image clip
            img_clip = img_clip.with_fps(24)
            
            # Add audio explicitly (MoviePy 2.x)
            img_clip = img_clip.with_audio(audio_clip)
            
            clips.append(img_clip)
        
        print("   🔗 Concatenating clips...")
        
        # Concatenate all clips
        final = concatenate_videoclips(clips, method="compose")
        
        print("   💾 Writing video file...")
        
        # Write video with explicit audio settings
        final.write_videofile(
            str(output_path),
            fps=24,
            codec="libx264",
            audio_codec="aac",
            temp_audiofile=str(output_path.with_suffix('.temp-audio.m4a')),
            remove_temp=True,
            logger=None
        )
        
        # Cleanup
        final.close()
        for clip in clips:
            clip.close()
        
        return output_path
    
    def process(
        self,
        story_text: str,
        title: str,
        language: str,
        num_scenes: int = None,
        scene_mode: str = "auto",
        aspect_ratio: str = "16:9",
        style_prefix: str = "",
        extract_characters: bool = True,
        speaker_wav: str = None,
        progress_callback=None
    ) -> Path:
        """
        Full pipeline: text → scenes → images → audio → video
        
        Args:
            story_text: The story text to convert
            title: Video title
            language: Language code for TTS
            num_scenes: Number of scenes (None for auto)
            scene_mode: "auto", "short", "medium", "long", "custom"
            aspect_ratio: "16:9", "9:16", "1:1"
            style_prefix: Style tokens for image generation
            extract_characters: Whether to extract characters for consistency
            speaker_wav: Optional reference audio for voice cloning
            progress_callback: Progress callback function
        """
        # Create project directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:30]
        project_dir = self.output_dir / f"{timestamp}_{safe_title}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Calculate scene count
        if num_scenes is None or scene_mode != "custom":
            num_scenes = self.calculate_scenes(story_text, scene_mode, num_scenes or 5)
        
        print(f"\n{'='*60}")
        print(f"📚 Text to Video: {title}")
        print(f"   Scenes: {num_scenes} (mode: {scene_mode})")
        print(f"   Aspect Ratio: {aspect_ratio}")
        print(f"   Language: {config.LANGUAGES.get(language, language)}")
        print(f"   Character Extraction: {'Yes' if extract_characters else 'No'}")
        print(f"{'='*60}\n")
        
        # Step 1: Split story into scenes with character extraction
        if progress_callback:
            progress_callback(0, 5, "Analyzing story and extracting characters...")
        print("📝 Step 1: Splitting story into scenes...")
        scenes = self.split_into_scenes(
            story_text, 
            num_scenes, 
            extract_characters=extract_characters,
            style_prefix=style_prefix,
        )
        print(f"   ✅ Created {len(scenes)} scenes")
        
        # Log character info if extracted
        if self.current_analysis and self.current_analysis.characters:
            print(f"   👥 Found {len(self.current_analysis.characters)} characters:")
            for char in self.current_analysis.characters[:5]:
                print(f"      - {char.name}: {char.description[:50]}...")
        
        # Save scenes and analysis
        (project_dir / "scenes.json").write_text(json.dumps(scenes, indent=2, ensure_ascii=False))
        if self.current_analysis:
            analysis_data = {
                "title": self.current_analysis.title,
                "total_scenes": self.current_analysis.total_scenes,
                "characters": [
                    {"name": c.name, "description": c.description, "traits": c.visual_traits}
                    for c in self.current_analysis.characters
                ],
                "estimated_duration": self.current_analysis.estimated_duration,
            }
            (project_dir / "analysis.json").write_text(json.dumps(analysis_data, indent=2, ensure_ascii=False))
        
        # Step 2: Generate images
        if progress_callback:
            progress_callback(1, 5, "Generating images...")
        print("\n🎨 Step 2: Generating images...")
        
        images = []
        image_paths = []
        pipe = None
        
        for i, scene in enumerate(scenes):
            print(f"   🖼️ Image {i+1}/{len(scenes)}...")
            image, pipe = self.generate_image(scene["image_prompt"], pipe, aspect_ratio)
            
            # Save image
            img_path = project_dir / f"scene_{i:02d}.png"
            image.save(img_path)
            images.append(str(img_path))
            image_paths.append(img_path)
            print(f"      ✅ Saved: {img_path.name}")
        
        # Cleanup SDXL
        del pipe
        cleanup()
        
        # Step 3: Generate audio with upgraded TTS
        if progress_callback:
            progress_callback(2, 5, "Generating narration...")
        print("\n🎙️ Step 3: Generating narration...")
        
        audio_files = []
        for i, scene in enumerate(scenes):
            print(f"   🔊 Audio {i+1}/{len(scenes)}...")
            audio_path = project_dir / f"audio_{i:02d}.wav"
            self.generate_audio(scene["narration"], language, audio_path, speaker_wav)
            audio_files.append(audio_path)
            print(f"      ✅ Saved: {audio_path.name}")
        
        # Cleanup TTS
        self.cleanup_tts()
        
        # Step 4: Create video
        if progress_callback:
            progress_callback(3, 5, "Creating video...")
        print("\n🎬 Step 4: Creating video...")
        
        output_video = project_dir / f"{safe_title}.mp4"
        self.create_video_from_scenes(
            scenes=scenes,
            images=images,
            audio_files=audio_files,
            output_path=output_video,
            progress_callback=progress_callback
        )
        
        print(f"\n✅ Video created: {output_video}")
        
        if progress_callback:
            progress_callback(5, 5, "Complete!")
        
        return output_video


# Create Gradio interface
def create_interface():
    """Build the Gradio UI with enhanced options."""
    
    pipeline = TextToVideo()
    
    def process_story(
        story_text, title, language, scene_mode, custom_scenes, 
        aspect_ratio, style_preset, extract_chars, speaker_wav,
        progress=gr.Progress()
    ):
        if not story_text.strip():
            return None, "❌ Please enter a story"
        if not title.strip():
            return None, "❌ Please enter a title"
        
        def update_progress(current, total, message):
            progress((current / total), desc=message)
        
        # Get style prefix from preset
        style_presets = {
            "anime": "masterpiece, best quality, anime style, vibrant colors",
            "realistic": "photorealistic, highly detailed, cinematic lighting",
            "illustration": "digital illustration, artstation trending, detailed",
            "watercolor": "watercolor painting, soft colors, artistic",
            "comic": "comic book style, bold lines, dynamic composition",
            "none": "",
        }
        style_prefix = style_presets.get(style_preset, "")
        
        try:
            video_path = pipeline.process(
                story_text=story_text.strip(),
                title=title.strip(),
                language=language,
                num_scenes=custom_scenes if scene_mode == "custom" else None,
                scene_mode=scene_mode,
                aspect_ratio=aspect_ratio,
                style_prefix=style_prefix,
                extract_characters=extract_chars,
                speaker_wav=speaker_wav if speaker_wav else None,
                progress_callback=update_progress
            )
            
            # Get actual scene count
            actual_scenes = len(pipeline.current_analysis.scenes) if pipeline.current_analysis else custom_scenes
            char_info = ""
            if pipeline.current_analysis and pipeline.current_analysis.characters:
                chars = [c.name for c in pipeline.current_analysis.characters[:5]]
                char_info = f"\n👥 **Characters:** {', '.join(chars)}"
            
            status = f"""
✅ **Video Generated!**

📁 **Location:** `{video_path.parent}`
🎬 **File:** `{video_path.name}`
📊 **Scenes:** {actual_scenes}
📐 **Aspect Ratio:** {aspect_ratio}
🎨 **Style:** {style_preset}{char_info}
"""
            return str(video_path), status
            
        except Exception as e:
            cleanup()
            if hasattr(pipeline, '_tts') and pipeline._tts:
                pipeline.cleanup_tts()
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {str(e)}"
    
    def estimate_scenes(story_text, scene_mode):
        """Estimate scene count based on text and mode."""
        if not story_text.strip():
            return "Enter story text to estimate scenes"
        count = pipeline.calculate_scenes(story_text.strip(), scene_mode)
        word_count = len(story_text.split())
        return f"📊 ~{word_count} words → **{count} scenes** estimated"
    
    lang_choices = [(v, k) for k, v in config.LANGUAGES.items()]
    
    with gr.Blocks(
        title="AutoKatha - Text to Video",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
# 📖 AutoKatha - Text Story to Video
### Transform your stories into animated videos with AI

**Features:** Auto scene splitting • Character consistency • Multiple aspect ratios • Voice cloning
""")
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📝 Your Story")
                
                story_input = gr.Textbox(
                    label="Story Text",
                    placeholder="""Paste your story here...

Example:
Once upon a time, in a kingdom far away, there lived a wise king named Vikram. 
He was known for his justice and kindness throughout the land.

One day, a mysterious sage arrived at the palace gates with a magical challenge...""",
                    lines=12
                )
                
                title_input = gr.Textbox(
                    label="Title",
                    placeholder="e.g., The Wise King Vikram"
                )
                
                with gr.Row():
                    language = gr.Dropdown(
                        choices=lang_choices,
                        value="en",
                        label="Narration Language"
                    )
                    aspect_ratio = gr.Dropdown(
                        choices=[("Landscape (16:9)", "16:9"), ("Portrait (9:16)", "9:16"), ("Square (1:1)", "1:1")],
                        value="16:9",
                        label="Aspect Ratio"
                    )
                
                with gr.Row():
                    scene_mode = gr.Dropdown(
                        choices=[
                            ("Auto (based on length)", "auto"),
                            ("Short (3-5 scenes)", "short"),
                            ("Medium (6-10 scenes)", "medium"),
                            ("Long (11-20 scenes)", "long"),
                            ("Custom", "custom"),
                        ],
                        value="auto",
                        label="Scene Count"
                    )
                    custom_scenes = gr.Slider(
                        minimum=3,
                        maximum=30,
                        value=5,
                        step=1,
                        label="Custom Scenes",
                        visible=True
                    )
                
                scene_estimate = gr.Markdown("📊 Enter story to estimate scenes")
                
                with gr.Accordion("🎨 Advanced Options", open=False):
                    style_preset = gr.Dropdown(
                        choices=[
                            ("Anime Style", "anime"),
                            ("Realistic", "realistic"),
                            ("Digital Illustration", "illustration"),
                            ("Watercolor", "watercolor"),
                            ("Comic Book", "comic"),
                            ("No Style Prefix", "none"),
                        ],
                        value="anime",
                        label="Art Style Preset"
                    )
                    extract_chars = gr.Checkbox(
                        value=True,
                        label="Extract Characters (for consistency)",
                        info="AI will identify characters and maintain visual consistency"
                    )
                    speaker_wav = gr.Audio(
                        label="Voice Clone Reference (optional)",
                        type="filepath",
                        sources=["upload"],
                    )
                
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Output")
                
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown("*Upload a story to begin...*")
        
        # Update scene estimate when text or mode changes
        story_input.change(
            fn=estimate_scenes,
            inputs=[story_input, scene_mode],
            outputs=[scene_estimate]
        )
        scene_mode.change(
            fn=estimate_scenes,
            inputs=[story_input, scene_mode],
            outputs=[scene_estimate]
        )
        
        # Example stories
        gr.Markdown("### 📚 Example Stories")
        gr.Examples(
            examples=[
                [
                    """Long ago, in the dense forests of ancient India, there lived a young boy named Arjun. 
He was raised by sages who taught him the ways of wisdom and courage.

One fateful day, Arjun discovered a magical bow hidden in a sacred cave. 
The bow glowed with divine light, and a voice spoke: "Only the pure of heart may wield this weapon."

Arjun's journey had just begun. He would face demons, rescue kingdoms, 
and ultimately learn that true strength comes from compassion, not conquest.""",
                    "The Legend of Arjun",
                    "en",
                    "auto",
                    5,
                    "16:9",
                    "anime",
                    True,
                    None,
                ],
                [
                    """In the beautiful valleys of Kashmir, there lived a clever fox named Raja.
Raja was known for solving problems that stumped even the wisest elders.

When a drought threatened the village, Raja embarked on a quest to find the legendary Rain Stone.
His journey took him through enchanted forests, across treacherous mountains, 
and into the lair of the Cloud Dragon.

With wit and courage, Raja retrieved the Rain Stone and saved his village.
The animals celebrated, and Raja learned that helping others brings the greatest joy.""",
                    "Raja the Clever Fox",
                    "en",
                    "auto",
                    5,
                    "16:9",
                    "illustration",
                    True,
                    None,
                ]
            ],
            inputs=[story_input, title_input, language, scene_mode, custom_scenes, aspect_ratio, style_preset, extract_chars, speaker_wav]
        )
        
        generate_btn.click(
            fn=process_story,
            inputs=[story_input, title_input, language, scene_mode, custom_scenes, aspect_ratio, style_preset, extract_chars, speaker_wav],
            outputs=[video_output, status_output]
        )
    
    return demo


if __name__ == "__main__":
    print("🖥️  AutoKatha Text-to-Video using device:", config.DEVICE)
    
    # Check dependencies
    import subprocess
    import sys
    
    if sys.platform == 'win32':
        result = subprocess.run(["where", "ffmpeg"], capture_output=True)
    else:
        result = subprocess.run(["which", "ffmpeg"], capture_output=True)
    
    if result.returncode == 0:
        print("✅ FFmpeg available")
    else:
        print("⚠️ FFmpeg not found - video composition may fail")
    
    # Check Ollama
    try:
        import requests
        r = requests.get("http://localhost:11434/api/tags", timeout=5)
        if r.status_code == 200:
            print("✅ Ollama running")
    except:
        print("⚠️ Ollama not running - scene splitting will use fallback")
    
    # Check TTS backends
    tts_backends = []
    try:
        from f5_tts.api import F5TTS
        tts_backends.append("F5-TTS")
    except:
        pass
    try:
        from TTS.api import TTS
        tts_backends.append("XTTS")
    except:
        pass
    try:
        import edge_tts
        tts_backends.append("Edge-TTS")
    except:
        pass
    
    if tts_backends:
        print(f"✅ TTS backends available: {', '.join(tts_backends)}")
    else:
        print("⚠️ No TTS backend found - install edge-tts: pip install edge-tts")
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,
        share=False
    )
