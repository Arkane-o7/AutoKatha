"""
AutoKatha - Text Story to Video
Simplified interface: paste story text → get animated video
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
import config


class TextToVideo:
    """Simple text story to video pipeline."""
    
    def __init__(self):
        self.output_dir = config.OUTPUT_DIR
        self.device = config.DEVICE
        self.comfyui_client = None
        self.use_comfyui = False
        self.diffusers_pipe = None
        
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
        
    def split_into_scenes(self, story_text: str, num_scenes: int = 5) -> list:
        """
        Use Gemma3 via Ollama to split story into scenes with image prompts.
        """
        import requests
        
        prompt = f"""You are a storyboard artist. Split this story into exactly {num_scenes} scenes.
For each scene, provide:
1. A brief narration text (1-2 sentences to be read aloud)
2. An image description for AI art generation (detailed visual description)

Story:
{story_text}

Respond in JSON format:
[
  {{"narration": "...", "image_prompt": "..."}},
  ...
]

Only output the JSON array, nothing else."""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "gemma3:4b",
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.7}
                },
                timeout=120
            )
            
            result = response.json().get("response", "")
            
            # Extract JSON from response
            import re
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                scenes = json.loads(json_match.group())
                return scenes[:num_scenes]
            else:
                raise ValueError("No JSON found in response")
                
        except Exception as e:
            print(f"❌ Scene splitting error: {e}")
            # Fallback: split text evenly
            words = story_text.split()
            chunk_size = len(words) // num_scenes
            scenes = []
            for i in range(num_scenes):
                start = i * chunk_size
                end = start + chunk_size if i < num_scenes - 1 else len(words)
                text = " ".join(words[start:end])
                scenes.append({
                    "narration": text,
                    "image_prompt": f"Scene {i+1}: {text[:100]}"
                })
            return scenes
    
    def generate_image(self, prompt: str, pipe=None) -> Image.Image:
        """
        Generate image using ComfyUI (preferred) or SDXL Lightning (fallback).
        """
        # Try ComfyUI first
        if self.use_comfyui and self.comfyui_client:
            return self._generate_with_comfyui(prompt), None
        
        # Fallback to diffusers
        return self._generate_with_diffusers(prompt, pipe)
    
    def _generate_with_comfyui(self, prompt: str) -> Image.Image:
        """Generate image using ComfyUI API."""
        print(f"   🎨 Generating with ComfyUI...")
        
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
    
    def _generate_with_diffusers(self, prompt: str, pipe=None) -> tuple:
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
                width=1024,
                height=576,  # 16:9 aspect ratio
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
                    width=1024,
                    height=576,
                )
                image = result.images[0]
        
        return image, pipe
    
    def generate_audio(self, text: str, language: str, output_path: Path) -> Path:
        """Generate TTS audio using edge-tts (simpler than XTTS)."""
        import edge_tts
        import asyncio
        
        # Map language codes to edge-tts voices
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
        }
        
        voice = voice_map.get(language, "en-US-AriaNeural")
        
        async def generate():
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(output_path))
        
        asyncio.run(generate())
        return output_path
    
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
        num_scenes: int,
        progress_callback=None
    ) -> Path:
        """
        Full pipeline: text → scenes → images → audio → video
        """
        # Create project directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:30]
        project_dir = self.output_dir / f"{timestamp}_{safe_title}"
        project_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"📚 Text to Video: {title}")
        print(f"   Scenes: {num_scenes}")
        print(f"   Language: {config.LANGUAGES.get(language, language)}")
        print(f"{'='*60}\n")
        
        # Step 1: Split story into scenes
        if progress_callback:
            progress_callback(0, 5, "Splitting story into scenes...")
        print("📝 Step 1: Splitting story into scenes...")
        scenes = self.split_into_scenes(story_text, num_scenes)
        print(f"   ✅ Created {len(scenes)} scenes")
        
        # Save scenes
        (project_dir / "scenes.json").write_text(json.dumps(scenes, indent=2, ensure_ascii=False))
        
        # Step 2: Generate images
        if progress_callback:
            progress_callback(1, 5, "Generating images...")
        print("\n🎨 Step 2: Generating images...")
        
        images = []
        image_paths = []
        pipe = None
        
        for i, scene in enumerate(scenes):
            print(f"   🖼️ Image {i+1}/{len(scenes)}...")
            image, pipe = self.generate_image(scene["image_prompt"], pipe)
            
            # Save image
            img_path = project_dir / f"scene_{i:02d}.png"
            image.save(img_path)
            images.append(str(img_path))
            image_paths.append(img_path)
            print(f"      ✅ Saved: {img_path.name}")
        
        # Cleanup SDXL
        del pipe
        cleanup()
        
        # Step 3: Generate audio
        if progress_callback:
            progress_callback(2, 5, "Generating narration...")
        print("\n🎙️ Step 3: Generating narration...")
        
        audio_files = []
        for i, scene in enumerate(scenes):
            print(f"   🔊 Audio {i+1}/{len(scenes)}...")
            audio_path = project_dir / f"audio_{i:02d}.mp3"
            self.generate_audio(scene["narration"], language, audio_path)
            audio_files.append(audio_path)
            print(f"      ✅ Saved: {audio_path.name}")
        
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
    """Build the Gradio UI."""
    
    pipeline = TextToVideo()
    
    def process_story(story_text, title, language, num_scenes, progress=gr.Progress()):
        if not story_text.strip():
            return None, "❌ Please enter a story"
        if not title.strip():
            return None, "❌ Please enter a title"
        
        def update_progress(current, total, message):
            progress((current / total), desc=message)
        
        try:
            video_path = pipeline.process(
                story_text=story_text.strip(),
                title=title.strip(),
                language=language,
                num_scenes=num_scenes,
                progress_callback=update_progress
            )
            
            status = f"""
✅ **Video Generated!**

📁 **Location:** `{video_path.parent}`
🎬 **File:** `{video_path.name}`
📊 **Scenes:** {num_scenes}
"""
            return str(video_path), status
            
        except Exception as e:
            cleanup()
            import traceback
            traceback.print_exc()
            return None, f"❌ Error: {str(e)}"
    
    lang_choices = [(v, k) for k, v in config.LANGUAGES.items()]
    
    with gr.Blocks(
        title="AutoKatha - Text to Video",
        theme=gr.themes.Soft()
    ) as demo:
        
        gr.Markdown("""
# 📖 AutoKatha - Text Story to Video
### Transform your stories into animated videos with AI

Paste your story → AI generates scenes, artwork, and narration → Get video!
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
                    lines=15
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
                
                num_scenes = gr.Slider(
                    minimum=3,
                    maximum=15,
                    value=5,
                    step=1,
                    label="Number of Scenes"
                )
                
                generate_btn = gr.Button("🎬 Generate Video", variant="primary", size="lg")
            
            with gr.Column(scale=1):
                gr.Markdown("### 🎬 Output")
                
                video_output = gr.Video(label="Generated Video")
                status_output = gr.Markdown("*Upload a story to begin...*")
        
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
                    5
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
                    5
                ]
            ],
            inputs=[story_input, title_input, language, num_scenes]
        )
        
        generate_btn.click(
            fn=process_story,
            inputs=[story_input, title_input, language, num_scenes],
            outputs=[video_output, status_output]
        )
    
    return demo


if __name__ == "__main__":
    print("🖥️  AutoKatha Text-to-Video using device:", config.DEVICE)
    
    # Check dependencies
    import subprocess
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
    
    demo = create_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7861,  # Different port
        share=False
    )
