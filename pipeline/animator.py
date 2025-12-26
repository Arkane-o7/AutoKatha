"""
Animator - Uses Stable Video Diffusion to animate still images.
Creates subtle motion from styled comic panels.
"""
import torch
from PIL import Image
from pathlib import Path
from typing import List, Optional
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import export_to_video
import numpy as np

from pipeline.memory_manager import MemoryManager, model_loader
import config


class Animator:
    """
    Animates static images using Stable Video Diffusion.
    Creates short video clips from styled comic panels.
    
    Note: This is the most memory-intensive model. 
    All other models should be unloaded before using this.
    """
    
    def __init__(self):
        self.pipe = None
        self.device = config.DEVICE
        
    @model_loader("stable_video_diffusion", required_gb=20.0)
    def _load_model(self):
        """Load Stable Video Diffusion model."""
        print("🎬 Loading Stable Video Diffusion...")
        print("   ⚠️  This is memory-intensive, please wait...")
        
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            config.SVD_MODEL,
            torch_dtype=config.TORCH_DTYPE,
            variant="fp16",
        )
        
        # Move to device
        pipe = pipe.to(self.device)
        
        # Enable memory optimizations
        pipe.enable_attention_slicing()
        
        # For MPS, we might need to be more aggressive
        if self.device == "mps":
            # Use sequential CPU offload if needed
            # pipe.enable_sequential_cpu_offload()
            pass
        
        print("✅ Stable Video Diffusion loaded!")
        return pipe
    
    def load(self):
        """Initialize the animator."""
        self.pipe = self._load_model()
        return self
    
    def unload(self):
        """Unload the model to free memory."""
        MemoryManager.unload_model("stable_video_diffusion")
        self.pipe = None
    
    def animate_image(
        self,
        image: Image.Image,
        num_frames: int = None,
        fps: int = None,
        motion_bucket_id: int = 127,
        noise_aug_strength: float = 0.02,
        decode_chunk_size: int = 4,
    ) -> List[Image.Image]:
        """
        Animate a single image.
        
        Args:
            image: Source image to animate
            num_frames: Number of frames (default from config)
            fps: Frames per second (default from config)
            motion_bucket_id: Controls motion amount (1-255, higher = more motion)
            noise_aug_strength: Amount of noise augmentation
            decode_chunk_size: Chunk size for decoding (lower = less memory)
        
        Returns:
            List of PIL Image frames
        """
        if self.pipe is None:
            self.load()
        
        num_frames = num_frames or config.SVD_FRAMES
        fps = fps or config.SVD_FPS
        
        # SVD expects 1024x576 images
        target_size = (1024, 576)
        image_resized = self._prepare_image(image, target_size)
        
        print(f"   🎬 Generating {num_frames} frames...")
        
        with torch.inference_mode():
            frames = self.pipe(
                image_resized,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_bucket_id,
                noise_aug_strength=noise_aug_strength,
                decode_chunk_size=decode_chunk_size,
            ).frames[0]
        
        return frames
    
    def animate_panels(
        self,
        images: List[Image.Image],
        output_dir: Path,
        num_frames: int = None,
        fps: int = None,
        motion_bucket_id: int = 100,
        progress_callback=None
    ) -> List[Path]:
        """
        Animate multiple images and save as video clips.
        
        Args:
            images: List of images to animate
            output_dir: Directory to save video clips
            num_frames: Frames per animation
            fps: Frames per second
            motion_bucket_id: Motion amount (lower = subtler motion for comics)
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of paths to generated video clips
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.pipe is None:
            self.load()
        
        num_frames = num_frames or config.SVD_FRAMES
        fps = fps or config.SVD_FPS
        
        print(f"\n🎬 Animating {len(images)} panels...")
        print(f"   Frames per panel: {num_frames}")
        print(f"   FPS: {fps}")
        
        video_paths = []
        total = len(images)
        
        for idx, image in enumerate(images):
            if progress_callback:
                progress_callback(idx, total, f"Animating panel {idx + 1}/{total}")
            
            print(f"\n🎬 Panel {idx + 1}/{total}...")
            
            # Generate frames
            frames = self.animate_image(
                image,
                num_frames=num_frames,
                fps=fps,
                motion_bucket_id=motion_bucket_id
            )
            
            # Save as video
            output_path = output_dir / f"clip_{idx:03d}.mp4"
            export_to_video(frames, str(output_path), fps=fps)
            video_paths.append(output_path)
            
            print(f"   💾 Saved: {output_path.name}")
            
            # Clear cache
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        if progress_callback:
            progress_callback(total, total, "Animation complete!")
        
        return video_paths
    
    def _prepare_image(self, image: Image.Image, target_size: tuple) -> Image.Image:
        """Prepare image for SVD (resize and pad to 1024x576)."""
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Calculate scaling
        ratio = min(target_size[0] / image.width, target_size[1] / image.height)
        new_size = (int(image.width * ratio), int(image.height * ratio))
        
        # Resize
        resized = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Create canvas and center
        canvas = Image.new('RGB', target_size, (0, 0, 0))
        offset = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
        canvas.paste(resized, offset)
        
        return canvas


# Simple Ken Burns fallback (for when SVD is too slow/memory intensive)
class KenBurnsAnimator:
    """
    Fallback animator using simple Ken Burns effects.
    Much faster and uses no GPU memory.
    """
    
    def __init__(self):
        from moviepy import ImageClip
        self.ImageClip = ImageClip
    
    def animate_image(
        self,
        image: Image.Image,
        duration: float = 5.0,
        effect: str = "zoom_in",  # zoom_in, zoom_out, pan_left, pan_right
    ):
        """
        Apply Ken Burns effect to an image.
        
        Returns a moviepy VideoClip.
        """
        import numpy as np
        from moviepy import ImageClip
        
        # Convert PIL to numpy
        img_array = np.array(image)
        
        clip = ImageClip(img_array, duration=duration)
        
        # Apply effect
        if effect == "zoom_in":
            clip = clip.resized(lambda t: 1 + 0.1 * (t / duration))
        elif effect == "zoom_out":
            clip = clip.resized(lambda t: 1.1 - 0.1 * (t / duration))
        elif effect == "pan_left":
            # Implemented via position
            clip = clip.with_position(lambda t: (-50 * t / duration, 0))
        elif effect == "pan_right":
            clip = clip.with_position(lambda t: (50 * t / duration, 0))
        
        return clip
    
    def animate_panels(
        self,
        images: List[Image.Image],
        output_dir: Path,
        duration_per_panel: float = 5.0,
        effects: List[str] = None,
        progress_callback=None
    ) -> List[Path]:
        """
        Apply Ken Burns effects to all panels and save as clips.
        """
        from moviepy import ImageClip, CompositeVideoClip
        import numpy as np
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Alternate effects if not specified
        if effects is None:
            effects = ["zoom_in", "zoom_out", "pan_left", "pan_right"]
        
        video_paths = []
        total = len(images)
        
        for idx, image in enumerate(images):
            if progress_callback:
                progress_callback(idx, total, f"Animating panel {idx + 1}/{total}")
            
            effect = effects[idx % len(effects)]
            print(f"🎬 Panel {idx + 1}/{total} - {effect}")
            
            # Convert to numpy array
            img_array = np.array(image.convert('RGB'))
            
            # Create clip
            clip = ImageClip(img_array, duration=duration_per_panel)
            
            # Apply zoom effect
            if effect == "zoom_in":
                def zoom_in(get_frame, t):
                    scale = 1 + 0.15 * (t / duration_per_panel)
                    frame = get_frame(t)
                    from scipy.ndimage import zoom as scipy_zoom
                    return scipy_zoom(frame, [scale, scale, 1], order=1)[:frame.shape[0], :frame.shape[1]]
                
                # Simpler approach - just resize
                clip = clip.resized(lambda t: 1 + 0.15 * (t / duration_per_panel))
            
            elif effect == "zoom_out":
                clip = clip.resized(lambda t: 1.15 - 0.15 * (t / duration_per_panel))
            
            # Save clip
            output_path = output_dir / f"clip_{idx:03d}.mp4"
            clip.write_videofile(
                str(output_path), 
                fps=24, 
                codec='libx264',
                audio=False,
                logger=None
            )
            video_paths.append(output_path)
            clip.close()
        
        if progress_callback:
            progress_callback(total, total, "Animation complete!")
        
        return video_paths


# Convenience functions
def animate_with_svd(
    images: List[Image.Image],
    output_dir: str,
    progress_callback=None
) -> List[Path]:
    """Animate using Stable Video Diffusion."""
    animator = Animator()
    try:
        return animator.animate_panels(
            images, 
            Path(output_dir),
            progress_callback=progress_callback
        )
    finally:
        animator.unload()


def animate_with_kenburns(
    images: List[Image.Image],
    output_dir: str,
    duration: float = 5.0,
    progress_callback=None
) -> List[Path]:
    """Animate using Ken Burns effects (faster, no GPU)."""
    animator = KenBurnsAnimator()
    return animator.animate_panels(
        images,
        Path(output_dir),
        duration_per_panel=duration,
        progress_callback=progress_callback
    )
