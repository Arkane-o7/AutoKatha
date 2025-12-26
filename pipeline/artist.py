"""
Artist - Uses SDXL Lightning to generate images from prompts.
Simplified version without art style transformation.
"""
import torch
from PIL import Image
from pathlib import Path
from typing import List, Optional, Union
from diffusers import StableDiffusionXLPipeline, DPMSolverMultistepScheduler
import gc

from pipeline.memory_manager import MemoryManager, model_loader
from pipeline.comic_reader import PanelData
import config


class Artist:
    """
    Generates images using SDXL Lightning.
    """
    
    def __init__(self):
        self.pipe = None
        self.device = config.DEVICE
        
    @model_loader("sdxl_lightning", required_gb=12.0)
    def _load_model(self):
        """Load SDXL Lightning model."""
        print("🎨 Loading SDXL Lightning model...")
        
        try:
            # Use float32 for MPS to avoid NaN issues
            dtype = torch.float32 if self.device == "mps" else torch.float16
            
            # Load base SDXL
            pipe = StableDiffusionXLPipeline.from_pretrained(
                "stabilityai/stable-diffusion-xl-base-1.0",
                torch_dtype=dtype,
                variant="fp16" if dtype == torch.float16 else None,
                use_safetensors=True,
            )
            
            # Load Lightning LoRA for fast inference
            print("   Loading Lightning LoRA...")
            pipe.load_lora_weights(
                "ByteDance/SDXL-Lightning",
                weight_name=f"sdxl_lightning_{config.SDXL_LIGHTNING_STEPS}step_lora.safetensors"
            )
            pipe.fuse_lora()
            
            # Use appropriate scheduler for Lightning
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True,
            )
            
            # Move to device
            print(f"   Moving to {self.device}...")
            pipe = pipe.to(self.device)
            
            # Enable memory optimizations
            if self.device == "mps":
                pipe.enable_attention_slicing()
                # Ensure VAE is in float32 for MPS
                pipe.vae = pipe.vae.to(torch.float32)
            
            print("✅ SDXL Lightning loaded!")
            return pipe
            
        except Exception as e:
            print(f"❌ Error loading SDXL Lightning: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def load(self):
        """Initialize the artist."""
        self.pipe = self._load_model()
        return self
    
    def unload(self):
        """Unload the model to free memory."""
        MemoryManager.unload_model("sdxl_lightning")
        self.pipe = None
    
    def generate_image(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 576,
        guidance_scale: float = 2.0,
    ) -> Image.Image:
        """
        Generate an image from a prompt.
        
        Args:
            prompt: Text prompt for image generation
            width: Image width (default 1024)
            height: Image height (default 576 for 16:9)
            guidance_scale: How closely to follow the prompt (1.5-3.0 for Lightning)
        
        Returns:
            Generated PIL Image
        """
        if self.pipe is None:
            self.load()
        
        # Truncate prompt to avoid CLIP token limit
        full_prompt = prompt[:300] if len(prompt) > 300 else prompt
        negative_prompt = config.DEFAULT_NEGATIVE_PROMPT
        
        print(f"   🎨 Generating image...")
        
        # Generate
        with torch.inference_mode():
            result = self.pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=config.SDXL_LIGHTNING_STEPS,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
            ).images[0]
        
        return result
    
    def generate_images_for_panels(
        self,
        panels: List[PanelData],
        output_dir: Optional[Path] = None,
        progress_callback=None
    ) -> List[Image.Image]:
        """
        Generate images for multiple panels based on their descriptions.
        
        Args:
            panels: List of PanelData from comic reader
            output_dir: Optional directory to save images
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of generated PIL Images
        """
        if output_dir:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Ensure model is loaded
        if self.pipe is None:
            self.load()
        
        print(f"\n🎨 Generating {len(panels)} images...")
        
        generated = []
        total = len(panels)
        
        for idx, panel in enumerate(panels):
            if progress_callback:
                progress_callback(idx, total, f"Generating image {idx + 1}/{total}")
            
            print(f"\n🎨 Image {idx + 1}/{total}...")
            
            # Generate image from description
            image = self.generate_image(
                prompt=panel.visual_description[:300],
            )
            
            generated.append(image)
            
            # Save if output directory provided
            if output_dir:
                output_path = output_dir / f"panel_{idx:03d}.png"
                image.save(output_path)
                print(f"   💾 Saved: {output_path.name}")
            
            # Clear cache periodically
            if idx % 5 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                elif torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        if progress_callback:
            progress_callback(total, total, "Image generation complete!")
        
        return generated


# Convenience function
def generate_images(
    panels: List[PanelData],
    output_dir: Optional[str] = None,
    progress_callback=None
) -> List[Image.Image]:
    """Quick function to generate images for panels."""
    artist = Artist()
    try:
        return artist.generate_images_for_panels(
            panels, 
            Path(output_dir) if output_dir else None,
            progress_callback=progress_callback
        )
    finally:
        artist.unload()
