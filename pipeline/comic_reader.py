"""
Comic Reader - Uses Moondream2 VLM to understand comic panels
Extracts both visual descriptions and text from each panel.
"""
import torch
from PIL import Image
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from pdf2image import convert_from_path
import re

from pipeline.memory_manager import MemoryManager, model_loader
import config


@dataclass
class PanelData:
    """Data extracted from a single comic panel/page."""
    index: int
    image: Image.Image
    image_path: Optional[Path]
    visual_description: str
    extracted_text: str
    narration: str  # Combined description for TTS
    

class ComicReader:
    """
    Reads comic books using Moondream2 Vision-Language Model.
    Can understand visual content and extract text from panels.
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.device = config.DEVICE
        
    @model_loader("moondream2", required_gb=4.0)
    def _load_model(self):
        """Load Moondream2 model."""
        print("📖 Loading Moondream2 Vision-Language Model...")
        
        model = AutoModelForCausalLM.from_pretrained(
            config.MOONDREAM_MODEL,
            revision=config.MOONDREAM_REVISION,
            trust_remote_code=True,
            torch_dtype=config.TORCH_DTYPE,
            device_map={"": self.device}
        )
        
        return model
    
    def load(self):
        """Initialize the comic reader."""
        self.model = self._load_model()
        return self
    
    def unload(self):
        """Unload the model to free memory."""
        MemoryManager.unload_model("moondream2")
        self.model = None
    
    def _load_images_from_path(self, input_path: Union[str, Path]) -> List[Image.Image]:
        """Load images from a file path (PDF or image)."""
        path = Path(input_path)
        images = []
        
        if path.suffix.lower() == '.pdf':
            print(f"📄 Converting PDF to images: {path.name}")
            images = convert_from_path(str(path), dpi=150)
            print(f"   Found {len(images)} pages")
        elif path.suffix.lower() in ['.png', '.jpg', '.jpeg', '.webp', '.bmp']:
            images = [Image.open(path).convert('RGB')]
        elif path.is_dir():
            # Load all images from directory
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.webp']:
                for img_path in sorted(path.glob(ext)):
                    images.append(Image.open(img_path).convert('RGB'))
            print(f"📁 Found {len(images)} images in directory")
        
        return images
    
    def _analyze_panel(self, image: Image.Image) -> Dict[str, str]:
        """Analyze a single panel using Moondream2."""
        if self.model is None:
            self.load()
        
        # Resize if too large (save memory)
        max_size = 1024
        if max(image.size) > max_size:
            ratio = max_size / max(image.size)
            new_size = (int(image.size[0] * ratio), int(image.size[1] * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Ensure RGB mode
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Query 1: Get visual description using caption() for detailed description
        try:
            caption_result = self.model.caption(image, length="normal")
            visual_description = caption_result.get("caption", "")
        except Exception as e:
            print(f"   ⚠️ Caption failed, trying query: {e}")
            visual_description = ""
        
        # Query 2: Get detailed scene description
        visual_prompt = """Describe this comic panel in detail for a narrator. 
Include: characters present, their actions, expressions, setting/background, 
and any visual storytelling elements. Be vivid and descriptive."""
        
        try:
            query_result = self.model.query(image, visual_prompt)
            scene_description = query_result.get("answer", "")
            if scene_description:
                visual_description = scene_description  # Prefer detailed query
        except Exception as e:
            print(f"   ⚠️ Visual query failed: {e}")
        
        # Query 3: Extract text/dialogue
        text_prompt = """Extract ALL text visible in this comic panel.
Include dialogue, narration boxes, sound effects, and any written text.
List each text element on a new line, preserving reading order."""
        
        try:
            text_result = self.model.query(image, text_prompt)
            extracted_text = text_result.get("answer", "")
        except Exception as e:
            print(f"   ⚠️ Text extraction failed: {e}")
            extracted_text = ""
        
        return {
            "visual_description": visual_description.strip() if visual_description else "",
            "extracted_text": extracted_text.strip() if extracted_text else ""
        }
    
    def read_comic(
        self, 
        input_source: Union[str, Path, List[Image.Image]],
        progress_callback=None
    ) -> List[PanelData]:
        """
        Read a comic and extract data from all panels.
        
        Args:
            input_source: Path to PDF, image, directory, or list of PIL Images
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of PanelData objects
        """
        # Load images
        if isinstance(input_source, list):
            images = input_source
        else:
            images = self._load_images_from_path(input_source)
        
        if not images:
            raise ValueError("No images found in input source")
        
        # Ensure model is loaded
        if self.model is None:
            self.load()
        
        # Process each panel
        panels = []
        total = len(images)
        
        for idx, image in enumerate(images):
            if progress_callback:
                progress_callback(idx, total, f"Analyzing panel {idx + 1}/{total}")
            
            print(f"\n📖 Analyzing panel {idx + 1}/{total}...")
            
            # Analyze the panel
            analysis = self._analyze_panel(image)
            
            # Create combined narration
            narration = self._create_narration(
                analysis['visual_description'],
                analysis['extracted_text']
            )
            
            panel = PanelData(
                index=idx,
                image=image,
                image_path=None,
                visual_description=analysis['visual_description'],
                extracted_text=analysis['extracted_text'],
                narration=narration
            )
            panels.append(panel)
            
            print(f"   ✅ Text found: {len(analysis['extracted_text'])} chars")
            print(f"   ✅ Description: {len(analysis['visual_description'])} chars")
        
        if progress_callback:
            progress_callback(total, total, "Comic reading complete!")
        
        return panels
    
    def _create_narration(self, visual_desc: str, text: str) -> str:
        """
        Combine visual description and extracted text into narration.
        This is a simple version - the Screenwriter will enhance it.
        """
        parts = []
        
        # Add scene description
        if visual_desc:
            parts.append(visual_desc)
        
        # Add dialogue/text
        if text and text.lower() not in ['no text', 'none', 'n/a']:
            parts.append(f"\n\n{text}")
        
        return "\n\n".join(parts)


# Convenience function
def read_comic(input_path: str, progress_callback=None) -> List[PanelData]:
    """
    Quick function to read a comic.
    Remember to call cleanup() after processing.
    """
    reader = ComicReader()
    try:
        panels = reader.read_comic(input_path, progress_callback)
        return panels
    finally:
        reader.unload()
