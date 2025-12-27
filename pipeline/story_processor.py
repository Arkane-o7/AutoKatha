"""
AutoKatha Story Processor
Handles intelligent scene splitting, character extraction, and text processing.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import requests


class AspectRatio(Enum):
    """Supported aspect ratios for video output."""
    LANDSCAPE = "16:9"      # 1920x1080, 1024x576
    PORTRAIT = "9:16"       # 1080x1920, 576x1024
    SQUARE = "1:1"          # 1080x1080, 1024x1024
    CINEMATIC = "21:9"      # 2560x1080, 1024x440
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get default dimensions for this aspect ratio."""
        dims = {
            "16:9": (1024, 576),
            "9:16": (576, 1024),
            "1:1": (1024, 1024),
            "21:9": (1024, 440),
        }
        return dims[self.value]
    
    @property
    def video_dimensions(self) -> Tuple[int, int]:
        """Get video export dimensions."""
        dims = {
            "16:9": (1920, 1080),
            "9:16": (1080, 1920),
            "1:1": (1080, 1080),
            "21:9": (2560, 1080),
        }
        return dims[self.value]


@dataclass
class Character:
    """A character extracted from the story."""
    name: str
    description: str = ""
    visual_traits: List[str] = field(default_factory=list)
    personality: str = ""
    aliases: List[str] = field(default_factory=list)
    
    def to_prompt(self) -> str:
        """Convert character to prompt-friendly description."""
        parts = [self.name]
        if self.visual_traits:
            parts.append(", ".join(self.visual_traits))
        if self.description:
            parts.append(self.description)
        return ", ".join(parts)


@dataclass
class Scene:
    """A single scene from the story."""
    index: int
    narration: str
    image_prompt: str
    characters: List[str] = field(default_factory=list)
    setting: str = ""
    mood: str = "neutral"
    duration_hint: float = 5.0


@dataclass
class StoryAnalysis:
    """Complete analysis of a story."""
    title: str
    total_scenes: int
    characters: List[Character]
    scenes: List[Scene]
    themes: List[str] = field(default_factory=list)
    genre: str = "general"
    estimated_duration: float = 0.0  # seconds


class StoryProcessor:
    """
    Intelligent story processor with character extraction and scene splitting.
    
    Features:
    - Dynamic scene count based on text length
    - Character extraction and consistency
    - Narrative structure awareness
    - Enhanced image prompts with style tokens
    """
    
    def __init__(
        self,
        ollama_model: str = None,
        ollama_host: str = None,
        groq_api_key: str = None,
        groq_model: str = None,
    ):
        import config
        
        # Use config values if not specified
        self.ollama_model = ollama_model or config.OLLAMA_MODEL
        self.ollama_host = ollama_host or config.OLLAMA_HOST
        self.groq_api_key = groq_api_key or config.GROQ_API_KEY
        self.groq_model = groq_model or config.GROQ_MODEL
        
        # Determine backend
        self.backend = "groq" if self.groq_api_key else "ollama"
        
        if self.backend == "groq":
            print(f"🧠 Using Groq API with {self.groq_model}")
        else:
            print(f"🧠 Using Ollama with {self.ollama_model}")
            self._check_ollama()
    
    def _check_ollama(self):
        """Verify Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code != 200:
                raise ConnectionError("Ollama not responding")
        except Exception as e:
            print(f"⚠️ Ollama check failed: {e}")
    
    def _call_llm(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Call LLM API (Groq or Ollama)."""
        if self.backend == "groq":
            return self._call_groq(prompt, system, temperature)
        else:
            return self._call_ollama(prompt, system, temperature)
    
    def _call_groq(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Call Groq API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.groq_api_key}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": self.groq_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 8000,
                },
                timeout=120
            )
            result = response.json()
            if "choices" in result:
                return result["choices"][0]["message"]["content"]
            elif "error" in result:
                print(f"❌ Groq error: {result['error']}")
                # Fallback to Ollama
                print("   Falling back to Ollama...")
                self.backend = "ollama"
                return self._call_ollama(prompt, system, temperature)
            return ""
        except Exception as e:
            print(f"❌ Groq error: {e}")
            print("   Falling back to Ollama...")
            self.backend = "ollama"
            return self._call_ollama(prompt, system, temperature)
    
    def _call_ollama(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Call Ollama API."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        
        try:
            response = requests.post(
                f"{self.ollama_host}/api/chat",
                json={
                    "model": self.ollama_model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature}
                },
                timeout=120
            )
            return response.json().get("message", {}).get("content", "")
        except Exception as e:
            print(f"❌ Ollama error: {e}")
            return ""
    
    def calculate_scene_count(
        self,
        text: str,
        min_scenes: int = 3,
        max_scenes: int = 50,
        words_per_scene: int = 150,
    ) -> int:
        """
        Dynamically calculate optimal scene count based on text.
        
        Args:
            text: Story text
            min_scenes: Minimum scenes
            max_scenes: Maximum scenes
            words_per_scene: Target words per scene
        
        Returns:
            Recommended scene count
        """
        # Word count based calculation
        word_count = len(text.split())
        base_scenes = max(min_scenes, word_count // words_per_scene)
        
        # Adjust for paragraph/chapter structure
        paragraphs = len([p for p in text.split('\n\n') if p.strip()])
        chapters = len(re.findall(r'(?:chapter|part|section)\s+\d+', text, re.I))
        
        # If text has clear structure, use that
        if chapters > 2:
            # Chapter-based: 2-4 scenes per chapter
            structured_scenes = chapters * 3
        elif paragraphs > 5:
            # Paragraph-based: roughly 1 scene per 2-3 paragraphs
            structured_scenes = paragraphs // 2
        else:
            structured_scenes = base_scenes
        
        # Take weighted average
        recommended = int(0.6 * base_scenes + 0.4 * structured_scenes)
        
        # Clamp to bounds
        return max(min_scenes, min(max_scenes, recommended))
    
    def extract_characters(self, text: str) -> List[Character]:
        """
        Extract characters from the story text.
        
        Uses Ollama to identify characters and their descriptions.
        """
        system_prompt = """You are a literary analyst. Extract all named characters from the story.
For each character, provide:
- name: The character's name
- description: Brief physical/visual description
- visual_traits: List of specific visual traits (hair color, clothing, etc.)
- personality: One-word personality trait

Output as JSON array. Example:
[
  {
    "name": "Aria",
    "description": "young woman with silver hair",
    "visual_traits": ["silver hair", "blue eyes", "white dress", "slender"],
    "personality": "brave"
  }
]

Only output the JSON array, nothing else."""

        user_prompt = f"""Extract characters from this story:

{text[:4000]}

Characters (JSON):"""

        result = self._call_llm(user_prompt, system_prompt, temperature=0.3)
        
        # Parse JSON
        try:
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                data = json.loads(json_match.group())
                return [Character(**c) for c in data]
        except Exception as e:
            print(f"⚠️ Character extraction failed: {e}")
        
        return []
    
    def split_into_scenes(
        self,
        text: str,
        num_scenes: int,
        characters: List[Character] = None,
        style_prefix: str = "",
    ) -> List[Scene]:
        """
        Split story into scenes with enhanced prompts.
        
        Args:
            text: Story text
            num_scenes: Number of scenes to create
            characters: Pre-extracted characters for consistency
            style_prefix: Style tokens to add to image prompts
        
        Returns:
            List of Scene objects
        """
        # Build character reference for prompt
        char_reference = ""
        if characters:
            char_reference = "\n\nKnown Characters:\n"
            for c in characters:
                char_reference += f"- {c.name}: {c.to_prompt()}\n"
        
        system_prompt = f"""You are a professional storyboard artist and AI art prompt engineer creating scenes for an animated video.

Your task:
1. Split the story into exactly {num_scenes} visually distinct scenes
2. Each scene needs narration text (to be spoken) and a HIGHLY DETAILED image prompt
3. Maintain character consistency using the provided character descriptions
4. Image prompts must be EXTREMELY detailed and specific for AI image generation

{char_reference}

Output JSON array with this structure:
[
  {{
    "narration": "The spoken narration text (1-3 sentences)",
    "image_prompt": "EXTREMELY detailed visual description - see guidelines below",
    "characters": ["character names appearing in this scene"],
    "setting": "location/environment",
    "mood": "emotional tone (happy, tense, mysterious, etc.)"
  }}
]

CRITICAL GUIDELINES for image_prompt (MUST follow all):
1. COMPOSITION: Specify shot type (close-up, medium shot, wide shot, bird's eye view, low angle, etc.)
2. CHARACTERS: Describe exact poses, expressions, clothing, accessories. Include ALL visual traits.
3. ENVIRONMENT: Detailed background description - architecture, nature, weather, time of day
4. LIGHTING: Specify lighting type (dramatic side lighting, soft ambient, golden hour, moonlit, neon glow, etc.)
5. ATMOSPHERE: Include mood elements (mist, dust particles, rays of light, shadows, etc.)
6. COLORS: Mention dominant color palette (warm tones, cool blues, vibrant, muted pastels, etc.)
7. ART STYLE: Include style keywords (anime style, detailed illustration, cinematic, painterly, etc.)
8. DETAILS: Add specific visual details that make the scene unique and interesting
9. LENGTH: Each image_prompt should be 80-150 words minimum

Example of a GOOD image_prompt:
"Dramatic wide shot of a young warrior with flowing silver hair and piercing blue eyes, wearing ornate black armor with gold trim, standing at the edge of a cliff overlooking a vast misty valley. Sunset lighting with golden rays breaking through storm clouds. Wind blowing cape and hair to the left. Ancient ruins visible in the distance. Atmospheric fog in the valley below. Dramatic shadows and rim lighting on the character. Epic fantasy illustration style, highly detailed, cinematic composition, vibrant orange and purple sky contrasting with dark foreground."

{f'Always start prompts with: {style_prefix}' if style_prefix else 'Use vivid, specific descriptive language'}

Only output the JSON array."""

        user_prompt = f"""Create {num_scenes} scenes from this story:

{text[:8000]}

Scenes (JSON):"""

        result = self._call_llm(user_prompt, system_prompt, temperature=0.7)
        
        # Parse JSON
        try:
            json_match = re.search(r'\[[\s\S]*\]', result)
            if json_match:
                data = json.loads(json_match.group())
                scenes = []
                for i, s in enumerate(data[:num_scenes]):
                    # Add style prefix to image prompt
                    img_prompt = s.get("image_prompt", "")
                    if style_prefix and not img_prompt.startswith(style_prefix):
                        img_prompt = f"{style_prefix}, {img_prompt}"
                    
                    scenes.append(Scene(
                        index=i,
                        narration=s.get("narration", ""),
                        image_prompt=img_prompt,
                        characters=s.get("characters", []),
                        setting=s.get("setting", ""),
                        mood=s.get("mood", "neutral"),
                    ))
                return scenes
        except Exception as e:
            print(f"⚠️ Scene parsing failed: {e}")
        
        # Fallback: simple text splitting
        return self._fallback_split(text, num_scenes, style_prefix)
    
    def _fallback_split(self, text: str, num_scenes: int, style_prefix: str = "") -> List[Scene]:
        """Fallback scene splitting without LLM."""
        words = text.split()
        chunk_size = len(words) // num_scenes
        scenes = []
        
        for i in range(num_scenes):
            start = i * chunk_size
            end = start + chunk_size if i < num_scenes - 1 else len(words)
            chunk_text = " ".join(words[start:end])
            
            scenes.append(Scene(
                index=i,
                narration=chunk_text[:500],
                image_prompt=f"{style_prefix}, scene depicting: {chunk_text[:150]}" if style_prefix else chunk_text[:200],
            ))
        
        return scenes
    
    def process_story(
        self,
        text: str,
        title: str = "Untitled",
        num_scenes: int = None,
        style_prefix: str = "",
        extract_characters: bool = True,
    ) -> StoryAnalysis:
        """
        Complete story processing pipeline.
        
        Args:
            text: Story text
            title: Story title
            num_scenes: Number of scenes (None for auto-calculate)
            style_prefix: Style tokens for image prompts
            extract_characters: Whether to extract characters
        
        Returns:
            StoryAnalysis with all processed data
        """
        print(f"📖 Processing story: {title}")
        
        # Calculate scene count if not specified
        if num_scenes is None:
            num_scenes = self.calculate_scene_count(text)
            print(f"   Auto-calculated scenes: {num_scenes}")
        
        # Extract characters
        characters = []
        if extract_characters:
            print("   Extracting characters...")
            characters = self.extract_characters(text)
            print(f"   Found {len(characters)} characters")
        
        # Split into scenes
        print(f"   Splitting into {num_scenes} scenes...")
        scenes = self.split_into_scenes(text, num_scenes, characters, style_prefix)
        
        # Estimate duration (5s per scene average)
        estimated_duration = len(scenes) * 5.0
        
        return StoryAnalysis(
            title=title,
            total_scenes=len(scenes),
            characters=characters,
            scenes=scenes,
            estimated_duration=estimated_duration,
        )
    
    def enhance_image_prompt(
        self,
        base_prompt: str,
        characters: List[Character] = None,
        style_tokens: str = "",
        aspect_ratio: AspectRatio = AspectRatio.LANDSCAPE,
        quality_tokens: str = "masterpiece, best quality, highly detailed",
    ) -> str:
        """
        Enhance an image prompt with style and character consistency.
        
        Args:
            base_prompt: Base scene description
            characters: Characters to include
            style_tokens: Art style tokens
            aspect_ratio: Target aspect ratio
            quality_tokens: Quality enhancement tokens
        
        Returns:
            Enhanced prompt string
        """
        parts = []
        
        # Add quality tokens first
        if quality_tokens:
            parts.append(quality_tokens)
        
        # Add style tokens
        if style_tokens:
            parts.append(style_tokens)
        
        # Add character descriptions if referenced
        if characters:
            for char in characters:
                if char.name.lower() in base_prompt.lower():
                    char_desc = char.to_prompt()
                    # Replace character name with full description
                    base_prompt = re.sub(
                        rf'\b{re.escape(char.name)}\b',
                        char_desc,
                        base_prompt,
                        flags=re.IGNORECASE
                    )
        
        parts.append(base_prompt)
        
        # Add composition hints based on aspect ratio
        if aspect_ratio == AspectRatio.PORTRAIT:
            parts.append("vertical composition, portrait orientation")
        elif aspect_ratio == AspectRatio.CINEMATIC:
            parts.append("cinematic wide shot, letterbox composition")
        
        return ", ".join(parts)


# Utility functions for chapter detection
def detect_chapters(text: str) -> List[Tuple[str, int]]:
    """
    Detect chapter boundaries in text.
    
    Returns:
        List of (chapter_title, start_position) tuples
    """
    patterns = [
        r'^(chapter\s+\d+[:\.\s]*.*?)$',
        r'^(part\s+\d+[:\.\s]*.*?)$',
        r'^(section\s+\d+[:\.\s]*.*?)$',
        r'^(\d+\.\s+.*?)$',
        r'^(#{1,3}\s+.*?)$',  # Markdown headers
    ]
    
    chapters = []
    for pattern in patterns:
        for match in re.finditer(pattern, text, re.MULTILINE | re.IGNORECASE):
            chapters.append((match.group(1).strip(), match.start()))
    
    # Sort by position
    chapters.sort(key=lambda x: x[1])
    return chapters


def chunk_large_text(
    text: str,
    max_chunk_size: int = 10000,
    overlap: int = 500,
) -> List[str]:
    """
    Split large text into overlapping chunks.
    
    Args:
        text: Text to chunk
        max_chunk_size: Maximum characters per chunk
        overlap: Overlap between chunks
    
    Returns:
        List of text chunks
    """
    if len(text) <= max_chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + max_chunk_size
        
        # Try to break at paragraph boundary
        if end < len(text):
            # Look for paragraph break
            para_break = text.rfind('\n\n', start, end)
            if para_break > start + max_chunk_size // 2:
                end = para_break + 2
            else:
                # Look for sentence break
                sent_break = text.rfind('. ', start, end)
                if sent_break > start + max_chunk_size // 2:
                    end = sent_break + 2
        
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks
