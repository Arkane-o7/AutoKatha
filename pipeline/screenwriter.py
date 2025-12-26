"""
Screenwriter - Uses Ollama (Gemma 3) to enhance narration
Transforms raw panel descriptions into engaging storytelling.
"""
import ollama
from typing import List, Optional
from dataclasses import dataclass
import re

from pipeline.comic_reader import PanelData


@dataclass 
class ScriptSegment:
    """A segment of the final script."""
    panel_index: int
    narration: str  # The text to be spoken
    duration_hint: float  # Estimated duration in seconds
    emotion: str  # Emotional tone for TTS
    

class Screenwriter:
    """
    Enhances comic panel descriptions into engaging narration.
    Uses Gemma 3 via Ollama for local LLM processing.
    """
    
    def __init__(self, model: str = "gemma3:4b"):
        """
        Initialize the screenwriter.
        
        Args:
            model: Ollama model to use. Options:
                   - "gemma3:4b" (recommended, balanced)
                   - "gemma3:12b" (better quality, slower)
                   - "llama3.2" (alternative)
                   - "mistral" (alternative)
        """
        self.model = model
        self._check_ollama()
    
    def _check_ollama(self):
        """Check if Ollama is running and model is available."""
        try:
            models = ollama.list()
            model_names = [m.model for m in models.models] if models.models else []
            
            # Check for base model name (without size suffix variations)
            base_model = self.model.split(':')[0]
            has_model = any(base_model in m for m in model_names)
            
            if not has_model:
                print(f"⚠️  Model '{self.model}' not found. Pulling...")
                ollama.pull(self.model)
                print(f"✅ Model '{self.model}' ready")
            else:
                print(f"✅ Ollama model '{self.model}' available")
                
        except Exception as e:
            raise RuntimeError(
                f"Ollama not available. Please install and start Ollama:\n"
                f"  1. Install: https://ollama.ai\n"
                f"  2. Start: 'ollama serve'\n"
                f"  3. Pull model: 'ollama pull {self.model}'\n"
                f"Error: {e}"
            )
    
    def _generate(self, prompt: str, system: str = None) -> str:
        """Generate text using Ollama."""
        messages = []
        
        if system:
            messages.append({"role": "system", "content": system})
        
        messages.append({"role": "user", "content": prompt})
        
        response = ollama.chat(
            model=self.model,
            messages=messages,
            options={
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
        
        return response['message']['content']
    
    def enhance_panel(
        self, 
        panel: PanelData, 
        story_context: str = "",
        language: str = "en",
        style: str = "narrative"
    ) -> ScriptSegment:
        """
        Enhance a single panel's description into narration.
        
        Args:
            panel: The panel data from comic reader
            story_context: Overall story context for coherence
            language: Target language code
            style: Narration style (narrative, dramatic, children, etc.)
        """
        
        system_prompt = f"""You are a master storyteller adapting comics into audio narration.
Your task is to transform visual descriptions and dialogue into engaging spoken narration.

Guidelines:
- Write in {self._get_language_name(language)}
- Style: {style}
- Make it vivid and immersive for audio-only consumption
- Include emotional cues in [brackets] for the voice actor
- Keep dialogue natural and conversational
- Describe actions smoothly without "In this panel..."
- Duration should be 15-30 seconds when read aloud

Output ONLY the narration text, nothing else."""

        user_prompt = f"""Story Context: {story_context if story_context else "Opening scene"}

Panel {panel.index + 1}:
Visual Description: {panel.visual_description}

Extracted Text/Dialogue: {panel.extracted_text}

Transform this into engaging narration:"""

        narration = self._generate(user_prompt, system_prompt)
        
        # Clean up the response
        narration = self._clean_narration(narration)
        
        # Estimate duration (rough: 150 words per minute)
        word_count = len(narration.split())
        duration_hint = (word_count / 150) * 60
        
        # Extract emotion if present in brackets
        emotion = self._extract_emotion(narration)
        
        return ScriptSegment(
            panel_index=panel.index,
            narration=narration,
            duration_hint=max(5, min(duration_hint, 60)),  # 5-60 seconds
            emotion=emotion
        )
    
    def write_script(
        self,
        panels: List[PanelData],
        title: str = "Untitled Story",
        language: str = "en",
        style: str = "narrative",
        progress_callback=None
    ) -> List[ScriptSegment]:
        """
        Create a complete script from all panels.
        
        Args:
            panels: List of panel data from comic reader
            title: Story title for context
            language: Target language code
            style: Narration style
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of ScriptSegment objects
        """
        print(f"\n✍️  Writing script for: {title}")
        print(f"   Language: {self._get_language_name(language)}")
        print(f"   Style: {style}")
        print(f"   Panels: {len(panels)}")
        
        # First, create story context
        story_context = self._create_story_context(panels, title)
        
        # Process each panel
        script = []
        total = len(panels)
        
        for idx, panel in enumerate(panels):
            if progress_callback:
                progress_callback(idx, total, f"Writing panel {idx + 1}/{total}")
            
            print(f"\n✍️  Processing panel {idx + 1}/{total}...")
            
            segment = self.enhance_panel(
                panel,
                story_context=story_context,
                language=language,
                style=style
            )
            script.append(segment)
            
            # Update context with what we've written
            story_context += f"\n[Panel {idx + 1}]: {segment.narration[:100]}..."
            
            print(f"   ✅ Generated {len(segment.narration)} chars")
            print(f"   ⏱️  Est. duration: {segment.duration_hint:.1f}s")
        
        if progress_callback:
            progress_callback(total, total, "Script complete!")
        
        return script
    
    def _create_story_context(self, panels: List[PanelData], title: str) -> str:
        """Create initial story context from all panels."""
        
        # Gather all extracted text
        all_text = "\n".join([
            f"Panel {p.index + 1}: {p.extracted_text[:200]}"
            for p in panels[:5]  # First 5 panels for context
        ])
        
        prompt = f"""Based on this comic titled "{title}", provide a brief story context (2-3 sentences):

{all_text}

Context:"""
        
        context = self._generate(prompt)
        return context.strip()
    
    def _clean_narration(self, text: str) -> str:
        """Clean up generated narration."""
        # Remove markdown artifacts
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # Bold
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # Italic
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Headers
        
        # Remove "Panel X:" prefixes
        text = re.sub(r'^Panel \d+:?\s*', '', text, flags=re.MULTILINE)
        
        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = text.strip()
        
        return text
    
    def _extract_emotion(self, text: str) -> str:
        """Extract emotional cue from text if present."""
        match = re.search(r'\[([^\]]+)\]', text)
        if match:
            return match.group(1).lower()
        return "neutral"
    
    def _get_language_name(self, code: str) -> str:
        """Convert language code to name."""
        from config import LANGUAGES
        return LANGUAGES.get(code, code)


# Convenience function
def write_script(
    panels: List[PanelData],
    title: str = "Untitled Story",
    language: str = "en",
    model: str = "gemma3:4b",
    progress_callback=None
) -> List[ScriptSegment]:
    """Quick function to generate a script."""
    writer = Screenwriter(model=model)
    return writer.write_script(
        panels, 
        title=title, 
        language=language,
        progress_callback=progress_callback
    )
