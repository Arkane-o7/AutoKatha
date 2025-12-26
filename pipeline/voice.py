"""
Voice - Uses Coqui XTTS v2 for multilingual text-to-speech.
Supports voice cloning and multiple Indian languages.
"""
import torch
from TTS.api import TTS
from pathlib import Path
from typing import List, Optional, Union
from pydub import AudioSegment
import numpy as np
import tempfile
import os

from pipeline.memory_manager import MemoryManager, model_loader
from pipeline.screenwriter import ScriptSegment
import config


class Voice:
    """
    Text-to-Speech using Coqui XTTS v2.
    Supports multilingual synthesis and voice cloning.
    """
    
    def __init__(self):
        self.tts = None
        self.device = config.DEVICE
        self.speaker_wav = None
        
    @model_loader("xtts_v2", required_gb=6.0)
    def _load_model(self):
        """Load XTTS v2 model."""
        print("🎙️ Loading XTTS v2 Text-to-Speech model...")
        
        tts = TTS(config.TTS_MODEL)
        tts = tts.to(self.device)
        
        print("✅ XTTS v2 loaded!")
        return tts
    
    def load(self):
        """Initialize the voice module."""
        self.tts = self._load_model()
        return self
    
    def unload(self):
        """Unload the model to free memory."""
        MemoryManager.unload_model("xtts_v2")
        self.tts = None
    
    def set_speaker_voice(self, audio_path: Union[str, Path]):
        """
        Set a reference voice for cloning.
        
        Args:
            audio_path: Path to a WAV file (6-30 seconds recommended)
        """
        self.speaker_wav = str(audio_path)
        print(f"🎙️ Voice reference set: {Path(audio_path).name}")
    
    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        language: str = "en",
        speaker_wav: Optional[str] = None,
    ) -> Path:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            output_path: Path to save the audio
            language: Language code (en, hi, ta, etc.)
            speaker_wav: Optional path to voice reference for cloning
        
        Returns:
            Path to generated audio file
        """
        if self.tts is None:
            self.load()
        
        output_path = Path(output_path)
        speaker = speaker_wav or self.speaker_wav
        
        # XTTS v2 supported languages
        xtts_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 
                         'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi']
        
        # Map some language codes
        lang_map = {
            'zh': 'zh-cn',
            'ta': 'en',  # Fallback - XTTS doesn't support Tamil natively
            'te': 'en',  # Fallback
            'bn': 'en',  # Fallback
            'mr': 'hi',  # Use Hindi for Marathi (similar)
            'gu': 'hi',  # Use Hindi for Gujarati
            'kn': 'en',  # Fallback
            'ml': 'en',  # Fallback
            'pa': 'hi',  # Use Hindi for Punjabi
        }
        
        actual_lang = lang_map.get(language, language)
        if actual_lang not in xtts_languages:
            print(f"⚠️  Language '{language}' not fully supported, using 'en'")
            actual_lang = 'en'
        
        # Generate speech
        if speaker:
            # With voice cloning
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker,
                language=actual_lang
            )
        else:
            # Without voice cloning - use default speaker
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                language=actual_lang
            )
        
        return output_path
    
    def synthesize_script(
        self,
        script: List[ScriptSegment],
        output_dir: Union[str, Path],
        language: str = "en",
        speaker_wav: Optional[str] = None,
        progress_callback=None
    ) -> List[Path]:
        """
        Synthesize speech for an entire script.
        
        Args:
            script: List of ScriptSegment from screenwriter
            output_dir: Directory to save audio files
            language: Language code
            speaker_wav: Optional voice reference
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of paths to generated audio files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if self.tts is None:
            self.load()
        
        speaker = speaker_wav or self.speaker_wav
        
        print(f"\n🎙️ Synthesizing {len(script)} audio segments...")
        print(f"   Language: {config.LANGUAGES.get(language, language)}")
        print(f"   Voice cloning: {'Yes' if speaker else 'No'}")
        
        audio_paths = []
        total = len(script)
        
        for idx, segment in enumerate(script):
            if progress_callback:
                progress_callback(idx, total, f"Synthesizing panel {idx + 1}/{total}")
            
            print(f"\n🎙️ Panel {idx + 1}/{total}...")
            
            # Clean up narration for TTS
            narration = self._clean_for_tts(segment.narration)
            
            if not narration.strip():
                print("   ⚠️  Empty narration, creating silence...")
                audio_path = self._create_silence(
                    output_dir / f"audio_{idx:03d}.wav",
                    duration=2.0
                )
            else:
                audio_path = output_dir / f"audio_{idx:03d}.wav"
                self.synthesize(
                    text=narration,
                    output_path=audio_path,
                    language=language,
                    speaker_wav=speaker
                )
            
            audio_paths.append(audio_path)
            
            # Get duration
            audio = AudioSegment.from_wav(str(audio_path))
            duration = len(audio) / 1000.0
            print(f"   ✅ Duration: {duration:.1f}s")
        
        if progress_callback:
            progress_callback(total, total, "Voice synthesis complete!")
        
        return audio_paths
    
    def _clean_for_tts(self, text: str) -> str:
        """Clean text for TTS synthesis."""
        import re
        
        # Remove emotional cues in brackets for now
        # (Could be used for SSML in the future)
        text = re.sub(r'\[([^\]]+)\]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove markdown
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        
        return text.strip()
    
    def _create_silence(self, output_path: Path, duration: float) -> Path:
        """Create a silent audio file."""
        silence = AudioSegment.silent(duration=int(duration * 1000))
        silence.export(str(output_path), format="wav")
        return output_path
    
    def get_audio_duration(self, audio_path: Union[str, Path]) -> float:
        """Get duration of an audio file in seconds."""
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0


# Alternative: MeloTTS (faster, less expressive)
class MeloVoice:
    """
    Alternative TTS using MeloTTS.
    Faster but less expressive than XTTS.
    
    Note: MeloTTS needs to be installed separately:
    pip install git+https://github.com/myshell-ai/MeloTTS.git
    """
    
    def __init__(self):
        self.model = None
        self.device = config.DEVICE
    
    def load(self):
        """Load MeloTTS model."""
        try:
            from melo.api import TTS
            print("🎙️ Loading MeloTTS...")
            self.model = TTS(language='EN', device=self.device)
            print("✅ MeloTTS loaded!")
        except ImportError:
            raise ImportError(
                "MeloTTS not installed. Install with:\n"
                "pip install git+https://github.com/myshell-ai/MeloTTS.git"
            )
        return self
    
    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        language: str = "en",
        speed: float = 1.0
    ) -> Path:
        """Synthesize speech using MeloTTS."""
        if self.model is None:
            self.load()
        
        output_path = Path(output_path)
        
        # MeloTTS language mapping
        lang_map = {
            'en': 'EN',
            'es': 'ES', 
            'fr': 'FR',
            'zh': 'ZH',
            'ja': 'JP',
            'ko': 'KR',
        }
        
        speaker_ids = self.model.hps.data.spk2id
        
        self.model.tts_to_file(
            text,
            speaker_ids['EN-US'],  # Default speaker
            str(output_path),
            speed=speed
        )
        
        return output_path


# Convenience function
def synthesize_script(
    script: List[ScriptSegment],
    output_dir: str,
    language: str = "en",
    speaker_wav: Optional[str] = None,
    progress_callback=None
) -> List[Path]:
    """Quick function to synthesize a script."""
    voice = Voice()
    try:
        return voice.synthesize_script(
            script,
            Path(output_dir),
            language=language,
            speaker_wav=speaker_wav,
            progress_callback=progress_callback
        )
    finally:
        voice.unload()
