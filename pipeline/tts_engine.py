"""
AutoKatha TTS Engine - Multi-backend Text-to-Speech
Supports: F5-TTS, XTTS v2, Edge-TTS with automatic fallback

Priority order:
1. F5-TTS (highest quality, voice cloning)
2. XTTS v2 (good quality, multilingual)
3. Edge-TTS (fast, cloud-based fallback)
"""
import torch
from pathlib import Path
from typing import Optional, Union, List, Literal
from dataclasses import dataclass
from enum import Enum
import tempfile
import os

import config


class TTSBackend(Enum):
    """Available TTS backends."""
    F5_TTS = "f5-tts"
    XTTS = "xtts"
    EDGE_TTS = "edge-tts"
    AUTO = "auto"


@dataclass
class TTSConfig:
    """TTS configuration options."""
    backend: TTSBackend = TTSBackend.AUTO
    language: str = "en"
    speaker_wav: Optional[str] = None  # Reference audio for voice cloning
    speed: float = 1.0
    
    # F5-TTS specific
    f5_model: str = "F5-TTS"  # or "E2-TTS"
    
    # XTTS specific  
    xtts_model: str = "tts_models/multilingual/multi-dataset/xtts_v2"


class F5TTSEngine:
    """
    F5-TTS Engine - State-of-the-art zero-shot voice cloning.
    
    Install: pip install f5-tts
    """
    
    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        self.model = None
        self.available = self._check_available()
    
    def _check_available(self) -> bool:
        """Check if F5-TTS is installed."""
        try:
            from f5_tts.api import F5TTS
            return True
        except ImportError:
            return False
    
    def load(self):
        """Load F5-TTS model."""
        if not self.available:
            raise ImportError("F5-TTS not installed. Run: pip install f5-tts")
        
        from f5_tts.api import F5TTS
        
        print("🎙️ Loading F5-TTS model...")
        self.model = F5TTS(device=self.device)
        print("✅ F5-TTS loaded!")
        return self
    
    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        ref_audio: Optional[str] = None,
        ref_text: Optional[str] = None,
        speed: float = 1.0,
    ) -> Path:
        """
        Synthesize speech using F5-TTS.
        
        Args:
            text: Text to synthesize
            output_path: Output audio file path
            ref_audio: Reference audio for voice cloning
            ref_text: Transcript of reference audio (optional, auto-transcribed if None)
            speed: Speech speed multiplier
        
        Returns:
            Path to generated audio
        """
        if self.model is None:
            self.load()
        
        output_path = Path(output_path)
        
        # Generate audio
        audio, sr = self.model.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            speed=speed,
        )
        
        # Save audio
        import soundfile as sf
        sf.write(str(output_path), audio, sr)
        
        return output_path
    
    def unload(self):
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


class XTTSEngine:
    """
    Coqui XTTS v2 Engine - Multilingual with voice cloning.
    
    Already included in requirements.txt (TTS package)
    """
    
    def __init__(self, device: str = None):
        self.device = device or config.DEVICE
        self.tts = None
        self.available = self._check_available()
    
    def _check_available(self) -> bool:
        """Check if XTTS is available."""
        try:
            from TTS.api import TTS
            return True
        except ImportError:
            return False
    
    def load(self):
        """Load XTTS model."""
        if not self.available:
            raise ImportError("TTS not installed. Run: pip install TTS")
        
        from TTS.api import TTS
        
        print("🎙️ Loading XTTS v2 model...")
        self.tts = TTS(config.TTS_MODEL)
        self.tts = self.tts.to(self.device)
        print("✅ XTTS v2 loaded!")
        return self
    
    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        language: str = "en",
        speaker_wav: Optional[str] = None,
    ) -> Path:
        """Synthesize speech using XTTS."""
        if self.tts is None:
            self.load()
        
        output_path = Path(output_path)
        
        # XTTS supported languages
        xtts_languages = ['en', 'es', 'fr', 'de', 'it', 'pt', 'pl', 'tr', 'ru', 
                         'nl', 'cs', 'ar', 'zh-cn', 'ja', 'hu', 'ko', 'hi']
        
        # Language mapping for unsupported languages
        lang_map = {
            'zh': 'zh-cn',
            'ta': 'en', 'te': 'en', 'bn': 'en',  # South Indian - fallback
            'mr': 'hi', 'gu': 'hi', 'pa': 'hi',  # North Indian - use Hindi
            'kn': 'en', 'ml': 'en',
        }
        
        actual_lang = lang_map.get(language, language)
        if actual_lang not in xtts_languages:
            actual_lang = 'en'
        
        if speaker_wav:
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                speaker_wav=speaker_wav,
                language=actual_lang
            )
        else:
            self.tts.tts_to_file(
                text=text,
                file_path=str(output_path),
                language=actual_lang
            )
        
        return output_path
    
    def unload(self):
        """Unload model."""
        if self.tts is not None:
            del self.tts
            self.tts = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()


class EdgeTTSEngine:
    """
    Edge-TTS Engine - Microsoft's cloud TTS (fast, no GPU needed).
    
    Install: pip install edge-tts
    """
    
    # Voice mappings for different languages
    VOICE_MAP = {
        "en": "en-US-AriaNeural",
        "en-us": "en-US-AriaNeural",
        "en-gb": "en-GB-SoniaNeural",
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
        "ko": "ko-KR-SunHiNeural",
        "pt": "pt-BR-FranciscaNeural",
        "it": "it-IT-ElsaNeural",
        "ru": "ru-RU-SvetlanaNeural",
        "ar": "ar-SA-ZariyahNeural",
    }
    
    def __init__(self):
        self.available = self._check_available()
    
    def _check_available(self) -> bool:
        """Check if edge-tts is installed."""
        try:
            import edge_tts
            return True
        except ImportError:
            return False
    
    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        language: str = "en",
        voice: Optional[str] = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
    ) -> Path:
        """
        Synthesize speech using Edge-TTS.
        
        Args:
            text: Text to synthesize
            output_path: Output file path
            language: Language code
            voice: Specific voice name (overrides language default)
            rate: Speech rate adjustment (e.g., "+10%", "-20%")
            pitch: Pitch adjustment (e.g., "+5Hz", "-10Hz")
        """
        import edge_tts
        import asyncio
        
        output_path = Path(output_path)
        selected_voice = voice or self.VOICE_MAP.get(language, "en-US-AriaNeural")
        
        async def _generate():
            communicate = edge_tts.Communicate(text, selected_voice, rate=rate, pitch=pitch)
            await communicate.save(str(output_path))
        
        asyncio.run(_generate())
        return output_path
    
    def list_voices(self, language: str = None) -> List[dict]:
        """List available voices."""
        import edge_tts
        import asyncio
        
        async def _list():
            voices = await edge_tts.list_voices()
            if language:
                voices = [v for v in voices if v['Locale'].startswith(language)]
            return voices
        
        return asyncio.run(_list())


class UnifiedTTS:
    """
    Unified TTS interface with automatic backend selection and fallback.
    
    Usage:
        tts = UnifiedTTS()
        tts.synthesize("Hello world", "output.wav", language="en")
        
        # With voice cloning (requires F5-TTS or XTTS)
        tts = UnifiedTTS(speaker_wav="reference.wav")
        tts.synthesize("Hello world", "output.wav")
    """
    
    def __init__(
        self,
        backend: TTSBackend = TTSBackend.AUTO,
        speaker_wav: Optional[str] = None,
        device: str = None,
    ):
        """
        Initialize unified TTS.
        
        Args:
            backend: Preferred backend (AUTO for automatic selection)
            speaker_wav: Reference audio for voice cloning
            device: PyTorch device (auto-detected if None)
        """
        self.device = device or config.DEVICE
        self.speaker_wav = speaker_wav
        self.preferred_backend = backend
        
        # Initialize engines lazily
        self._f5 = None
        self._xtts = None
        self._edge = None
        
        # Determine available backends
        self._available_backends = self._detect_backends()
        self._active_backend = None
        
        print(f"🎙️ TTS initialized")
        print(f"   Available backends: {[b.value for b in self._available_backends]}")
        if speaker_wav:
            print(f"   Voice cloning enabled: {Path(speaker_wav).name}")
    
    def _detect_backends(self) -> List[TTSBackend]:
        """Detect which backends are available."""
        available = []
        
        # Check F5-TTS
        try:
            from f5_tts.api import F5TTS
            available.append(TTSBackend.F5_TTS)
        except ImportError:
            pass
        
        # Check XTTS
        try:
            from TTS.api import TTS
            available.append(TTSBackend.XTTS)
        except ImportError:
            pass
        
        # Check Edge-TTS
        try:
            import edge_tts
            available.append(TTSBackend.EDGE_TTS)
        except ImportError:
            pass
        
        return available
    
    def _select_backend(self, needs_cloning: bool = False) -> TTSBackend:
        """Select the best available backend."""
        if self.preferred_backend != TTSBackend.AUTO:
            if self.preferred_backend in self._available_backends:
                return self.preferred_backend
            print(f"⚠️ Preferred backend {self.preferred_backend.value} not available")
        
        # If voice cloning is needed, prefer F5-TTS or XTTS
        if needs_cloning:
            if TTSBackend.F5_TTS in self._available_backends:
                return TTSBackend.F5_TTS
            if TTSBackend.XTTS in self._available_backends:
                return TTSBackend.XTTS
            print("⚠️ Voice cloning requested but no compatible backend available")
        
        # Default priority: F5-TTS > XTTS > Edge-TTS
        for backend in [TTSBackend.F5_TTS, TTSBackend.XTTS, TTSBackend.EDGE_TTS]:
            if backend in self._available_backends:
                return backend
        
        raise RuntimeError("No TTS backend available. Install at least one: pip install edge-tts")
    
    def _get_engine(self, backend: TTSBackend):
        """Get or create engine instance."""
        if backend == TTSBackend.F5_TTS:
            if self._f5 is None:
                self._f5 = F5TTSEngine(self.device)
            return self._f5
        elif backend == TTSBackend.XTTS:
            if self._xtts is None:
                self._xtts = XTTSEngine(self.device)
            return self._xtts
        elif backend == TTSBackend.EDGE_TTS:
            if self._edge is None:
                self._edge = EdgeTTSEngine()
            return self._edge
        else:
            raise ValueError(f"Unknown backend: {backend}")
    
    def synthesize(
        self,
        text: str,
        output_path: Union[str, Path],
        language: str = "en",
        speaker_wav: Optional[str] = None,
        speed: float = 1.0,
    ) -> Path:
        """
        Synthesize speech with automatic backend selection.
        
        Args:
            text: Text to synthesize
            output_path: Output audio path
            language: Language code
            speaker_wav: Reference audio for voice cloning (overrides instance default)
            speed: Speech speed multiplier
        
        Returns:
            Path to generated audio
        """
        ref_audio = speaker_wav or self.speaker_wav
        needs_cloning = ref_audio is not None
        
        # Select backend
        backend = self._select_backend(needs_cloning)
        
        if self._active_backend != backend:
            print(f"   Using TTS backend: {backend.value}")
            self._active_backend = backend
        
        output_path = Path(output_path)
        
        try:
            if backend == TTSBackend.F5_TTS:
                engine = self._get_engine(backend)
                return engine.synthesize(
                    text=text,
                    output_path=output_path,
                    ref_audio=ref_audio,
                    speed=speed,
                )
            
            elif backend == TTSBackend.XTTS:
                engine = self._get_engine(backend)
                return engine.synthesize(
                    text=text,
                    output_path=output_path,
                    language=language,
                    speaker_wav=ref_audio,
                )
            
            elif backend == TTSBackend.EDGE_TTS:
                engine = self._get_engine(backend)
                # Convert speed to rate string
                rate_percent = int((speed - 1.0) * 100)
                rate = f"+{rate_percent}%" if rate_percent >= 0 else f"{rate_percent}%"
                return engine.synthesize(
                    text=text,
                    output_path=output_path,
                    language=language,
                    rate=rate,
                )
        
        except Exception as e:
            print(f"⚠️ {backend.value} failed: {e}")
            # Remove failed backend from available list to prevent infinite retry
            if backend in self._available_backends:
                self._available_backends.remove(backend)
            # Try fallback with remaining backends
            if self._available_backends:
                remaining = [b.value for b in self._available_backends]
                print(f"   Trying fallback: {remaining[0]}")
                return self.synthesize(text, output_path, language, speaker_wav, speed)
            raise RuntimeError(f"All TTS backends failed for text: {text[:50]}...")
    
    def synthesize_batch(
        self,
        texts: List[str],
        output_dir: Union[str, Path],
        language: str = "en",
        speaker_wav: Optional[str] = None,
        progress_callback=None,
    ) -> List[Path]:
        """
        Synthesize multiple texts.
        
        Args:
            texts: List of texts to synthesize
            output_dir: Directory for output files
            language: Language code
            speaker_wav: Reference audio
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            List of output paths
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = []
        total = len(texts)
        
        for i, text in enumerate(texts):
            if progress_callback:
                progress_callback(i, total, f"Synthesizing {i+1}/{total}")
            
            output_path = output_dir / f"audio_{i:03d}.wav"
            
            if not text.strip():
                # Create silence for empty text
                self._create_silence(output_path, duration=1.0)
            else:
                self.synthesize(
                    text=text,
                    output_path=output_path,
                    language=language,
                    speaker_wav=speaker_wav,
                )
            
            outputs.append(output_path)
        
        if progress_callback:
            progress_callback(total, total, "Synthesis complete!")
        
        return outputs
    
    def _create_silence(self, output_path: Path, duration: float):
        """Create a silent audio file."""
        from pydub import AudioSegment
        silence = AudioSegment.silent(duration=int(duration * 1000))
        silence.export(str(output_path), format="wav")
    
    def unload(self):
        """Unload all loaded models."""
        if self._f5 is not None:
            self._f5.unload()
            self._f5 = None
        if self._xtts is not None:
            self._xtts.unload()
            self._xtts = None
        self._active_backend = None
        print("🎙️ TTS models unloaded")


# Convenience function for simple usage
def synthesize_text(
    text: str,
    output_path: str,
    language: str = "en",
    speaker_wav: Optional[str] = None,
) -> Path:
    """
    Quick function to synthesize text to speech.
    
    Args:
        text: Text to synthesize
        output_path: Output file path
        language: Language code
        speaker_wav: Optional reference audio for voice cloning
    
    Returns:
        Path to generated audio
    """
    tts = UnifiedTTS(speaker_wav=speaker_wav)
    try:
        return tts.synthesize(text, output_path, language)
    finally:
        tts.unload()
