"""
Captioner - Uses Whisper for automatic subtitle generation.
Generates word-level timestamps for accurate captions.
"""
import whisper
from pathlib import Path
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass
import json
import re

import config


@dataclass
class Caption:
    """A single caption/subtitle entry."""
    index: int
    start_time: float  # seconds
    end_time: float  # seconds
    text: str


class Captioner:
    """
    Generates captions/subtitles from audio using Whisper.
    Supports multiple output formats (SRT, VTT, JSON).
    """
    
    def __init__(self, model_size: str = None):
        """
        Initialize captioner.
        
        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
                       Larger = more accurate but slower
        """
        self.model_size = model_size or config.WHISPER_MODEL
        self.model = None
        self.device = config.DEVICE
    
    def load(self):
        """Load Whisper model."""
        print(f"📝 Loading Whisper ({self.model_size})...")
        
        # Whisper works well on MPS
        self.model = whisper.load_model(self.model_size, device=self.device)
        
        print("✅ Whisper loaded!")
        return self
    
    def transcribe(
        self,
        audio_path: Union[str, Path],
        language: str = None,
        word_timestamps: bool = True
    ) -> dict:
        """
        Transcribe audio file.
        
        Args:
            audio_path: Path to audio file
            language: Language code (auto-detect if None)
            word_timestamps: Whether to include word-level timestamps
        
        Returns:
            Whisper transcription result
        """
        if self.model is None:
            self.load()
        
        result = self.model.transcribe(
            str(audio_path),
            language=language,
            word_timestamps=word_timestamps,
            verbose=False
        )
        
        return result
    
    def generate_captions(
        self,
        audio_paths: List[Path],
        output_path: Path,
        language: str = None,
        format: str = "srt",
        max_chars_per_line: int = 42,
        progress_callback=None
    ) -> Path:
        """
        Generate captions for multiple audio files.
        
        Args:
            audio_paths: List of audio file paths (in order)
            output_path: Path for output caption file
            language: Language code (auto-detect if None)
            format: Output format (srt, vtt, json)
            max_chars_per_line: Maximum characters per caption line
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            Path to generated caption file
        """
        if self.model is None:
            self.load()
        
        print(f"\n📝 Generating captions for {len(audio_paths)} audio files...")
        
        all_captions = []
        current_time = 0.0
        caption_index = 1
        total = len(audio_paths)
        
        for idx, audio_path in enumerate(audio_paths):
            if progress_callback:
                progress_callback(idx, total, f"Transcribing audio {idx + 1}/{total}")
            
            print(f"📝 Processing {audio_path.name}...")
            
            # Transcribe
            result = self.transcribe(audio_path, language=language)
            
            # Convert to captions
            for segment in result['segments']:
                # Split long segments
                text = segment['text'].strip()
                
                if len(text) > max_chars_per_line * 2:
                    # Split into multiple captions
                    chunks = self._split_text(text, max_chars_per_line * 2)
                    chunk_duration = (segment['end'] - segment['start']) / len(chunks)
                    
                    for i, chunk in enumerate(chunks):
                        caption = Caption(
                            index=caption_index,
                            start_time=current_time + segment['start'] + (i * chunk_duration),
                            end_time=current_time + segment['start'] + ((i + 1) * chunk_duration),
                            text=chunk
                        )
                        all_captions.append(caption)
                        caption_index += 1
                else:
                    caption = Caption(
                        index=caption_index,
                        start_time=current_time + segment['start'],
                        end_time=current_time + segment['end'],
                        text=text
                    )
                    all_captions.append(caption)
                    caption_index += 1
            
            # Get audio duration for offset
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            current_time += len(audio) / 1000.0
        
        if progress_callback:
            progress_callback(total, total, "Caption generation complete!")
        
        # Export
        output_path = Path(output_path)
        if format == "srt":
            self._export_srt(all_captions, output_path)
        elif format == "vtt":
            self._export_vtt(all_captions, output_path)
        elif format == "json":
            self._export_json(all_captions, output_path)
        
        print(f"✅ Captions saved: {output_path}")
        print(f"   Total captions: {len(all_captions)}")
        
        return output_path
    
    def _split_text(self, text: str, max_length: int) -> List[str]:
        """Split text into chunks at sentence/clause boundaries."""
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) + 1 <= max_length:
                current_chunk = f"{current_chunk} {sentence}".strip()
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks if chunks else [text]
    
    def _format_timestamp_srt(self, seconds: float) -> str:
        """Format timestamp for SRT (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    
    def _format_timestamp_vtt(self, seconds: float) -> str:
        """Format timestamp for VTT (HH:MM:SS.mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    
    def _export_srt(self, captions: List[Caption], output_path: Path):
        """Export captions in SRT format."""
        lines = []
        for cap in captions:
            lines.append(str(cap.index))
            lines.append(f"{self._format_timestamp_srt(cap.start_time)} --> {self._format_timestamp_srt(cap.end_time)}")
            lines.append(cap.text)
            lines.append("")
        
        output_path.write_text("\n".join(lines), encoding='utf-8')
    
    def _export_vtt(self, captions: List[Caption], output_path: Path):
        """Export captions in WebVTT format."""
        lines = ["WEBVTT", ""]
        for cap in captions:
            lines.append(f"{self._format_timestamp_vtt(cap.start_time)} --> {self._format_timestamp_vtt(cap.end_time)}")
            lines.append(cap.text)
            lines.append("")
        
        output_path.write_text("\n".join(lines), encoding='utf-8')
    
    def _export_json(self, captions: List[Caption], output_path: Path):
        """Export captions in JSON format."""
        data = [
            {
                "index": cap.index,
                "start": cap.start_time,
                "end": cap.end_time,
                "text": cap.text
            }
            for cap in captions
        ]
        output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')


# Convenience function
def generate_captions(
    audio_paths: List[Path],
    output_path: str,
    language: str = None,
    format: str = "srt",
    progress_callback=None
) -> Path:
    """Quick function to generate captions."""
    captioner = Captioner()
    captioner.load()
    return captioner.generate_captions(
        audio_paths,
        Path(output_path),
        language=language,
        format=format,
        progress_callback=progress_callback
    )
