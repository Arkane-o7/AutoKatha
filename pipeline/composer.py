"""
Composer - Merges video clips, audio, and subtitles into final video.
Uses FFmpeg for high-quality video processing.
"""
import subprocess
from pathlib import Path
from typing import List, Optional, Union, Tuple
from pydub import AudioSegment
import json
import tempfile
import shutil

import config


class Composer:
    """
    Composes final video from animated clips, audio, and captions.
    Uses FFmpeg for all video processing operations.
    """
    
    def __init__(self):
        self._check_ffmpeg()
        
    def _check_ffmpeg(self):
        """Verify FFmpeg is installed."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-version'], 
                capture_output=True, 
                text=True
            )
            if result.returncode != 0:
                raise RuntimeError("FFmpeg check failed")
            print("✅ FFmpeg available")
        except FileNotFoundError:
            raise RuntimeError(
                "FFmpeg not found. Please install it:\n"
                "  macOS: brew install ffmpeg\n"
                "  Ubuntu: sudo apt install ffmpeg"
            )
    
    def get_video_duration(self, video_path: Path) -> float:
        """Get duration of a video file in seconds."""
        cmd = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            str(video_path)
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(result.stdout)
        return float(data['format']['duration'])
    
    def get_audio_duration(self, audio_path: Path) -> float:
        """Get duration of an audio file in seconds."""
        audio = AudioSegment.from_file(str(audio_path))
        return len(audio) / 1000.0
    
    def adjust_video_to_audio(
        self,
        video_path: Path,
        audio_path: Path,
        output_path: Path
    ) -> Path:
        """
        Adjust video duration to match audio duration.
        Uses speed adjustment or looping as needed.
        """
        video_duration = self.get_video_duration(video_path)
        audio_duration = self.get_audio_duration(audio_path)
        
        if abs(video_duration - audio_duration) < 0.5:
            # Close enough, just copy
            shutil.copy(video_path, output_path)
            return output_path
        
        if video_duration < audio_duration:
            # Video is shorter - slow it down or loop
            speed_factor = video_duration / audio_duration
            
            if speed_factor > 0.5:
                # Slow down video
                cmd = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-filter:v', f'setpts={1/speed_factor}*PTS',
                    '-an',  # Remove audio
                    str(output_path)
                ]
            else:
                # Loop video to match audio
                loops = int(audio_duration / video_duration) + 1
                cmd = [
                    'ffmpeg', '-y',
                    '-stream_loop', str(loops),
                    '-i', str(video_path),
                    '-t', str(audio_duration),
                    '-an',
                    str(output_path)
                ]
        else:
            # Video is longer - speed it up or trim
            speed_factor = video_duration / audio_duration
            
            if speed_factor < 2.0:
                # Speed up video
                cmd = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-filter:v', f'setpts={1/speed_factor}*PTS',
                    '-an',
                    str(output_path)
                ]
            else:
                # Just trim video
                cmd = [
                    'ffmpeg', '-y', '-i', str(video_path),
                    '-t', str(audio_duration),
                    '-an',
                    str(output_path)
                ]
        
        subprocess.run(cmd, capture_output=True, check=True)
        return output_path
    
    def concatenate_videos(
        self,
        video_paths: List[Path],
        output_path: Path,
        transition: str = "none"  # none, fade, dissolve
    ) -> Path:
        """
        Concatenate multiple video clips into one.
        
        Args:
            video_paths: List of video file paths
            output_path: Output video path
            transition: Transition type between clips
        """
        if not video_paths:
            raise ValueError("No video paths provided")
        
        print(f"🎬 Concatenating {len(video_paths)} video clips...")
        
        # Create concat file
        concat_file = Path(tempfile.mktemp(suffix='.txt'))
        with open(concat_file, 'w') as f:
            for vp in video_paths:
                f.write(f"file '{vp.absolute()}'\n")
        
        if transition == "none":
            # Simple concatenation
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c', 'copy',
                str(output_path)
            ]
        else:
            # With transitions (more complex, re-encodes)
            # For now, use simple concat
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', str(concat_file),
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'medium',
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        concat_file.unlink()  # Cleanup
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("Video concatenation failed")
        
        print(f"✅ Concatenated video: {output_path}")
        return output_path
    
    def concatenate_audio(
        self,
        audio_paths: List[Path],
        output_path: Path,
        gap_ms: int = 500  # Gap between segments
    ) -> Path:
        """
        Concatenate multiple audio files into one.
        
        Args:
            audio_paths: List of audio file paths
            output_path: Output audio path
            gap_ms: Milliseconds of silence between segments
        """
        if not audio_paths:
            raise ValueError("No audio paths provided")
        
        print(f"🎵 Concatenating {len(audio_paths)} audio segments...")
        
        combined = AudioSegment.empty()
        gap = AudioSegment.silent(duration=gap_ms)
        
        for i, path in enumerate(audio_paths):
            audio = AudioSegment.from_file(str(path))
            combined += audio
            if i < len(audio_paths) - 1:
                combined += gap
        
        combined.export(str(output_path), format="wav")
        
        print(f"✅ Combined audio: {output_path}")
        print(f"   Total duration: {len(combined)/1000:.1f}s")
        
        return output_path
    
    def merge_video_audio_subtitles(
        self,
        video_path: Path,
        audio_path: Path,
        subtitle_path: Optional[Path],
        output_path: Path,
        burn_subtitles: bool = True
    ) -> Path:
        """
        Merge video, audio, and subtitles into final video.
        
        Args:
            video_path: Path to video file
            audio_path: Path to audio file
            subtitle_path: Path to subtitle file (SRT/VTT)
            output_path: Output video path
            burn_subtitles: If True, burn subtitles into video
                           If False, include as separate track
        """
        print(f"\n🎬 Composing final video...")
        print(f"   Video: {video_path.name}")
        print(f"   Audio: {audio_path.name}")
        print(f"   Subtitles: {subtitle_path.name if subtitle_path else 'None'}")
        
        if subtitle_path and burn_subtitles:
            # Burn subtitles into video
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-vf', f"subtitles='{subtitle_path}'",
                '-c:v', 'libx264',
                '-crf', '18',
                '-preset', 'medium',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v',
                '-map', '1:a',
                str(output_path)
            ]
        elif subtitle_path:
            # Include subtitles as separate track
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-i', str(subtitle_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-c:s', 'mov_text',
                '-map', '0:v',
                '-map', '1:a',
                '-map', '2:s',
                str(output_path)
            ]
        else:
            # No subtitles
            cmd = [
                'ffmpeg', '-y',
                '-i', str(video_path),
                '-i', str(audio_path),
                '-c:v', 'copy',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-map', '0:v',
                '-map', '1:a',
                str(output_path)
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"FFmpeg error: {result.stderr}")
            raise RuntimeError("Video composition failed")
        
        print(f"✅ Final video: {output_path}")
        
        # Get final duration
        duration = self.get_video_duration(output_path)
        print(f"   Duration: {duration:.1f}s ({duration/60:.1f} minutes)")
        
        return output_path
    
    def compose(
        self,
        video_clips: List[Path],
        audio_files: List[Path],
        subtitle_path: Optional[Path],
        output_path: Path,
        sync_to_audio: bool = True,
        burn_subtitles: bool = True,
        progress_callback=None
    ) -> Path:
        """
        Full composition pipeline.
        
        Args:
            video_clips: List of video clip paths (one per panel)
            audio_files: List of audio file paths (one per panel)
            subtitle_path: Path to subtitle file
            output_path: Final output video path
            sync_to_audio: Adjust video timing to match audio
            burn_subtitles: Burn subtitles into video
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            Path to final video
        """
        if len(video_clips) != len(audio_files):
            raise ValueError(
                f"Mismatch: {len(video_clips)} video clips vs {len(audio_files)} audio files"
            )
        
        temp_dir = Path(tempfile.mkdtemp())
        total_steps = len(video_clips) + 3  # clips + concat video + concat audio + merge
        current_step = 0
        
        try:
            # Step 1: Sync each video clip to its audio
            synced_clips = []
            for i, (vid, aud) in enumerate(zip(video_clips, audio_files)):
                if progress_callback:
                    progress_callback(current_step, total_steps, f"Syncing clip {i+1}/{len(video_clips)}")
                
                if sync_to_audio:
                    synced_path = temp_dir / f"synced_{i:03d}.mp4"
                    self.adjust_video_to_audio(vid, aud, synced_path)
                    synced_clips.append(synced_path)
                else:
                    synced_clips.append(vid)
                
                current_step += 1
            
            # Step 2: Concatenate all video clips
            if progress_callback:
                progress_callback(current_step, total_steps, "Concatenating video clips...")
            
            combined_video = temp_dir / "combined_video.mp4"
            self.concatenate_videos(synced_clips, combined_video)
            current_step += 1
            
            # Step 3: Concatenate all audio
            if progress_callback:
                progress_callback(current_step, total_steps, "Concatenating audio...")
            
            combined_audio = temp_dir / "combined_audio.wav"
            self.concatenate_audio(audio_files, combined_audio, gap_ms=300)
            current_step += 1
            
            # Step 4: Merge everything
            if progress_callback:
                progress_callback(current_step, total_steps, "Merging final video...")
            
            self.merge_video_audio_subtitles(
                combined_video,
                combined_audio,
                subtitle_path,
                output_path,
                burn_subtitles=burn_subtitles
            )
            
            if progress_callback:
                progress_callback(total_steps, total_steps, "Composition complete!")
            
            return output_path
            
        finally:
            # Cleanup temp files
            shutil.rmtree(temp_dir, ignore_errors=True)


# Convenience function
def compose_video(
    video_clips: List[Path],
    audio_files: List[Path],
    subtitle_path: Optional[Path],
    output_path: str,
    progress_callback=None
) -> Path:
    """Quick function to compose final video."""
    composer = Composer()
    return composer.compose(
        video_clips,
        audio_files,
        subtitle_path,
        Path(output_path),
        progress_callback=progress_callback
    )
