"""
AutoKatha Orchestrator - Main pipeline that coordinates all modules.
Handles memory management and sequential model loading.
"""
from pathlib import Path
from typing import List, Optional, Union, Callable
from dataclasses import dataclass
from datetime import datetime
import json
import shutil

from pipeline.memory_manager import MemoryManager, cleanup
from pipeline.comic_reader import ComicReader, PanelData
from pipeline.screenwriter import Screenwriter, ScriptSegment
from pipeline.artist import Artist
from pipeline.animator import Animator, KenBurnsAnimator
from pipeline.voice import Voice
from pipeline.captioner import Captioner
from pipeline.composer import Composer
import config


@dataclass
class ProjectState:
    """Tracks the state of a video generation project."""
    project_dir: Path
    title: str
    language: str
    
    # Intermediate outputs
    panels: List[PanelData] = None
    script: List[ScriptSegment] = None
    generated_images: List[Path] = None
    video_clips: List[Path] = None
    audio_files: List[Path] = None
    subtitle_path: Path = None
    final_video: Path = None
    
    # Metadata
    created_at: str = None
    status: str = "initialized"
    current_step: str = None
    error: str = None


class AutoKatha:
    """
    Main orchestrator for the comic-to-video pipeline.
    
    Coordinates all modules and handles memory management to ensure
    only one heavy model is loaded at a time.
    """
    
    def __init__(self, output_base_dir: Path = None):
        """
        Initialize AutoKatha.
        
        Args:
            output_base_dir: Base directory for all outputs
        """
        self.output_base_dir = output_base_dir or config.OUTPUT_DIR
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize modules (but don't load models yet)
        self.comic_reader = ComicReader()
        self.screenwriter = None  # Lazy load with chosen model
        self.artist = Artist()
        self.animator = Animator()
        self.kenburns = KenBurnsAnimator()
        self.voice = Voice()
        self.captioner = Captioner()
        self.composer = Composer()
        
        # Current project state
        self.state: ProjectState = None
        
        # Progress callback
        self.progress_callback: Callable = None
    
    def set_progress_callback(self, callback: Callable[[int, int, str], None]):
        """Set a callback for progress updates."""
        self.progress_callback = callback
    
    def _update_progress(self, current: int, total: int, message: str):
        """Update progress if callback is set."""
        if self.progress_callback:
            self.progress_callback(current, total, message)
        print(f"[{current}/{total}] {message}")
    
    def create_project(
        self,
        title: str,
        language: str = "en"
    ) -> ProjectState:
        """
        Create a new video generation project.
        
        Args:
            title: Title of the story/video
            language: Language code for narration
        
        Returns:
            ProjectState object
        """
        # Create project directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_title = "".join(c if c.isalnum() else "_" for c in title)[:30]
        project_name = f"{timestamp}_{safe_title}"
        project_dir = self.output_base_dir / project_name
        project_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (project_dir / "panels").mkdir()
        (project_dir / "images").mkdir()
        (project_dir / "clips").mkdir()
        (project_dir / "audio").mkdir()
        
        self.state = ProjectState(
            project_dir=project_dir,
            title=title,
            language=language,
            created_at=timestamp,
            status="created"
        )
        
        # Save project metadata
        self._save_state()
        
        print(f"\n{'='*60}")
        print(f"📁 Project created: {project_dir.name}")
        print(f"   Title: {title}")
        print(f"   Language: {config.LANGUAGES.get(language, language)}")
        print(f"{'='*60}\n")
        
        return self.state
    
    def _save_state(self):
        """Save project state to disk."""
        if self.state is None:
            return
        
        state_file = self.state.project_dir / "project_state.json"
        state_dict = {
            "title": self.state.title,
            "language": self.state.language,
            "created_at": self.state.created_at,
            "status": self.state.status,
            "current_step": self.state.current_step,
            "error": self.state.error,
            "panels_count": len(self.state.panels) if self.state.panels else 0,
            "final_video": str(self.state.final_video) if self.state.final_video else None
        }
        state_file.write_text(json.dumps(state_dict, indent=2))
    
    def step_read_comic(
        self,
        input_path: Union[str, Path],
    ) -> List[PanelData]:
        """
        Step 1: Read comic and extract panel data.
        
        Args:
            input_path: Path to PDF, image, or directory of images
        """
        self.state.current_step = "reading_comic"
        self.state.status = "processing"
        self._save_state()
        
        self._update_progress(0, 1, "Loading comic reader (Moondream2)...")
        
        try:
            # Read comic
            panels = self.comic_reader.read_comic(
                input_path,
                progress_callback=self._update_progress
            )
            
            # Save panel images
            for i, panel in enumerate(panels):
                panel_path = self.state.project_dir / "panels" / f"panel_{i:03d}.png"
                panel.image.save(panel_path)
                panel.image_path = panel_path
            
            self.state.panels = panels
            
            # Unload model
            self.comic_reader.unload()
            
            self._save_state()
            return panels
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._save_state()
            cleanup()
            raise
    
    def step_write_script(
        self,
        llm_model: str = "gemma3:4b"
    ) -> List[ScriptSegment]:
        """
        Step 2: Generate enhanced narration script.
        
        Args:
            llm_model: Ollama model to use for script generation
        """
        if not self.state.panels:
            raise ValueError("No panels loaded. Run step_read_comic first.")
        
        self.state.current_step = "writing_script"
        self._save_state()
        
        self._update_progress(0, 1, f"Loading screenwriter ({llm_model})...")
        
        try:
            self.screenwriter = Screenwriter(model=llm_model)
            
            script = self.screenwriter.write_script(
                panels=self.state.panels,
                title=self.state.title,
                language=self.state.language,
                progress_callback=self._update_progress
            )
            
            self.state.script = script
            
            # Save script
            script_file = self.state.project_dir / "script.json"
            script_data = [
                {
                    "panel": seg.panel_index,
                    "narration": seg.narration,
                    "duration_hint": seg.duration_hint,
                    "emotion": seg.emotion
                }
                for seg in script
            ]
            script_file.write_text(json.dumps(script_data, indent=2, ensure_ascii=False))
            
            self._save_state()
            return script
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._save_state()
            raise
    
    def step_generate_images(
        self,
    ) -> List[Path]:
        """
        Step 3: Generate images for panels based on their descriptions.
        """
        if not self.state.panels:
            raise ValueError("No panels loaded. Run step_read_comic first.")
        
        self.state.current_step = "generating_images"
        self._save_state()
        
        self._update_progress(0, 1, "Loading artist (SDXL Lightning)...")
        
        try:
            images_dir = self.state.project_dir / "images"
            
            generated_images = self.artist.generate_images_for_panels(
                panels=self.state.panels,
                output_dir=images_dir,
                progress_callback=self._update_progress
            )
            
            # Save paths
            self.state.generated_images = [
                images_dir / f"panel_{i:03d}.png"
                for i in range(len(generated_images))
            ]
            
            # Unload model
            self.artist.unload()
            
            self._save_state()
            return self.state.generated_images
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            print(f"❌ Error in step_generate_images: {e}")
            import traceback
            traceback.print_exc()
            self._save_state()
            cleanup()
            raise
    
    def step_animate(
        self,
        use_svd: bool = True,
        motion_amount: int = 100
    ) -> List[Path]:
        """
        Step 4: Animate generated images.
        
        Args:
            use_svd: Use Stable Video Diffusion (True) or Ken Burns (False)
            motion_amount: For SVD, controls motion (1-255, higher = more)
        """
        if not self.state.generated_images:
            raise ValueError("No generated images. Run step_generate_images first.")
        
        self.state.current_step = "animating"
        self._save_state()
        
        clips_dir = self.state.project_dir / "clips"
        
        try:
            if use_svd:
                self._update_progress(0, 1, "Loading animator (Stable Video Diffusion)...")
                
                # Load styled images
                from PIL import Image
                images = [Image.open(p) for p in self.state.generated_images]
                
                video_clips = self.animator.animate_panels(
                    images=images,
                    output_dir=clips_dir,
                    motion_bucket_id=motion_amount,
                    progress_callback=self._update_progress
                )
                
                # Unload SVD
                self.animator.unload()
                
            else:
                self._update_progress(0, 1, "Applying Ken Burns effects...")
                
                from PIL import Image
                images = [Image.open(p) for p in self.state.generated_images]
                
                # Estimate duration from script if available
                duration = 5.0
                if self.state.script:
                    avg_duration = sum(s.duration_hint for s in self.state.script) / len(self.state.script)
                    duration = avg_duration
                
                video_clips = self.kenburns.animate_panels(
                    images=images,
                    output_dir=clips_dir,
                    duration_per_panel=duration,
                    progress_callback=self._update_progress
                )
            
            self.state.video_clips = video_clips
            self._save_state()
            return video_clips
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._save_state()
            cleanup()
            raise
    
    def step_generate_voice(
        self,
        speaker_wav: Optional[str] = None
    ) -> List[Path]:
        """
        Step 5: Generate voice narration.
        
        Args:
            speaker_wav: Optional path to voice sample for cloning
        """
        if not self.state.script:
            raise ValueError("No script. Run step_write_script first.")
        
        self.state.current_step = "generating_voice"
        self._save_state()
        
        self._update_progress(0, 1, "Loading voice (XTTS v2)...")
        
        try:
            audio_dir = self.state.project_dir / "audio"
            
            audio_files = self.voice.synthesize_script(
                script=self.state.script,
                output_dir=audio_dir,
                language=self.state.language,
                speaker_wav=speaker_wav,
                progress_callback=self._update_progress
            )
            
            self.state.audio_files = audio_files
            
            # Unload model
            self.voice.unload()
            
            self._save_state()
            return audio_files
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._save_state()
            cleanup()
            raise
    
    def step_generate_captions(self) -> Path:
        """
        Step 6: Generate subtitles from audio.
        """
        if not self.state.audio_files:
            raise ValueError("No audio files. Run step_generate_voice first.")
        
        self.state.current_step = "generating_captions"
        self._save_state()
        
        self._update_progress(0, 1, "Loading captioner (Whisper)...")
        
        try:
            subtitle_path = self.state.project_dir / "subtitles.srt"
            
            self.captioner.load()
            self.captioner.generate_captions(
                audio_paths=self.state.audio_files,
                output_path=subtitle_path,
                language=self.state.language,
                progress_callback=self._update_progress
            )
            
            self.state.subtitle_path = subtitle_path
            self._save_state()
            return subtitle_path
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._save_state()
            raise
    
    def step_compose_video(
        self,
        burn_subtitles: bool = True
    ) -> Path:
        """
        Step 7: Compose final video.
        
        Args:
            burn_subtitles: Burn subtitles into video (vs separate track)
        """
        if not self.state.video_clips or not self.state.audio_files:
            raise ValueError("Missing video clips or audio. Run previous steps first.")
        
        self.state.current_step = "composing"
        self._save_state()
        
        self._update_progress(0, 1, "Composing final video...")
        
        try:
            output_path = self.state.project_dir / f"{self.state.title.replace(' ', '_')}.mp4"
            
            self.composer.compose(
                video_clips=self.state.video_clips,
                audio_files=self.state.audio_files,
                subtitle_path=self.state.subtitle_path,
                output_path=output_path,
                burn_subtitles=burn_subtitles,
                progress_callback=self._update_progress
            )
            
            self.state.final_video = output_path
            self.state.status = "completed"
            self._save_state()
            
            print(f"\n{'='*60}")
            print(f"🎉 VIDEO COMPLETE!")
            print(f"   📁 {output_path}")
            print(f"{'='*60}\n")
            
            return output_path
            
        except Exception as e:
            self.state.status = "error"
            self.state.error = str(e)
            self._save_state()
            raise
    
    def run_full_pipeline(
        self,
        input_path: Union[str, Path],
        title: str,
        language: str = "en",
        use_svd: bool = True,
        speaker_wav: Optional[str] = None,
        llm_model: str = "gemma3:4b",
        burn_subtitles: bool = True
    ) -> Path:
        """
        Run the complete pipeline from comic to video.
        
        Args:
            input_path: Path to comic (PDF/images)
            title: Video title
            language: Language code
            use_svd: Use SVD for animation
            speaker_wav: Voice sample for cloning
            llm_model: Ollama model for script
            burn_subtitles: Burn subtitles into video
        
        Returns:
            Path to final video
        """
        print(f"\n{'#'*60}")
        print(f"# AutoKatha - Full Pipeline")
        print(f"# Comic: {input_path}")
        print(f"# Title: {title}")
        print(f"{'#'*60}\n")
        
        # Create project
        self.create_project(title, language)
        
        # Run all steps
        self.step_read_comic(input_path)
        cleanup()
        
        self.step_write_script(llm_model)
        
        self.step_generate_images()
        cleanup()
        
        self.step_animate(use_svd)
        cleanup()
        
        self.step_generate_voice(speaker_wav)
        cleanup()
        
        self.step_generate_captions()
        
        final_video = self.step_compose_video(burn_subtitles)
        
        return final_video


# Convenience function
def generate_video(
    comic_path: str,
    title: str,
    language: str = "en",
    **kwargs
) -> Path:
    """
    Quick function to generate a video from a comic.
    
    Args:
        comic_path: Path to comic (PDF or images)
        title: Video title
        language: Language code
        **kwargs: Additional options for run_full_pipeline
    
    Returns:
        Path to final video
    """
    autokatha = AutoKatha()
    return autokatha.run_full_pipeline(
        input_path=comic_path,
        title=title,
        language=language,
        **kwargs
    )
