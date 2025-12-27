"""
AutoKatha Large Text Processor
Handles books (400+ pages) with intelligent chunking and chapter-aware processing.
"""
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Generator
from dataclasses import dataclass, field
import json

from pipeline.story_processor import (
    StoryProcessor, Character, Scene, StoryAnalysis,
    AspectRatio, detect_chapters, chunk_large_text
)


@dataclass
class Chapter:
    """Represents a chapter in a book."""
    index: int
    title: str
    content: str
    start_position: int
    word_count: int
    scenes: List[Scene] = field(default_factory=list)


@dataclass
class BookAnalysis:
    """Complete analysis of a book."""
    title: str
    author: str
    total_chapters: int
    total_words: int
    chapters: List[Chapter]
    characters: List[Character]
    estimated_duration: float  # minutes
    recommended_episodes: int


class LargeTextProcessor:
    """
    Processes large texts (books, long stories) with intelligent chunking.
    
    Features:
    - Chapter detection and extraction
    - Character tracking across chapters
    - Memory-efficient streaming processing
    - Episode/part splitting for very long content
    """
    
    # Rough estimates
    WORDS_PER_PAGE = 250
    SECONDS_PER_SCENE = 8  # Average narration + viewing time
    MAX_SCENES_PER_EPISODE = 30  # ~4 minute episode
    
    def __init__(
        self,
        ollama_model: str = "gemma3:4b",
        ollama_host: str = "http://localhost:11434",
    ):
        self.processor = StoryProcessor(ollama_model, ollama_host)
        self.characters: List[Character] = []
    
    def estimate_book_metrics(self, text: str) -> Dict:
        """
        Estimate metrics for a book/long text.
        
        Returns:
            Dict with word_count, page_count, chapter_count, 
            recommended_scenes, recommended_episodes
        """
        word_count = len(text.split())
        page_count = word_count / self.WORDS_PER_PAGE
        
        # Detect chapters
        chapters = detect_chapters(text)
        chapter_count = len(chapters) if chapters else max(1, int(page_count / 20))
        
        # Estimate scenes (roughly 1 scene per 150 words)
        total_scenes = max(10, word_count // 150)
        
        # Calculate episodes
        episodes = max(1, total_scenes // self.MAX_SCENES_PER_EPISODE)
        
        # Estimate duration
        total_duration = total_scenes * self.SECONDS_PER_SCENE / 60  # minutes
        
        return {
            "word_count": word_count,
            "page_count": int(page_count),
            "chapter_count": chapter_count,
            "recommended_scenes": total_scenes,
            "recommended_episodes": episodes,
            "estimated_duration_minutes": round(total_duration, 1),
            "scenes_per_episode": min(self.MAX_SCENES_PER_EPISODE, total_scenes // max(1, episodes)),
        }
    
    def extract_chapters(self, text: str, title: str = "Book") -> List[Chapter]:
        """
        Extract chapters from a book.
        
        Args:
            text: Full book text
            title: Book title for unnamed chapters
        
        Returns:
            List of Chapter objects
        """
        chapter_markers = detect_chapters(text)
        
        if not chapter_markers:
            # No chapters detected - create artificial divisions
            return self._create_artificial_chapters(text, title)
        
        chapters = []
        for i, (chapter_title, start_pos) in enumerate(chapter_markers):
            # Get end position (start of next chapter or end of text)
            end_pos = chapter_markers[i + 1][1] if i + 1 < len(chapter_markers) else len(text)
            
            content = text[start_pos:end_pos].strip()
            
            chapters.append(Chapter(
                index=i,
                title=chapter_title,
                content=content,
                start_position=start_pos,
                word_count=len(content.split()),
            ))
        
        return chapters
    
    def _create_artificial_chapters(
        self, 
        text: str, 
        title: str,
        target_words_per_chapter: int = 3000
    ) -> List[Chapter]:
        """Create artificial chapter divisions for texts without chapters."""
        words = text.split()
        total_words = len(words)
        num_chapters = max(1, total_words // target_words_per_chapter)
        
        chapters = []
        words_per_chapter = total_words // num_chapters
        
        current_pos = 0
        for i in range(num_chapters):
            start_word = i * words_per_chapter
            end_word = start_word + words_per_chapter if i < num_chapters - 1 else total_words
            
            content = " ".join(words[start_word:end_word])
            
            # Find actual character position
            char_start = len(" ".join(words[:start_word])) + 1 if start_word > 0 else 0
            
            chapters.append(Chapter(
                index=i,
                title=f"{title} - Part {i + 1}",
                content=content,
                start_position=char_start,
                word_count=end_word - start_word,
            ))
        
        return chapters
    
    def extract_all_characters(
        self, 
        text: str,
        sample_size: int = 10000
    ) -> List[Character]:
        """
        Extract characters from the entire text using sampling.
        
        For very long texts, samples multiple sections to build
        a comprehensive character list.
        """
        if len(text) <= sample_size:
            return self.processor.extract_characters(text)
        
        # Sample beginning, middle, and end
        samples = [
            text[:sample_size],  # Beginning
            text[len(text)//2 - sample_size//2 : len(text)//2 + sample_size//2],  # Middle
            text[-sample_size:],  # End
        ]
        
        all_characters = []
        seen_names = set()
        
        for sample in samples:
            chars = self.processor.extract_characters(sample)
            for char in chars:
                # Deduplicate by name
                if char.name.lower() not in seen_names:
                    seen_names.add(char.name.lower())
                    all_characters.append(char)
        
        self.characters = all_characters
        return all_characters
    
    def process_chapter(
        self,
        chapter: Chapter,
        scenes_per_chapter: int = None,
        style_prefix: str = "",
        progress_callback=None,
    ) -> List[Scene]:
        """
        Process a single chapter into scenes.
        
        Args:
            chapter: Chapter to process
            scenes_per_chapter: Override scene count (None for auto)
            style_prefix: Style tokens for prompts
            progress_callback: Progress callback
        
        Returns:
            List of Scene objects
        """
        # Calculate scene count for this chapter
        if scenes_per_chapter is None:
            scenes_per_chapter = max(2, chapter.word_count // 150)
        
        # Process with existing characters for consistency
        scenes = self.processor.split_into_scenes(
            text=chapter.content,
            num_scenes=scenes_per_chapter,
            characters=self.characters,
            style_prefix=style_prefix,
        )
        
        chapter.scenes = scenes
        return scenes
    
    def process_book(
        self,
        text: str,
        title: str = "Book",
        author: str = "Unknown",
        max_scenes_total: int = 100,
        style_prefix: str = "",
        progress_callback=None,
    ) -> BookAnalysis:
        """
        Process an entire book into scenes.
        
        Args:
            text: Full book text
            title: Book title
            author: Author name
            max_scenes_total: Maximum total scenes
            style_prefix: Style tokens
            progress_callback: Progress callback(current, total, message)
        
        Returns:
            BookAnalysis with all chapters and scenes
        """
        print(f"📚 Processing book: {title}")
        
        # Get metrics
        metrics = self.estimate_book_metrics(text)
        print(f"   📊 {metrics['word_count']:,} words, ~{metrics['page_count']} pages")
        print(f"   📖 {metrics['chapter_count']} chapters detected")
        
        # Extract chapters
        if progress_callback:
            progress_callback(0, 100, "Extracting chapters...")
        chapters = self.extract_chapters(text, title)
        print(f"   ✅ Extracted {len(chapters)} chapters")
        
        # Extract characters from full text
        if progress_callback:
            progress_callback(5, 100, "Extracting characters...")
        characters = self.extract_all_characters(text)
        print(f"   👥 Found {len(characters)} characters")
        
        # Calculate scenes per chapter
        total_chapters = len(chapters)
        scenes_per_chapter = max(2, min(10, max_scenes_total // total_chapters))
        
        # Process each chapter
        all_scenes = []
        for i, chapter in enumerate(chapters):
            if progress_callback:
                progress = 10 + int(85 * (i / total_chapters))
                progress_callback(progress, 100, f"Processing chapter {i+1}/{total_chapters}...")
            
            print(f"\n📖 Chapter {i+1}/{total_chapters}: {chapter.title}")
            
            scenes = self.process_chapter(
                chapter=chapter,
                scenes_per_chapter=scenes_per_chapter,
                style_prefix=style_prefix,
            )
            
            # Offset scene indices
            for scene in scenes:
                scene.index = len(all_scenes) + scene.index
            
            all_scenes.extend(scenes)
            
            # Check if we've hit the limit
            if len(all_scenes) >= max_scenes_total:
                print(f"   ⚠️ Reached scene limit ({max_scenes_total})")
                break
        
        # Calculate final metrics
        total_duration = len(all_scenes) * self.SECONDS_PER_SCENE / 60
        num_episodes = max(1, len(all_scenes) // self.MAX_SCENES_PER_EPISODE)
        
        if progress_callback:
            progress_callback(100, 100, "Processing complete!")
        
        return BookAnalysis(
            title=title,
            author=author,
            total_chapters=len(chapters),
            total_words=metrics['word_count'],
            chapters=chapters,
            characters=characters,
            estimated_duration=total_duration,
            recommended_episodes=num_episodes,
        )
    
    def split_into_episodes(
        self,
        analysis: BookAnalysis,
        scenes_per_episode: int = None,
    ) -> List[List[Scene]]:
        """
        Split book scenes into episodes for sequential video generation.
        
        Args:
            analysis: BookAnalysis from process_book
            scenes_per_episode: Scenes per episode (None for auto)
        
        Returns:
            List of episode scene lists
        """
        if scenes_per_episode is None:
            scenes_per_episode = self.MAX_SCENES_PER_EPISODE
        
        # Gather all scenes
        all_scenes = []
        for chapter in analysis.chapters:
            all_scenes.extend(chapter.scenes)
        
        # Split into episodes
        episodes = []
        for i in range(0, len(all_scenes), scenes_per_episode):
            episode_scenes = all_scenes[i:i + scenes_per_episode]
            episodes.append(episode_scenes)
        
        return episodes
    
    def generate_episode(
        self,
        episode_scenes: List[Scene],
        episode_number: int,
        book_title: str,
        output_dir: Path,
        language: str = "en",
        aspect_ratio: str = "16:9",
        progress_callback=None,
    ) -> Path:
        """
        Generate a video for a single episode.
        
        This is meant to be called from the main pipeline for each episode.
        """
        # This would integrate with the main TextToVideo pipeline
        # For now, return the scenes formatted for processing
        
        episode_data = {
            "episode": episode_number,
            "book_title": book_title,
            "scenes": [
                {
                    "index": s.index,
                    "narration": s.narration,
                    "image_prompt": s.image_prompt,
                    "characters": s.characters,
                    "setting": s.setting,
                    "mood": s.mood,
                }
                for s in episode_scenes
            ]
        }
        
        output_path = output_dir / f"episode_{episode_number:02d}_data.json"
        output_path.write_text(json.dumps(episode_data, indent=2, ensure_ascii=False))
        
        return output_path


def process_book_file(
    file_path: str,
    output_dir: str = None,
    max_scenes: int = 100,
    style_prefix: str = "",
) -> BookAnalysis:
    """
    Convenience function to process a book file.
    
    Supports: .txt, .md files
    """
    path = Path(file_path)
    
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Read text
    text = path.read_text(encoding='utf-8')
    title = path.stem.replace('_', ' ').replace('-', ' ').title()
    
    # Process
    processor = LargeTextProcessor()
    analysis = processor.process_book(
        text=text,
        title=title,
        max_scenes_total=max_scenes,
        style_prefix=style_prefix,
    )
    
    # Save analysis
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        analysis_file = output_path / f"{path.stem}_analysis.json"
        
        analysis_data = {
            "title": analysis.title,
            "author": analysis.author,
            "total_chapters": analysis.total_chapters,
            "total_words": analysis.total_words,
            "estimated_duration_minutes": analysis.estimated_duration,
            "recommended_episodes": analysis.recommended_episodes,
            "characters": [
                {"name": c.name, "description": c.description, "traits": c.visual_traits}
                for c in analysis.characters
            ],
            "chapters": [
                {
                    "index": ch.index,
                    "title": ch.title,
                    "word_count": ch.word_count,
                    "scene_count": len(ch.scenes),
                }
                for ch in analysis.chapters
            ],
        }
        
        analysis_file.write_text(json.dumps(analysis_data, indent=2, ensure_ascii=False))
        print(f"📁 Analysis saved to: {analysis_file}")
    
    return analysis


# Streaming processor for very large files
class StreamingBookProcessor:
    """
    Memory-efficient streaming processor for very large books.
    Processes chapters one at a time without loading entire book into memory.
    """
    
    def __init__(self, ollama_model: str = "gemma3:4b"):
        self.processor = LargeTextProcessor(ollama_model)
    
    def stream_chapters(
        self, 
        file_path: str,
        chunk_size: int = 100000
    ) -> Generator[Chapter, None, None]:
        """
        Stream chapters from a file without loading entire content.
        
        Args:
            file_path: Path to text file
            chunk_size: Read buffer size
        
        Yields:
            Chapter objects
        """
        path = Path(file_path)
        
        with open(path, 'r', encoding='utf-8') as f:
            buffer = ""
            chapter_pattern = re.compile(
                r'^(chapter|part|section)\s+\d+',
                re.IGNORECASE | re.MULTILINE
            )
            
            chapter_idx = 0
            current_chapter_start = 0
            
            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    # Yield final chapter
                    if buffer.strip():
                        yield Chapter(
                            index=chapter_idx,
                            title=f"Chapter {chapter_idx + 1}",
                            content=buffer.strip(),
                            start_position=current_chapter_start,
                            word_count=len(buffer.split()),
                        )
                    break
                
                buffer += chunk
                
                # Look for chapter breaks
                matches = list(chapter_pattern.finditer(buffer))
                
                if len(matches) > 1:
                    # Found a new chapter start
                    for i in range(len(matches) - 1):
                        chapter_content = buffer[matches[i].start():matches[i+1].start()]
                        chapter_title = matches[i].group().strip()
                        
                        yield Chapter(
                            index=chapter_idx,
                            title=chapter_title,
                            content=chapter_content.strip(),
                            start_position=current_chapter_start,
                            word_count=len(chapter_content.split()),
                        )
                        
                        chapter_idx += 1
                        current_chapter_start += len(chapter_content)
                    
                    # Keep last chapter in buffer (incomplete)
                    buffer = buffer[matches[-1].start():]
    
    def process_streaming(
        self,
        file_path: str,
        output_dir: str,
        scenes_per_chapter: int = 5,
        style_prefix: str = "",
        progress_callback=None,
    ) -> List[Path]:
        """
        Process a large book with streaming, generating output incrementally.
        
        Args:
            file_path: Path to book file
            output_dir: Output directory
            scenes_per_chapter: Scenes per chapter
            style_prefix: Style tokens
            progress_callback: Progress callback
        
        Returns:
            List of generated episode data files
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        episode_files = []
        all_scenes = []
        current_episode = 1
        
        for chapter in self.stream_chapters(file_path):
            print(f"📖 Processing: {chapter.title}")
            
            # Process chapter
            scenes = self.processor.process_chapter(
                chapter=chapter,
                scenes_per_chapter=scenes_per_chapter,
                style_prefix=style_prefix,
            )
            
            all_scenes.extend(scenes)
            
            # Check if we have enough for an episode
            if len(all_scenes) >= self.processor.MAX_SCENES_PER_EPISODE:
                # Save episode
                episode_scenes = all_scenes[:self.processor.MAX_SCENES_PER_EPISODE]
                episode_file = output_path / f"episode_{current_episode:02d}.json"
                
                episode_data = {
                    "episode": current_episode,
                    "scenes": [
                        {
                            "narration": s.narration,
                            "image_prompt": s.image_prompt,
                            "characters": s.characters,
                        }
                        for s in episode_scenes
                    ]
                }
                
                episode_file.write_text(json.dumps(episode_data, indent=2))
                episode_files.append(episode_file)
                
                print(f"   ✅ Saved episode {current_episode}")
                
                all_scenes = all_scenes[self.processor.MAX_SCENES_PER_EPISODE:]
                current_episode += 1
        
        # Save remaining scenes as final episode
        if all_scenes:
            episode_file = output_path / f"episode_{current_episode:02d}.json"
            episode_data = {
                "episode": current_episode,
                "scenes": [
                    {
                        "narration": s.narration,
                        "image_prompt": s.image_prompt,
                        "characters": s.characters,
                    }
                    for s in all_scenes
                ]
            }
            episode_file.write_text(json.dumps(episode_data, indent=2))
            episode_files.append(episode_file)
        
        return episode_files
