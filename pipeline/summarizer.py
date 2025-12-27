"""
AutoKatha Book Summarizer
Intelligent chapter-by-chapter summarization with importance weighting.
Designed for creating 30-35 minute YouTube recap/summary videos.
"""
import re
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import requests


@dataclass
class ChapterSummary:
    """Summary of a single chapter."""
    chapter_index: int
    chapter_title: str
    original_word_count: int
    summary: str
    summary_word_count: int
    importance_score: float  # 0.0 to 1.0
    key_events: List[str] = field(default_factory=list)
    characters_involved: List[str] = field(default_factory=list)


@dataclass
class BookSummary:
    """Complete summarized book ready for video generation."""
    title: str
    author: str
    total_chapters: int
    original_word_count: int
    summary_word_count: int
    chapter_summaries: List[ChapterSummary]
    combined_narrative: str
    main_characters: List[str]
    themes: List[str]
    estimated_video_duration: float  # minutes


class BookSummarizer:
    """
    Summarizes books chapter-by-chapter with importance weighting.
    
    Strategy:
    - Beginning chapters (setup): Higher importance (600-800 words)
    - Middle chapters: Standard importance (300-500 words)
    - Climax/ending chapters: Higher importance (600-800 words)
    - Importance is position-based for speed and reliability
    """
    
    # Target word counts based on importance
    IMPORTANCE_WORD_COUNTS = {
        "high": (600, 800),      # Beginning, climax, ending
        "medium": (400, 500),    # Key middle chapters
        "low": (250, 350),       # Transitional chapters
    }
    
    # Video timing constants
    WORDS_PER_MINUTE_NARRATION = 150  # Average speaking rate
    TARGET_VIDEO_MINUTES = 32         # Target 30-35 min
    
    def __init__(
        self,
        groq_api_key: str = None,
        groq_model: str = None,
        ollama_model: str = None,
        ollama_host: str = None,
    ):
        import config
        
        self.groq_api_key = groq_api_key or config.GROQ_API_KEY
        self.groq_model = groq_model or config.GROQ_MODEL
        self.ollama_model = ollama_model or config.OLLAMA_MODEL
        self.ollama_host = ollama_host or config.OLLAMA_HOST
        
        # Determine backend
        self.backend = "groq" if self.groq_api_key else "ollama"
        
        if self.backend == "groq":
            print(f"📝 Summarizer using Groq API with {self.groq_model}")
        else:
            print(f"📝 Summarizer using Ollama with {self.ollama_model}")
    
    def _call_llm(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Call LLM API (Groq or Ollama)."""
        if self.backend == "groq":
            return self._call_groq(prompt, system, temperature)
        else:
            return self._call_ollama(prompt, system, temperature)
    
    def _call_groq(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Call Groq API."""
        print(f"🧠 Calling Groq API for summarization...")
        
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
                content = result["choices"][0]["message"]["content"]
                print(f"✅ Groq response received ({len(content)} chars)")
                return content
            elif "error" in result:
                print(f"\n{'='*60}")
                print(f"❌ GROQ API ERROR: {result['error']}")
                print(f"🔄 FALLING BACK TO OLLAMA ({self.ollama_model})")
                print(f"{'='*60}\n")
                self.backend = "ollama"
                return self._call_ollama(prompt, system, temperature)
            return ""
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ GROQ API EXCEPTION: {e}")
            print(f"🔄 FALLING BACK TO OLLAMA ({self.ollama_model})")
            print(f"{'='*60}\n")
            self.backend = "ollama"
            return self._call_ollama(prompt, system, temperature)
    
    def _call_ollama(self, prompt: str, system: str = None, temperature: float = 0.7) -> str:
        """Call Ollama API."""
        print(f"🤖 Calling Ollama ({self.ollama_model}) for summarization...")
        
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
                timeout=180  # Longer timeout for local models
            )
            result = response.json()
            content = result.get("message", {}).get("content", "")
            if content:
                print(f"✅ Ollama response received ({len(content)} chars)")
            else:
                print(f"⚠️ Ollama returned empty response: {result}")
            return content
        except Exception as e:
            print(f"\n{'='*60}")
            print(f"❌ OLLAMA ERROR: {e}")
            print(f"   Make sure Ollama is running: ollama serve")
            print(f"   Model needed: {self.ollama_model}")
            print(f"{'='*60}\n")
            return ""
    
    def calculate_chapter_importance(
        self,
        chapter_index: int,
        total_chapters: int,
        chapter_word_count: int = 0,
    ) -> Tuple[float, str]:
        """
        Calculate importance score for a chapter based on position.
        
        Returns:
            Tuple of (importance_score, importance_level)
        """
        if total_chapters <= 3:
            # Short book - all chapters are important
            return 1.0, "high"
        
        # Position-based importance
        position_ratio = chapter_index / (total_chapters - 1) if total_chapters > 1 else 0.5
        
        # First 15% = beginning (high importance)
        if position_ratio < 0.15:
            return 0.9, "high"
        
        # Last 20% = climax/ending (high importance)
        if position_ratio > 0.80:
            return 0.95, "high"
        
        # 70-80% range = building to climax (medium-high)
        if position_ratio > 0.70:
            return 0.75, "medium"
        
        # Middle chapters (lower importance, but still needed)
        return 0.5, "low"
    
    def get_target_word_count(self, importance_level: str) -> int:
        """Get target summary word count based on importance."""
        min_words, max_words = self.IMPORTANCE_WORD_COUNTS.get(
            importance_level, 
            self.IMPORTANCE_WORD_COUNTS["medium"]
        )
        # Return middle of range
        return (min_words + max_words) // 2
    
    def summarize_chapter(
        self,
        chapter_content: str,
        chapter_title: str,
        chapter_index: int,
        total_chapters: int,
        book_title: str = "",
        known_characters: List[str] = None,
    ) -> ChapterSummary:
        """
        Summarize a single chapter with importance weighting.
        
        Args:
            chapter_content: Full chapter text
            chapter_title: Chapter title
            chapter_index: 0-based chapter index
            total_chapters: Total number of chapters
            book_title: Title of the book
            known_characters: List of known character names for consistency
        
        Returns:
            ChapterSummary object
        """
        original_word_count = len(chapter_content.split())
        importance_score, importance_level = self.calculate_chapter_importance(
            chapter_index, total_chapters, original_word_count
        )
        target_words = self.get_target_word_count(importance_level)
        
        print(f"   📖 Chapter {chapter_index + 1}: {chapter_title}")
        print(f"      Original: {original_word_count:,} words → Target: {target_words} words ({importance_level} importance)")
        
        # Build character context
        char_context = ""
        if known_characters:
            char_context = f"\nKnown characters: {', '.join(known_characters[:10])}"
        
        system_prompt = f"""You are an expert book summarizer creating content for a YouTube video summary.
Your goal is to capture the essential plot points, character development, and emotional moments.

Book: {book_title}
Chapter: {chapter_title} (Chapter {chapter_index + 1} of {total_chapters})
Importance Level: {importance_level.upper()} - {"This is a key chapter (beginning/climax/ending)" if importance_level == "high" else "Standard chapter"}
{char_context}

Guidelines:
1. Write in narrative prose, not bullet points
2. Focus on KEY EVENTS that move the plot forward
3. Include important dialogue or emotional moments
4. Maintain character names consistently
5. Write approximately {target_words} words
6. Use present tense for immediacy
7. Make it engaging for viewers who haven't read the book"""

        # Truncate very long chapters
        max_input_chars = 15000
        truncated_content = chapter_content[:max_input_chars]
        if len(chapter_content) > max_input_chars:
            truncated_content += f"\n\n[Chapter continues for {len(chapter_content) - max_input_chars} more characters...]"
        
        user_prompt = f"""Summarize this chapter in approximately {target_words} words:

{truncated_content}

Also extract:
1. KEY_EVENTS: List 3-5 most important events (one line each)
2. CHARACTERS: List all characters who appear in this chapter

Format your response as:
SUMMARY:
[Your narrative summary here]

KEY_EVENTS:
- Event 1
- Event 2
- Event 3

CHARACTERS:
- Character 1
- Character 2"""

        response = self._call_llm(user_prompt, system_prompt, temperature=0.5)
        
        # Parse response
        summary = ""
        key_events = []
        characters = []
        
        if "SUMMARY:" in response:
            parts = response.split("KEY_EVENTS:")
            summary = parts[0].replace("SUMMARY:", "").strip()
            
            if len(parts) > 1:
                events_and_chars = parts[1].split("CHARACTERS:")
                events_text = events_and_chars[0].strip()
                key_events = [
                    e.strip().lstrip("- ").lstrip("• ")
                    for e in events_text.split("\n")
                    if e.strip() and not e.strip().startswith("KEY_EVENTS")
                ]
                
                if len(events_and_chars) > 1:
                    chars_text = events_and_chars[1].strip()
                    characters = [
                        c.strip().lstrip("- ").lstrip("• ")
                        for c in chars_text.split("\n")
                        if c.strip() and not c.strip().startswith("CHARACTERS")
                    ]
        else:
            # Fallback - use entire response as summary
            summary = response.strip()
        
        summary_word_count = len(summary.split())
        print(f"      ✅ Summary: {summary_word_count} words, {len(key_events)} events, {len(characters)} characters")
        
        return ChapterSummary(
            chapter_index=chapter_index,
            chapter_title=chapter_title,
            original_word_count=original_word_count,
            summary=summary,
            summary_word_count=summary_word_count,
            importance_score=importance_score,
            key_events=key_events[:5],  # Limit to 5
            characters_involved=characters[:10],  # Limit to 10
        )
    
    def combine_summaries(
        self,
        chapter_summaries: List[ChapterSummary],
        book_title: str,
        author: str,
    ) -> str:
        """
        Combine chapter summaries into a cohesive narrative.
        
        Args:
            chapter_summaries: List of chapter summaries
            book_title: Book title
            author: Author name
        
        Returns:
            Combined narrative text
        """
        print(f"\n📚 Combining {len(chapter_summaries)} chapter summaries...")
        
        # Build combined text
        combined_parts = []
        for cs in chapter_summaries:
            combined_parts.append(f"## {cs.chapter_title}\n\n{cs.summary}")
        
        raw_combined = "\n\n".join(combined_parts)
        total_words = sum(cs.summary_word_count for cs in chapter_summaries)
        
        # Calculate target for final narrative
        target_final_words = min(
            total_words,  # Don't expand
            self.TARGET_VIDEO_MINUTES * self.WORDS_PER_MINUTE_NARRATION
        )
        
        print(f"   Combined summaries: {total_words:,} words")
        print(f"   Target final narrative: {target_final_words:,} words (~{target_final_words // self.WORDS_PER_MINUTE_NARRATION} min)")
        
        # If already at target, return as-is
        if total_words <= target_final_words * 1.1:
            return raw_combined
        
        # Otherwise, ask LLM to polish and condense
        system_prompt = f"""You are creating the final narrative for a YouTube book summary video.
Book: "{book_title}" by {author}

Your task:
1. Combine these chapter summaries into ONE flowing narrative
2. Smooth transitions between chapters
3. Remove redundancy while keeping key plot points
4. Maintain consistent character names
5. Keep the narrative engaging and dramatic
6. Target approximately {target_final_words} words"""

        user_prompt = f"""Polish and combine these chapter summaries into a cohesive narrative:

{raw_combined}

Create a smooth, engaging narrative of approximately {target_final_words} words."""

        polished = self._call_llm(user_prompt, system_prompt, temperature=0.6)
        
        if polished and len(polished.split()) > 100:
            print(f"   ✅ Polished narrative: {len(polished.split())} words")
            return polished
        else:
            print(f"   ⚠️ Using raw combined summaries (polish failed)")
            return raw_combined
    
    def extract_main_characters(
        self,
        chapter_summaries: List[ChapterSummary],
    ) -> List[str]:
        """Extract main characters from all chapter summaries."""
        character_counts = {}
        for cs in chapter_summaries:
            for char in cs.characters_involved:
                char_lower = char.lower()
                character_counts[char_lower] = character_counts.get(char_lower, 0) + 1
        
        # Sort by frequency and return top characters
        sorted_chars = sorted(character_counts.items(), key=lambda x: x[1], reverse=True)
        return [char for char, count in sorted_chars[:10] if count >= 2]
    
    def extract_themes(
        self,
        combined_narrative: str,
        book_title: str,
    ) -> List[str]:
        """Extract main themes from the narrative."""
        system_prompt = "You are a literary analyst. Extract the main themes from this book summary."
        
        user_prompt = f"""From this summary of "{book_title}", list the 3-5 main themes:

{combined_narrative[:5000]}

Return ONLY the themes, one per line, no numbering or bullets."""

        response = self._call_llm(user_prompt, system_prompt, temperature=0.3)
        
        themes = [
            t.strip().lstrip("- ").lstrip("• ").lstrip("1234567890. ")
            for t in response.split("\n")
            if t.strip() and len(t.strip()) < 50
        ]
        
        return themes[:5]
    
    def summarize_book(
        self,
        chapters: List[Dict],  # [{"title": str, "content": str}, ...]
        book_title: str,
        author: str = "Unknown",
        progress_callback=None,
    ) -> BookSummary:
        """
        Summarize an entire book chapter by chapter.
        
        Args:
            chapters: List of chapter dicts with "title" and "content" keys
            book_title: Title of the book
            author: Author name
            progress_callback: Optional callback(current, total, message)
        
        Returns:
            BookSummary object ready for video generation
        """
        print(f"\n{'='*60}")
        print(f"📚 SUMMARIZING: {book_title} by {author}")
        print(f"   {len(chapters)} chapters to process")
        print(f"{'='*60}\n")
        
        total_chapters = len(chapters)
        original_word_count = sum(len(ch.get("content", "").split()) for ch in chapters)
        
        # Process each chapter
        chapter_summaries = []
        known_characters = []
        
        for i, chapter in enumerate(chapters):
            if progress_callback:
                progress = int(90 * (i / total_chapters))
                progress_callback(progress, 100, f"Summarizing chapter {i+1}/{total_chapters}...")
            
            summary = self.summarize_chapter(
                chapter_content=chapter.get("content", ""),
                chapter_title=chapter.get("title", f"Chapter {i+1}"),
                chapter_index=i,
                total_chapters=total_chapters,
                book_title=book_title,
                known_characters=known_characters,
            )
            
            chapter_summaries.append(summary)
            
            # Update known characters for consistency
            for char in summary.characters_involved:
                if char.lower() not in [c.lower() for c in known_characters]:
                    known_characters.append(char)
        
        # Combine into final narrative
        if progress_callback:
            progress_callback(92, 100, "Combining summaries...")
        
        combined_narrative = self.combine_summaries(
            chapter_summaries=chapter_summaries,
            book_title=book_title,
            author=author,
        )
        
        # Extract metadata
        if progress_callback:
            progress_callback(95, 100, "Extracting themes...")
        
        main_characters = self.extract_main_characters(chapter_summaries)
        themes = self.extract_themes(combined_narrative, book_title)
        
        # Calculate metrics
        summary_word_count = len(combined_narrative.split())
        estimated_duration = summary_word_count / self.WORDS_PER_MINUTE_NARRATION
        
        if progress_callback:
            progress_callback(100, 100, "Summarization complete!")
        
        print(f"\n{'='*60}")
        print(f"✅ SUMMARIZATION COMPLETE")
        print(f"   Original: {original_word_count:,} words")
        print(f"   Summary: {summary_word_count:,} words ({100*summary_word_count/original_word_count:.1f}% of original)")
        print(f"   Estimated video: {estimated_duration:.1f} minutes")
        print(f"   Main characters: {', '.join(main_characters[:5])}")
        print(f"   Themes: {', '.join(themes)}")
        print(f"{'='*60}\n")
        
        return BookSummary(
            title=book_title,
            author=author,
            total_chapters=total_chapters,
            original_word_count=original_word_count,
            summary_word_count=summary_word_count,
            chapter_summaries=chapter_summaries,
            combined_narrative=combined_narrative,
            main_characters=main_characters,
            themes=themes,
            estimated_video_duration=estimated_duration,
        )


def summarize_from_text(
    text: str,
    title: str = "Book",
    author: str = "Unknown",
) -> BookSummary:
    """
    Convenience function to summarize a book from raw text.
    Automatically detects chapters.
    """
    from pipeline.story_processor import detect_chapters
    
    # Detect chapters
    chapter_markers = detect_chapters(text)
    
    if not chapter_markers:
        # Create artificial chapters
        words = text.split()
        chunk_size = 3000
        chapters = []
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            chapters.append({
                "title": f"Part {len(chapters) + 1}",
                "content": chunk,
            })
    else:
        chapters = []
        for i, (chapter_title, start_pos) in enumerate(chapter_markers):
            end_pos = chapter_markers[i + 1][1] if i + 1 < len(chapter_markers) else len(text)
            chapters.append({
                "title": chapter_title,
                "content": text[start_pos:end_pos].strip(),
            })
    
    summarizer = BookSummarizer()
    return summarizer.summarize_book(chapters, title, author)
