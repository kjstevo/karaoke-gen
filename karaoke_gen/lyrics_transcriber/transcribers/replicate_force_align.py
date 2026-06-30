from dataclasses import dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from karaoke_gen.lyrics_transcriber.transcribers.base_transcriber import BaseTranscriber, TranscriptionError
from karaoke_gen.lyrics_transcriber.types import TranscriptionData, LyricsSegment, Word
from karaoke_gen.lyrics_transcriber.utils.word_utils import WordUtils
import replicate

MODEL_VERSION = "cureau/force-align-wordstamps:44dedb84066ba1e00761f45c1003c5c19ed3b12ae9d42c1c1883ca4c016ffa85"


@dataclass
class ReplicateForceAlignConfig:
    """Configuration for Replicate force-align transcription."""

    api_token: str
    reference_text: str


class ReplicateForceAlignTranscriber(BaseTranscriber):
    """Transcriber that uses Replicate force-align-wordstamps when reference lyrics exist."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        config: ReplicateForceAlignConfig,
        logger: Optional[logging.Logger] = None,
    ):
        super().__init__(cache_dir=cache_dir, logger=logger)
        self.config = config

    def get_name(self) -> str:
        return "ReplicateForceAlign"

    def _perform_transcription(self, audio_filepath: str) -> Dict[str, Any]:
        """Call Replicate force-align API with audio file and reference text."""
        self.logger.info(f"Calling Replicate force-align for {audio_filepath}")
        client = replicate.Client(api_token=self.config.api_token)

        with open(audio_filepath, "rb") as audio_file:
            output = client.run(
                MODEL_VERSION,
                input={
                    "audio": audio_file,
                    "text": self.config.reference_text,
                },
            )

        word_list = list(output) if not isinstance(output, list) else output
        if not word_list:
            raise TranscriptionError("Replicate force-align returned empty output")

        self.logger.info(f"Replicate returned {len(word_list)} aligned words")
        return {"words": word_list}

    def _convert_result_format(self, raw_data: Dict[str, Any]) -> TranscriptionData:
        """Convert Replicate word list to TranscriptionData, grouped by reference text lines."""
        aligned_words = raw_data.get("words", [])
        lines = [line for line in self.config.reference_text.splitlines() if line.strip()]

        segments: List[LyricsSegment] = []
        all_words: List[Word] = []
        word_cursor = 0

        for line in lines:
            line_word_count = len(line.split())
            line_aligned = aligned_words[word_cursor: word_cursor + line_word_count]
            word_cursor += line_word_count

            if not line_aligned:
                continue

            seg_words = [
                Word(
                    id=WordUtils.generate_id(),
                    text=w["word"],
                    start_time=w["start"],
                    end_time=w["end"],
                    confidence=w.get("score"),
                )
                for w in line_aligned
            ]
            all_words.extend(seg_words)
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),
                    text=" ".join(w.text for w in seg_words),
                    words=seg_words,
                    start_time=seg_words[0].start_time,
                    end_time=seg_words[-1].end_time,
                )
            )

        full_text = " ".join(seg.text for seg in segments)
        return TranscriptionData(
            segments=segments,
            words=all_words,
            text=full_text,
            source=self.get_name(),
            metadata={"model": MODEL_VERSION},
        )
