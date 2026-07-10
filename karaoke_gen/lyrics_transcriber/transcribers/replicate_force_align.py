from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydub import AudioSegment

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

    @staticmethod
    def _normalize_word_list(words: List[Any]) -> List[Any]:
        """Parse JSON string items and unwrap single-element list-of-list; skip empty strings."""
        result = []
        for w in words:
            if isinstance(w, str):
                w = w.strip()
                if not w:
                    continue
                w = json.loads(w)
            result.append(w)
        if len(result) == 1 and isinstance(result[0], list):
            result = result[0]
        return result

    def _perform_transcription(self, audio_filepath: str) -> Dict[str, Any]:
        """Call Replicate force-align API with audio file and reference text."""
        self.logger.info(f"Calling Replicate force-align for {audio_filepath}")
        client = replicate.Client(api_token=self.config.api_token, timeout=600.0)

        flac_path = audio_filepath + ".replicate_tmp.flac"
        AudioSegment.from_file(audio_filepath).set_channels(1).set_frame_rate(16000).export(flac_path, format="flac")
        flac_size_mb = os.path.getsize(flac_path) / (1024 * 1024)
        self.logger.info(f"Uploading mono 16kHz FLAC ({flac_size_mb:.1f} MB) to Replicate")
        try:
            with open(flac_path, "rb") as audio_file:
                # wait=False avoids the 60-second per-request read timeout that
                # client.run() adds when wait=True (Prefer: wait header). The
                # model takes several minutes so we need to poll instead.
                # show_probabilities is omitted: requesting it triggers the model's
                # probability-refinement pass, which crashes (torch.stft on a
                # zero-length tensor) whenever any word fails initial alignment.
                # We don't use the probability/confidence field downstream, and the
                # primary word-timing output completes without it.
                output = client.run(
                    MODEL_VERSION,
                    input={
                        "audio_file": audio_file,
                        "transcript": self.config.reference_text,
                    },
                    wait=False,
                )
        finally:
            if os.path.exists(flac_path):
                os.remove(flac_path)

        if output is None:
            raise TranscriptionError("Replicate force-align returned None output")

        # Model returns {"wordstamps": [...]} dict, not a bare list
        if isinstance(output, dict):
            output = output.get("wordstamps", [])

        word_list = list(output) if not isinstance(output, list) else output
        if not word_list:
            raise TranscriptionError("Replicate force-align returned empty output")

        word_list = self._normalize_word_list(word_list)

        self.logger.info(f"Replicate returned {len(word_list)} aligned words")
        return {"words": word_list}

    def _convert_result_format(self, raw_data: Dict[str, Any]) -> TranscriptionData:
        """Convert Replicate word list to TranscriptionData, grouped by reference text lines."""
        aligned_words = raw_data.get("words", [])
        aligned_words = self._normalize_word_list(aligned_words)
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

            if len(line_aligned) < line_word_count:
                self.logger.warning(
                    f"Line '{line[:40]}' expected {line_word_count} aligned words, "
                    f"got {len(line_aligned)}. Alignment may be incomplete."
                )

            seg_words = [
                Word(
                    id=WordUtils.generate_id(),
                    text=w["word"],
                    start_time=w["start"],
                    end_time=w["end"],
                    confidence=w.get("probability"),
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
