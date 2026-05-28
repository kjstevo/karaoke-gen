"""Handles adding countdown intro to songs that start too quickly for karaoke singers."""

import logging
import os
import subprocess
from typing import List, Optional, Tuple
from copy import deepcopy

import numpy as np

from karaoke_gen.lyrics_transcriber.types import CorrectionResult, LyricsSegment, Word
from karaoke_gen.lyrics_transcriber.utils.word_utils import WordUtils


class CountdownProcessor:
    """
    Processes corrected lyrics and audio to add countdown intro for songs that start too quickly.
    
    For songs where vocals start within the first 3 seconds, this processor:
    - Adds 3 seconds of silence to the start of the audio file
    - Shifts all timestamps in corrected lyrics by 3 seconds
    - Adds a countdown segment "3... 2... 1..." spanning 0.1s to 2.9s
    """

    # Configuration constants
    COUNTDOWN_THRESHOLD_SECONDS = 3.0  # Trigger countdown if first word is within this time
    COUNTDOWN_PADDING_SECONDS = 3.0    # Amount of silence to add
    COUNTDOWN_START_TIME = 0.1         # When countdown text starts
    COUNTDOWN_END_TIME = 2.9           # When countdown text ends
    COUNTDOWN_TEXT = "3... 2... 1..."  # The countdown text to display

    def __init__(
        self,
        cache_dir: str,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize CountdownProcessor.

        Args:
            cache_dir: Directory for temporary files (padded audio)
            logger: Optional logger instance
        """
        self.cache_dir = cache_dir
        self.logger = logger or logging.getLogger(__name__)

        # Ensure cache directory exists
        os.makedirs(self.cache_dir, exist_ok=True)

    def process(
        self,
        correction_result: CorrectionResult,
        audio_filepath: str,
    ) -> Tuple[CorrectionResult, str, bool, float]:
        """
        Process correction result and audio file, adding countdown if needed.

        Args:
            correction_result: The CorrectionResult to potentially modify
            audio_filepath: Path to the original audio file

        Returns:
            Tuple of:
            - potentially modified CorrectionResult
            - potentially padded audio filepath
            - whether padding was added (bool)
            - amount of padding in seconds (float)
        """
        # Check if countdown is needed
        if not self._needs_countdown(correction_result):
            self.logger.info(
                f"First word starts after {self.COUNTDOWN_THRESHOLD_SECONDS}s - "
                "no countdown needed"
            )
            return correction_result, audio_filepath, False, 0.0

        self.logger.info(
            f"First word starts within {self.COUNTDOWN_THRESHOLD_SECONDS}s - "
            "adding countdown intro"
        )

        # Detect the true first vocal onset to correct Whisper's 0.0 snap bug
        first_vocal_onset = self._get_true_first_vocal_onset(audio_filepath)

        # Create padded audio file
        padded_audio_path = self._create_padded_audio(audio_filepath)

        # Create modified correction result with adjusted timestamps
        modified_result = self._add_countdown_to_result(correction_result, first_vocal_onset=first_vocal_onset)

        self.logger.info(
            f"Countdown intro added successfully. "
            f"Padded audio: {os.path.basename(padded_audio_path)}"
        )

        return modified_result, padded_audio_path, True, self.COUNTDOWN_PADDING_SECONDS

    def _needs_countdown(self, correction_result: CorrectionResult) -> bool:
        """
        Check if the song needs a countdown intro.

        Args:
            correction_result: The correction result to check

        Returns:
            True if first word starts within threshold, False otherwise
        """
        if not correction_result.corrected_segments:
            return False

        # Find the first segment with words
        for segment in correction_result.corrected_segments:
            if segment.words:
                first_word_start = segment.words[0].start_time
                return first_word_start < self.COUNTDOWN_THRESHOLD_SECONDS

        return False

    def _create_padded_audio(self, audio_filepath: str) -> str:
        """
        Create a new audio file with silence prepended.

        Args:
            audio_filepath: Path to original audio file

        Returns:
            Path to padded audio file

        Raises:
            FileNotFoundError: If input audio file doesn't exist
            RuntimeError: If ffmpeg command fails
        """
        if not os.path.isfile(audio_filepath):
            raise FileNotFoundError(f"Audio file not found: {audio_filepath}")

        # Create output path in cache directory
        # Always use .flac extension since we encode with FLAC codec for quality
        basename = os.path.basename(audio_filepath)
        name, _ = os.path.splitext(basename)
        padded_filename = f"{name}_padded.flac"
        padded_filepath = os.path.join(self.cache_dir, padded_filename)

        self.logger.info(f"Creating padded audio file: {padded_filename}")

        # Build ffmpeg command to prepend silence
        # We use the anullsrc filter to generate silence and concat it with the original audio
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-hide_banner",
            "-loglevel", "error",
            "-f", "lavfi",
            "-t", str(self.COUNTDOWN_PADDING_SECONDS),
            "-i", f"anullsrc=channel_layout=stereo:sample_rate=44100",
            "-i", audio_filepath,
            "-filter_complex", "[0:a][1:a]concat=n=2:v=0:a=1[out]",
            "-map", "[out]",
            "-c:a", "flac",  # Use FLAC to preserve quality
            padded_filepath,
        ]

        try:
            self.logger.debug(f"Running ffmpeg command: {' '.join(cmd)}")
            output = subprocess.check_output(
                cmd,
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )
            self.logger.debug(f"ffmpeg output: {output}")

            if not os.path.isfile(padded_filepath):
                raise RuntimeError(
                    f"ffmpeg command succeeded but output file not created: {padded_filepath}"
                )

            return padded_filepath

        except subprocess.CalledProcessError as e:
            self.logger.error(f"Failed to create padded audio: {e.output}")
            raise RuntimeError(f"ffmpeg command failed: {e.output}")

    def _add_countdown_to_result(
        self, correction_result: CorrectionResult, first_vocal_onset: float = 0.0
    ) -> CorrectionResult:
        """
        Create a new CorrectionResult with countdown segment and adjusted timestamps.

        Args:
            correction_result: The original correction result
            first_vocal_onset: True onset time of the first vocal (seconds).
                When Whisper snaps the first word to 0.0, this corrects it before
                the countdown shift is applied.

        Returns:
            A new CorrectionResult with countdown and shifted timestamps
        """
        # Deep copy the result to avoid modifying the original
        modified_result = deepcopy(correction_result)

        # Correct first-word onset before shifting (fixes Whisper's 0.0 snap bug)
        if first_vocal_onset > 0.0:
            self._apply_first_word_correction(modified_result.corrected_segments, first_vocal_onset)
            if modified_result.resized_segments:
                self._apply_first_word_correction(modified_result.resized_segments, first_vocal_onset)

        # Shift all timestamps in corrected_segments
        self._shift_segments_timestamps(
            modified_result.corrected_segments,
            self.COUNTDOWN_PADDING_SECONDS
        )

        # Shift timestamps in resized_segments if they exist
        if modified_result.resized_segments:
            self._shift_segments_timestamps(
                modified_result.resized_segments,
                self.COUNTDOWN_PADDING_SECONDS
            )

        # Create and prepend countdown segment
        countdown_segment = self._create_countdown_segment()
        modified_result.corrected_segments.insert(0, countdown_segment)

        # Also add to resized_segments if present
        if modified_result.resized_segments:
            modified_result.resized_segments.insert(0, countdown_segment)

        self.logger.debug(
            f"Added countdown segment and shifted {len(modified_result.corrected_segments)} segments "
            f"by {self.COUNTDOWN_PADDING_SECONDS}s"
        )

        return modified_result

    def _shift_segments_timestamps(
        self,
        segments: List[LyricsSegment],
        offset_seconds: float
    ) -> None:
        """
        Shift all timestamps in segments by the given offset (in-place).

        Args:
            segments: List of segments to modify
            offset_seconds: Amount to shift timestamps (in seconds)
        """
        for segment in segments:
            # Shift segment timestamps
            segment.start_time += offset_seconds
            segment.end_time += offset_seconds

            # Shift all word timestamps
            for word in segment.words:
                word.start_time += offset_seconds
                word.end_time += offset_seconds

    def _get_true_first_vocal_onset(self, audio_filepath: str, analysis_duration: float = 15.0) -> float:
        """
        Detect when significant audio content first begins using RMS energy analysis.

        Whisper sometimes assigns start_time=0.0 to the first word even when there is
        silence or a quiet intro before the vocals. This method analyzes the raw audio
        to find the actual onset of meaningful content.

        Args:
            audio_filepath: Path to the audio file to analyze
            analysis_duration: How many seconds to analyze from the start

        Returns:
            Time in seconds of the first significant onset, or 0.0 if detection fails
        """
        sample_rate = 16000
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", audio_filepath,
            "-t", str(analysis_duration),
            "-ac", "1",
            "-ar", str(sample_rate),
            "-f", "s16le",
            "-",
        ]

        try:
            pcm = subprocess.check_output(cmd, stderr=subprocess.DEVNULL)
            audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
        except Exception as e:
            self.logger.warning(f"Audio onset detection failed, using 0.0: {e}")
            return 0.0

        if len(audio) < sample_rate:
            return 0.0

        window_size = int(sample_rate * 0.025)  # 25 ms
        hop_size = int(sample_rate * 0.010)      # 10 ms

        rms_values = []
        times = []
        for start in range(0, len(audio) - window_size, hop_size):
            chunk = audio[start : start + window_size]
            rms = float(np.sqrt(np.mean(chunk**2)))
            rms_values.append(rms)
            times.append(start / sample_rate)

        if not rms_values:
            return 0.0

        rms_arr = np.array(rms_values)
        max_rms = float(rms_arr.max())

        if max_rms < 1e-6:
            return 0.0

        # First frame exceeding 1% of peak energy
        threshold = max_rms * 0.01
        above = np.where(rms_arr > threshold)[0]

        if len(above) == 0:
            return 0.0

        onset = times[above[0]]
        self.logger.debug(f"Detected audio onset at {onset:.3f}s (threshold={threshold:.6f}, max_rms={max_rms:.6f})")
        return onset

    def _apply_first_word_correction(self, segments: List[LyricsSegment], t_true: float) -> None:
        """
        Correct only the first lyric word's start_time from Whisper's erroneous 0.0 to t_true.

        Whisper snaps the first word to 0.0 when the audio starts with silence or a quiet
        intro. Only the first word is corrected; all subsequent word timestamps from Whisper
        are accurately aligned and must not be changed.

        Args:
            segments: List of segments to correct (modified in-place)
            t_true: The true onset time to apply to the first word
        """
        for segment in segments:
            if not segment.words:
                continue
            first_word = segment.words[0]
            # Only correct when Whisper clearly snapped the timestamp to near-zero
            # and the detected onset is actually later (we never move timestamps backward)
            if first_word.start_time >= 0.5 or t_true <= first_word.start_time:
                break
            self.logger.info(
                f"Correcting first word '{first_word.text}' start_time: "
                f"{first_word.start_time:.3f}s → {t_true:.3f}s"
            )
            first_word.start_time = t_true
            if segment.start_time < 0.5:
                segment.start_time = t_true
            break

    def _create_countdown_segment(self) -> LyricsSegment:
        """
        Create a countdown segment with the countdown text.

        Returns:
            A LyricsSegment containing the countdown
        """
        # Create a single word for the countdown text
        countdown_word = Word(
            id=WordUtils.generate_id(),
            text=self.COUNTDOWN_TEXT,
            start_time=self.COUNTDOWN_START_TIME,
            end_time=self.COUNTDOWN_END_TIME,
            confidence=1.0,
            created_during_correction=True,
        )

        # Create the segment
        countdown_segment = LyricsSegment(
            id=WordUtils.generate_id(),
            text=self.COUNTDOWN_TEXT,
            words=[countdown_word],
            start_time=self.COUNTDOWN_START_TIME,
            end_time=self.COUNTDOWN_END_TIME,
        )

        return countdown_segment

    def has_countdown(self, correction_result: CorrectionResult) -> bool:
        """
        Check if a CorrectionResult already has a countdown segment.
        
        This is used to detect if countdown padding was applied to corrections
        that were loaded from a saved JSON file (where the padding state is not
        explicitly stored).

        Args:
            correction_result: The correction result to check

        Returns:
            True if the first segment is a countdown, False otherwise
        """
        if not correction_result.corrected_segments:
            return False

        first_segment = correction_result.corrected_segments[0]
        return first_segment.text == self.COUNTDOWN_TEXT

    def create_padded_audio_only(self, audio_filepath: str) -> str:
        """
        Create a padded audio file without modifying the correction result.
        
        This is used when loading existing corrections that already have countdown
        timestamps, but we need to create the padded audio file for video rendering.

        Args:
            audio_filepath: Path to original audio file

        Returns:
            Path to padded audio file

        Raises:
            FileNotFoundError: If input audio file doesn't exist
            RuntimeError: If ffmpeg command fails
        """
        return self._create_padded_audio(audio_filepath)

