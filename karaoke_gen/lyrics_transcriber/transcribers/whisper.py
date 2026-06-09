#! /usr/bin/env python3
from dataclasses import dataclass
import base64
import os
import json
import requests
import tempfile
import time
from typing import Optional, Dict, Any, Union
from pathlib import Path
from pydub import AudioSegment
from karaoke_gen.lyrics_transcriber.types import TranscriptionData, LyricsSegment, Word
from karaoke_gen.lyrics_transcriber.transcribers.base_transcriber import BaseTranscriber, TranscriptionError
from karaoke_gen.lyrics_transcriber.utils.word_utils import WordUtils


@dataclass
class WhisperConfig:
    """Configuration for Whisper transcription service."""

    runpod_api_key: Optional[str] = None
    endpoint_id: Optional[str] = None
    timeout_minutes: int = 10


class RunPodWhisperAPI:
    """Handles interactions with RunPod API."""

    def __init__(self, config: WhisperConfig, logger):
        self.config = config
        self.logger = logger
        self._validate_config()

    def _validate_config(self) -> None:
        """Validate API configuration."""
        if not self.config.runpod_api_key or not self.config.endpoint_id:
            raise ValueError("RunPod API key and endpoint ID must be provided")

    def submit_job(self, audio_base64: str) -> str:
        """Submit transcription job and return job ID."""
        run_url = f"https://api.runpod.ai/v2/{self.config.endpoint_id}/run"
        headers = {"Authorization": f"Bearer {self.config.runpod_api_key}"}

        payload = {
            "input": {
                "audio_file": audio_base64,
                "word_timestamps": True,
                "model": "medium",
                "temperature": 0.2,
                "best_of": 5,
                "compression_ratio_threshold": 2.8,
                "no_speech_threshold": 1,
                "condition_on_previous_text": True,
                "enable_vad": True,
            }
        }

        self.logger.info("Submitting transcription job...")
        response = requests.post(run_url, json=payload, headers=headers)

        self.logger.debug(f"Response status code: {response.status_code}")

        # Try to parse and log the JSON response
        try:
            response_json = response.json()
            self.logger.debug(f"Response content: {json.dumps(response_json, indent=2)}")
        except ValueError:
            self.logger.debug(f"Raw response content: {response.text}")
            # Re-raise if we can't parse the response at all
            raise TranscriptionError(f"Invalid JSON response: {response.text}")

        response.raise_for_status()
        return response_json["id"]

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get job status and results."""
        status_url = f"https://api.runpod.ai/v2/{self.config.endpoint_id}/status/{job_id}"
        headers = {"Authorization": f"Bearer {self.config.runpod_api_key}"}

        response = requests.get(status_url, headers=headers)
        response.raise_for_status()
        return response.json()

    def cancel_job(self, job_id: str) -> None:
        """Cancel a running job."""
        cancel_url = f"https://api.runpod.ai/v2/{self.config.endpoint_id}/cancel/{job_id}"
        headers = {"Authorization": f"Bearer {self.config.runpod_api_key}"}

        try:
            response = requests.post(cancel_url, headers=headers)
            response.raise_for_status()
        except Exception as e:
            self.logger.warning(f"Failed to cancel job {job_id}: {e}")

    def wait_for_job_result(self, job_id: str) -> Dict[str, Any]:
        """Poll for job completion and return results."""
        self.logger.info(f"Getting job result for job {job_id}")

        start_time = time.time()
        last_status_log = start_time
        timeout_seconds = self.config.timeout_minutes * 60

        while True:
            current_time = time.time()
            elapsed_time = current_time - start_time

            if elapsed_time > timeout_seconds:
                self.cancel_job(job_id)
                raise TranscriptionError(f"Transcription timed out after {self.config.timeout_minutes} minutes")

            # Log status periodically
            if current_time - last_status_log >= 60:
                self.logger.info(f"Still waiting for transcription... Elapsed time: {int(elapsed_time/60)} minutes")
                last_status_log = current_time

            status_data = self.get_job_status(job_id)

            if status_data["status"] == "COMPLETED":
                return status_data["output"]
            elif status_data["status"] == "FAILED":
                error_msg = status_data.get("error", "Unknown error")
                self.logger.error(f"Job failed with error: {error_msg}")
                raise TranscriptionError(f"Transcription failed: {error_msg}")

            time.sleep(5)


class AudioProcessor:
    """Handles audio file processing."""

    def __init__(self, logger):
        self.logger = logger

    def convert_to_flac(self, filepath: str) -> str:
        """Convert WAV to FLAC if needed to reduce encoded size."""
        if not filepath.lower().endswith(".wav"):
            return filepath

        self.logger.info("Converting WAV to FLAC...")
        audio = AudioSegment.from_wav(filepath)

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as tmp:
            flac_path = tmp.name
        audio.export(flac_path, format="flac")

        return flac_path

    def compress_for_transcription(self, filepath: str) -> str:
        """Compress to mono 16kHz MP3 to fit within RunPod's 10MB payload limit."""
        self.logger.info("Compressing audio to fit RunPod 10MB limit...")
        audio = AudioSegment.from_file(filepath).set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
            out_path = tmp.name
        audio.export(out_path, format="mp3", bitrate="64k")

        return out_path


class WhisperTranscriber(BaseTranscriber):
    """Transcription service using Whisper API via RunPod."""

    def __init__(
        self,
        cache_dir: Union[str, Path],
        config: Optional[WhisperConfig] = None,
        logger: Optional[Any] = None,
        runpod_client: Optional[RunPodWhisperAPI] = None,
        audio_processor: Optional[AudioProcessor] = None,
    ):
        """Initialize Whisper transcriber."""
        super().__init__(cache_dir=cache_dir, logger=logger)

        self.config = config or WhisperConfig(
            runpod_api_key=os.getenv("RUNPOD_API_KEY"),
            endpoint_id=os.getenv("WHISPER_RUNPOD_ID"),
        )

        self.runpod = runpod_client or RunPodWhisperAPI(self.config, self.logger)
        self.audio_processor = audio_processor or AudioProcessor(self.logger)

    def get_name(self) -> str:
        return "Whisper"

    def _perform_transcription(self, audio_filepath: str) -> TranscriptionData:
        """Actually perform the whisper transcription using Whisper API."""
        self.logger.info(f"Starting transcription for {audio_filepath}")

        # Start transcription and get results
        job_id = self.start_transcription(audio_filepath)
        result = self.get_transcription_result(job_id)
        return result

    def start_transcription(self, audio_filepath: str) -> str:
        """Prepare audio and start whisper transcription job."""
        audio_base64, temp_filepath = self._encode_audio_as_base64(audio_filepath)
        try:
            return self.runpod.submit_job(audio_base64)
        except Exception as e:
            if temp_filepath:
                self._cleanup_temporary_files(temp_filepath)
            raise TranscriptionError(f"Failed to submit job: {str(e)}") from e

    _RUNPOD_MAX_BYTES = 10 * 1024 * 1024  # 10MB RunPod payload limit

    def _encode_audio_as_base64(self, audio_filepath: str) -> tuple[str, Optional[str]]:
        """Convert audio file to base64 string for direct API submission.

        Compresses to mono 16kHz MP3 if the base64-encoded size would exceed RunPod's 10MB limit.
        """
        working_path = self.audio_processor.convert_to_flac(audio_filepath)
        flac_is_temp = working_path != audio_filepath

        # base64 expands size by ~4/3; check before encoding
        if os.path.getsize(working_path) * 4 // 3 > self._RUNPOD_MAX_BYTES:
            compressed_path = self.audio_processor.compress_for_transcription(working_path)
            if flac_is_temp:
                self._cleanup_temporary_files(working_path)
            working_path = compressed_path
            is_temp = True
        else:
            is_temp = flac_is_temp

        self.logger.info("Encoding audio as base64...")
        with open(working_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        return audio_base64, working_path if is_temp else None

    def get_transcription_result(self, job_id: str) -> Dict[str, Any]:
        """Poll for whisper job completion and return raw results."""
        raw_data = self.runpod.wait_for_job_result(job_id)

        # Add job_id to raw data for later use
        raw_data["job_id"] = job_id

        return raw_data

    def _convert_result_format(self, raw_data: Dict[str, Any]) -> TranscriptionData:
        """Convert Whisper API response to standard format."""
        self._validate_response(raw_data)

        job_id = raw_data.get("job_id")
        all_words = []

        # First collect all words from word_timestamps
        word_list = [
            Word(
                id=WordUtils.generate_id(),  # Generate unique ID for each word
                text=word["word"].strip(),
                start_time=word["start"],
                end_time=word["end"],
                confidence=word.get("probability"),  # Only set if provided
            )
            for word in raw_data.get("word_timestamps", [])
        ]
        all_words.extend(word_list)

        # Then create segments, using the words that fall within each segment's time range
        segments = []
        for seg in raw_data["segments"]:
            segment_words = [word for word in word_list if seg["start"] <= word.start_time < seg["end"]]
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),  # Generate unique ID for each segment
                    text=seg["text"].strip(),
                    words=segment_words,
                    start_time=seg["start"],
                    end_time=seg["end"],
                )
            )

        return TranscriptionData(
            segments=segments,
            words=all_words,
            text=raw_data["transcription"],
            source=self.get_name(),
            metadata={
                "language": raw_data.get("detected_language", "en"),
                "model": raw_data.get("model"),
                "job_id": job_id,
            },
        )

    def _cleanup_temporary_files(self, *filepaths: Optional[str]) -> None:
        """Clean up any temporary files that were created during transcription."""
        for filepath in filepaths:
            if filepath and os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    self.logger.debug(f"Cleaned up temporary file: {filepath}")
                except Exception as e:
                    self.logger.warning(f"Failed to clean up temporary file {filepath}: {e}")

    def _validate_response(self, raw_data: Dict[str, Any]) -> None:
        """Validate the response contains required fields."""
        if "segments" not in raw_data:
            raise TranscriptionError("Response missing required 'segments' field")
        if "transcription" not in raw_data:
            raise TranscriptionError("Response missing required 'transcription' field")
