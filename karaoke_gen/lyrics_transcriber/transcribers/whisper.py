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
                "align_output": True,
                "batch_size": 32,
                "debug": True,
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
                output = status_data.get("output")
                if output is None:
                    self.logger.error(f"RunPod COMPLETED response missing 'output': {status_data}")
                    raise TranscriptionError("RunPod job completed but returned no output")
                return output
            elif status_data["status"] == "FAILED":
                error_msg = status_data.get("error", "Unknown error")
                self.logger.error(f"Job failed with error: {error_msg}")
                raise TranscriptionError(f"Transcription failed: {error_msg}")

            time.sleep(5)


class AudioProcessor:
    """Handles audio file processing."""

    def __init__(self, logger):
        self.logger = logger

    def to_mono_16k_wav(self, filepath: str) -> str:
        """Convert audio to mono 16kHz WAV (the format the RunPod handler expects)."""
        self.logger.info("Converting audio to mono 16kHz WAV...")
        audio = AudioSegment.from_file(filepath).set_channels(1).set_frame_rate(16000)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name
        audio.export(out_path, format="wav")

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
        """Convert audio to mono 16kHz WAV data URI for RunPod submission."""
        wav_path = self.audio_processor.to_mono_16k_wav(audio_filepath)

        raw_size = os.path.getsize(wav_path)
        self.logger.info(f"Mono 16kHz WAV size: {raw_size / 1024 / 1024:.1f}MB (base64: ~{raw_size * 4 // 3 / 1024 / 1024:.1f}MB)")

        if raw_size * 4 // 3 > self._RUNPOD_MAX_BYTES:
            self.logger.warning(f"WAV exceeds RunPod limit — audio is unusually long ({raw_size / 1024 / 1024:.1f}MB mono 16kHz)")

        self.logger.info("Encoding audio as base64...")
        with open(wav_path, "rb") as f:
            audio_base64 = base64.b64encode(f.read()).decode("utf-8")

        data_uri = f"data:audio/wav;base64,{audio_base64}"
        return data_uri, wav_path

    def get_transcription_result(self, job_id: str) -> Dict[str, Any]:
        """Poll for whisper job completion and return raw results."""
        raw_data = self.runpod.wait_for_job_result(job_id)

        # Add job_id to raw data for later use
        raw_data["job_id"] = job_id

        return raw_data

    def _convert_result_format(self, raw_data: Dict[str, Any]) -> TranscriptionData:
        """Convert WhisperX API response to standard format.

        WhisperX returns segments with words embedded (start/end/word/score),
        not a separate word_timestamps list or a top-level transcription field.
        """
        self._validate_response(raw_data)

        job_id = raw_data.get("job_id")
        all_words = []
        segments = []

        for seg in raw_data["segments"]:
            seg_words = [
                Word(
                    id=WordUtils.generate_id(),
                    text=w["word"].strip(),
                    start_time=w["start"],
                    end_time=w["end"],
                    confidence=w.get("score"),
                )
                for w in seg.get("words", [])
                if w.get("start") is not None and w.get("end") is not None
            ]
            all_words.extend(seg_words)
            segments.append(
                LyricsSegment(
                    id=WordUtils.generate_id(),
                    text=seg["text"].strip(),
                    words=seg_words,
                    start_time=seg["start"],
                    end_time=seg["end"],
                )
            )

        full_text = " ".join(seg["text"].strip() for seg in raw_data["segments"])

        return TranscriptionData(
            segments=segments,
            words=all_words,
            text=full_text,
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
