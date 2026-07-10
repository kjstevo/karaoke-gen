"""Tests for BaseTranscriber.clear_cache."""
from unittest.mock import MagicMock

from karaoke_gen.lyrics_transcriber.transcribers.base_transcriber import BaseTranscriber
from karaoke_gen.lyrics_transcriber.types import TranscriptionData


class _FakeTranscriber(BaseTranscriber):
    def get_name(self) -> str:
        return "Fake"

    def _perform_transcription(self, audio_filepath):
        raise NotImplementedError

    def _convert_result_format(self, raw_data):
        raise NotImplementedError


def _make_transcriber(tmp_path):
    return _FakeTranscriber(cache_dir=tmp_path, logger=MagicMock())


def _make_audio_file(tmp_path):
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)
    return str(audio_file)


def test_clear_cache_removes_existing_raw_and_converted_files(tmp_path):
    transcriber = _make_transcriber(tmp_path)
    audio_filepath = _make_audio_file(tmp_path)
    file_hash = transcriber._get_file_hash(audio_filepath)
    raw_path = transcriber._get_cache_path(file_hash, "raw")
    converted_path = transcriber._get_cache_path(file_hash, "converted")

    with open(raw_path, "w") as f:
        f.write("{}")
    with open(converted_path, "w") as f:
        f.write("{}")

    transcriber.clear_cache(audio_filepath)

    assert not __import__("os").path.exists(raw_path)
    assert not __import__("os").path.exists(converted_path)


def test_clear_cache_is_noop_when_no_cache_exists(tmp_path):
    transcriber = _make_transcriber(tmp_path)
    audio_filepath = _make_audio_file(tmp_path)

    # Should not raise even though no cache files exist.
    transcriber.clear_cache(audio_filepath)


def test_clear_cache_does_not_touch_other_transcribers_cache(tmp_path):
    """clear_cache only removes this transcriber's own cache files, never another's."""
    transcriber = _make_transcriber(tmp_path)
    audio_filepath = _make_audio_file(tmp_path)
    file_hash = transcriber._get_file_hash(audio_filepath)

    other_cache_path = tmp_path / f"otherprovider_{file_hash}_converted.json"
    other_cache_path.write_text("{}")

    transcriber.clear_cache(audio_filepath)

    assert other_cache_path.exists()
