"""Tests for ReplicateForceAlignTranscriber."""
import importlib
import pytest
from unittest.mock import MagicMock, patch

from karaoke_gen.lyrics_transcriber.transcribers import replicate_force_align
from karaoke_gen.lyrics_transcriber.transcribers.replicate_force_align import (
    ReplicateForceAlignConfig,
    ReplicateForceAlignTranscriber,
)
from karaoke_gen.lyrics_transcriber.transcribers.base_transcriber import TranscriptionError
from karaoke_gen.lyrics_transcriber.types import TranscriptionData


@pytest.fixture
def config():
    return ReplicateForceAlignConfig(
        api_token="r8_test",
        reference_text="Hello world\nGoodbye world",
    )


@pytest.fixture
def transcriber(config, tmp_path):
    logger = MagicMock()
    return ReplicateForceAlignTranscriber(
        cache_dir=tmp_path,
        config=config,
        logger=logger,
    )


def test_config_stores_api_token():
    config = ReplicateForceAlignConfig(api_token="r8_abc", reference_text="hi")
    assert config.api_token == "r8_abc"


def test_config_stores_reference_text():
    config = ReplicateForceAlignConfig(api_token="r8_abc", reference_text="hi there")
    assert config.reference_text == "hi there"


def test_get_name_returns_expected_string(transcriber):
    assert transcriber.get_name() == "ReplicateForceAlign"


def test_convert_result_format_produces_correct_segment_count(transcriber):
    """Reference text has 2 lines -> 2 segments."""
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "world", "start": 1.4, "end": 1.8},
        ]
    }
    result = transcriber._convert_result_format(raw_data)
    assert isinstance(result, TranscriptionData)
    assert len(result.segments) == 2


def test_convert_result_format_segment_word_counts(transcriber):
    """Each segment gets the words for that line."""
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "world", "start": 1.4, "end": 1.8},
        ]
    }
    result = transcriber._convert_result_format(raw_data)
    assert len(result.segments[0].words) == 2  # "Hello world"
    assert len(result.segments[1].words) == 2  # "Goodbye world"


def test_convert_result_format_segment_timing(transcriber):
    """Segment start/end times come from first/last word in the segment."""
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "world", "start": 1.4, "end": 1.8},
        ]
    }
    result = transcriber._convert_result_format(raw_data)
    assert result.segments[0].start_time == pytest.approx(0.1)
    assert result.segments[0].end_time == pytest.approx(0.9)
    assert result.segments[1].start_time == pytest.approx(1.0)
    assert result.segments[1].end_time == pytest.approx(1.8)


def test_convert_result_format_word_text(transcriber):
    """Word text is taken from aligned output."""
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "world", "start": 1.4, "end": 1.8},
        ]
    }
    result = transcriber._convert_result_format(raw_data)
    assert result.segments[0].words[0].text == "Hello"
    assert result.segments[0].words[1].text == "world"


def test_convert_result_format_skips_empty_lines(tmp_path):
    """Empty lines in reference text are skipped (no empty segments)."""
    config = ReplicateForceAlignConfig(
        api_token="r8_test",
        reference_text="Hello world\n\nGoodbye world",
    )
    t = ReplicateForceAlignTranscriber(
        cache_dir=tmp_path, config=config, logger=MagicMock()
    )
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "world", "start": 1.4, "end": 1.8},
        ]
    }
    result = t._convert_result_format(raw_data)
    assert len(result.segments) == 2


def test_convert_result_format_full_text(transcriber):
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            {"word": "world", "start": 0.5, "end": 0.9},
            {"word": "Goodbye", "start": 1.0, "end": 1.3},
            {"word": "world", "start": 1.4, "end": 1.8},
        ]
    }
    result = transcriber._convert_result_format(raw_data)
    assert "Hello" in result.text
    assert "Goodbye" in result.text


def test_perform_transcription_calls_replicate_run(transcriber, tmp_path):
    """Calls replicate.Client.run with correct model, audio, and text."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)

    word_stamps = [
        {"word": "Hello", "start": 0.1, "end": 0.4, "probability": 0.9},
        {"word": "world", "start": 0.5, "end": 0.9, "probability": 0.8},
        {"word": "Goodbye", "start": 1.0, "end": 1.3, "probability": 0.95},
        {"word": "world", "start": 1.4, "end": 1.8, "probability": 0.85},
    ]
    # Model returns a dict with "wordstamps" key
    fake_output = {"wordstamps": word_stamps}

    flac_path = str(audio_file) + ".replicate_tmp.flac"

    def write_stub_flac(*args, **kwargs):
        open(flac_path, "wb").close()

    with patch("karaoke_gen.lyrics_transcriber.transcribers.replicate_force_align.AudioSegment") as mock_seg, \
         patch("karaoke_gen.lyrics_transcriber.transcribers.replicate_force_align.replicate") as mock_replicate:
        mock_seg.from_file.return_value.set_channels.return_value.set_frame_rate.return_value.export.side_effect = write_stub_flac
        mock_client = MagicMock()
        mock_replicate.Client.return_value = mock_client
        mock_client.run.return_value = fake_output
        result = transcriber._perform_transcription(str(audio_file))

    mock_replicate.Client.assert_called_once_with(api_token="r8_test", timeout=600.0)
    call_args = mock_client.run.call_args
    assert call_args[0][0] == "cureau/force-align-wordstamps:44dedb84066ba1e00761f45c1003c5c19ed3b12ae9d42c1c1883ca4c016ffa85"
    assert call_args[1]["input"]["transcript"] == "Hello world\nGoodbye world"
    assert hasattr(call_args[1]["input"]["audio_file"], "read")
    assert "show_probabilities" not in call_args[1]["input"]
    assert call_args[1]["wait"] is False
    assert result == {"words": word_stamps}


def test_input_field_names_default(monkeypatch):
    """Without env vars, audio/transcript input field names use the documented defaults."""
    monkeypatch.delenv("REPLICATE_FORCE_ALIGN_AUDIO_FIELD", raising=False)
    monkeypatch.delenv("REPLICATE_FORCE_ALIGN_TRANSCRIPT_FIELD", raising=False)
    module = importlib.reload(replicate_force_align)
    try:
        assert module.AUDIO_INPUT_FIELD == "audio_file"
        assert module.TRANSCRIPT_INPUT_FIELD == "transcript"
    finally:
        importlib.reload(replicate_force_align)


def test_input_field_names_overridden_by_env_vars(monkeypatch):
    """REPLICATE_FORCE_ALIGN_AUDIO_FIELD / _TRANSCRIPT_FIELD override the default input keys."""
    monkeypatch.setenv("REPLICATE_FORCE_ALIGN_AUDIO_FIELD", "input_audio")
    monkeypatch.setenv("REPLICATE_FORCE_ALIGN_TRANSCRIPT_FIELD", "lyrics_text")
    module = importlib.reload(replicate_force_align)
    try:
        assert module.AUDIO_INPUT_FIELD == "input_audio"
        assert module.TRANSCRIPT_INPUT_FIELD == "lyrics_text"
    finally:
        importlib.reload(replicate_force_align)


def test_perform_transcription_raises_on_empty_output(transcriber, tmp_path):
    """Raises TranscriptionError if Replicate returns empty list."""
    audio_file = tmp_path / "audio.wav"
    audio_file.write_bytes(b"RIFF" + b"\x00" * 40)
    flac_path = str(audio_file) + ".replicate_tmp.flac"

    def write_stub_flac(*args, **kwargs):
        open(flac_path, "wb").close()

    with patch("karaoke_gen.lyrics_transcriber.transcribers.replicate_force_align.AudioSegment") as mock_seg, \
         patch("karaoke_gen.lyrics_transcriber.transcribers.replicate_force_align.replicate") as mock_replicate:
        mock_seg.from_file.return_value.set_channels.return_value.set_frame_rate.return_value.export.side_effect = write_stub_flac
        mock_client = MagicMock()
        mock_replicate.Client.return_value = mock_client
        mock_client.run.return_value = {"wordstamps": []}
        with pytest.raises(TranscriptionError):
            transcriber._perform_transcription(str(audio_file))


def test_convert_result_format_partial_alignment_does_not_crash(transcriber):
    """When fewer aligned words are returned than expected, produces partial segments without raising."""
    raw_data = {
        "words": [
            {"word": "Hello", "start": 0.1, "end": 0.4},
            # "world" missing — alignment gave only 1 of 2 words for line 1
        ]
    }
    result = transcriber._convert_result_format(raw_data)
    # First segment has 1 word, second segment is empty (skipped by `if not line_aligned`)
    assert len(result.segments) == 1
    assert len(result.segments[0].words) == 1
