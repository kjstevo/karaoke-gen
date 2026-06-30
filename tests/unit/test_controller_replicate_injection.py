"""Tests for Replicate transcriber injection in controller.process()."""
import pytest
from unittest.mock import MagicMock, patch
from karaoke_gen.lyrics_transcriber.core.controller import LyricsTranscriber
from karaoke_gen.lyrics_transcriber.core.config import TranscriberConfig, LyricsConfig, OutputConfig
from karaoke_gen.lyrics_transcriber.types import LyricsData, LyricsSegment, LyricsMetadata
from karaoke_gen.lyrics_transcriber.utils.word_utils import WordUtils


def make_lyrics_data(text: str, source: str = "lrclib") -> LyricsData:
    segment = LyricsSegment(
        id=WordUtils.generate_id(),
        text=text,
        words=[],
        start_time=0.0,
        end_time=5.0,
    )
    return LyricsData(
        segments=[segment],
        metadata=LyricsMetadata(source=source, track_name="Test", artist_names=["Test"]),
        source=source,
    )


@pytest.fixture
def controller(tmp_path):
    config = TranscriberConfig(
        replicate_api_token="r8_test",
        runpod_api_key="rp_key",
        whisper_runpod_id="ep_id",
    )
    output_config = OutputConfig(
        output_styles_json="nonexistent.json",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        fetch_lyrics=False,
        run_transcription=True,
        run_correction=False,
        enable_review=False,
        generate_plain_text=False,
        generate_lrc=False,
        generate_cdg=False,
        render_video=False,
    )
    c = LyricsTranscriber(
        audio_filepath=str(tmp_path / "audio.wav"),
        artist="Test",
        title="Song",
        transcriber_config=config,
        output_config=output_config,
        logger=MagicMock(),
    )
    return c


def test_whisper_replaced_by_replicate_when_lyrics_found(controller):
    """When lyrics exist and replicate_api_token is set, whisper transcriber is removed."""
    controller.results.lyrics_results = {
        "lrclib": make_lyrics_data("Hello world\nGoodbye world"),
    }
    controller._inject_replicate_if_applicable()

    assert "whisper" not in controller.transcribers
    assert "replicate_force_align" in controller.transcribers


def test_replicate_transcriber_gets_longest_lyrics(controller):
    """Transcriber is initialized with the longest lyrics text."""
    short_lyrics = make_lyrics_data("Hi", "lrclib")
    long_lyrics = make_lyrics_data("Hello world\nGoodbye world\nExtra line here", "genius")
    controller.results.lyrics_results = {
        "lrclib": short_lyrics,
        "genius": long_lyrics,
    }
    controller._inject_replicate_if_applicable()

    replicate_instance = controller.transcribers["replicate_force_align"]["instance"]
    assert "Extra line here" in replicate_instance.config.reference_text


def test_replicate_not_injected_when_no_lyrics(controller):
    """Replicate transcriber is NOT injected when no lyrics were found."""
    controller.results.lyrics_results = {}
    controller._inject_replicate_if_applicable()

    assert "replicate_force_align" not in controller.transcribers
    assert "whisper" in controller.transcribers


def test_replicate_not_injected_when_no_token(tmp_path):
    """Replicate transcriber is NOT injected when replicate_api_token is absent."""
    config = TranscriberConfig(
        replicate_api_token=None,
        runpod_api_key="rp_key",
        whisper_runpod_id="ep_id",
    )
    output_config = OutputConfig(
        output_styles_json="nonexistent.json",
        output_dir=str(tmp_path),
        cache_dir=str(tmp_path / "cache"),
        fetch_lyrics=False,
        run_transcription=True,
        run_correction=False,
        enable_review=False,
        generate_plain_text=False,
        generate_lrc=False,
        generate_cdg=False,
        render_video=False,
    )
    c = LyricsTranscriber(
        audio_filepath=str(tmp_path / "audio.wav"),
        transcriber_config=config,
        output_config=output_config,
        logger=MagicMock(),
    )
    c.results.lyrics_results = {"lrclib": make_lyrics_data("Hello world")}
    c._inject_replicate_if_applicable()

    assert "replicate_force_align" not in c.transcribers
    assert "whisper" in c.transcribers


def test_audioshake_untouched_when_replicate_injected(controller):
    """AudioShake transcriber is not removed when Replicate is injected."""
    mock_audioshake = {"instance": MagicMock(), "priority": 1}
    controller.transcribers["audioshake"] = mock_audioshake
    controller.results.lyrics_results = {"lrclib": make_lyrics_data("Hello world")}
    controller._inject_replicate_if_applicable()

    assert "audioshake" in controller.transcribers
    assert controller.transcribers["audioshake"] is mock_audioshake
