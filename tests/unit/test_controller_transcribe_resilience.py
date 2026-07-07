"""Tests for controller.transcribe() resilience to a single provider failing."""
import pytest
from unittest.mock import MagicMock
from karaoke_gen.lyrics_transcriber.core.controller import LyricsTranscriber
from karaoke_gen.lyrics_transcriber.core.config import TranscriberConfig, OutputConfig
from karaoke_gen.lyrics_transcriber.types import TranscriptionData


def make_transcription_data(source: str) -> TranscriptionData:
    return TranscriptionData(segments=[], words=[], text="hello", source=source)


@pytest.fixture
def output_config(tmp_path):
    return OutputConfig(
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


def make_controller(tmp_path, output_config, transcribers):
    return LyricsTranscriber(
        audio_filepath=str(tmp_path / "audio.wav"),
        artist="Test",
        title="Song",
        transcriber_config=TranscriberConfig(),
        output_config=output_config,
        transcribers=transcribers,
        logger=MagicMock(),
    )


def test_transcribe_continues_after_one_provider_raises(tmp_path, output_config):
    """A provider raising should not prevent other providers from running."""
    failing = {"instance": MagicMock(), "priority": 2}
    failing["instance"].transcribe.side_effect = RuntimeError("boom")

    succeeding = {"instance": MagicMock(), "priority": 1}
    succeeding["instance"].transcribe.return_value = make_transcription_data("audioshake")

    controller = make_controller(
        tmp_path, output_config, {"replicate_force_align": failing, "audioshake": succeeding}
    )

    controller.transcribe()

    assert len(controller.results.transcription_results) == 1
    assert controller.results.transcription_results[0].name == "audioshake"


def test_transcribe_records_successful_result_when_no_providers_fail(tmp_path, output_config):
    succeeding = {"instance": MagicMock(), "priority": 1}
    succeeding["instance"].transcribe.return_value = make_transcription_data("audioshake")

    controller = make_controller(tmp_path, output_config, {"audioshake": succeeding})

    controller.transcribe()

    assert len(controller.results.transcription_results) == 1


def test_transcribe_warns_when_all_providers_fail(tmp_path, output_config):
    failing = {"instance": MagicMock(), "priority": 2}
    failing["instance"].transcribe.side_effect = RuntimeError("boom")

    controller = make_controller(tmp_path, output_config, {"replicate_force_align": failing})

    controller.transcribe()

    assert controller.results.transcription_results == []
