"""Tests for LyricsTranscriber._transcribe_with_fallback cache-clearing behavior."""
import pytest
from unittest.mock import MagicMock

from karaoke_gen.lyrics_transcriber.core.controller import LyricsTranscriber
from karaoke_gen.lyrics_transcriber.core.config import TranscriberConfig, OutputConfig


@pytest.fixture
def controller(tmp_path):
    config = TranscriberConfig()
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
    return LyricsTranscriber(
        audio_filepath=str(tmp_path / "audio.wav"),
        artist="Test",
        title="Song",
        transcriber_config=config,
        output_config=output_config,
        logger=MagicMock(),
    )


def test_fallback_clears_fallback_transcriber_cache_before_transcribing(controller):
    """When the primary transcriber fails, the fallback's stale cache is cleared first."""
    primary_instance = MagicMock()
    primary_instance.transcribe.side_effect = RuntimeError("primary failed")
    primary_info = {"instance": primary_instance, "priority": 2}

    fallback_instance = MagicMock()
    fallback_info = {"instance": fallback_instance, "priority": 3}
    controller._transcriber_fallbacks["primary"] = {"name": "fallback", "info": fallback_info}

    call_order = []
    fallback_instance.clear_cache.side_effect = lambda *_: call_order.append("clear_cache")
    fallback_instance.transcribe.side_effect = lambda *_: call_order.append("transcribe") or "result"

    result, used_name, used_info = controller._transcribe_with_fallback("primary", primary_info)

    fallback_instance.clear_cache.assert_called_once_with(controller.audio_filepath)
    fallback_instance.transcribe.assert_called_once_with(controller.audio_filepath)
    assert call_order == ["clear_cache", "transcribe"]
    assert result == "result"
    assert used_name == "fallback"
    assert used_info is fallback_info


def test_no_fallback_registered_does_not_clear_any_cache(controller):
    """When the primary fails and no fallback is registered, nothing is cleared and result is None."""
    primary_instance = MagicMock()
    primary_instance.transcribe.side_effect = RuntimeError("primary failed")
    primary_info = {"instance": primary_instance, "priority": 2}

    result, used_name, used_info = controller._transcribe_with_fallback("primary", primary_info)

    assert result is None
    assert used_name == "primary"
    assert used_info is primary_info


def test_primary_success_does_not_touch_fallback(controller):
    """When the primary succeeds, the fallback (and its cache) is never touched."""
    primary_instance = MagicMock()
    primary_instance.transcribe.return_value = "primary_result"
    primary_info = {"instance": primary_instance, "priority": 2}

    fallback_instance = MagicMock()
    fallback_info = {"instance": fallback_instance, "priority": 3}
    controller._transcriber_fallbacks["primary"] = {"name": "fallback", "info": fallback_info}

    result, used_name, used_info = controller._transcribe_with_fallback("primary", primary_info)

    fallback_instance.clear_cache.assert_not_called()
    fallback_instance.transcribe.assert_not_called()
    assert result == "primary_result"
    assert used_name == "primary"
