"""Tests for RunPodWhisperAPI.submit_job payload."""
from unittest.mock import MagicMock, patch

from karaoke_gen.lyrics_transcriber.transcribers.whisper import RunPodWhisperAPI, WhisperConfig


def _make_api():
    config = WhisperConfig(runpod_api_key="rp_test", endpoint_id="ep_test")
    return RunPodWhisperAPI(config=config, logger=MagicMock())


def test_submit_job_forces_english_language():
    """The RunPod payload should force language=en rather than relying on auto-detection."""
    api = _make_api()

    mock_response = MagicMock()
    mock_response.json.return_value = {"id": "job-123"}
    mock_response.raise_for_status.return_value = None

    with patch("karaoke_gen.lyrics_transcriber.transcribers.whisper.requests.post", return_value=mock_response) as mock_post:
        job_id = api.submit_job("data:audio/wav;base64,AAAA")

    assert job_id == "job-123"
    call_kwargs = mock_post.call_args
    payload = call_kwargs.kwargs["json"]
    assert payload["input"]["language"] == "en"
    assert payload["input"]["align_output"] is True
    assert payload["input"]["audio_file"] == "data:audio/wav;base64,AAAA"
