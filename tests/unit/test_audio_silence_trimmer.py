import os
import shutil
import subprocess

import pytest

from karaoke_gen.audio_silence_trimmer import (
    compute_trim_window,
    parse_input_duration,
    parse_silence_intervals,
    trim_silence,
)

FFMPEG_AVAILABLE = shutil.which("ffmpeg") is not None


class TestParseSilenceIntervals:
    def test_leading_and_trailing_silence_parses_both(self):
        stderr = (
            "[silencedetect @ 0000] silence_start: 0\n"
            "[silencedetect @ 0000] silence_end: 2.001 | silence_duration: 2.001\n"
            "[silencedetect @ 0000] silence_start: 4.002\n"
        )

        intervals = parse_silence_intervals(stderr)

        assert intervals == [(0.0, 2.001), (4.002, None)]

    def test_no_silence_detected_returns_empty(self):
        assert parse_silence_intervals("...no silence lines at all...\n") == []

    def test_only_middle_silence_leaves_both_ends_unflagged(self):
        stderr = (
            "[silencedetect @ 0000] silence_start: 3.0\n"
            "[silencedetect @ 0000] silence_end: 3.6 | silence_duration: 0.6\n"
        )

        assert parse_silence_intervals(stderr) == [(3.0, 3.6)]


class TestParseInputDuration:
    def test_normal_line_parses_correctly(self):
        stderr = "  Duration: 00:03:45.67, start: 0.000000, bitrate: 192 kb/s\n"

        assert parse_input_duration(stderr) == pytest.approx(3 * 60 + 45.67, abs=0.01)

    def test_missing_line_returns_none(self):
        assert parse_input_duration("no duration line here") is None


class TestComputeTrimWindow:
    def test_no_silence_intervals_keeps_whole_file(self):
        assert compute_trim_window([], total_duration=10) == (0.0, 10)

    def test_short_leading_and_trailing_silence_leaves_both_untouched(self):
        # Both runs are under the truncate-to floor, so neither needs any action.
        intervals = [(0.0, 0.3), (9.8, None)]

        assert compute_trim_window(intervals, total_duration=10) == (0.0, 10)

    def test_long_leading_silence_compresses_excess(self):
        # 2s leading silence, truncate_to=0.5, compress=0.5 -> kept = 0.5 + (2-0.5)*0.5 = 1.25
        intervals = [(0.0, 2.0)]

        start, end = compute_trim_window(intervals, total_duration=10)

        assert start == pytest.approx(0.75, abs=1e-6)
        assert end == 10

    def test_long_trailing_silence_compresses_excess(self):
        # Silence from 8.0 to EOF at 10.0 -> 2s trailing, kept = 1.25 -> end = 8.0 + 1.25 = 9.25
        intervals = [(8.0, None)]

        start, end = compute_trim_window(intervals, total_duration=10)

        assert start == 0
        assert end == pytest.approx(9.25, abs=1e-6)

    def test_middle_silence_only_leaves_file_untouched(self):
        # Silence strictly between real audio must never be touched, no matter how long.
        intervals = [(4.0, 7.0)]

        assert compute_trim_window(intervals, total_duration=10) == (0.0, 10)

    def test_entirely_silent_file_falls_back_to_untrimmed(self):
        # Degenerate case: one interval is simultaneously the "leading" and "trailing" run,
        # which would otherwise compute an inverted/empty window.
        intervals = [(0.0, None)]

        assert compute_trim_window(intervals, total_duration=10) == (0.0, 10)


@pytest.mark.skipif(not FFMPEG_AVAILABLE, reason="requires a real ffmpeg binary on PATH")
class TestTrimSilenceRealFfmpeg:
    def test_trims_silent_ends_but_keeps_tone_intact(self, temp_dir, mock_logger):
        # silence[0,2) -> tone[2,4) -> silence[4,6): a gated sine wave built via a single lavfi
        # filter graph, avoiding any concat-boundary artifacts a multi-file concat could introduce.
        #
        # Deliberately .m4a/AAC, not .flac: FLAC stores total sample count directly in its
        # STREAMINFO header, which -c:a copy remuxing (what trim_silence uses to avoid
        # re-encoding loss) doesn't rewrite -- so a FLAC file's reported Duration stays wrong
        # after trimming even though the actual audio content is correctly shortened. m4a's MP4
        # muxer recalculates duration correctly on remux, and matches what karaoke-gen actually
        # downloads in production.
        path = os.path.join(temp_dir, "gated_tone.m4a")
        subprocess.run(
            [
                "ffmpeg", "-y",
                "-f", "lavfi", "-i", "sine=frequency=440:duration=6",
                "-af", "volume=enable='not(between(t,2,4))':volume=0",
                "-c:a", "aac", "-b:a", "128k",
                path,
            ],
            check=True, capture_output=True,
        )

        trim_silence(path, mock_logger)

        probe = subprocess.run(["ffmpeg", "-i", path, "-f", "null", "-"], capture_output=True, text=True)
        new_duration = parse_input_duration(probe.stderr)

        assert new_duration is not None
        # Expected kept window is ~[0.75, 5.25] = 4.5s (both silent ends compressed from 2s to
        # 1.25s each), well under the original 6s and comfortably above the untouched 2s tone.
        assert new_duration < 5.5, f"Expected trimmed duration well under original 6s, got {new_duration}"
        assert new_duration > 3.5, f"Expected trimmed duration to still hold the full tone plus margin, got {new_duration}"

        # Duration alone doesn't prove the tone survived uncut -- confirm real energy at 440Hz remains.
        bandpass = subprocess.run(
            ["ffmpeg", "-i", path, "-af", "bandpass=f=440:width_type=h:w=50,volumedetect", "-f", "null", "-"],
            capture_output=True, text=True,
        )
        match = None
        for line in bandpass.stderr.splitlines():
            if "max_volume" in line:
                match = line
        assert match is not None, "Expected volumedetect output in ffmpeg stderr"

    def test_no_silence_to_trim_leaves_file_unchanged(self, temp_dir, mock_logger):
        path = os.path.join(temp_dir, "plain_tone.flac")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "lavfi", "-i", "sine=frequency=440:duration=2", "-c:a", "flac", path],
            check=True, capture_output=True,
        )

        with open(path, "rb") as f:
            original_bytes = f.read()

        trim_silence(path, mock_logger)

        with open(path, "rb") as f:
            assert f.read() == original_bytes
