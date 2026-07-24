"""
Trims excess leading/trailing silence from the master downloaded/copied audio file, once, before
it's converted to WAV or read by anything downstream (lyrics transcription/alignment, audio
separation, video rendering). karaoke_gen.py's prep_single_track sets processed_track["input_media"]
once (from a local file copy or an AudioFetcher download) and then derives processed_track
["input_audio_wav"] from that same file via FileHandler.convert_to_wav -- trimming input_media in
place, before that conversion, keeps every downstream stage on one consistent timeline instead of
trimming a copy fed to just one stage. Trimming a copy used by only one stage would be unsafe for
leading silence specifically: it would shift word timestamps against a video still built from the
untrimmed original.

Added after a real report of misaligned lyrics traced to the forced-alignment model returning
suspiciously uniform per-word timings -- long dead air at the start/end of a track is a known way
to confuse a forced aligner into spreading words evenly rather than aligning them for real.
Parameters match Audacity's own "Truncate Silence" effect (Threshold/Duration/Truncate to/
Compress by), verified against a Replicate force-align model before this port existed, rather than
an invented threshold. Ported from the native C# rewrite's AudioSilenceTrimmer (KJMasterX.Core),
which carries the same parameters and the same leading/trailing-only contract.
"""

import logging
import os
import re
import subprocess

# Audacity "Truncate Silence" defaults, as verified against a force-align model before this port
# existed -- not independently derived, so kept exactly as given rather than "improved".
SILENCE_THRESHOLD_DB = -20.0
MIN_SILENCE_DURATION_SECONDS = 0.5
TRUNCATE_TO_SECONDS = 0.5
COMPRESS_RATIO = 0.5

# How close (in seconds) a detected interval's start/end must be to the file's own start/total
# duration to count as "leading"/"trailing" rather than silence strictly between real audio.
_BOUNDARY_EPSILON = 0.05

_SILENCE_LINE_RE = re.compile(r"silence_(start|end):\s*(-?[\d.]+)")
_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


def parse_silence_intervals(ffmpeg_stderr):
    """
    Parses ffmpeg's silencedetect filter output into a list of (start, end) tuples, in the order
    ffmpeg reported them. end is None only if ffmpeg never printed a matching silence_end line at
    all for that run (e.g. a build/version that omits it) -- compute_trim_window treats that the
    same as an end exactly at the file's total duration, since ffmpeg does normally flush a final
    silence_end at true end-of-file even if the signal never actually rises back above the
    threshold (verified empirically against a real ffmpeg build).
    """
    intervals = []
    for match in _SILENCE_LINE_RE.finditer(ffmpeg_stderr):
        kind, value = match.group(1), float(match.group(2))
        if kind == "start":
            intervals.append([value, None])
        elif intervals and intervals[-1][1] is None:
            intervals[-1][1] = value

    return [tuple(interval) for interval in intervals]


def parse_input_duration(ffmpeg_stderr):
    """Parses ffmpeg's own input-probe "Duration: HH:MM:SS.ss" line; None if it isn't present."""
    match = _DURATION_RE.search(ffmpeg_stderr)
    if not match:
        return None

    hours, minutes, seconds = int(match.group(1)), int(match.group(2)), float(match.group(3))
    return hours * 3600 + minutes * 60 + seconds


def _compress_excess(original_duration, truncate_to_seconds, compress_ratio):
    if original_duration <= truncate_to_seconds:
        return original_duration

    return truncate_to_seconds + (original_duration - truncate_to_seconds) * compress_ratio


def compute_trim_window(
    intervals,
    total_duration,
    truncate_to_seconds=TRUNCATE_TO_SECONDS,
    compress_ratio=COMPRESS_RATIO,
):
    """
    Computes the (start, end) window, in seconds relative to the original file, to keep. Only the
    FIRST detected interval (if it starts within _BOUNDARY_EPSILON of the file's own start) and the
    LAST detected interval (if its end -- or total_duration, when end is None -- lands within
    _BOUNDARY_EPSILON of total_duration) are ever touched; any silence strictly between real audio
    is left completely alone.

    For whichever of those two applies, silence beyond truncate_to_seconds is compressed by
    compress_ratio (Audacity's own "Truncate Silence" rule):
    kept = truncate_to + (original - truncate_to) * compress_ratio
    for a run longer than truncate_to_seconds, otherwise the run is left as-is.

    Falls back to the full, untrimmed (0, total_duration) window if the computed window would be
    empty or inverted -- only possible for a degenerate single-interval file that is silent from
    start to end, where there's no real audio content to preserve a trim around.
    """
    start = 0.0
    end = total_duration

    if intervals:
        first_start, first_end = intervals[0]
        is_leading = first_start <= _BOUNDARY_EPSILON

        last_start, last_end = intervals[-1]
        resolved_last_end = last_end if last_end is not None else total_duration
        is_trailing = (total_duration - resolved_last_end) <= _BOUNDARY_EPSILON

        # Degenerate case: a single interval spans (approximately) the entire file -- no real
        # audio content exists to anchor a leading/trailing split around.
        if len(intervals) == 1 and is_leading and is_trailing:
            return (0.0, total_duration)

        if is_leading:
            leading_end = first_end if first_end is not None else total_duration
            leading_duration = leading_end - first_start
            kept_leading = _compress_excess(leading_duration, truncate_to_seconds, compress_ratio)
            start = leading_end - kept_leading

        if is_trailing:
            trailing_duration = total_duration - last_start
            kept_trailing = _compress_excess(trailing_duration, truncate_to_seconds, compress_ratio)
            end = last_start + kept_trailing

    if start >= end:
        return (0.0, total_duration)

    return (start, end)


def trim_silence(file_path, logger=None):
    """
    Detects leading/trailing silence in file_path via ffmpeg's silencedetect filter and, if either
    end has more than TRUNCATE_TO_SECONDS worth of it, re-writes the file in place with the excess
    compressed away (see compute_trim_window). Leaves the file untouched if duration can't be
    determined, the detection/trim ffmpeg commands fail, or there's nothing worth trimming at
    either end.

    Uses -c:a copy for the actual trim -- no re-encoding, so audio quality is unaffected. Runs its
    own explicit ffmpeg invocation rather than the app's configured ffmpeg_base_command, since that
    command's default "-loglevel fatal" would suppress the silencedetect/Duration output this
    function depends on parsing.
    """
    log = logger or logging.getLogger(__name__)
    log.info(f"Detecting leading/trailing silence in {file_path}")

    detect_result = subprocess.run(
        [
            "ffmpeg", "-i", file_path,
            "-af", f"silencedetect=noise={SILENCE_THRESHOLD_DB}dB:d={MIN_SILENCE_DURATION_SECONDS}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
    )
    stderr = detect_result.stderr

    total_duration = parse_input_duration(stderr)
    if total_duration is None:
        log.warning(f"Could not determine duration for {file_path}; skipping silence trim")
        return

    intervals = parse_silence_intervals(stderr)
    start, end = compute_trim_window(intervals, total_duration)

    if start <= 0.001 and end >= total_duration - 0.001:
        log.info(f"No excess leading/trailing silence detected in {file_path}")
        return

    root, ext = os.path.splitext(file_path)
    temp_path = f"{root}.trimming{ext}"

    try:
        trim_result = subprocess.run(
            [
                "ffmpeg", "-y",
                "-ss", str(start),
                "-i", file_path,
                "-t", str(end - start),
                "-c:a", "copy",
                temp_path,
            ],
            capture_output=True,
            text=True,
        )
        if trim_result.returncode != 0:
            log.warning(f"Silence trim failed for {file_path}, leaving unchanged: {trim_result.stderr}")
            return

        os.replace(temp_path, file_path)
    finally:
        if os.path.isfile(temp_path):
            os.remove(temp_path)

    log.info(
        f"Trimmed silence from {file_path}: kept [{start:.3f}s, {end:.3f}s] of {total_duration:.3f}s original"
    )
