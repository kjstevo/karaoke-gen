# make1video: switch single-file output from 4K lossless to 720p lossless-audio

## Problem

`--make1video` (env `MAKE1VIDEO=true`) currently outputs a single **4K lossless MP4**
(video copied at full 3840x2160, FLAC audio) and skips the lossy 4K MP4, lossless MKV,
and lossy 720p MP4. The 4K file is too large and unnecessary for the intended use case;
audio quality must be preserved.

## Decision

When `make_one_video` is set, produce a **720p MP4 with lossless (FLAC) audio** instead
of the 4K lossless MP4. Non-`make1video` behavior is unchanged (still produces the
4K lossless MP4, 4K lossless MKV, 4K lossy MP4, and lossy 720p MP4).

- Encoded in a single ffmpeg pass: concat (title + karaoke + optional end credits) +
  scale to 1280x720 + FLAC audio. The 4K intermediate is never written to disk.
- Video quality: standard 720p bitrate (~2000k), matching the existing lossy 720p
  encoder's settings — only audio needs to be lossless.
- Output filename: `Artist - Title (Karaoke lossless 720 final).mp4`

## Implementation

### `karaoke_gen/karaoke_finalise/karaoke_finalise.py`

1. Add new suffix/filename entry: `final_karaoke_lossless_720p_mp4` ->
   `" (Karaoke lossless 720 final).mp4"`.
2. Extend `prepare_concat_filter()` with `scale_to_720p: bool = False`. When set, the
   filter_complex graph adds a scale stage after concat (`[concatv]scale=1280:720[outv]`)
   instead of mapping the concat output directly to `[outv]`.
3. Add `encode_lossless_720p_mp4(title_mov_file, karaoke_mp4_file, env_mov_input,
   ffmpeg_filter, output_file)` — same shape as `encode_lossless_mp4`, but uses
   `get_nvenc_quality_settings("medium")` + `-b:v 2000k` for video (matching
   `encode_720p_version`'s settings) and `-c:a flac` for audio.
4. In `remux_and_encode_output_video_files()`:
   - When `make_one_video`: call `prepare_concat_filter(input_files, scale_to_720p=True)`
     and `encode_lossless_720p_mp4(...)` writing to
     `output_files["final_karaoke_lossless_720p_mp4"]`.
   - When not `make_one_video`: unchanged, uses `prepare_concat_filter(input_files)`
     (scale_to_720p defaults False) and `encode_lossless_mp4(...)` as today.
   - Update the existing-file check (currently checks `final_karaoke_lossless_mp4`) to
     check the 720p file when `make_one_video`.
   - Update the confirmation message to reference the 720p lossless file.
5. In `process()`: when `make_one_video`, set
   `result["final_video"] = output_files["final_karaoke_lossless_720p_mp4"]`. No other
   result keys change — `final_video_mkv/lossy/720p` remain `None` as today.

### `karaoke_gen/utils/gen_cli.py`

Three near-identical summary-logging blocks (~L611, L726, L1202) hardcode
"Lossless 4K MP4 (PCM)". Change to say "Lossless 720p MP4 (FLAC)" when `make1video` is
set, else keep the existing 4K line unchanged.

### `karaoke_gen/utils/cli_args.py`

Update `--make1video` help text to describe the new 720p-lossless-audio behavior.

## Testing

- New unit test for `encode_lossless_720p_mp4` ffmpeg command construction (GPU + CPU
  fallback), mirroring `test_encode_lossless_mp4` / `test_encode_720p_version_aac` in
  `tests/unit/test_karaoke_finalise/test_ffmpeg_commands.py`.
- Update `test_prepare_concat_filter_*` tests to cover `scale_to_720p=True` (with and
  without end credits).
- Update/add orchestration tests in `test_orchestration.py` confirming
  `make_one_video=True` calls only `encode_lossless_720p_mp4` (not
  `encode_lossless_mp4`, `encode_lossy_mp4`, `encode_lossless_mkv`, or
  `encode_720p_version`).

## Out of scope

- Not touching upstream 4K video generation (title/karaoke rendering stays at 4K;
  only the finalise-stage single-file output changes).
- Not fixing the pre-existing "(PCM)" mislabel in the non-`make1video` 4K MP4 log line
  (audio is actually FLAC there too) — unrelated to this change.
