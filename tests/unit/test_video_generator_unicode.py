"""Unit tests for VideoGenerator Unicode filename handling.

Tests the fix for job 1eab1172 where Cyrillic characters in temp subtitle
filenames caused FFmpeg to fail with exit code 183.
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch
from karaoke_gen.lyrics_transcriber.output.video import VideoGenerator


class TestVideoGeneratorUnicodeSanitization:
    """Test that VideoGenerator properly sanitizes Unicode characters in temp filenames."""

    @pytest.fixture
    def video_generator(self):
        """Create a VideoGenerator instance for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = os.path.join(temp_dir, "output")
            cache_dir = os.path.join(temp_dir, "cache")
            os.makedirs(output_dir)
            os.makedirs(cache_dir)

            styles = {
                "karaoke": {
                    "background_color": "black",
                    "font_path": None,
                }
            }

            generator = VideoGenerator(
                output_dir=output_dir,
                cache_dir=cache_dir,
                video_resolution=(1920, 1080),
                styles=styles,
                logger=MagicMock(),
            )

            yield generator

    def test_generate_video_sanitizes_cyrillic_characters(self, video_generator):
        """Test that Cyrillic characters are replaced with underscores in temp filename.

        This was the bug in job 1eab1172 with Russian title "я куплю тебе дом".
        """
        output_prefix = "zivert - я куплю тебе дом"

        # Create mock ASS and audio files
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as ass_file:
            ass_path = ass_file.name
            ass_file.write(b"[Script Info]\nTitle: Test\n")

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as audio_file:
            audio_path = audio_file.name

        try:
            with patch.object(video_generator, '_run_ffmpeg_command') as mock_ffmpeg:
                # This should not raise an exception
                video_generator.generate_video(ass_path, audio_path, output_prefix)

                # Verify FFmpeg was called
                assert mock_ffmpeg.called

                # Get the temp file path that was created
                cmd = mock_ffmpeg.call_args[0][0]
                ass_filter_arg = None
                for i, arg in enumerate(cmd):
                    if arg == "-vf":
                        ass_filter_arg = cmd[i + 1]
                        break

                assert ass_filter_arg is not None, "Should have ASS filter in FFmpeg command"

                # Extract the temp ASS path from the filter.
                # Format: ass='/path/to/temp_subtitles_...':fontsdir='/fonts/dir'
                path_part = ass_filter_arg.split("=", 1)[1].split(":fontsdir=")[0]
                temp_ass_path = path_part.strip("'")  # Remove surrounding single quotes

                # Verify the temp filename only contains ASCII characters
                temp_filename = os.path.basename(temp_ass_path)
                assert temp_filename.isascii(), f"Temp filename should be ASCII-only: {temp_filename}"

                # Verify Cyrillic characters were replaced with underscores
                assert "я" not in temp_filename
                assert "куплю" not in temp_filename
                assert "дом" not in temp_filename
                assert "_" in temp_filename  # Should have underscores as replacements

        finally:
            # Cleanup
            if os.path.exists(ass_path):
                os.unlink(ass_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_generate_preview_video_sanitizes_unicode(self, video_generator):
        """Test that preview video generation also sanitizes Unicode characters."""
        output_prefix = "artist - 中文标题"  # Chinese characters

        # Create mock ASS and audio files
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as ass_file:
            ass_path = ass_file.name
            ass_file.write(b"[Script Info]\nTitle: Test\n")

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as audio_file:
            audio_path = audio_file.name

        try:
            # Mock the LocalPreviewEncodingService
            with patch('backend.services.local_preview_encoding_service.LocalPreviewEncodingService') as mock_service_class:
                mock_service = MagicMock()
                mock_result = MagicMock()
                mock_result.success = True
                mock_service.encode_preview.return_value = mock_result
                mock_service_class.return_value = mock_service

                # This should not raise an exception
                video_generator.generate_preview_video(ass_path, audio_path, output_prefix)

                # Verify the service was called
                assert mock_service.encode_preview.called

                # Get the config that was passed
                config = mock_service.encode_preview.call_args[0][0]
                temp_ass_path = config.ass_path

                # Verify the temp filename only contains ASCII characters
                temp_filename = os.path.basename(temp_ass_path)
                assert temp_filename.isascii(), f"Temp filename should be ASCII-only: {temp_filename}"

                # Verify Chinese characters were replaced
                assert "中文标题" not in temp_filename

        finally:
            # Cleanup
            if os.path.exists(ass_path):
                os.unlink(ass_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    @pytest.mark.parametrize("output_prefix,unicode_chars,should_contain_ascii", [
        ("artist - я куплю тебе дом", ["я", "к", "у", "п", "л", "ю", "т", "е", "б", "д", "о", "м"], "artist"),  # Cyrillic
        ("artist - 中文标题", ["中", "文", "标", "题"], "artist"),  # Chinese
        ("artist - العربية", ["ا", "ل", "ع", "ر", "ب", "ي", "ة"], "artist"),  # Arabic
        ("artist - 😀🎵", ["😀", "🎵"], "artist"),  # Emoji
        ("artist - título", ["í"], "tulo"),  # Spanish with accent - 't' and 'ulo' should remain
        ("normal title", [], "normal_title"),  # ASCII-only (control)
    ])
    def test_various_unicode_characters_sanitized(self, video_generator, output_prefix, unicode_chars, should_contain_ascii):
        """Test that various Unicode character sets are properly sanitized."""
        import re

        # Create mock ASS and audio files
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as ass_file:
            ass_path = ass_file.name
            ass_file.write(b"[Script Info]\nTitle: Test\n")

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as audio_file:
            audio_path = audio_file.name

        try:
            with patch.object(video_generator, '_run_ffmpeg_command') as mock_ffmpeg:
                video_generator.generate_video(ass_path, audio_path, output_prefix)

                # Get the temp file path
                cmd = mock_ffmpeg.call_args[0][0]
                ass_filter_arg = None
                for i, arg in enumerate(cmd):
                    if arg == "-vf":
                        ass_filter_arg = cmd[i + 1]
                        break

                # Format: ass='/path/to/temp_subtitles_...':fontsdir='/fonts/dir'
                path_part = ass_filter_arg.split("=", 1)[1].split(":fontsdir=")[0]
                temp_ass_path = path_part.strip("'")  # Remove surrounding single quotes
                temp_filename = os.path.basename(temp_ass_path)

                # Verify filename only contains ASCII
                assert temp_filename.isascii(), f"Should be ASCII-only: {temp_filename}"

                # Verify all Unicode characters were removed
                for unicode_char in unicode_chars:
                    assert unicode_char not in temp_filename, \
                        f"Unicode character '{unicode_char}' should not be in filename: {temp_filename}"

                # Verify ASCII parts were preserved
                if should_contain_ascii:
                    assert should_contain_ascii in temp_filename, \
                        f"Expected '{should_contain_ascii}' to be in filename: {temp_filename}"

                # Verify it matches the general pattern (starts with temp_subtitles_, ends with .ass)
                assert temp_filename.startswith("temp_subtitles_"), \
                    f"Filename should start with 'temp_subtitles_': {temp_filename}"
                assert temp_filename.endswith(".ass"), \
                    f"Filename should end with '.ass': {temp_filename}"

        finally:
            if os.path.exists(ass_path):
                os.unlink(ass_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_sanitization_preserves_ascii_alphanumeric(self, video_generator):
        """Test that ASCII alphanumeric characters are preserved."""
        output_prefix = "ABC123xyz"

        # Create mock ASS and audio files
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as ass_file:
            ass_path = ass_file.name
            ass_file.write(b"[Script Info]\nTitle: Test\n")

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as audio_file:
            audio_path = audio_file.name

        try:
            with patch.object(video_generator, '_run_ffmpeg_command') as mock_ffmpeg:
                video_generator.generate_video(ass_path, audio_path, output_prefix)

                # Get the temp file path
                cmd = mock_ffmpeg.call_args[0][0]
                ass_filter_arg = None
                for i, arg in enumerate(cmd):
                    if arg == "-vf":
                        ass_filter_arg = cmd[i + 1]
                        break

                # Format: ass='/path/to/temp_subtitles_...':fontsdir='/fonts/dir'
                path_part = ass_filter_arg.split("=", 1)[1].split(":fontsdir=")[0]
                temp_ass_path = path_part.strip("'")  # Remove surrounding single quotes
                temp_filename = os.path.basename(temp_ass_path)

                # Verify ASCII alphanumeric characters are preserved
                assert "ABC123xyz" in temp_filename

        finally:
            if os.path.exists(ass_path):
                os.unlink(ass_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)

    def test_fontsdir_relative_to_cwd_avoids_space_in_path(self):
        """Regression test: an absolute font_path under a directory containing a
        space (e.g. a Windows user profile like 'C:\\Users\\John Doe\\...') must not
        leak into the -vf filtergraph string as an absolute fontsdir. FFmpeg's
        filtergraph parser can misparse such paths even when single-quoted,
        producing 'Could not create a libass track when reading file <fontsdir>'
        because fontsdir gets mistaken for the ass filename. fontsdir must be made
        relative to ffmpeg's cwd, the same way the temp ASS path already is (see
        the 'Pass only the basename' comment in generate_video).
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            space_dir = os.path.join(tmpdir, "dir with space")
            output_dir = os.path.join(space_dir, "output")
            cache_dir = os.path.join(space_dir, "cache")
            fonts_dir = os.path.join(space_dir, "Resources")
            os.makedirs(output_dir)
            os.makedirs(cache_dir)
            os.makedirs(fonts_dir)

            font_path = os.path.join(fonts_dir, "custom.ttf")
            with open(font_path, "w") as f:
                f.write("fake font")

            styles = {
                "karaoke": {
                    "background_color": "black",
                    "font_path": font_path,
                }
            }

            generator = VideoGenerator(
                output_dir=output_dir,
                cache_dir=cache_dir,
                video_resolution=(1920, 1080),
                styles=styles,
                logger=MagicMock(),
            )

            with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as ass_file:
                ass_path = ass_file.name
                ass_file.write(b"[Script Info]\nTitle: Test\n")

            with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as audio_file:
                audio_path = audio_file.name

            try:
                with patch.object(generator, '_run_ffmpeg_command') as mock_ffmpeg:
                    generator.generate_video(ass_path, audio_path, "Test Song")

                    cmd = mock_ffmpeg.call_args[0][0]
                    vf_value = cmd[cmd.index("-vf") + 1]

                    assert "fontsdir=" in vf_value
                    # The space-containing absolute directory must not appear in the
                    # -vf value (in either separator form — _escape_ffmpeg_filter_path
                    # converts backslashes to forward slashes).
                    assert space_dir not in vf_value
                    assert space_dir.replace("\\", "/") not in vf_value
            finally:
                if os.path.exists(ass_path):
                    os.unlink(ass_path)
                if os.path.exists(audio_path):
                    os.unlink(audio_path)

    def test_special_characters_replaced_with_underscores(self, video_generator):
        """Test that special characters (spaces, hyphens, etc.) are replaced."""
        output_prefix = "artist - title (version) [2024]"

        # Create mock ASS and audio files
        with tempfile.NamedTemporaryFile(suffix=".ass", delete=False) as ass_file:
            ass_path = ass_file.name
            ass_file.write(b"[Script Info]\nTitle: Test\n")

        with tempfile.NamedTemporaryFile(suffix=".flac", delete=False) as audio_file:
            audio_path = audio_file.name

        try:
            with patch.object(video_generator, '_run_ffmpeg_command') as mock_ffmpeg:
                video_generator.generate_video(ass_path, audio_path, output_prefix)

                # Get the temp file path
                cmd = mock_ffmpeg.call_args[0][0]
                ass_filter_arg = None
                for i, arg in enumerate(cmd):
                    if arg == "-vf":
                        ass_filter_arg = cmd[i + 1]
                        break

                # Format: ass='/path/to/temp_subtitles_...':fontsdir='/fonts/dir'
                path_part = ass_filter_arg.split("=", 1)[1].split(":fontsdir=")[0]
                temp_ass_path = path_part.strip("'")  # Remove surrounding single quotes
                temp_filename = os.path.basename(temp_ass_path)

                # Verify special characters were replaced
                assert " " not in temp_filename
                assert "-" not in temp_filename
                assert "(" not in temp_filename
                assert ")" not in temp_filename
                assert "[" not in temp_filename
                assert "]" not in temp_filename

                # But alphanumeric should remain
                assert "artist" in temp_filename
                assert "title" in temp_filename
                assert "version" in temp_filename
                assert "2024" in temp_filename

        finally:
            if os.path.exists(ass_path):
                os.unlink(ass_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
