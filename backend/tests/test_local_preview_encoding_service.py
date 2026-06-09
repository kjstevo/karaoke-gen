"""
Tests for LocalPreviewEncodingService.

Tests cover:
- Service initialization
- Hardware acceleration detection (NVENC)
- FFmpeg filter path escaping
- ASS filter building
- Preview FFmpeg command construction
- Preview encoding execution
- Error handling
"""

import os
import tempfile
import pytest
from unittest.mock import MagicMock, patch

from backend.services.local_preview_encoding_service import (
    LocalPreviewEncodingService,
    PreviewEncodingConfig,
    PreviewEncodingResult,
    get_local_preview_encoding_service,
)


class TestLocalPreviewEncodingServiceInit:
    """Test service initialization."""

    def test_init_default_values(self):
        """Test default initialization."""
        service = LocalPreviewEncodingService()
        assert service._nvenc_available is None  # Lazy detection
        assert service._video_encoder is None  # Lazy detection
        assert service._hwaccel_flags is None  # Lazy detection

    def test_init_with_custom_logger(self):
        """Test initialization with custom logger."""
        import logging
        custom_logger = logging.getLogger("test")
        service = LocalPreviewEncodingService(logger=custom_logger)
        assert service.logger is custom_logger


class TestLocalPreviewEncodingServiceHWAccel:
    """Test hardware acceleration detection."""

    @patch("subprocess.run")
    def test_detect_nvenc_available(self, mock_run):
        """Test NVENC detection when available."""
        mock_run.return_value = MagicMock(returncode=0)

        service = LocalPreviewEncodingService()
        result = service._detect_nvenc_support()

        assert result is True

    @patch("subprocess.run")
    def test_detect_nvenc_not_available(self, mock_run):
        """Test fallback when NVENC not available."""
        import subprocess
        mock_run.side_effect = subprocess.CalledProcessError(1, "test")

        service = LocalPreviewEncodingService()
        result = service._detect_nvenc_support()

        assert result is False

    @patch("subprocess.run")
    def test_detect_nvenc_timeout(self, mock_run):
        """Test NVENC detection timeout handling."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("test", 30)

        service = LocalPreviewEncodingService()
        result = service._detect_nvenc_support()

        assert result is False

    @patch.object(LocalPreviewEncodingService, "_detect_nvenc_support")
    def test_configure_nvenc_available(self, mock_detect):
        """Test configuration when NVENC is available."""
        mock_detect.return_value = True

        service = LocalPreviewEncodingService()
        service._configure_hardware_acceleration()

        assert service._nvenc_available is True
        assert service._video_encoder == "h264_nvenc"
        assert "-hwaccel" in service._hwaccel_flags

    @patch.object(LocalPreviewEncodingService, "_detect_nvenc_support")
    def test_configure_no_nvenc(self, mock_detect):
        """Test configuration when NVENC not available."""
        mock_detect.return_value = False

        service = LocalPreviewEncodingService()
        service._configure_hardware_acceleration()

        assert service._nvenc_available is False
        assert service._video_encoder == "libx264"
        assert service._hwaccel_flags == []


class TestLocalPreviewEncodingServiceProperties:
    """Test property accessors that trigger lazy detection."""

    @patch.object(LocalPreviewEncodingService, "_configure_hardware_acceleration")
    def test_nvenc_available_triggers_detection(self, mock_config):
        """Test that nvenc_available property triggers detection."""
        service = LocalPreviewEncodingService()
        service._nvenc_available = None

        # Access property
        _ = service.nvenc_available

        mock_config.assert_called_once()

    @patch.object(LocalPreviewEncodingService, "_configure_hardware_acceleration")
    def test_video_encoder_triggers_detection(self, mock_config):
        """Test that video_encoder property triggers detection."""
        service = LocalPreviewEncodingService()
        service._video_encoder = None

        # Access property
        _ = service.video_encoder

        mock_config.assert_called_once()

    @patch.object(LocalPreviewEncodingService, "_configure_hardware_acceleration")
    def test_hwaccel_flags_triggers_detection(self, mock_config):
        """Test that hwaccel_flags property triggers detection."""
        service = LocalPreviewEncodingService()
        service._hwaccel_flags = None

        # Access property
        _ = service.hwaccel_flags

        mock_config.assert_called_once()


class TestLocalPreviewEncodingServicePathEscaping:
    """Test FFmpeg filter path escaping."""

    def test_escape_simple_path(self):
        """Test escaping a simple path."""
        service = LocalPreviewEncodingService()
        result = service._escape_ffmpeg_filter_path("/simple/path.ass")
        assert result == "/simple/path.ass"

    def test_escape_path_with_spaces(self):
        """Test escaping path with spaces."""
        service = LocalPreviewEncodingService()
        result = service._escape_ffmpeg_filter_path("/path/with spaces/file.ass")
        assert "\\ " in result  # Spaces should be escaped

    def test_escape_path_with_apostrophe(self):
        """Test escaping path with apostrophe (common in song titles)."""
        service = LocalPreviewEncodingService()
        result = service._escape_ffmpeg_filter_path("/path/I'm With You/file.ass")
        assert "\\\\\\'" in result  # Apostrophe should be triple-backslash escaped

    def test_escape_path_with_special_chars(self):
        """Test escaping path with FFmpeg special characters."""
        service = LocalPreviewEncodingService()
        result = service._escape_ffmpeg_filter_path("/path:with[special];chars,here.ass")
        # Should escape :,[];
        assert "\\:" in result
        assert "\\[" in result
        assert "\\]" in result
        assert "\\;" in result
        assert "\\," in result

    def test_escape_path_with_backslashes(self):
        """Test that backslashes are converted to forward slashes (Windows path safety)."""
        service = LocalPreviewEncodingService()
        result = service._escape_ffmpeg_filter_path("/path\\with\\backslashes.ass")
        assert "\\\\" not in result
        assert "/" in result


class TestLocalPreviewEncodingServiceASSFilter:
    """Test ASS filter building."""

    def test_build_ass_filter_simple(self):
        """Test building ASS filter without font."""
        service = LocalPreviewEncodingService()
        result = service._build_ass_filter("/path/to/file.ass")
        assert result.startswith("ass=")
        assert "fontsdir" not in result

    def test_build_ass_filter_with_font(self):
        """Test building ASS filter with custom font."""
        with tempfile.TemporaryDirectory() as tmpdir:
            font_path = os.path.join(tmpdir, "custom.ttf")
            with open(font_path, "w") as f:
                f.write("fake font")

            service = LocalPreviewEncodingService()
            result = service._build_ass_filter("/path/to/file.ass", font_path)

            assert "fontsdir=" in result

    def test_build_ass_filter_nonexistent_font(self):
        """Test building ASS filter with nonexistent font file."""
        service = LocalPreviewEncodingService()
        result = service._build_ass_filter("/path/to/file.ass", "/nonexistent/font.ttf")
        # Should not include fontsdir for nonexistent font
        assert "fontsdir" not in result


class TestLocalPreviewEncodingServiceFFmpegCommand:
    """Test FFmpeg command building."""

    def test_build_command_solid_background_libx264(self):
        """Test command building with solid background and libx264."""
        service = LocalPreviewEncodingService()
        service._nvenc_available = False
        service._video_encoder = "libx264"
        service._hwaccel_flags = []

        config = PreviewEncodingConfig(
            ass_path="/path/to/subs.ass",
            audio_path="/path/to/audio.flac",
            output_path="/path/to/output.mp4",
            background_color="black"
        )

        cmd = service._build_preview_ffmpeg_command(config)

        assert "ffmpeg" in cmd
        assert "-r" in cmd and "24" in cmd  # Frame rate
        assert "-c:v" in cmd
        assert "libx264" in cmd
        assert "-preset" in cmd and "superfast" in cmd
        assert "-crf" in cmd and "28" in cmd
        assert "color=c=black" in " ".join(cmd)

    def test_build_command_with_nvenc(self):
        """Test command building with NVENC hardware acceleration."""
        service = LocalPreviewEncodingService()
        service._nvenc_available = True
        service._video_encoder = "h264_nvenc"
        service._hwaccel_flags = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        config = PreviewEncodingConfig(
            ass_path="/path/to/subs.ass",
            audio_path="/path/to/audio.flac",
            output_path="/path/to/output.mp4"
        )

        cmd = service._build_preview_ffmpeg_command(config)

        assert "h264_nvenc" in cmd
        assert "-hwaccel" in cmd
        assert "-preset" in cmd and "p1" in cmd  # Fastest NVENC preset

    def test_build_command_with_background_image(self):
        """Test command building with background image."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bg_path = os.path.join(tmpdir, "background.png")
            with open(bg_path, "wb") as f:
                f.write(b"fake png")

            service = LocalPreviewEncodingService()
            service._nvenc_available = False
            service._video_encoder = "libx264"
            service._hwaccel_flags = []

            config = PreviewEncodingConfig(
                ass_path="/path/to/subs.ass",
                audio_path="/path/to/audio.flac",
                output_path="/path/to/output.mp4",
                background_image_path=bg_path
            )

            cmd = service._build_preview_ffmpeg_command(config)

            assert "-loop" in cmd
            assert "1" in cmd
            assert bg_path in cmd
            # Should have scale filter for background image
            vf_index = cmd.index("-vf")
            vf_value = cmd[vf_index + 1]
            assert "scale=" in vf_value
            assert "pad=" in vf_value

    def test_build_command_resolution(self):
        """Test that command uses correct preview resolution."""
        service = LocalPreviewEncodingService()
        service._nvenc_available = False
        service._video_encoder = "libx264"
        service._hwaccel_flags = []

        config = PreviewEncodingConfig(
            ass_path="/path/to/subs.ass",
            audio_path="/path/to/audio.flac",
            output_path="/path/to/output.mp4"
        )

        cmd = service._build_preview_ffmpeg_command(config)
        cmd_str = " ".join(cmd)

        # Resolution should be 480x270
        assert "480" in cmd_str
        assert "270" in cmd_str

    def test_build_command_audio_settings(self):
        """Test that command uses correct audio settings."""
        service = LocalPreviewEncodingService()
        service._nvenc_available = False
        service._video_encoder = "libx264"
        service._hwaccel_flags = []

        config = PreviewEncodingConfig(
            ass_path="/path/to/subs.ass",
            audio_path="/path/to/audio.flac",
            output_path="/path/to/output.mp4"
        )

        cmd = service._build_preview_ffmpeg_command(config)

        assert "-c:a" in cmd
        assert "aac" in cmd
        assert "-b:a" in cmd
        assert "96k" in cmd


class TestLocalPreviewEncodingServiceEncode:
    """Test preview encoding execution."""

    @patch("subprocess.run")
    def test_encode_preview_success(self, mock_run):
        """Test successful preview encoding."""
        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = os.path.join(tmpdir, "subs.ass")
            audio_path = os.path.join(tmpdir, "audio.flac")
            output_path = os.path.join(tmpdir, "output.mp4")

            # Create input files
            with open(ass_path, "w") as f:
                f.write("[Script Info]\n")
            with open(audio_path, "wb") as f:
                f.write(b"fake audio")

            service = LocalPreviewEncodingService()
            service._nvenc_available = False
            service._video_encoder = "libx264"
            service._hwaccel_flags = []

            config = PreviewEncodingConfig(
                ass_path=ass_path,
                audio_path=audio_path,
                output_path=output_path
            )

            result = service.encode_preview(config)

            assert result.success is True
            assert result.output_path == output_path
            assert result.error is None
            mock_run.assert_called_once()

    @patch("subprocess.run")
    def test_encode_preview_ffmpeg_failure(self, mock_run):
        """Test preview encoding with FFmpeg failure."""
        mock_run.return_value = MagicMock(returncode=1, stderr="FFmpeg error message")

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = os.path.join(tmpdir, "subs.ass")
            audio_path = os.path.join(tmpdir, "audio.flac")
            output_path = os.path.join(tmpdir, "output.mp4")

            with open(ass_path, "w") as f:
                f.write("[Script Info]\n")
            with open(audio_path, "wb") as f:
                f.write(b"fake audio")

            service = LocalPreviewEncodingService()
            service._nvenc_available = False
            service._video_encoder = "libx264"
            service._hwaccel_flags = []

            config = PreviewEncodingConfig(
                ass_path=ass_path,
                audio_path=audio_path,
                output_path=output_path
            )

            result = service.encode_preview(config)

            assert result.success is False
            assert "FFmpeg preview encoding failed" in result.error

    def test_encode_preview_missing_ass_file(self):
        """Test preview encoding with missing ASS file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "audio.flac")
            output_path = os.path.join(tmpdir, "output.mp4")

            with open(audio_path, "wb") as f:
                f.write(b"fake audio")

            service = LocalPreviewEncodingService()

            config = PreviewEncodingConfig(
                ass_path="/nonexistent/subs.ass",
                audio_path=audio_path,
                output_path=output_path
            )

            result = service.encode_preview(config)

            assert result.success is False
            assert "not found" in result.error

    def test_encode_preview_missing_audio_file(self):
        """Test preview encoding with missing audio file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = os.path.join(tmpdir, "subs.ass")
            output_path = os.path.join(tmpdir, "output.mp4")

            with open(ass_path, "w") as f:
                f.write("[Script Info]\n")

            service = LocalPreviewEncodingService()

            config = PreviewEncodingConfig(
                ass_path=ass_path,
                audio_path="/nonexistent/audio.flac",
                output_path=output_path
            )

            result = service.encode_preview(config)

            assert result.success is False
            assert "not found" in result.error

    @patch("subprocess.run")
    def test_encode_preview_timeout(self, mock_run):
        """Test preview encoding timeout."""
        import subprocess
        mock_run.side_effect = subprocess.TimeoutExpired("ffmpeg", 300)

        with tempfile.TemporaryDirectory() as tmpdir:
            ass_path = os.path.join(tmpdir, "subs.ass")
            audio_path = os.path.join(tmpdir, "audio.flac")
            output_path = os.path.join(tmpdir, "output.mp4")

            with open(ass_path, "w") as f:
                f.write("[Script Info]\n")
            with open(audio_path, "wb") as f:
                f.write(b"fake audio")

            service = LocalPreviewEncodingService()
            service._nvenc_available = False
            service._video_encoder = "libx264"
            service._hwaccel_flags = []

            config = PreviewEncodingConfig(
                ass_path=ass_path,
                audio_path=audio_path,
                output_path=output_path
            )

            result = service.encode_preview(config)

            assert result.success is False
            assert "timed out" in result.error


class TestPreviewEncodingConfig:
    """Test PreviewEncodingConfig dataclass."""

    def test_config_required_fields(self):
        """Test that required fields must be provided."""
        config = PreviewEncodingConfig(
            ass_path="/path/to/subs.ass",
            audio_path="/path/to/audio.flac",
            output_path="/path/to/output.mp4"
        )
        assert config.ass_path == "/path/to/subs.ass"
        assert config.background_color == "black"  # Default
        assert config.background_image_path is None  # Optional

    def test_config_all_fields(self):
        """Test config with all fields."""
        config = PreviewEncodingConfig(
            ass_path="/path/to/subs.ass",
            audio_path="/path/to/audio.flac",
            output_path="/path/to/output.mp4",
            background_image_path="/path/to/bg.png",
            background_color="red",
            font_path="/path/to/font.ttf"
        )
        assert config.background_image_path == "/path/to/bg.png"
        assert config.background_color == "red"
        assert config.font_path == "/path/to/font.ttf"


class TestPreviewEncodingResult:
    """Test PreviewEncodingResult dataclass."""

    def test_result_success(self):
        """Test successful result."""
        result = PreviewEncodingResult(
            success=True,
            output_path="/path/to/output.mp4"
        )
        assert result.success is True
        assert result.error is None
        assert result.output_path == "/path/to/output.mp4"

    def test_result_failure(self):
        """Test failure result."""
        result = PreviewEncodingResult(
            success=False,
            error="Something went wrong"
        )
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.output_path is None


class TestGetLocalPreviewEncodingService:
    """Test factory function."""

    def test_get_service_creates_instance(self):
        """Test that factory function creates a new instance."""
        import backend.services.local_preview_encoding_service as module
        module._local_preview_encoding_service = None

        service = get_local_preview_encoding_service()

        assert service is not None
        assert isinstance(service, LocalPreviewEncodingService)

    def test_get_service_returns_singleton(self):
        """Test that factory function returns the same instance."""
        import backend.services.local_preview_encoding_service as module
        module._local_preview_encoding_service = None

        service1 = get_local_preview_encoding_service()
        service2 = get_local_preview_encoding_service()

        assert service1 is service2


class TestPreviewEncodingConstants:
    """Test that encoding constants match expected values."""

    def test_preview_resolution(self):
        """Test preview resolution constants."""
        service = LocalPreviewEncodingService()
        assert service.PREVIEW_WIDTH == 480
        assert service.PREVIEW_HEIGHT == 270

    def test_preview_fps(self):
        """Test preview frame rate constant."""
        service = LocalPreviewEncodingService()
        assert service.PREVIEW_FPS == 24

    def test_preview_audio_bitrate(self):
        """Test preview audio bitrate constant."""
        service = LocalPreviewEncodingService()
        assert service.PREVIEW_AUDIO_BITRATE == "96k"
