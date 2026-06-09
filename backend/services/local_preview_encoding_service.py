"""
Local Preview Encoding Service.

Provides local FFmpeg-based preview video encoding functionality, extracted from
VideoGenerator for use by the GCE worker. This ensures the same encoding logic
is used across local CLI, Cloud Run, and GCE worker environments.

This service handles:
- Generating preview videos with ASS subtitle overlay
- Hardware acceleration detection (NVENC) with fallback to software encoding
- Background image/color support
- Custom font support
- Optimized settings for fast preview generation
"""

import logging
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List

logger = logging.getLogger(__name__)


@dataclass
class PreviewEncodingConfig:
    """Configuration for preview video encoding."""
    ass_path: str  # Path to ASS subtitles file
    audio_path: str  # Path to audio file
    output_path: str  # Path for output video file

    # Optional background settings
    background_image_path: Optional[str] = None  # Path to background image
    background_color: str = "black"  # Fallback background color

    # Optional font settings
    font_path: Optional[str] = None  # Path to custom font file


@dataclass
class PreviewEncodingResult:
    """Result of preview encoding operation."""
    success: bool
    output_path: Optional[str] = None
    error: Optional[str] = None


class LocalPreviewEncodingService:
    """
    Service for local FFmpeg-based preview video encoding.

    This is the single source of truth for preview encoding logic, used by:
    - Local CLI (via VideoGenerator which delegates here)
    - Cloud Run (when GCE is unavailable)
    - GCE worker (via installed wheel)

    Supports hardware acceleration (NVENC) with automatic fallback
    to software encoding (libx264) when unavailable.
    """

    # Preview video settings - these are the canonical values
    PREVIEW_WIDTH = 480
    PREVIEW_HEIGHT = 270
    PREVIEW_FPS = 24
    PREVIEW_AUDIO_BITRATE = "96k"

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize the local preview encoding service.

        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

        # Hardware acceleration settings (detected on first use)
        self._nvenc_available: Optional[bool] = None
        self._video_encoder: Optional[str] = None
        self._hwaccel_flags: Optional[List[str]] = None

    def _detect_nvenc_support(self) -> bool:
        """
        Detect if NVENC hardware encoding is available.

        Returns:
            True if NVENC is available, False otherwise
        """
        try:
            self.logger.info("Detecting NVENC hardware acceleration...")

            # Test h264_nvenc encoder directly
            test_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "testsrc=duration=1:size=320x240:rate=1",
                "-c:v", "h264_nvenc", "-f", "null", "-"
            ]

            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                self.logger.info("NVENC hardware acceleration available")
                return True

            # Try alternative test with different source
            alt_test_cmd = [
                "ffmpeg", "-hide_banner", "-loglevel", "error",
                "-f", "lavfi", "-i", "color=red:size=320x240:duration=0.1",
                "-c:v", "h264_nvenc", "-preset", "fast", "-f", "null", "-"
            ]

            alt_result = subprocess.run(
                alt_test_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )

            if alt_result.returncode == 0:
                self.logger.info("NVENC hardware acceleration available")
                return True

            self.logger.info("NVENC not available, using software encoding")
            return False

        except subprocess.TimeoutExpired:
            self.logger.debug("NVENC detection timed out")
            return False
        except Exception as e:
            self.logger.debug(f"NVENC detection failed: {e}")
            return False

    def _configure_hardware_acceleration(self) -> None:
        """Configure hardware acceleration settings based on detected capabilities."""
        if self._nvenc_available is None:
            self._nvenc_available = self._detect_nvenc_support()

        if self._nvenc_available:
            self._video_encoder = "h264_nvenc"
            self._hwaccel_flags = ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
            self.logger.info("Using NVENC hardware acceleration for preview encoding")
        else:
            self._video_encoder = "libx264"
            self._hwaccel_flags = []
            self.logger.info("Using software encoding (libx264) for preview")

    @property
    def nvenc_available(self) -> bool:
        """Check if NVENC hardware acceleration is available."""
        if self._nvenc_available is None:
            self._configure_hardware_acceleration()
        return self._nvenc_available

    @property
    def video_encoder(self) -> str:
        """Get the video encoder to use."""
        if self._video_encoder is None:
            self._configure_hardware_acceleration()
        return self._video_encoder

    @property
    def hwaccel_flags(self) -> List[str]:
        """Get hardware acceleration flags."""
        if self._hwaccel_flags is None:
            self._configure_hardware_acceleration()
        return self._hwaccel_flags

    def _escape_ffmpeg_filter_path(self, path: str) -> str:
        """
        Escape a path for FFmpeg filter expressions (for subprocess without shell).

        When using subprocess with a command list (no shell), FFmpeg receives the
        filter string directly. FFmpeg's filter parser requires escaping:
        - Backslashes: convert to forward slashes (both work on Windows, avoids
          double-escape issues where \\t in filter strings gets consumed as 't')
        - Single quotes/apostrophes: escape with three backslashes (' -> \\\\')
        - Spaces: escape with backslash ( -> \\ )
        - Special characters: :,[];

        Example: "I'm With You" becomes "I\\\\'m\\ With\\ You"
        """
        # Convert Windows backslashes to forward slashes to avoid FFmpeg filter
        # parser consuming \t, \c, etc. as escape sequences. Windows accepts
        # forward slashes in all path APIs (fopen, CreateFile, etc.).
        escaped = path.replace("\\", "/")
        # Escape single quotes
        escaped = escaped.replace("'", "\\\\\\'")
        # Escape spaces
        escaped = escaped.replace(" ", "\\ ")
        # Escape FFmpeg filter special characters
        escaped = escaped.replace(":", "\\:")
        escaped = escaped.replace(",", "\\,")
        escaped = escaped.replace("[", "\\[")
        escaped = escaped.replace("]", "\\]")
        escaped = escaped.replace(";", "\\;")
        return escaped

    def _build_ass_filter(self, ass_path: str, font_path: Optional[str] = None) -> str:
        """
        Build ASS filter with optional font directory support.

        Args:
            ass_path: Path to ASS subtitles file
            font_path: Optional path to custom font file

        Returns:
            FFmpeg ASS filter string
        """
        escaped_ass_path = self._escape_ffmpeg_filter_path(ass_path)
        ass_filter = f"ass={escaped_ass_path}"

        if font_path and os.path.isfile(font_path):
            font_dir = os.path.dirname(font_path)
            escaped_font_dir = self._escape_ffmpeg_filter_path(font_dir)
            ass_filter += f":fontsdir={escaped_font_dir}"
            self.logger.debug(f"Using font directory: {font_dir}")

        return ass_filter

    def _build_preview_ffmpeg_command(self, config: PreviewEncodingConfig) -> List[str]:
        """
        Build FFmpeg command for preview video generation.

        This is the canonical preview encoding command, ensuring consistency
        across all environments (local CLI, Cloud Run, GCE worker).

        Args:
            config: Preview encoding configuration

        Returns:
            FFmpeg command as a list of arguments
        """
        width, height = self.PREVIEW_WIDTH, self.PREVIEW_HEIGHT

        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-r", str(self.PREVIEW_FPS),
        ]

        # Add hardware acceleration flags if available
        cmd.extend(self.hwaccel_flags)

        # Input source (background image or solid color)
        if config.background_image_path and os.path.isfile(config.background_image_path):
            self.logger.debug(f"Using background image: {config.background_image_path}")
            cmd.extend([
                "-loop", "1",
                "-i", config.background_image_path,
            ])
            # Build video filter with scaling and ASS subtitles
            video_filter = (
                f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
                f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2,"
                f"{self._build_ass_filter(config.ass_path, config.font_path)}"
            )
        else:
            self.logger.debug(f"Using solid {config.background_color} background")
            cmd.extend([
                "-f", "lavfi",
                "-i", f"color=c={config.background_color}:s={width}x{height}:r={self.PREVIEW_FPS}",
            ])
            # Just ASS subtitles, no scaling needed
            video_filter = self._build_ass_filter(config.ass_path, config.font_path)

        cmd.extend([
            "-i", config.audio_path,
            "-vf", video_filter,
            "-c:a", "aac",
            "-b:a", self.PREVIEW_AUDIO_BITRATE,
            "-c:v", self.video_encoder,
        ])

        # Add encoder-specific settings optimized for speed
        if self.nvenc_available:
            cmd.extend([
                "-preset", "p1",        # Fastest NVENC preset
                "-tune", "ll",          # Low latency
                "-rc", "cbr",           # Constant bitrate for speed
                "-b:v", "800k",         # Lower bitrate for speed
                "-profile:v", "baseline",  # Most compatible profile
                "-level", "3.1",        # Lower level for speed
            ])
            self.logger.debug("Using NVENC with maximum speed settings")
        else:
            cmd.extend([
                "-profile:v", "baseline",
                "-level", "3.0",
                "-preset", "superfast",
                "-tune", "fastdecode",
                "-b:v", "600k",
                "-maxrate", "800k",
                "-bufsize", "1200k",
                "-crf", "28",
            ])
            self.logger.debug("Using software encoding with maximum speed settings")

        cmd.extend([
            "-pix_fmt", "yuv420p",  # Required for browser compatibility
            "-movflags", "+faststart",
            "-g", "48",             # Keyframe every 48 frames (2 seconds at 24fps)
            "-keyint_min", "48",
            "-sc_threshold", "0",   # Disable scene change detection for speed
            "-threads", "0",        # Use all available CPU threads
            "-shortest",
            "-y",
            config.output_path,
        ])

        return cmd

    def encode_preview(self, config: PreviewEncodingConfig) -> PreviewEncodingResult:
        """
        Encode a preview video.

        Args:
            config: Preview encoding configuration

        Returns:
            PreviewEncodingResult with success status and output path
        """
        self.logger.info(f"Encoding preview video: {config.output_path}")

        # Validate input files
        if not os.path.isfile(config.ass_path):
            return PreviewEncodingResult(
                success=False,
                error=f"ASS subtitles file not found: {config.ass_path}"
            )

        if not os.path.isfile(config.audio_path):
            return PreviewEncodingResult(
                success=False,
                error=f"Audio file not found: {config.audio_path}"
            )

        # Ensure output directory exists
        output_dir = os.path.dirname(config.output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        try:
            # Build and execute FFmpeg command
            cmd = self._build_preview_ffmpeg_command(config)
            self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout for preview encoding
            )

            if result.returncode != 0:
                error_msg = result.stderr[-500:] if result.stderr else "Unknown error"
                self.logger.error(f"FFmpeg failed: {error_msg}")
                return PreviewEncodingResult(
                    success=False,
                    error=f"FFmpeg preview encoding failed: {error_msg}"
                )

            self.logger.info(f"Preview encoded successfully: {config.output_path}")
            return PreviewEncodingResult(
                success=True,
                output_path=config.output_path
            )

        except subprocess.TimeoutExpired:
            self.logger.error("Preview encoding timed out")
            return PreviewEncodingResult(
                success=False,
                error="Preview encoding timed out after 5 minutes"
            )
        except Exception as e:
            self.logger.error(f"Preview encoding failed: {e}")
            return PreviewEncodingResult(
                success=False,
                error=str(e)
            )


# Singleton instance and factory function
_local_preview_encoding_service: Optional[LocalPreviewEncodingService] = None


def get_local_preview_encoding_service(**kwargs) -> LocalPreviewEncodingService:
    """
    Get a local preview encoding service instance.

    Args:
        **kwargs: Arguments passed to LocalPreviewEncodingService

    Returns:
        LocalPreviewEncodingService instance
    """
    global _local_preview_encoding_service

    if _local_preview_encoding_service is None:
        _local_preview_encoding_service = LocalPreviewEncodingService(**kwargs)

    return _local_preview_encoding_service
