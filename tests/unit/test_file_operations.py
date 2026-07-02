import os
import pytest
import glob
import shutil
from unittest.mock import MagicMock, patch, mock_open, call, DEFAULT
from karaoke_gen.karaoke_gen import KaraokePrep
from karaoke_gen.utils import sanitize_filename # Import utility

class TestFileOperations:
    def test_copy_input_media(self, basic_karaoke_gen, temp_dir):
        """Test copying input media to a new location."""
        # Setup
        source_file = os.path.join(temp_dir, "source.mp4")
        with open(source_file, "w") as f:
            f.write("test content")
        
        output_filename = os.path.join(temp_dir, "output")
        
        # Test with mocked shutil.copy2
        with patch('shutil.copy2') as mock_copy:
            result = basic_karaoke_gen.file_handler.copy_input_media(source_file, output_filename)
            
            # Verify the correct file path was returned
            assert result == output_filename + ".mp4"
            
            # Verify shutil.copy2 was called with correct arguments
            mock_copy.assert_called_once_with(source_file, output_filename + ".mp4")
    
    def test_copy_input_media_same_file(self, basic_karaoke_gen, temp_dir):
        """Test copying input media when source and destination are the same."""
        # Setup
        file_path = os.path.join(temp_dir, "file.mp4")
        with open(file_path, "w") as f:
            f.write("test content")
        
        # Test with mocked os.path.abspath to simulate same file
        with patch('os.path.abspath', side_effect=lambda x: x):
            result = basic_karaoke_gen.file_handler.copy_input_media(file_path, file_path[:-4])
            
            # Verify the correct file path was returned
            assert result == file_path
    
    def test_download_audio_from_fetcher_result(self, basic_karaoke_gen, temp_dir):
        """Test processing downloaded audio from flacfetch."""
        # Create a mock source file
        source_file = os.path.join(temp_dir, "source.flac")
        with open(source_file, "w") as f:
            f.write("test audio content")
        
        output_filename = os.path.join(temp_dir, "output")
        
        # Test with mocked shutil.copy2
        with patch('shutil.copy2') as mock_copy:
            result = basic_karaoke_gen.file_handler.download_audio_from_fetcher_result(
                source_file, output_filename
            )
            
            # Verify the correct file path was returned
            assert result == output_filename + ".flac"
            
            # Verify shutil.copy2 was called with correct arguments
            mock_copy.assert_called_once_with(source_file, output_filename + ".flac")
    
    def test_download_audio_from_fetcher_result_same_location(self, basic_karaoke_gen, temp_dir):
        """Test processing when source and target are effectively the same."""
        # Create a source file
        source_file = os.path.join(temp_dir, "output.flac")
        with open(source_file, "w") as f:
            f.write("test audio content")
        
        output_filename = os.path.join(temp_dir, "output")
        
        # Test when source and target are the same
        with patch('shutil.copy2') as mock_copy:
            result = basic_karaoke_gen.file_handler.download_audio_from_fetcher_result(
                source_file, output_filename
            )
            
            # Should return the source file without copying
            assert result == source_file
            mock_copy.assert_not_called()
    
    def test_extract_still_image_from_video(self, basic_karaoke_gen):
        """Test extracting a still image from a video."""
        input_filename = "input.mp4"
        output_filename = "output"
        
        # Patch os.system directly within this test
        with patch('os.system') as mock_os_system:
            result = basic_karaoke_gen.file_handler.extract_still_image_from_video(input_filename, output_filename)
            
            # Verify the correct file path was returned
            assert result == output_filename + ".png"
            
            # Verify os.system was called with correct arguments
            expected_command = f'{basic_karaoke_gen.file_handler.ffmpeg_base_command} -i "{input_filename}" -ss 00:00:30 -vframes 1 "{output_filename}.png"'
            mock_os_system.assert_called_once_with(expected_command)
    
    def test_convert_to_wav_success(self, basic_karaoke_gen, temp_dir):
        """Test converting input audio to WAV format successfully."""
        # Create a test input file
        input_filename = os.path.join(temp_dir, "input.mp3")
        with open(input_filename, "w") as f:
            f.write("test audio content")
        
        # Mock os.path.isfile and os.path.getsize
        with patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.popen') as mock_popen:
            
            # Mock the ffprobe output
            mock_popen.return_value.read.return_value = "codec_type=audio"
            
            output_filename = os.path.join(temp_dir, "output")
            
            # Patch os.system directly
            with patch('os.system') as mock_os_system:
                result = basic_karaoke_gen.file_handler.convert_to_wav(input_filename, output_filename)
                
                # Verify the correct file path was returned
                assert result == output_filename + ".wav"
                
                # Verify os.system was called
                mock_os_system.assert_called_once()
    
    def test_convert_to_wav_file_not_found(self, basic_karaoke_gen):
        """Test converting input audio when the file is not found."""
        input_filename = "nonexistent.mp3"
        output_filename = "output"
        
        # Mock os.path.isfile to return False
        with patch('os.path.isfile', return_value=False):
            with pytest.raises(Exception, match=f"Input audio file not found: {input_filename}"):
                basic_karaoke_gen.file_handler.convert_to_wav(input_filename, output_filename)
    
    def test_convert_to_wav_empty_file(self, basic_karaoke_gen):
        """Test converting input audio when the file is empty."""
        input_filename = "empty.mp3"
        output_filename = "output"
        
        # Mock os.path.isfile to return True and os.path.getsize to return 0
        with patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=0):
            with pytest.raises(Exception, match=f"Input audio file is empty: {input_filename}"):
                basic_karaoke_gen.file_handler.convert_to_wav(input_filename, output_filename)
    
    def test_convert_to_wav_no_audio_stream(self, basic_karaoke_gen):
        """Test converting input audio when no audio stream is found."""
        input_filename = "no_audio.mp4"
        output_filename = "output"
        
        # Mock os.path.isfile, os.path.getsize, and os.popen
        with patch('os.path.isfile', return_value=True), \
             patch('os.path.getsize', return_value=100), \
             patch('os.popen') as mock_popen:
            
            # Mock the ffprobe output to indicate no audio stream
            mock_popen.return_value.read.return_value = "codec_type=video"
            
            with pytest.raises(Exception, match=f"No valid audio stream found in file: {input_filename}"):
                basic_karaoke_gen.file_handler.convert_to_wav(input_filename, output_filename)
    
    def test_valid_video_exists_file_missing(self, basic_karaoke_gen):
        """A file that doesn't exist is never treated as a valid cached video."""
        with patch('os.path.isfile', return_value=False):
            assert basic_karaoke_gen.file_handler._valid_video_exists("missing.mov") is False

    def test_valid_video_exists_valid_video(self, basic_karaoke_gen):
        """A complete video file with a video stream is treated as valid and skipped."""
        with patch('os.path.isfile', return_value=True), \
             patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="video\n")

            assert basic_karaoke_gen.file_handler._valid_video_exists("title.mov") is True

    def test_valid_video_exists_truncated_video(self, basic_karaoke_gen):
        """A truncated/corrupt video file (e.g. missing moov atom) is not treated as valid,
        so an interrupted previous run doesn't get silently reused."""
        with patch('os.path.isfile', return_value=True), \
             patch('subprocess.run') as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stdout="")

            assert basic_karaoke_gen.file_handler._valid_video_exists("title.mov") is False

    def test_valid_video_exists_ffprobe_missing(self, basic_karaoke_gen):
        """If ffprobe isn't available, fall back to trusting the existing file rather than failing."""
        with patch('os.path.isfile', return_value=True), \
             patch('subprocess.run', side_effect=FileNotFoundError):
            assert basic_karaoke_gen.file_handler._valid_video_exists("title.mov") is True

    def test_sanitize_filename(self, basic_karaoke_gen):
        """Test sanitizing filenames."""
        # Test with various problematic characters
        assert sanitize_filename('file/with\\chars:*?"<>|') == 'file_with_chars_'
        assert sanitize_filename("  leading spaces") == "leading spaces"
        assert sanitize_filename("trailing spaces  ") == "trailing spaces"
        assert sanitize_filename("trailing dots...") == "trailing dots"
        assert sanitize_filename("multiple   spaces") == "multiple spaces"
        assert sanitize_filename("valid_filename_123") == "valid_filename_123"
        assert sanitize_filename("file_with__multiple___underscores") == "file_with_multiple_underscores"
    
    def test_setup_output_paths(self, basic_karaoke_gen, temp_dir):
        """Test setting up output paths."""
        # Test with both artist and title
        with patch('os.makedirs') as mock_makedirs:
            basic_karaoke_gen.file_handler.output_dir = temp_dir
            track_output_dir, artist_title = basic_karaoke_gen.file_handler.setup_output_paths(temp_dir, "Test Artist", "Test Title")
            
            assert artist_title == "Test Artist - Test Title"
            assert track_output_dir == temp_dir
            
            # Test with create_track_subfolders=True
            basic_karaoke_gen.file_handler.create_track_subfolders = True
            track_output_dir, artist_title = basic_karaoke_gen.file_handler.setup_output_paths(temp_dir, "Test Artist", "Test Title")
            
            expected_dir = os.path.join(temp_dir, "Test Artist - Test Title")
            assert track_output_dir == expected_dir
            mock_makedirs.assert_called_with(expected_dir)
    
    def test_setup_output_paths_title_only(self, basic_karaoke_gen, temp_dir):
        """Test setting up output paths with only title."""
        with patch('os.makedirs'):
            basic_karaoke_gen.file_handler.output_dir = temp_dir
            track_output_dir, artist_title = basic_karaoke_gen.file_handler.setup_output_paths(temp_dir, None, "Test Title")
            
            assert artist_title == "Test Title"
            assert track_output_dir == temp_dir
    
    def test_setup_output_paths_no_inputs(self, basic_karaoke_gen):
        """Test setting up output paths with no inputs."""
        with pytest.raises(ValueError, match="Error: At least title or artist must be provided"):
            basic_karaoke_gen.file_handler.setup_output_paths(basic_karaoke_gen.output_dir, None, None)

    def test_download_audio_from_fetcher_result_file_not_found(self, basic_karaoke_gen, temp_dir):
        """Test download_audio_from_fetcher_result when file doesn't exist."""
        fake_path = os.path.join(temp_dir, "nonexistent.flac")
        output_filename = os.path.join(temp_dir, "output")
        
        result = basic_karaoke_gen.file_handler.download_audio_from_fetcher_result(
            fake_path, output_filename
        )
        
        assert result is None


class TestDownloadVideo:
    """Tests for download_video method."""

    def test_download_video_yt_dlp_not_available(self, basic_karaoke_gen):
        """Test download_video when yt-dlp is not available."""
        with patch.dict('karaoke_gen.file_handler.__dict__', {'YT_DLP_AVAILABLE': False}):
            # Re-import to apply patch
            from karaoke_gen.file_handler import FileHandler
            handler = FileHandler(
                logger=basic_karaoke_gen.logger,
                ffmpeg_base_command="ffmpeg",
                create_track_subfolders=False,
                dry_run=False,
            )
            
            # Mock the YT_DLP_AVAILABLE at module level
            with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', False):
                result = handler.download_video("https://youtube.com/watch?v=test", "/output/file")
                assert result is None

    def test_download_video_success(self, basic_karaoke_gen, temp_dir):
        """Test successful video download."""
        output_filename = os.path.join(temp_dir, "output")
        
        # Create a mock downloaded file
        downloaded_file = f"{output_filename}.m4a"
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {"title": "Test Video"}
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                # Create the file that would be downloaded
                with open(downloaded_file, 'w') as f:
                    f.write("audio content")
                
                result = basic_karaoke_gen.file_handler.download_video(
                    "https://youtube.com/watch?v=test",
                    output_filename
                )
                
                assert result == downloaded_file

    def test_download_video_extract_info_returns_none(self, basic_karaoke_gen, temp_dir):
        """Test download_video when extract_info returns None."""
        output_filename = os.path.join(temp_dir, "output")
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = None
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.download_video(
                    "https://youtube.com/watch?v=test",
                    output_filename
                )
                
                assert result is None

    def test_download_video_with_cookies(self, basic_karaoke_gen, temp_dir):
        """Test download_video with cookies string."""
        output_filename = os.path.join(temp_dir, "output")
        downloaded_file = f"{output_filename}.m4a"
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {"title": "Test Video"}
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                # Create the file
                with open(downloaded_file, 'w') as f:
                    f.write("audio content")
                
                result = basic_karaoke_gen.file_handler.download_video(
                    "https://youtube.com/watch?v=test",
                    output_filename,
                    cookies_str="cookie_data_here"
                )
                
                assert result == downloaded_file

    def test_download_video_download_error(self, basic_karaoke_gen, temp_dir):
        """Test download_video when yt-dlp raises DownloadError."""
        output_filename = os.path.join(temp_dir, "output")
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.DownloadError = Exception
                mock_ydl_instance.extract_info.side_effect = mock_yt_dlp.DownloadError("Download failed")
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.download_video(
                    "https://youtube.com/watch?v=test",
                    output_filename
                )
                
                assert result is None

    def test_download_video_generic_exception(self, basic_karaoke_gen, temp_dir):
        """Test download_video when generic exception occurs."""
        output_filename = os.path.join(temp_dir, "output")
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.side_effect = Exception("Generic error")
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.DownloadError = type('DownloadError', (Exception,), {})
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.download_video(
                    "https://youtube.com/watch?v=test",
                    output_filename
                )
                
                assert result is None

    def test_download_video_file_not_found_after_download(self, basic_karaoke_gen, temp_dir):
        """Test download_video when downloaded file is not found."""
        output_filename = os.path.join(temp_dir, "output")
        
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {"title": "Test Video"}
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                # Don't create the file - simulate download failure
                
                result = basic_karaoke_gen.file_handler.download_video(
                    "https://youtube.com/watch?v=test",
                    output_filename
                )
                
                assert result is None


class TestExtractMetadataFromUrl:
    """Tests for extract_metadata_from_url method."""

    def test_extract_metadata_yt_dlp_not_available(self, basic_karaoke_gen):
        """Test extract_metadata_from_url when yt-dlp is not available."""
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', False):
            result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                "https://youtube.com/watch?v=test"
            )
            assert result is None

    def test_extract_metadata_success_with_dash_format(self, basic_karaoke_gen):
        """Test extracting metadata with 'Artist - Title' format."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {
            "title": "Test Artist - Test Song",
            "uploader": "Test Channel",
            "duration": 240,
        }
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                    "https://youtube.com/watch?v=test"
                )
                
                assert result is not None
                assert result['artist'] == "Test Artist"
                assert result['title'] == "Test Song"
                assert result['duration'] == 240

    def test_extract_metadata_success_without_dash(self, basic_karaoke_gen):
        """Test extracting metadata without 'Artist - Title' format."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {
            "title": "Just A Song Title",
            "uploader": "The Artist",
            "duration": 180,
        }
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                    "https://youtube.com/watch?v=test"
                )
                
                assert result is not None
                assert result['artist'] == "The Artist"
                assert result['title'] == "Just A Song Title"

    def test_extract_metadata_cleans_title_suffixes(self, basic_karaoke_gen):
        """Test that common suffixes are cleaned from title."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {
            "title": "Artist - Song Title (Official Video)",
            "uploader": "VEVO",
            "duration": 200,
        }
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                    "https://youtube.com/watch?v=test"
                )
                
                assert result is not None
                assert result['title'] == "Song Title"

    def test_extract_metadata_cleans_multiple_suffixes(self, basic_karaoke_gen):
        """Test cleaning multiple suffixes from title."""
        test_titles = [
            ("Song (Official Music Video)", "Song"),
            ("Song (Official Audio)", "Song"),
            ("Song (Lyric Video)", "Song"),
            ("Song (HD)", "Song"),
            ("Song (4K)", "Song"),
            ("Song (Remastered)", "Song"),
            ("Song | Official Video", "Song"),
            ("Song [Official Video]", "Song"),
        ]
        
        for input_title, expected_title in test_titles:
            mock_ydl_instance = MagicMock()
            mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
            mock_ydl_instance.__exit__ = MagicMock(return_value=False)
            mock_ydl_instance.extract_info.return_value = {
                "title": f"Artist - {input_title}",
                "uploader": "Channel",
                "duration": 200,
            }
            
            with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
                with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                    mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                    
                    result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                        "https://youtube.com/watch?v=test"
                    )
                    
                    assert result['title'] == expected_title, f"Failed for input: {input_title}"

    def test_extract_metadata_returns_none_on_error(self, basic_karaoke_gen):
        """Test that None is returned on extraction error."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.side_effect = Exception("Network error")
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                    "https://youtube.com/watch?v=test"
                )
                
                assert result is None

    def test_extract_metadata_returns_none_when_info_is_none(self, basic_karaoke_gen):
        """Test that None is returned when extract_info returns None."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = None
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                    "https://youtube.com/watch?v=test"
                )
                
                assert result is None

    def test_extract_metadata_uses_channel_as_fallback(self, basic_karaoke_gen):
        """Test that channel is used as artist when uploader is not available."""
        mock_ydl_instance = MagicMock()
        mock_ydl_instance.__enter__ = MagicMock(return_value=mock_ydl_instance)
        mock_ydl_instance.__exit__ = MagicMock(return_value=False)
        mock_ydl_instance.extract_info.return_value = {
            "title": "Song Title",
            "channel": "The Channel",
            "duration": 200,
        }
        
        with patch('karaoke_gen.file_handler.YT_DLP_AVAILABLE', True):
            with patch('karaoke_gen.file_handler.yt_dlp') as mock_yt_dlp:
                mock_yt_dlp.YoutubeDL.return_value = mock_ydl_instance
                
                result = basic_karaoke_gen.file_handler.extract_metadata_from_url(
                    "https://youtube.com/watch?v=test"
                )
                
                assert result['artist'] == "The Channel"


class TestBackupExistingOutputs:
    """Tests for backup_existing_outputs method."""

    def test_backup_existing_outputs_creates_version_dir(self, basic_karaoke_gen, temp_dir):
        """Test that backup creates a versioned directory."""
        # Setup artist and title
        artist = "Test Artist"
        title = "Test Song"
        
        # Create the necessary files
        base_name = f"{artist} - {title}"
        wav_file = os.path.join(temp_dir, f"{base_name}.wav")
        with open(wav_file, 'w') as f:
            f.write("audio content")
        
        # Create a file to be moved
        karaoke_file = os.path.join(temp_dir, f"{base_name} (Karaoke).mp4")
        with open(karaoke_file, 'w') as f:
            f.write("video content")
        
        result = basic_karaoke_gen.file_handler.backup_existing_outputs(
            temp_dir, artist, title
        )
        
        # Check that version directory was created
        version_dir = os.path.join(temp_dir, "version-1")
        assert os.path.exists(version_dir)
        
        # Check that the file was moved
        moved_file = os.path.join(version_dir, f"{base_name} (Karaoke).mp4")
        assert os.path.exists(moved_file)
        assert not os.path.exists(karaoke_file)
        
        # Check that the WAV file path was returned
        assert result == wav_file

    def test_backup_existing_outputs_increments_version(self, basic_karaoke_gen, temp_dir):
        """Test that backup increments version number."""
        artist = "Test Artist"
        title = "Test Song"
        
        # Create existing version directories
        os.makedirs(os.path.join(temp_dir, "version-1"))
        os.makedirs(os.path.join(temp_dir, "version-2"))
        
        # Create the WAV file
        base_name = f"{artist} - {title}"
        wav_file = os.path.join(temp_dir, f"{base_name}.wav")
        with open(wav_file, 'w') as f:
            f.write("audio content")
        
        basic_karaoke_gen.file_handler.backup_existing_outputs(
            temp_dir, artist, title
        )
        
        # Check that version-3 was created
        assert os.path.exists(os.path.join(temp_dir, "version-3"))

    def test_backup_existing_outputs_raises_on_missing_wav(self, basic_karaoke_gen, temp_dir):
        """Test that exception is raised when no WAV file is found."""
        artist = "Test Artist"
        title = "Test Song"
        
        with pytest.raises(Exception, match="No input audio file found"):
            basic_karaoke_gen.file_handler.backup_existing_outputs(
                temp_dir, artist, title
            )

    def test_backup_existing_outputs_backs_up_lyrics_dir(self, basic_karaoke_gen, temp_dir):
        """Test that lyrics directory is backed up."""
        artist = "Test Artist"
        title = "Test Song"
        
        # Create WAV file
        base_name = f"{artist} - {title}"
        wav_file = os.path.join(temp_dir, f"{base_name}.wav")
        with open(wav_file, 'w') as f:
            f.write("audio content")
        
        # Create lyrics directory with a file
        lyrics_dir = os.path.join(temp_dir, "lyrics")
        os.makedirs(lyrics_dir)
        lyrics_file = os.path.join(lyrics_dir, "lyrics.txt")
        with open(lyrics_file, 'w') as f:
            f.write("lyrics content")
        
        basic_karaoke_gen.file_handler.backup_existing_outputs(
            temp_dir, artist, title
        )
        
        # Check that lyrics were backed up
        version_dir = os.path.join(temp_dir, "version-1")
        backed_up_lyrics = os.path.join(version_dir, "lyrics", "lyrics.txt")
        assert os.path.exists(backed_up_lyrics)
        
        # Original lyrics directory should be removed
        assert not os.path.exists(lyrics_dir)
