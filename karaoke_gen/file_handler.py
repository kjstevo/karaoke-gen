import os
import glob
import logging
import shutil
import subprocess
import tempfile
from .utils import sanitize_filename

try:
    import yt_dlp
    YT_DLP_AVAILABLE = True
except ImportError:
    YT_DLP_AVAILABLE = False


# Placeholder class or functions for file handling
class FileHandler:
    def __init__(self, logger, ffmpeg_base_command, create_track_subfolders, dry_run):
        self.logger = logger
        self.ffmpeg_base_command = ffmpeg_base_command
        self.create_track_subfolders = create_track_subfolders
        self.dry_run = dry_run

    def _file_exists(self, file_path):
        """Check if a file exists and log the result."""
        exists = os.path.isfile(file_path)
        if exists:
            self.logger.info(f"File already exists, skipping creation: {file_path}")
        return exists

    def _valid_video_exists(self, file_path):
        """
        Check if a video file exists and is a complete, readable video (not truncated).

        A generation run that gets interrupted mid-encode (crash, Ctrl+C, etc.) can leave
        a non-empty but corrupt video file on disk (e.g. missing moov atom). A resumed run
        must not mistake that for a valid cached result, since it would be passed on to
        later ffmpeg steps and fail there instead.
        """
        if not os.path.isfile(file_path):
            return False

        try:
            result = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0", file_path],
                capture_output=True,
                text=True,
            )
        except FileNotFoundError:
            self.logger.warning("ffprobe not found on PATH, cannot validate cached video file, assuming valid")
            return True

        if result.returncode != 0 or "video" not in result.stdout:
            self.logger.warning(f"Existing file is not a valid video, will regenerate: {file_path}")
            return False

        self.logger.info(f"File already exists, skipping creation: {file_path}")
        return True

    # Placeholder methods - to be filled by user moving code
    def copy_input_media(self, input_media, output_filename_no_extension):
        self.logger.debug(f"Copying media from local path {input_media} to filename {output_filename_no_extension} + existing extension")

        copied_file_name = output_filename_no_extension + os.path.splitext(input_media)[1]
        self.logger.debug(f"Target filename: {copied_file_name}")

        # Check if source and destination are the same
        if os.path.abspath(input_media) == os.path.abspath(copied_file_name):
            self.logger.info("Source and destination are the same file, skipping copy")
            return input_media

        self.logger.debug(f"Copying {input_media} to {copied_file_name}")
        shutil.copy2(input_media, copied_file_name)

        return copied_file_name

    def download_audio_from_fetcher_result(self, filepath, output_filename_no_extension):
        """
        Handle audio that was downloaded via the AudioFetcher.
        
        This method copies/moves the downloaded file to the expected location
        and returns the path with the correct naming convention.
        
        Args:
            filepath: Path to the downloaded audio file from AudioFetcher
            output_filename_no_extension: Desired output filename without extension
            
        Returns:
            Path to the renamed/copied audio file
        """
        if not os.path.isfile(filepath):
            self.logger.error(f"Downloaded file not found: {filepath}")
            return None
            
        # Get the extension from the downloaded file
        ext = os.path.splitext(filepath)[1]
        target_path = f"{output_filename_no_extension}{ext}"
        
        # If source and target are the same, no action needed
        if os.path.abspath(filepath) == os.path.abspath(target_path):
            self.logger.debug(f"Downloaded file already at target location: {target_path}")
            return target_path
            
        # Copy the file to the target location
        self.logger.debug(f"Copying downloaded file from {filepath} to {target_path}")
        shutil.copy2(filepath, target_path)
        
        return target_path

    def download_video(self, url, output_filename_no_extension, cookies_str=None):
        """
        Download audio from a URL (YouTube, etc.) using yt-dlp.
        
        This method downloads the best quality audio from a URL and saves it
        to the specified output path. It handles YouTube and other video platforms
        supported by yt-dlp.
        
        Args:
            url: URL to download from (YouTube, Vimeo, etc.)
            output_filename_no_extension: Output filename without extension
            cookies_str: Optional cookies string for authenticated downloads
            
        Returns:
            Path to downloaded audio file, or None if failed
        """
        if not YT_DLP_AVAILABLE:
            self.logger.error("yt-dlp is not installed. Install with: pip install yt-dlp")
            return None
        
        self.logger.info(f"Downloading audio from URL: {url}")
        
        # Configure yt-dlp options
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': output_filename_no_extension + '.%(ext)s',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'best',
                'preferredquality': '0',  # Best quality
            }],
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            # Anti-detection options
            'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'retries': 3,
            'fragment_retries': 3,
            'http_headers': {
                'Accept-Language': 'en-US,en;q=0.9',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            },
        }
        
        # Handle cookies if provided - use safe tempfile pattern to avoid leaks
        cookie_file_path = None
        if cookies_str:
            try:
                # Use context manager to safely write cookies file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as cookie_file:
                    cookie_file.write(cookies_str)
                    cookie_file_path = cookie_file.name
                ydl_opts['cookiefile'] = cookie_file_path
            except Exception as e:
                self.logger.warning(f"Failed to write cookies file: {e}")
                cookie_file_path = None
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                # Extract info first to get actual filename
                info = ydl.extract_info(url, download=True)
                
                if info is None:
                    self.logger.error("Failed to extract info from URL")
                    return None
                
                # Find the downloaded file
                # The actual filename might differ from template due to post-processing
                downloaded_file = None
                
                # Check common extensions
                for ext in ['m4a', 'opus', 'webm', 'mp3', 'flac', 'wav', 'ogg', 'aac']:
                    candidate = f"{output_filename_no_extension}.{ext}"
                    if os.path.exists(candidate):
                        downloaded_file = candidate
                        break
                
                if downloaded_file is None:
                    # Try to find any audio file with matching prefix
                    import glob
                    matches = glob.glob(f"{output_filename_no_extension}.*")
                    audio_extensions = ['.m4a', '.opus', '.webm', '.mp3', '.flac', '.wav', '.ogg', '.aac']
                    for match in matches:
                        if any(match.endswith(ext) for ext in audio_extensions):
                            downloaded_file = match
                            break
                
                if downloaded_file and os.path.exists(downloaded_file):
                    self.logger.info(f"Successfully downloaded: {downloaded_file}")
                    return downloaded_file
                else:
                    self.logger.error("Downloaded file not found after yt-dlp completed")
                    return None
                    
        except yt_dlp.DownloadError as e:
            self.logger.error(f"yt-dlp download error: {e}")
            return None
        except Exception as e:
            self.logger.error(f"Failed to download from URL: {e}")
            return None
        finally:
            # Clean up cookie file if we created one
            if cookie_file_path is not None:
                try:
                    os.unlink(cookie_file_path)
                except Exception:
                    pass

    def extract_metadata_from_url(self, url):
        """
        Extract metadata (artist, title) from a URL without downloading.
        
        Uses yt-dlp to fetch video metadata including title, uploader/artist,
        and other information that can be used for the karaoke generation.
        
        Args:
            url: URL to extract metadata from
            
        Returns:
            Dict with 'artist', 'title', 'duration', and 'raw_info', or None if failed
        """
        if not YT_DLP_AVAILABLE:
            self.logger.error("yt-dlp is not installed. Install with: pip install yt-dlp")
            return None
        
        self.logger.info(f"Extracting metadata from URL: {url}")
        
        ydl_opts = {
            'quiet': True,
            'no_warnings': True,
            'extract_flat': False,
            'skip_download': True,
        }
        
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                
                if info is None:
                    self.logger.error("Failed to extract metadata from URL")
                    return None
                
                # Try to extract artist and title from various fields
                raw_title = info.get('title', '')
                uploader = info.get('uploader', '') or info.get('channel', '') or info.get('artist', '')
                duration = info.get('duration', 0)
                
                # Attempt to parse "Artist - Title" format from title
                artist = None
                title = raw_title
                
                if ' - ' in raw_title:
                    parts = raw_title.split(' - ', 1)
                    if len(parts) == 2:
                        artist = parts[0].strip()
                        title = parts[1].strip()
                
                # Fall back to uploader as artist if not found in title
                if not artist:
                    artist = uploader
                
                # Clean up title (remove common suffixes like "(Official Video)")
                title_cleanup_patterns = [
                    '(official video)', '(official music video)', '(official audio)',
                    '(lyric video)', '(lyrics)', '(visualizer)', '(music video)',
                    '[official video]', '[official music video]', '[official audio]',
                    '(hd)', '(4k)', '(remastered)', '| official video', '| official audio',
                ]
                title_lower = title.lower()
                for pattern in title_cleanup_patterns:
                    if pattern in title_lower:
                        idx = title_lower.find(pattern)
                        title = title[:idx].strip()
                        title_lower = title.lower()
                
                return {
                    'artist': artist,
                    'title': title,
                    'duration': duration,
                    'raw_info': info,
                }
                
        except Exception as e:
            self.logger.error(f"Failed to extract metadata from URL: {e}")
            return None

    def extract_still_image_from_video(self, input_filename, output_filename_no_extension):
        output_filename = output_filename_no_extension + ".png"
        self.logger.info(f"Extracting still image from position 30s input media")
        ffmpeg_command = f'{self.ffmpeg_base_command} -i "{input_filename}" -ss 00:00:30 -vframes 1 "{output_filename}"'
        self.logger.debug(f"Running command: {ffmpeg_command}")
        os.system(ffmpeg_command)
        return output_filename

    def convert_to_wav(self, input_filename, output_filename_no_extension):
        """Convert input audio to WAV format, with input validation."""
        # Validate input file exists and is readable
        if not os.path.isfile(input_filename):
            raise Exception(f"Input audio file not found: {input_filename}")

        if os.path.getsize(input_filename) == 0:
            raise Exception(f"Input audio file is empty: {input_filename}")

        # Validate input file format using ffprobe
        probe_command = f'ffprobe -v error -show_entries stream=codec_type -of default=noprint_wrappers=1 "{input_filename}"'
        probe_output = os.popen(probe_command).read()

        if "codec_type=audio" not in probe_output:
            raise Exception(f"No valid audio stream found in file: {input_filename}")

        output_filename = output_filename_no_extension + ".wav"
        self.logger.info(f"Converting input media to audio WAV file")
        ffmpeg_command = f'{self.ffmpeg_base_command} -n -i "{input_filename}" "{output_filename}"'
        self.logger.debug(f"Running command: {ffmpeg_command}")
        if not self.dry_run:
            os.system(ffmpeg_command)
        return output_filename

    def setup_output_paths(self, output_dir, artist, title):
        if title is None and artist is None:
            raise ValueError("Error: At least title or artist must be provided")

        # If only title is provided, use it for both artist and title portions of paths
        if artist is None:
            sanitized_title = sanitize_filename(title)
            artist_title = sanitized_title
        else:
            sanitized_artist = sanitize_filename(artist)
            sanitized_title = sanitize_filename(title)
            artist_title = f"{sanitized_artist} - {sanitized_title}"

        track_output_dir = output_dir
        if self.create_track_subfolders:
            track_output_dir = os.path.join(output_dir, f"{artist_title}")

        if not os.path.exists(track_output_dir):
            self.logger.debug(f"Output dir {track_output_dir} did not exist, creating")
            os.makedirs(track_output_dir)

        return track_output_dir, artist_title

    def backup_existing_outputs(self, track_output_dir, artist, title):
        """
        Backup existing outputs to a versioned folder.

        Args:
            track_output_dir: The directory containing the track outputs
            artist: The artist name
            title: The track title

        Returns:
            The path to the original input audio file
        """
        self.logger.info(f"Backing up existing outputs for {artist} - {title}")

        # Sanitize artist and title for filenames
        sanitized_artist = sanitize_filename(artist)
        sanitized_title = sanitize_filename(title)
        base_name = f"{sanitized_artist} - {sanitized_title}"

        # Find the next available version number
        version_num = 1
        while os.path.exists(os.path.join(track_output_dir, f"version-{version_num}")):
            version_num += 1

        version_dir = os.path.join(track_output_dir, f"version-{version_num}")
        self.logger.info(f"Creating backup directory: {version_dir}")
        os.makedirs(version_dir, exist_ok=True)

        # Find the input audio file (we'll need this for re-running the transcription)
        input_audio_wav = os.path.join(track_output_dir, f"{base_name}.wav")
        if not os.path.exists(input_audio_wav):
            self.logger.warning(f"Input audio file not found: {input_audio_wav}")
            # Try to find any WAV file
            wav_files = glob.glob(os.path.join(track_output_dir, "*.wav"))
            if wav_files:
                input_audio_wav = wav_files[0]
                self.logger.info(f"Using alternative input audio file: {input_audio_wav}")
            else:
                raise Exception(f"No input audio file found in {track_output_dir}")

        # List of file patterns to move
        file_patterns = [
            f"{base_name} (With Vocals).*",
            f"{base_name} (Karaoke).*",
            f"{base_name} (Final Karaoke*).*",
        ]

        # Move files matching patterns to version directory
        for pattern in file_patterns:
            for file_path in glob.glob(os.path.join(track_output_dir, pattern)):
                if os.path.isfile(file_path):
                    dest_path = os.path.join(version_dir, os.path.basename(file_path))
                    self.logger.info(f"Moving {file_path} to {dest_path}")
                    if not self.dry_run:
                        shutil.move(file_path, dest_path)

        # Also backup the lyrics directory
        lyrics_dir = os.path.join(track_output_dir, "lyrics")
        if os.path.exists(lyrics_dir):
            lyrics_backup_dir = os.path.join(version_dir, "lyrics")
            self.logger.info(f"Backing up lyrics directory to {lyrics_backup_dir}")
            if not self.dry_run:
                shutil.copytree(lyrics_dir, lyrics_backup_dir)
                # Remove the original lyrics directory
                shutil.rmtree(lyrics_dir)

        return input_audio_wav
