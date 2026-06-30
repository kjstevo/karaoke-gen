import os
import pytest
from unittest.mock import MagicMock, patch
import json
import logging
from karaoke_gen.karaoke_gen import KaraokePrep
import tempfile

class TestInitialization:
    def test_init_with_defaults(self, mock_logger):
        """Test initialization with default parameters."""
        karaoke_gen = KaraokePrep(logger=mock_logger)
        
        assert karaoke_gen.input_media is None
        assert karaoke_gen.artist is None
        assert karaoke_gen.title is None
        assert karaoke_gen.dry_run is False
        assert karaoke_gen.logger is mock_logger
        assert karaoke_gen.output_dir == "."
        assert karaoke_gen.lossless_output_format == "flac"
        assert karaoke_gen.create_track_subfolders is False
        
    def test_init_with_custom_params(self, mock_logger):
        """Test initialization with custom parameters."""
        with patch('os.makedirs'):
            karaoke_gen = KaraokePrep(
                input_media="test_media.mp4",
                artist="Test Artist",
                title="Test Title",
                dry_run=True,
                logger=mock_logger,
                output_dir="test_output",
                lossless_output_format="WAV",
                create_track_subfolders=True
            )
        
        assert karaoke_gen.input_media == "test_media.mp4"
        assert karaoke_gen.artist == "Test Artist"
        assert karaoke_gen.title == "Test Title"
        assert karaoke_gen.dry_run is True
        assert karaoke_gen.logger is mock_logger
        assert karaoke_gen.output_dir == "test_output"
        assert karaoke_gen.lossless_output_format == "wav"
        assert karaoke_gen.create_track_subfolders is True
    
    def test_init_creates_output_dir(self, mock_logger, temp_dir):
        """Test that initialization creates the output directory if it doesn't exist."""
        output_dir = os.path.join(temp_dir, "new_dir")
        
        with patch('os.makedirs') as mock_makedirs:
            karaoke_gen = KaraokePrep(
                logger=mock_logger,
                output_dir=output_dir
            )
            mock_makedirs.assert_called_once_with(output_dir)
    
    def test_init_with_custom_style_params(self, mock_logger, temp_dir):
        """Test initialization with custom style parameters."""
        # Create a temporary style params file
        style_params = {
            "intro": {
                "video_duration": 10,
                "background_color": "#FF0000",
                "background_image": None,
                "existing_image": None,
                "font": "CustomFont.ttf",
                "artist_color": "#ffffff",
                "artist_gradient": None,
                "artist_text_transform": None,
                "title_color": "#ffffff",
                "title_gradient": None,
                "title_text_transform": None,
                "title_region": "10, 10, 100, 50",
                "artist_region": "10, 70, 100, 50",
                "extra_text": None,
                "extra_text_color": "#ffffff",
                "extra_text_gradient": None,
                "extra_text_region": "10, 130, 100, 50",
                "extra_text_text_transform": None,
            },
            "end": {
                "video_duration": 8,
                "background_color": "#0000FF",
                "background_image": None,
                "existing_image": None,
                "font": "CustomFont.ttf",
                "artist_color": "#ffffff",
                "artist_gradient": None,
                "artist_text_transform": None,
                "title_color": "#ffffff",
                "title_gradient": None,
                "title_text_transform": None,
                "title_region": None,
                "artist_region": None,
                "extra_text": "Thank you!",
                "extra_text_color": "#ffffff",
                "extra_text_gradient": None,
                "extra_text_region": None,
                "extra_text_text_transform": None,
            },
            "karaoke": {
                "background_color": "#000000",
                "background_image": None,
                "font": "Noto Sans",
                "font_path": "",
                "ass_name": "Default",
                "primary_color": "112, 112, 247, 255",
                "secondary_color": "255, 255, 255, 255",
                "outline_color": "26, 58, 235, 255",
                "back_color": "0, 0, 0, 0",
                "bold": False,
                "italic": False,
                "underline": False,
                "strike_out": False,
                "scale_x": 100,
                "scale_y": 100,
                "spacing": 0,
                "angle": 0.0,
                "border_style": 1,
                "outline": 1,
                "shadow": 0,
                "margin_l": 0,
                "margin_r": 0,
                "margin_v": 0,
                "encoding": 0,
                "max_line_length": 40,
                "top_padding": 200,
                "font_size": 100,
            },
            "cdg": {
                "font_path": None,
                "instrumental_background": None,
                "title_screen_background": None,
                "outro_background": None,
            }
        }
        
        style_params_path = os.path.join(temp_dir, "style_params.json")
        
        with open(style_params_path, "w") as f:
            json.dump(style_params, f)
        
        karaoke_gen = KaraokePrep(
            logger=mock_logger,
            style_params_json=style_params_path
        )
        
        assert karaoke_gen.intro_video_duration == 10
        assert karaoke_gen.end_video_duration == 8
        assert karaoke_gen.title_format["background_color"] == "#FF0000"
        assert karaoke_gen.end_format["background_color"] == "#0000FF"
    
    def test_init_with_invalid_style_params_file(self, mock_logger):
        """Test initialization with an invalid style params file path."""
        with pytest.raises(SystemExit):
            KaraokePrep(
                logger=mock_logger,
                style_params_json="/nonexistent/path.json"
            )
    
    def test_init_with_invalid_style_params_json(self, mock_logger, temp_dir):
        """Test initialization with invalid JSON in style params file."""
        # Create a file with invalid JSON
        style_params_path = os.path.join(temp_dir, "invalid_style_params.json")
        
        with open(style_params_path, "w") as f:
            f.write("This is not valid JSON")
        
        with pytest.raises(SystemExit):
            KaraokePrep(
                logger=mock_logger,
                style_params_json=style_params_path
            )
    
    def test_parse_region(self):
        """Test the parse_region static method."""
        # Instantiate a basic KaraokePrep to access video_generator
        kp = KaraokePrep(logger=MagicMock())
        # Test valid region string
        # Call on video_generator
        region = kp.video_generator.parse_region("10,20,300,400")
        assert region == (10, 20, 300, 400)

        # Test None input
        assert kp.video_generator.parse_region(None) is None

        # Test invalid format
        with pytest.raises(ValueError, match="Invalid region format: 10,20,300. Expected 4 elements: 'x,y,width,height'"):
            kp.video_generator.parse_region("10,20,300")
        with pytest.raises(ValueError, match="Invalid region format: 10,twenty,300,400. Could not convert to integers. Expected format: 'x,y,width,height'"):
            kp.video_generator.parse_region("10,twenty,300,400")


def test_transcriber_config_accepts_replicate_api_token():
    from karaoke_gen.lyrics_transcriber.core.config import TranscriberConfig
    config = TranscriberConfig(replicate_api_token="r8_abc123")
    assert config.replicate_api_token == "r8_abc123"


def test_transcriber_config_replicate_api_token_defaults_to_none():
    from karaoke_gen.lyrics_transcriber.core.config import TranscriberConfig
    config = TranscriberConfig()
    assert config.replicate_api_token is None
