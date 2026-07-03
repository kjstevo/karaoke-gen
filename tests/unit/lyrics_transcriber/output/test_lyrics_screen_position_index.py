"""Regression test for position_to_line_index floating-point rounding (preview vs final mismatch).

Preview-mode scales top_padding by 1/6 (e.g. 200 -> 33.333...), which makes
calculate_line_positions() return values whose spacing isn't an exact multiple of
line_height once floating-point representation error accumulates. Floor division
in position_to_line_index truncated the affected index down by one, corrupting the
screen-transition "extra reading time" bonus and making preview video line entrances
diverge from the final (4k, integer top_padding) render for the same segments.
"""
from karaoke_gen.lyrics_transcriber.output.ass.config import ScreenConfig
from karaoke_gen.lyrics_transcriber.output.ass.lyrics_screen import PositionCalculator


def test_position_to_line_index_matches_for_fractional_top_padding():
    """Reproduces the real preview-mode config: top_padding = 200 * (1/6)."""
    config = ScreenConfig(
        line_height=50,
        max_visible_lines=4,
        top_padding=200 * (1 / 6),
        video_width=640,
        video_height=360,
    )

    positions = PositionCalculator.calculate_line_positions(config)

    for expected_index, y_position in enumerate(positions):
        assert PositionCalculator.position_to_line_index(y_position, config) == expected_index


def test_position_to_line_index_matches_for_integer_top_padding():
    """Sanity check: the clean-integer (final render) config always worked."""
    config = ScreenConfig(
        line_height=250,
        max_visible_lines=4,
        top_padding=200,
        video_width=3840,
        video_height=2160,
    )

    positions = PositionCalculator.calculate_line_positions(config)

    for expected_index, y_position in enumerate(positions):
        assert PositionCalculator.position_to_line_index(y_position, config) == expected_index
