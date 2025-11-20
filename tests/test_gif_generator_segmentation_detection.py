import pytest

from src.config_manager import ConfigManager
from src.gif_processing.gif_generator import GifGenerator


@pytest.fixture(scope="module")
def gif_generator():
    return GifGenerator(ConfigManager())


def test_is_segmented_video_detects_segment_tokens(gif_generator, tmp_path):
    video_path = tmp_path / "clip_part_01.mp4"
    assert gif_generator._is_segmented_video(str(video_path)) is True


def test_is_segmented_video_ignores_words_containing_part(gif_generator, tmp_path):
    video_path = tmp_path / "[HMV] It's Party Fuck Time - Rondoudou Media [ph632ad3da27577].mp4"
    assert gif_generator._is_segmented_video(str(video_path)) is False


def test_build_guardrail_overrides_clamps_and_reduces_settings(gif_generator):
    base_settings = {
        'width': 360,
        'fps': 22,
        'colors': 400,
        'palette_max_colors': 400,
    }

    overrides = gif_generator._build_guardrail_overrides(base_settings)

    assert overrides['colors'] <= 256
    assert overrides['palette_max_colors'] <= 256
    assert overrides['width'] < base_settings['width']
    assert overrides['fps'] < base_settings['fps']
