from pathlib import Path

from src.config_manager import ConfigManager
from src.gif_processing.gif_generator import GifGenerator
from src.gif_processing.gif_optimizer import GifOptimizer


def test_build_filter_profile_clamps_palette_colors():
    generator = GifGenerator(ConfigManager())
    settings = {
        'fps': 22,
        'palette_max_colors': 400,
        'colors': 400,
        'width': 360,
        'height': -1,
        'max_size_mb': 10,
    }

    profile = generator._build_filter_profile(settings, duration=5.0)

    assert profile['max_colors'] <= 256


def test_apply_settings_override_clamps_palette_colors():
    generator = GifGenerator(ConfigManager())
    base_settings = generator._get_platform_settings(platform=None, max_size_mb=10)
    overrides = {'colors': 999, 'palette_max_colors': 512}

    updated = generator._apply_settings_override(base_settings, overrides)

    assert updated['colors'] == 256
    assert updated['palette_max_colors'] == 256


def test_calculate_required_parameters_caps_colors():
    optimizer = GifOptimizer(ConfigManager())
    current_params = {'width': 360, 'height': 360, 'fps': 20, 'colors': 400, 'lossy': 0}
    gif_info = {'duration': 10.0, 'width': 360, 'height': 360, 'fps': 20}

    params = optimizer._calculate_required_parameters(40.0, 10.0, current_params, gif_info)

    assert params['colors'] <= 256


def test_reencode_with_params_clamps_palette_colors(monkeypatch, tmp_path):
    optimizer = GifOptimizer(ConfigManager())
    captured = {}

    def fake_create_single_gif(self, source_video, output_path, settings, start_time, duration, **kwargs):
        captured['colors'] = settings['colors']
        captured['palette_max_colors'] = settings.get('palette_max_colors')
        Path(output_path).write_bytes(b'GIF89a')
        return {'success': True}

    monkeypatch.setattr(GifGenerator, "_create_single_gif", fake_create_single_gif)

    params = {'width': 320, 'height': -1, 'fps': 15, 'colors': 500, 'lossy': 0}
    video_info = {'duration': 5.0}
    output_path = tmp_path / "out.gif"

    result = optimizer._reencode_with_params(
        source_video="input.mp4",
        output_path=str(output_path),
        target_size_mb=10.0,
        params=params,
        video_info=video_info,
    )

    assert result is True
    assert captured['colors'] == 256
    assert captured['palette_max_colors'] == 256

