from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.automated_workflow import AutomatedWorkflow
from src.config_manager import ConfigManager
from src.ffmpeg_utils import FFmpegUtils
from src.video_processing.video_compressor import DynamicVideoCompressor


@pytest.fixture
def config_manager():
    return ConfigManager()


def test_cae_guardrail_switches_to_segmentation(tmp_path, config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)
    video_info = {'width': 1920, 'height': 1080, 'duration': 600.0, 'fps': 24.0}
    input_path = tmp_path / "clip.mp4"
    output_path = tmp_path / "clip_out.mp4"
    input_path.write_bytes(b"data")

    segments_dir = tmp_path / "segments"
    segments_dir.mkdir(parents=True)

    guardrail_params = {'width': 320, 'height': 180, 'fps': 20}

    with patch.object(
        compressor,
        "_select_resolution_fps_by_bpp",
        return_value=guardrail_params,
    ), patch.object(
        compressor,
        "_compress_with_segmentation",
        return_value={
            'success': True,
            'method': 'segmentation',
            'segments': [{'path': str(segments_dir / "segment_001.mp4"), 'size_mb': 5.0}],
            'segments_folder': str(segments_dir),
            'is_segmented_output': True,
        }
    ) as mock_segmentation:
        result = compressor._compress_with_cae_discord_10mb(
            str(input_path),
            str(output_path),
            video_info,
            platform_config={},
            platform='discord'
        )

    assert result['guardrail_triggered'] is True
    assert result['guardrail_reason'] == 'discord_resolution_guardrail'
    mock_segmentation.assert_called_once()


def test_workflow_tracks_guardrail_segmentation(tmp_path, config_manager, monkeypatch):
    hardware = MagicMock()
    video_compressor = MagicMock()
    gif_generator = MagicMock()
    config_manager._set_nested_value('gif_settings.multiprocessing.use_dynamic_analysis', False)
    workflow = AutomatedWorkflow(config_manager, hardware, video_compressor=video_compressor, gif_generator=gif_generator)

    workflow.input_dir = tmp_path / "input"
    workflow.output_dir = tmp_path / "output"
    workflow.temp_dir = tmp_path / "temp"
    workflow.failures_dir = tmp_path / "failures"
    for directory in [workflow.input_dir, workflow.output_dir, workflow.temp_dir, workflow.failures_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    workflow._record_success_cache = MagicMock()

    category = workflow.input_dir / "category"
    category.mkdir()
    video_file = category / "clip.mp4"
    video_file.write_bytes(b"rawdata")

    segments_folder = workflow.output_dir / "category" / "clip_segments"
    segments_folder.mkdir(parents=True, exist_ok=True)
    segment_mp4 = segments_folder / "clip_part_01.mp4"
    segment_mp4.write_bytes(b"segmentdata")

    guardrail_result = {
        'success': True,
        'method': 'segmentation',
        'guardrail_triggered': True,
        'guardrail_reason': 'discord_resolution_guardrail',
        'segments_folder': str(segments_folder),
        'segments': [{'path': str(segment_mp4), 'size_mb': 4.5}],
        'is_segmented_output': True,
    }
    video_compressor.compress_video.return_value = guardrail_result

    monkeypatch.setattr(FFmpegUtils, "extract_thumbnail_image", lambda *args, **kwargs: None)

    result_path = workflow._ensure_mp4_format(video_file, max_size_mb=10.0)

    assert isinstance(result_path, Path)
    assert result_path == segments_folder

    guardrail_events = workflow.analysis_tracker.counts.get('guardrail_events', 0)
    segmentation_events = workflow.analysis_tracker.counts.get('segmentation_events', 0)

    assert guardrail_events == 1
    assert segmentation_events == 1


def test_cae_refine_respects_min_resolution(config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)

    video_info = {
        'width': 1920,
        'height': 1080,
        'duration': 86.9,
        'fps': 60.0,
        'motion_level': 'high'
    }
    initial_params = {
        'bitrate': 896,
        'width': 1280,
        'height': 720,
        'fps': 24,
        'audio_bitrate': 64,
        'encoder': 'libx264'
    }
    quality_result = {'ssim_score': 0.90, 'vmaf_score': None}

    strategy = compressor._calculate_refinement_strategy(
        quality_pass=False,
        output_size_mb=9.94,
        size_limit_mb=9.95,
        initial_params=initial_params,
        audio_bitrate_kbps=64,
        video_info=video_info,
        duration=video_info['duration'],
        bitrate_step=1.0,
        refine_pass=2,
        quality_result=quality_result
    )

    assert strategy['strategy'] != 'reduce_resolution'
    assert strategy['adjusted_params']['height'] == 720


def test_cae_guardrail_override_drops_resolution(tmp_path, config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)

    # Lower the BPP floor so the override produces an acceptable plan (no segmentation)
    discord_profile = compressor.config.config['video_compression']['profiles']['discord_10mb']
    discord_profile['bpp_floor']['normal'] = 0.015
    discord_profile.setdefault('guardrails', {})['min_short_side_px'] = 240
    assert compressor.config.get('video_compression.profiles.discord_10mb.bpp_floor.normal') == 0.015

    video_info = {
        'width': 1920,
        'height': 1080,
        'duration': 744.85,
        'fps': 24.0,
        'motion_level': 'low',
        'size_mb': 118.0,
    }
    input_path = tmp_path / "long_clip.mp4"
    input_path.write_bytes(b"raw")
    output_path = tmp_path / "long_clip_out.mp4"

    def fake_select(*args, **kwargs):
        enforce_guard = kwargs.get('enforce_min_resolution', False)
        if enforce_guard:
            return {'width': 1920, 'height': 1080, 'fps': 24}
        assert kwargs.get('min_resolution_override') == (640, 360)
        assert kwargs.get('min_fps_override') == 12
        return {'width': 640, 'height': 360, 'fps': 12}

    original_builder = compressor._build_cae_encode_params

    with patch.object(
        compressor,
        "_select_resolution_fps_by_bpp",
        side_effect=fake_select,
    ) as mock_select, patch.object(
        compressor,
        "_execute_two_pass_x264",
        side_effect=StopIteration("stop"),
    ), patch.object(
        compressor,
        "_build_cae_encode_params",
        wraps=original_builder,
    ) as mock_build, patch.object(
        compressor,
        "_compress_with_segmentation",
        return_value=None,
    ) as mock_seg:
        with pytest.raises(StopIteration):
            compressor._compress_with_cae_discord_10mb(
                str(input_path),
                str(output_path),
                video_info,
                platform_config={},
                platform='discord'
            )

    assert mock_select.call_count >= 2
    assert not mock_seg.called
    built_params = mock_build.call_args_list[0].args[0]
    assert built_params['width'] == 640
    assert built_params['height'] == 360
    assert built_params['fps'] == 12


def test_cae_bpp_guardrail_routes_to_segmentation(tmp_path, config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)

    video_info = {
        'width': 1920,
        'height': 1080,
        'duration': 744.85,
        'fps': 24.0,
        'motion_level': 'low',
        'size_mb': 118.0,
    }
    input_path = tmp_path / "bpp_fail.mp4"
    input_path.write_bytes(b"raw")
    output_path = tmp_path / "bpp_fail_out.mp4"

    def fake_select(*args, **kwargs):
        enforce_guard = kwargs.get('enforce_min_resolution', False)
        if enforce_guard:
            return {'width': 1920, 'height': 1080, 'fps': 24}
        return {'width': 640, 'height': 360, 'fps': 12}

    segmentation_payload = {
        'success': True,
        'method': 'segmentation',
        'is_segmented_output': True,
        'segments': [],
    }

    with patch.object(
        compressor,
        "_select_resolution_fps_by_bpp",
        side_effect=fake_select,
    ) as mock_select, patch.object(
        compressor,
        "_compress_with_segmentation",
        return_value=segmentation_payload,
    ) as mock_seg, patch.object(
        compressor,
        "_execute_two_pass_x264"
    ) as mock_execute:
        result = compressor._compress_with_cae_discord_10mb(
            str(input_path),
            str(output_path),
            video_info,
            platform_config={},
            platform='discord'
        )

    assert mock_select.call_count >= 2
    mock_execute.assert_not_called()
    assert result['guardrail_triggered'] is True
    assert result['guardrail_reason'] == 'discord_bpp_floor_guardrail'
    assert 'bpp_actual' in result['guardrail_details']
    seg_video_info = mock_seg.call_args.args[4]
    assert seg_video_info['segmentation_reason'] == 'discord_bpp_floor_guardrail'