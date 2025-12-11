from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest

from src.automated_workflow import AutomatedWorkflow
from src.config_manager import ConfigManager
from src.ffmpeg_utils import FFmpegUtils
from src.video_processing.video_compressor import DynamicVideoCompressor
from src.video_processing.video_segmenter import VideoSegmenter


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


def test_cae_guardrail_override_respects_minimums(tmp_path, config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)
    min_width, min_height = compressor._get_min_resolution()
    min_fps = compressor.config.get('video_compression.bitrate_validation.min_fps', 20)

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

    segmentation_payload = {
        'success': True,
        'method': 'segmentation',
        'is_segmented_output': True,
        'segments': [],
    }

    original_select = compressor._select_resolution_fps_by_bpp

    with patch.object(
        compressor,
        "_select_resolution_fps_by_bpp",
        wraps=original_select,
    ) as mock_select, patch.object(
        compressor,
        "_compress_with_segmentation",
        return_value=segmentation_payload,
    ) as mock_seg:
        result = compressor._compress_with_cae_discord_10mb(
            str(input_path),
            str(output_path),
            video_info,
            platform_config={},
            platform='discord'
        )

    assert mock_seg.called
    assert result['guardrail_triggered'] is True
    assert result['guardrail_reason'] == 'discord_bpp_floor_guardrail'
    guard_details = result['guardrail_details']
    
    # When guardrail overrides are active, the override values (640x360) are intentionally
    # smaller than the minimums (1280x720). Verify that override values are respected.
    override_cfg = config_manager.get('video_compression.bitrate_validation.guardrail_override', {})
    override_min_res = override_cfg.get('min_resolution', {}) or {}
    override_width = override_min_res.get('width', 640)
    override_height = override_min_res.get('height', 360)
    override_min_fps = override_cfg.get('min_fps', 12)
    
    # Guardrail details should respect override values (which are lower than minimums)
    assert guard_details['width'] >= override_width, f"Width {guard_details['width']} should be >= override {override_width}"
    assert guard_details['height'] >= override_height, f"Height {guard_details['height']} should be >= override {override_height}"
    assert guard_details['fps'] >= override_min_fps, f"FPS {guard_details['fps']} should be >= override {override_min_fps}"
    
    # Verify that guardrail override was attempted
    override_calls = [
        call_obj.kwargs for call_obj in mock_select.call_args_list
        if call_obj.kwargs.get('min_resolution_override') is not None
    ]
    assert override_calls, "Override path should be attempted"
    for kwargs in override_calls:
        # When guardrail overrides are active, enforce_min_resolution should be False
        # to allow the override values to take precedence
        assert kwargs.get('enforce_min_resolution') is False, \
            "enforce_min_resolution should be False when guardrail overrides are active"


def test_cae_bpp_guardrail_routes_to_segmentation(tmp_path, config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)
    min_width, min_height = compressor._get_min_resolution()
    min_fps = compressor.config.get('video_compression.bitrate_validation.min_fps', 20)

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

    segmentation_payload = {
        'success': True,
        'method': 'segmentation',
        'is_segmented_output': True,
        'segments': [],
    }

    original_select = compressor._select_resolution_fps_by_bpp

    with patch.object(
        compressor,
        "_select_resolution_fps_by_bpp",
        wraps=original_select,
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
    guard_details = result['guardrail_details']
    assert guard_details['width'] >= min_width
    assert guard_details['height'] >= min_height
    assert guard_details['fps'] >= min_fps
    seg_video_info = mock_seg.call_args.args[4]
    assert seg_video_info['segmentation_reason'] == 'discord_bpp_floor_guardrail'

    override_calls = [
        call.kwargs for call in mock_select.call_args_list
        if call.kwargs.get('min_resolution_override') is not None
    ]
    for kwargs in override_calls:
        assert kwargs.get('enforce_min_resolution') is True


def _build_test_video(width=1920, height=1080, duration=300.0, fps=30.0):
    return {
        'width': width,
        'height': height,
        'duration': duration,
        'fps': fps,
        'complexity_score': 5.0,
        'motion_level': 'medium',
        'size_bytes': 120 * 1024 * 1024,
    }


def test_optimal_resolution_enforces_config_floor(config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)
    video_info = _build_test_video()

    width, height = compressor._calculate_optimal_resolution(
        video_info,
        target_size_mb=10.0,
        platform_config={}
    )

    assert width >= 1280
    assert height >= 720


def test_segmenter_inherits_min_resolution_floor(config_manager):
    hardware = MagicMock()
    hardware.get_best_encoder.return_value = ("libx264", "software")
    hardware.analyze_optimal_segmentation_workers.return_value = {
        'recommended': 1,
        'maximum_safe': 1,
        'conservative': 1,
    }

    from src.video_processing.video_segmenter import VideoSegmenter

    segmenter = VideoSegmenter(config_manager, hardware)
    video_info = _build_test_video()

    params = segmenter._calculate_segment_compression_params(
        video_info,
        platform_config={},
        target_size_mb=10.0,
        duration=video_info['duration']
    )

    assert params['width'] >= 1280
    assert params['height'] >= 720


def test_minimum_viable_resolution_respects_guardrail_override(config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)
    video_info = _build_test_video()
    profile_cfg = config_manager.get('video_compression.profiles.discord_10mb')

    params = compressor._calculate_minimum_viable_resolution(
        video_info=video_info,
        target_bitrate_kbps=200,
        bpp_min=0.035,
        profile_cfg=profile_cfg,
        encoder='libx264',
        enforce_min_resolution=True,
        min_resolution_override=(640, 360)
    )

    assert params['width'] >= 640
    assert params['height'] >= 360
    assert params['width'] <= 1920
    assert params['height'] <= 1080


def test_guardrail_reroute_forces_segmentation_flag(tmp_path, config_manager):
    hardware = MagicMock()
    compressor = DynamicVideoCompressor(config_manager, hardware)
    video_info = _build_test_video()
    input_path = tmp_path / "guard.mp4"
    output_path = tmp_path / "guard_out.mp4"
    input_path.write_bytes(b"x")
    segments_dir = tmp_path / "segments"
    segments_dir.mkdir()

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
        compressor._compress_with_cae_discord_10mb(
            str(input_path),
            str(output_path),
            video_info,
            platform_config={},
            platform='discord'
        )

    assert mock_segmentation.called
    assert mock_segmentation.call_args.kwargs.get('force_segmentation') is True


def test_segmenter_force_segment_skips_single_path(tmp_path, config_manager):
    hardware = MagicMock()
    hardware.get_best_encoder.return_value = ("libx264", "software")
    hardware.analyze_optimal_segmentation_workers.return_value = {
        'recommended': 1,
        'maximum_safe': 1,
        'conservative': 1,
    }
    segmenter = VideoSegmenter(config_manager, hardware)
    input_video = tmp_path / "clip.mp4"
    input_video.write_bytes(b"\x00")
    output_base_path = tmp_path / "clip.mp4"

    with patch.object(segmenter, '_get_video_info', return_value=_build_test_video()), \
         patch.object(segmenter, 'should_segment_video', return_value=False) as mock_should, \
         patch.object(segmenter, '_split_video_into_segments', return_value={
            'success': True,
            'segments': [{'path': str(tmp_path / "seg.mp4"), 'size_mb': 5.0}],
            'temp_files': []
         }) as mock_split, \
         patch.object(segmenter, '_finalize_segments', return_value={
            'success': True,
            'segments': [{'path': str(tmp_path / "seg.mp4"), 'size_mb': 5.0}],
            'output_folder': str(tmp_path / "clip_segments"),
            'num_segments': 1
         }) as mock_finalize, \
         patch.object(segmenter, '_process_single_video') as mock_single:
        result = segmenter.segment_video(
            str(input_video),
            str(output_base_path),
            force_segment=True
        )

    mock_should.assert_not_called()
    mock_split.assert_called_once()
    mock_finalize.assert_called_once()
    mock_single.assert_not_called()
    assert result['success'] is True


def test_segmentation_fallback_errors_on_missing_segments(tmp_path, config_manager):
    hardware = MagicMock()
    workflow = AutomatedWorkflow(config_manager, hardware, video_compressor=MagicMock(), gif_generator=MagicMock())
    workflow._record_success_cache = MagicMock()
    workflow._generate_gifs_from_segments = MagicMock(return_value=True)
    segments_folder = tmp_path / "segments"
    segments_folder.mkdir()

    result = {
        'success': True,
        'method': 'segmentation',
        'is_segmented_output': True,
        'segments_folder': str(segments_folder),
        'segments': []
    }

    status = workflow._handle_segmentation_fallback_result(result, tmp_path / "input.mp4", 10.0)

    assert status == 'error'