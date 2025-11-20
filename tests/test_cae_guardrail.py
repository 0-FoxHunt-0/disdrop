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

