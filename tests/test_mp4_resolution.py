from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.automated_workflow import AutomatedWorkflow
from src.config_manager import ConfigManager
from src.ffmpeg_utils import FFmpegUtils


@pytest.fixture
def workflow(tmp_path):
    config = ConfigManager()
    hardware = MagicMock()
    video_compressor = MagicMock()
    gif_generator = MagicMock()
    aw = AutomatedWorkflow(config, hardware, video_compressor=video_compressor, gif_generator=gif_generator)
    aw.output_dir = tmp_path / "output"
    aw.output_dir.mkdir(parents=True, exist_ok=True)
    aw.temp_dir = tmp_path / "temp"
    aw.temp_dir.mkdir(parents=True, exist_ok=True)
    aw.input_dir = tmp_path / "input"
    aw.input_dir.mkdir(parents=True, exist_ok=True)
    aw.failures_dir = tmp_path / "failures"
    aw.failures_dir.mkdir(parents=True, exist_ok=True)
    validator = MagicMock()
    validator.is_valid_gif_with_enhanced_checks.return_value = (True, None)
    validator.get_file_size_mb.return_value = 5.0
    aw.file_validator = validator
    aw._record_success_cache = MagicMock()
    return aw


def test_resolve_mp4_source_path_finds_segments_copy(workflow):
    category_dir = workflow.output_dir / "clips"
    segments_dir = category_dir / "sample_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    fallback_mp4 = segments_dir / "sample.mp4"
    fallback_mp4.write_bytes(b"frame-data")

    original_mp4 = category_dir / "sample.mp4"
    resolved = workflow._resolve_mp4_source_path(original_mp4)

    assert resolved == fallback_mp4


class RecordingGifGenerator:
    def __init__(self):
        self.calls = []

    def create_gif(self, **kwargs):
        self.calls.append(kwargs)
        output_path = Path(kwargs['output_path'])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"GIF89a")
        return {'success': True, 'size_mb': 5.0}


def test_generate_and_optimize_gif_uses_resolved_source(monkeypatch, tmp_path):
    config = ConfigManager()
    hardware = MagicMock()
    gif_generator = RecordingGifGenerator()
    workflow = AutomatedWorkflow(config, hardware, video_compressor=MagicMock(), gif_generator=gif_generator)

    workflow.output_dir = tmp_path / "output"
    workflow.output_dir.mkdir(parents=True, exist_ok=True)
    workflow.temp_dir = tmp_path / "temp"
    workflow.temp_dir.mkdir(parents=True, exist_ok=True)
    workflow.input_dir = tmp_path / "input"
    workflow.input_dir.mkdir(parents=True, exist_ok=True)
    workflow.failures_dir = tmp_path / "failures"
    workflow.failures_dir.mkdir(parents=True, exist_ok=True)

    validator = MagicMock()
    validator.is_valid_gif_with_enhanced_checks.return_value = (True, None)
    validator.get_file_size_mb.return_value = 5.0
    workflow.file_validator = validator
    workflow._record_success_cache = MagicMock()

    category_dir = workflow.output_dir / "clips"
    category_dir.mkdir(parents=True, exist_ok=True)
    segments_dir = category_dir / "sample_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    fallback_mp4 = segments_dir / "sample.mp4"
    fallback_mp4.write_bytes(b"frame-data")

    original_mp4 = category_dir / "sample.mp4"

    monkeypatch.setattr(FFmpegUtils, "get_video_duration", lambda *_args, **_kwargs: 12.0)

    result = workflow._generate_and_optimize_gif(original_mp4, max_size_mb=10.0)

    assert result is True
    assert gif_generator.calls, "GIF generator should be invoked"
    assert gif_generator.calls[0]['input_video'] == str(fallback_mp4)
    final_gif = category_dir / "sample.gif"
    assert final_gif.exists(), "Final GIF should be written to the original category directory"

