from unittest.mock import MagicMock

import pytest

from src.automated_workflow import AutomatedWorkflow
from src.config_manager import ConfigManager


@pytest.fixture
def workflow(tmp_path):
    config = ConfigManager()
    hardware = MagicMock()
    mock_video = MagicMock()
    mock_gif = MagicMock()
    aw = AutomatedWorkflow(config, hardware, video_compressor=mock_video, gif_generator=mock_gif)

    # Ensure workflow directories are rooted in the temp workspace
    aw.output_dir = tmp_path / "output"
    aw.input_dir = tmp_path / "input"
    aw.temp_dir = tmp_path / "temp"
    aw.failures_dir = tmp_path / "failures"
    aw.output_dir.mkdir(parents=True, exist_ok=True)
    aw.input_dir.mkdir(parents=True, exist_ok=True)
    aw.temp_dir.mkdir(parents=True, exist_ok=True)
    aw.failures_dir.mkdir(parents=True, exist_ok=True)

    return aw


def test_mp4_under_output_is_moved_into_segments(workflow, tmp_path):
    category_dir = workflow.output_dir / "gooned" / "Nude"
    category_dir.mkdir(parents=True)
    mp4_file = category_dir / "example.mp4"
    mp4_file.write_bytes(b"mock-data")

    segments_dir = workflow.output_dir / "gooned" / "Nude" / "example_segments"

    workflow._ensure_mp4_in_segments(mp4_file, segments_dir)

    target = segments_dir / "example.mp4"
    assert target.exists(), "MP4 should be moved into the segments folder"
    assert not mp4_file.exists(), "Source MP4 should be removed from the category directory"
    assert target.read_bytes() == b"mock-data"


def test_mp4_outside_output_is_copied(workflow, tmp_path):
    source_dir = workflow.input_dir
    mp4_file = source_dir / "external.mp4"
    mp4_file.write_bytes(b"external-data")

    segments_dir = workflow.output_dir / "incoming_segments"

    workflow._ensure_mp4_in_segments(mp4_file, segments_dir)

    target = segments_dir / "external.mp4"
    assert target.exists(), "MP4 should be copied into segments folder"
    assert mp4_file.exists(), "Original MP4 outside output should remain in place"
    assert target.read_bytes() == mp4_file.read_bytes()

