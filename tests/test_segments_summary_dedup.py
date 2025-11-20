import os
import time
from pathlib import Path
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

    # Redirect workflow directories into the temp workspace for isolation
    aw.output_dir = tmp_path
    aw.input_dir = tmp_path / "input"
    aw.temp_dir = tmp_path / "temp"
    aw.failures_dir = tmp_path / "failures"
    aw.input_dir.mkdir(parents=True, exist_ok=True)
    aw.temp_dir.mkdir(parents=True, exist_ok=True)
    aw.failures_dir.mkdir(parents=True, exist_ok=True)

    return aw


def test_cleanup_segments_summary_files_collapses_duplicates(workflow, tmp_path):
    segments_folder = tmp_path / "~[HMV] Kawaii (ft. RicedOutCivic) - Rondoudou Media ⧸ HMV [643295266b81e]_segments"
    segments_folder.mkdir(parents=True)
    # Include a dummy GIF so the folder is not pruned during cleanup
    dummy_segment = segments_folder / "segment_001.gif"
    dummy_segment.write_bytes(b"GIF89a")

    duplicate_names = [
        "~[HMV] Kawaii (ft. RicedOutCivic) - Rondoudou Media [643295266b81e]_comprehensive_summary.txt",
        "~[HMV] Kawaii (ft_comprehensive_summary.txt",
        "~_HMV_ Kawaii _ft. RicedOutCivic_ - Rondoudou Media _643295266b81e__comprehensive_summary.txt",
    ]
    base_time = time.time()
    expected_content = "latest summary content"

    for idx, name in enumerate(duplicate_names):
        summary_path = segments_folder / name
        summary_path.write_text(f"content-{idx}", encoding="utf-8")
        os.utime(summary_path, (base_time + idx, base_time + idx))

    # Overwrite the last file to ensure we know which content should survive
    last_path = segments_folder / duplicate_names[-1]
    last_path.write_text(expected_content, encoding="utf-8")
    os.utime(last_path, (base_time + len(duplicate_names), base_time + len(duplicate_names)))

    workflow.cleanup_segments_summary_files()

    canonical_path = workflow._canonical_segments_summary_path(segments_folder)
    remaining = list(segments_folder.glob("~*_comprehensive_summary.txt"))

    assert canonical_path.exists(), "Canonical summary file should be created"
    assert remaining == [canonical_path], "Only the canonical summary should remain"
    assert canonical_path.read_text(encoding="utf-8") == expected_content


def test_ensure_mp4_in_segments_moves_files_under_output(workflow):
    source_dir = workflow.output_dir / "category"
    source_dir.mkdir(parents=True, exist_ok=True)
    source_mp4 = source_dir / "clip.mp4"
    source_mp4.write_bytes(b"data")

    segments_folder = source_dir / "clip_segments"
    workflow._ensure_mp4_in_segments(source_mp4, segments_folder)

    target = segments_folder / source_mp4.name
    assert target.exists(), "MP4 should be moved into the segments folder"
    assert not source_mp4.exists(), "Original MP4 inside output should be removed after move"
    assert workflow.analysis_tracker.counts.get('mp4_moves') == 1


def test_ensure_mp4_in_segments_copies_external_files(workflow, tmp_path_factory):
    external_dir = tmp_path_factory.mktemp("external_sources")
    source_mp4 = external_dir / "clip.mp4"
    source_mp4.write_bytes(b"data")

    segments_folder = workflow.output_dir / "category" / "clip_segments"
    segments_folder.parent.mkdir(parents=True, exist_ok=True)
    workflow._ensure_mp4_in_segments(source_mp4, segments_folder)

    target = segments_folder / source_mp4.name
    assert target.exists(), "MP4 copy should exist in segments folder"
    assert source_mp4.exists(), "External MP4 should remain untouched"
    assert workflow.analysis_tracker.counts.get('mp4_moves') == 1
