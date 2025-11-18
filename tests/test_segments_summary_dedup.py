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

