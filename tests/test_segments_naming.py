from src.utils.segments_naming import sanitize_segments_base_name, segments_summary_path


def test_sanitize_segments_base_name_replaces_reserved_characters():
    raw = "Alpha⧸Beta:Gamma"
    assert sanitize_segments_base_name(raw) == "Alpha_Beta_Gamma"


def test_sanitize_segments_base_name_uses_fallback_for_empty_results():
    raw = "***"
    assert sanitize_segments_base_name(raw) == "segments_summary"


def test_segments_summary_path_infers_base_from_folder(tmp_path):
    folder = tmp_path / "[HMV] Party ⧸ Time_segments"
    folder.mkdir()

    summary_path = segments_summary_path(folder)
    assert summary_path.parent == folder
    assert summary_path.name == "~[HMV] Party _ Time_comprehensive_summary.txt"


def test_segments_summary_path_respects_explicit_base(tmp_path):
    folder = tmp_path / "custom_segments"
    folder.mkdir()

    summary_path = segments_summary_path(folder, base_name="[Stage] Clip:01")
    assert summary_path.name == "~[Stage] Clip_01_comprehensive_summary.txt"

