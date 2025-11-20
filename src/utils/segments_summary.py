"""Utilities for writing canonical comprehensive segments summaries."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..ffmpeg_utils import FFmpegUtils
from .segments_naming import segments_summary_path

SegmentsFolder = Union[str, Path]


def write_segments_summary(
    segments_folder: SegmentsFolder,
    base_name: Optional[Union[str, Path]] = None,
    *,
    logger: Optional[logging.Logger] = None,
    analysis_tracker: Optional[Any] = None,
) -> Optional[Path]:
    """
    Generate (and atomically place) the canonical '~*_comprehensive_summary.txt' file.

    Args:
        segments_folder: Folder containing MP4 and/or GIF segments.
        base_name: Optional override for the summary's base name; defaults to folder stem.
        logger: Logger used for diagnostics; falls back to this module's logger.
        analysis_tracker: Optional tracker for recording summary probe timeouts.

    Returns:
        Path to the written summary, or None when no summary was produced.
    """

    log = logger or logging.getLogger(__name__)
    folder = Path(segments_folder)

    mp4_files = sorted(p for p in folder.glob("*.mp4") if p.is_file())
    gif_files = sorted(p for p in folder.glob("*.gif") if p.is_file())

    if not mp4_files and not gif_files:
        log.info("No media segments found in %s; skipping summary creation", folder)
        return None

    summary_path = segments_summary_path(folder, base_name)
    temp_path = summary_path.with_name(f"{summary_path.name}.tmp")

    _clear_windows_attributes(summary_path)
    _remove_file(temp_path)

    mp4_details, mp4_totals = _collect_mp4_details(mp4_files, log)
    gif_details, gif_totals = _collect_gif_details(gif_files, log, analysis_tracker)

    try:
        with open(temp_path, "w", encoding="utf-8") as handle:
            _write_header(handle, summary_path, mp4_details, gif_details)
            _write_mp4_section(handle, mp4_details, mp4_totals)
            _write_gif_section(handle, gif_details, gif_totals)
            _write_overall_section(handle, mp4_totals, gif_totals)

        _atomic_replace(temp_path, summary_path, log)
        return summary_path
    except Exception:
        log.exception("Failed to create comprehensive segments summary at %s", summary_path)
        _remove_file(temp_path)
        return None


def _collect_mp4_details(
    files: List[Path], logger: logging.Logger
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    details: List[Dict[str, Any]] = []
    total_size = 0.0
    total_duration = 0.0
    total_frames = 0

    for idx, path in enumerate(files, 1):
        entry: Dict[str, Any] = {
            "index": idx,
            "name": path.name,
            "size_mb": _safe_size_mb(path),
        }
        try:
            info = FFmpegUtils.get_video_info(str(path))
            entry.update(
                {
                    "duration": float(info.get("duration", 0.0) or 0.0),
                    "fps": float(info.get("fps", 0.0) or 0.0),
                    "frame_count": int(info.get("frame_count", 0) or 0),
                    "width": int(info.get("width", 0) or 0),
                    "height": int(info.get("height", 0) or 0),
                    "codec": info.get("codec", "unknown"),
                    "bitrate": info.get("bitrate", 0),
                }
            )

            total_size += entry["size_mb"]
            total_duration += entry["duration"]
            total_frames += entry["frame_count"]
        except Exception as exc:  # pragma: no cover - FFmpeg failures are environment specific
            entry["error"] = str(exc)
            logger.debug("Unable to gather MP4 info for %s: %s", path, exc, exc_info=True)

        details.append(entry)

    totals = {
        "count": len(details),
        "size_mb": total_size,
        "duration": total_duration,
        "frames": total_frames,
    }
    return details, totals


def _collect_gif_details(
    files: List[Path],
    logger: logging.Logger,
    analysis_tracker: Optional[Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    details: List[Dict[str, Any]] = []
    total_size = 0.0
    total_duration = 0.0
    total_frames = 0

    for idx, path in enumerate(files, 1):
        entry: Dict[str, Any] = {
            "index": idx,
            "name": path.name,
            "size_mb": _safe_size_mb(path),
        }
        try:
            gif_info = _probe_gif_info(path, analysis_tracker, logger)
            entry.update(
                {
                    "duration": gif_info["duration"],
                    "fps": gif_info["fps"],
                    "width": gif_info["width"],
                    "height": gif_info["height"],
                    "estimated_frames": gif_info["estimated_frames"],
                }
            )

            total_size += entry["size_mb"]
            total_duration += entry["duration"]
            total_frames += entry["estimated_frames"]
        except Exception as exc:  # pragma: no cover - FFmpeg failures are environment specific
            entry["error"] = str(exc)
            logger.debug("Unable to gather GIF info for %s: %s", path, exc, exc_info=True)

        details.append(entry)

    totals = {
        "count": len(details),
        "size_mb": total_size,
        "duration": total_duration,
        "frames": total_frames,
    }
    return details, totals


def _write_header(
    handle, summary_path: Path, mp4_details: List[Dict[str, Any]], gif_details: List[Dict[str, Any]]
) -> None:
    handle.write("Comprehensive Segments Summary\n")
    handle.write("=============================\n\n")
    base_name = summary_path.stem.replace("~", "").replace("_comprehensive_summary", "")
    handle.write(f"Base Name: {base_name}\n")
    handle.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    handle.write(f"Total MP4 Segments: {len(mp4_details)}\n")
    handle.write(f"Total GIF Segments: {len(gif_details)}\n")
    handle.write(f"Total Files: {len(mp4_details) + len(gif_details)}\n\n")


def _write_mp4_section(
    handle, details: List[Dict[str, Any]], totals: Dict[str, float]
) -> None:
    if not details:
        return

    handle.write("MP4 Segments Details:\n")
    handle.write("--------------------\n")
    for entry in details:
        handle.write(f"MP4 Segment {entry['index']:03d}: {entry['name']}\n")
        if entry.get("error"):
            handle.write(f"  Error reading info: {entry['error']}\n\n")
            continue
        handle.write(f"  Duration: {entry.get('duration', 0.0):.2f}s\n")
        handle.write(f"  FPS: {entry.get('fps', 0.0):.2f}\n")
        handle.write(f"  Frame Count: {entry.get('frame_count', 0)}\n")
        handle.write(f"  Resolution: {entry.get('width', 0)}x{entry.get('height', 0)}\n")
        handle.write(f"  Codec: {entry.get('codec', 'unknown')}\n")
        handle.write(f"  Bitrate: {entry.get('bitrate', 0)} kbps\n")
        handle.write(f"  Size: {entry.get('size_mb', 0.0):.2f}MB\n\n")

    count = max(totals.get("count", 0), 1)
    handle.write("MP4 Summary:\n")
    handle.write(f"  Total MP4 Segments: {int(totals.get('count', 0))}\n")
    handle.write(f"  Total Size: {totals.get('size_mb', 0.0):.2f}MB\n")
    handle.write(f"  Total Duration: {totals.get('duration', 0.0):.2f}s\n")
    handle.write(f"  Total Frames: {int(totals.get('frames', 0))}\n")
    handle.write(f"  Average Size: {totals.get('size_mb', 0.0)/count:.2f}MB\n")
    handle.write(f"  Average Duration: {totals.get('duration', 0.0)/count:.2f}s\n")
    avg_fps = (
        totals.get("frames", 0) / totals.get("duration", 1.0)
        if totals.get("duration", 0.0) > 0
        else 0.0
    )
    handle.write(f"  Average FPS: {avg_fps:.2f}\n\n")


def _write_gif_section(
    handle, details: List[Dict[str, Any]], totals: Dict[str, float]
) -> None:
    if not details:
        return

    handle.write("GIF Segments Details:\n")
    handle.write("--------------------\n")
    for entry in details:
        handle.write(f"GIF Segment {entry['index']:03d}: {entry['name']}\n")
        if entry.get("error"):
            handle.write(f"  Error reading info: {entry['error']}\n\n")
            continue
        handle.write(f"  Duration: {entry.get('duration', 0.0):.2f}s\n")
        handle.write(f"  FPS: {entry.get('fps', 0.0):.2f}\n")
        handle.write(f"  Estimated Frame Count: {entry.get('estimated_frames', 0)}\n")
        handle.write(f"  Resolution: {entry.get('width', 0)}x{entry.get('height', 0)}\n")
        handle.write(f"  Size: {entry.get('size_mb', 0.0):.2f}MB\n\n")

    count = max(totals.get("count", 0), 1)
    handle.write("GIF Summary:\n")
    handle.write(f"  Total GIF Segments: {int(totals.get('count', 0))}\n")
    handle.write(f"  Total Size: {totals.get('size_mb', 0.0):.2f}MB\n")
    handle.write(f"  Total Duration: {totals.get('duration', 0.0):.2f}s\n")
    handle.write(f"  Estimated Total Frames: {int(totals.get('frames', 0))}\n")
    handle.write(f"  Average Size: {totals.get('size_mb', 0.0)/count:.2f}MB\n")
    handle.write(f"  Average Duration: {totals.get('duration', 0.0)/count:.2f}s\n")
    avg_fps = (
        totals.get("frames", 0) / totals.get("duration", 1.0)
        if totals.get("duration", 0.0) > 0
        else 0.0
    )
    handle.write(f"  Average FPS: {avg_fps:.2f}\n\n")


def _write_overall_section(
    handle,
    mp4_totals: Dict[str, float],
    gif_totals: Dict[str, float],
) -> None:
    total_files = int(mp4_totals.get("count", 0) + gif_totals.get("count", 0))
    total_size = mp4_totals.get("size_mb", 0.0) + gif_totals.get("size_mb", 0.0)
    total_duration = mp4_totals.get("duration", 0.0) + gif_totals.get("duration", 0.0)
    total_frames = int(mp4_totals.get("frames", 0) + gif_totals.get("frames", 0))

    handle.write("Overall Summary:\n")
    handle.write("---------------\n")
    handle.write(f"Total Files: {total_files}\n")
    handle.write(
        f"File Types: MP4 ({int(mp4_totals.get('count', 0))}), GIF ({int(gif_totals.get('count', 0))})\n"
    )
    handle.write(f"Total Size: {total_size:.2f}MB\n")
    handle.write(f"Total Duration: {total_duration:.2f}s\n")
    handle.write(f"Total Frames: {total_frames}\n")
    if total_duration > 0:
        handle.write(f"Overall Average FPS: {total_frames/total_duration:.2f}\n")


def _atomic_replace(temp_path: Path, summary_path: Path, logger: logging.Logger) -> None:
    try:
        os.replace(str(temp_path), str(summary_path))
    except Exception:
        _clear_windows_attributes(summary_path)
        os.replace(str(temp_path), str(summary_path))
    finally:
        _clear_windows_attributes(summary_path)
        logger.debug("Wrote canonical segments summary to %s", summary_path)


def _clear_windows_attributes(path: Path) -> None:
    if not path.exists():
        return
    try:
        subprocess.run(
            ["attrib", "-R", "-S", "-H", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
    except Exception:
        pass


def _remove_file(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        pass


def _safe_size_mb(path: Path) -> float:
    try:
        return path.stat().st_size / (1024 * 1024)
    except OSError:
        return 0.0


def _probe_gif_info(
    path: Path,
    analysis_tracker: Optional[Any],
    logger: logging.Logger,
) -> Dict[str, Any]:
    cmd = ["ffmpeg", "-i", str(path)]
    duration = 0.0
    fps = 12.0
    width = 320
    height = 240

    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30,
        )
    except subprocess.TimeoutExpired as timeout_exc:
        logger.warning("FFmpeg summary probe timed out for %s: %s", path, timeout_exc)
        if analysis_tracker:
            try:
                analysis_tracker.record_timeout("summary_probe")
            except Exception:
                logger.debug("Analysis tracker summary timeout recording failed", exc_info=True)
        return {
            "duration": 0.0,
            "fps": fps,
            "width": width,
            "height": height,
            "estimated_frames": 0,
        }

    if result.stderr:
        for line in result.stderr.splitlines():
            if "Duration:" in line:
                match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})", line)
                if match:
                    h, m, s, cs = map(int, match.groups())
                    duration = h * 3600 + m * 60 + s + cs / 100
            if "Video:" in line:
                res_match = re.search(r"(\d+)x(\d+)", line)
                if res_match:
                    width, height = map(int, res_match.groups())
                fps_match = re.search(r"(\d+(?:\.\d+)?) fps", line)
                if fps_match:
                    fps = float(fps_match.group(1))

    estimated_frames = int(round(duration * fps)) if duration > 0 and fps > 0 else 0
    return {
        "duration": duration,
        "fps": fps,
        "width": width,
        "height": height,
        "estimated_frames": estimated_frames,
    }

