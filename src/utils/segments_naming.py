"""Utilities for generating canonical segment folder and summary filenames."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

INVALID_WINDOWS_CHARS = {'<', '>', ':', '"', '/', '\\', '|', '?', '*'}
FALLBACK_BASE_NAME = 'segments_summary'
MAX_BASE_LENGTH = 180

__all__ = [
    'sanitize_segments_base_name',
    'segments_summary_filename',
    'segments_summary_path',
]


def sanitize_segments_base_name(name: Optional[Union[str, Path]]) -> str:
    """
    Normalize potentially messy titles into filesystem-safe base names.

    - Replaces Windows-reserved characters and the problematic Unicode division slash (⧸).
    - Collapses non-ASCII glyphs into underscores for portability.
    - Trims leading/trailing whitespace and dots, enforcing a deterministic fallback.
    """
    text = str(name or '').strip()
    if not text:
        text = FALLBACK_BASE_NAME

    safe_chars: list[str] = []
    for char in text:
        codepoint = ord(char)
        if char == '⧸':
            safe_chars.append('_')
        elif char in INVALID_WINDOWS_CHARS:
            safe_chars.append('_')
        elif codepoint < 32:
            # Control characters are dropped entirely
            continue
        elif codepoint < 128:
            safe_chars.append(char)
        else:
            safe_chars.append('_')

    sanitized = ''.join(safe_chars).strip().strip('._ ')
    if not sanitized:
        sanitized = FALLBACK_BASE_NAME

    if len(sanitized) > MAX_BASE_LENGTH:
        sanitized = sanitized[:MAX_BASE_LENGTH].rstrip('._ ')
        if not sanitized:
            sanitized = FALLBACK_BASE_NAME

    return sanitized


def segments_summary_filename(base_name: Optional[Union[str, Path]] = None) -> str:
    """Construct the standard '~<base>_comprehensive_summary.txt' filename."""
    safe_base = sanitize_segments_base_name(base_name)
    return f"~{safe_base}_comprehensive_summary.txt"


def segments_summary_path(segments_folder: Union[str, Path],
                          base_name: Optional[Union[str, Path]] = None) -> Path:
    """
    Return the canonical summary path for a segments folder.

    If base_name is omitted, derive it from the folder name (stripping any '_segments' suffix)
    before applying sanitization.
    """
    folder = Path(segments_folder)
    if base_name is None or not str(base_name).strip():
        candidate = folder.stem
        if candidate.endswith('_segments'):
            candidate = candidate[:-len('_segments')]
    else:
        candidate = base_name

    filename = segments_summary_filename(candidate)
    return folder / filename

