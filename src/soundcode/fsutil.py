"""Filesystem utilities for handling OS-level quirks.

On exFAT (external drives), macOS keeps AppleDouble companions `._<name>` next to
files, deleting them automatically when `<name>` is deleted. shutil.rmtree lists
both, deletes the main file, then fails with FileNotFoundError on the already-vanished
companion. This module provides rmtree() that tolerates such vanishing acts.

AppleDouble files must also be filtered when reading directories; wavs() returns
only actual WAV audio files, skipping AppleDouble companions and directories.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


def wavs(directory: str | Path) -> list[Path]:
    """List WAV files in a directory, excluding AppleDouble companions.

    Ignores files whose name starts with '._' (macOS AppleDouble format) and
    directories. Returns paths sorted by their string representation.

    Args:
        directory: Directory to scan.

    Returns:
        Sorted list of Path objects for '*.wav' files, excluding companions.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.glob("*.wav") if p.is_file() and not p.name.startswith("._"))


def rmtree(path: str | Path) -> None:
    """Remove a directory tree, tolerating files that vanish mid-deletion.

    On exFAT and other filesystems where companion files (e.g., AppleDouble `._*`)
    are auto-deleted alongside the main file, shutil.rmtree may try to explicitly
    delete the now-missing companion and raise FileNotFoundError. This function
    ignores such errors while re-raising any other exception.

    No-op if the path does not exist.

    Args:
        path: Directory to remove (string or Path).

    Raises:
        Any exception other than FileNotFoundError encountered during deletion.
    """
    path = Path(path)
    if not path.exists():
        return

    def handle_error(func, path_str, exc_info):
        # exc_info is (type, value, traceback)
        exc_type, exc_value, _ = exc_info
        if exc_type is FileNotFoundError:
            # File disappeared between listing and deletion (e.g., exFAT AppleDouble)
            return
        # Re-raise any other error
        raise exc_value

    shutil.rmtree(path, onerror=handle_error)
