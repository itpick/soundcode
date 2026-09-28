"""Filesystem utilities for handling OS-level quirks.

On exFAT (external drives), macOS keeps AppleDouble companions `._<name>` next to
files, deleting them automatically when `<name>` is deleted. shutil.rmtree lists
both, deletes the main file, then fails with FileNotFoundError on the already-vanished
companion. This module provides rmtree() that tolerates such vanishing acts.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path


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
