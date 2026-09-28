"""Test fsutil.rmtree() for exFAT AppleDouble companion file handling."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from soundcode import fsutil  # noqa: E402


def test_rmtree_removes_normal_tree():
    """Normal directory tree removal works."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test")
        (test_dir / "subdir").mkdir()
        (test_dir / "subdir" / "file2.txt").write_text("test")

        fsutil.rmtree(test_dir)

        assert not test_dir.exists()


def test_rmtree_file_disappears_between_listing_and_deletion():
    """FileNotFoundError on a vanishing companion file is silently ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test")
        (test_dir / "._file1.txt").write_text("test")

        # Monkeypatch os.unlink to simulate AppleDouble deletion on first call to ._file1.txt
        original_unlink = os.unlink
        unlink_calls = []

        def failing_unlink(*args, **kwargs):
            unlink_calls.append(args[0] if args else kwargs.get('path', ''))
            # On first call to ._file1.txt, simulate it vanishing (exFAT auto-deletion)
            if '._file1.txt' in str(args[0] if args else ''):
                if len([c for c in unlink_calls if '._file1.txt' in c]) == 1:
                    raise FileNotFoundError(f"File not found: {args[0]}")
            return original_unlink(*args, **kwargs)

        with patch('os.unlink', side_effect=failing_unlink):
            # This should not raise even though ._file1.txt disappears
            fsutil.rmtree(test_dir)

        assert not test_dir.exists()


def test_rmtree_permission_error_propagates():
    """PermissionError during deletion is re-raised, not silently ignored."""
    with tempfile.TemporaryDirectory() as tmpdir:
        test_dir = Path(tmpdir) / "test"
        test_dir.mkdir()
        (test_dir / "file1.txt").write_text("test")

        # Monkeypatch os.unlink to raise PermissionError
        original_unlink = os.unlink

        def failing_unlink(*args, **kwargs):
            raise PermissionError("Permission denied")

        with patch('os.unlink', side_effect=failing_unlink):
            try:
                fsutil.rmtree(test_dir)
                assert False, "Should have raised PermissionError"
            except PermissionError as e:
                assert "Permission denied" in str(e)


def test_rmtree_missing_path_is_noop():
    """Calling rmtree on a missing path is a no-op (does not raise)."""
    missing_path = Path("/tmp/nonexistent-soundcode-test-" + os.urandom(8).hex())
    assert not missing_path.exists()

    # Should not raise
    fsutil.rmtree(missing_path)
