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


def test_wavs_returns_sorted_wav_files():
    """wavs() returns sorted WAV files in a directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "kick_0.wav").write_text("audio")
        (tmpdir / "hat_1.wav").write_text("audio")
        (tmpdir / "snare_0.wav").write_text("audio")

        result = fsutil.wavs(tmpdir)

        assert len(result) == 3
        assert result[0].name == "hat_1.wav"
        assert result[1].name == "kick_0.wav"
        assert result[2].name == "snare_0.wav"


def test_wavs_skips_appledouble_files():
    """wavs() excludes files starting with '._' (AppleDouble companions)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "kick_0.wav").write_text("audio")
        (tmpdir / "._kick_0.wav").write_text("companion")
        (tmpdir / "._hat_1.wav").write_text("companion")
        (tmpdir / "hat_1.wav").write_text("audio")

        result = fsutil.wavs(tmpdir)

        assert len(result) == 2
        assert all(not p.name.startswith("._") for p in result)
        assert result[0].name == "hat_1.wav"
        assert result[1].name == "kick_0.wav"


def test_wavs_skips_directories():
    """wavs() excludes directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        (tmpdir / "kick_0.wav").write_text("audio")
        (tmpdir / "wav_dir").mkdir()
        (tmpdir / "wav_dir" / "file.txt").write_text("not wav")

        result = fsutil.wavs(tmpdir)

        assert len(result) == 1
        assert result[0].name == "kick_0.wav"


def test_wavs_handles_missing_directory():
    """wavs() returns an empty list for a non-existent directory."""
    missing = Path("/tmp/nonexistent-soundcode-wavs-" + os.urandom(8).hex())
    assert not missing.exists()

    result = fsutil.wavs(missing)

    assert result == []


def test_wavs_empty_directory():
    """wavs() returns an empty list for an empty directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        result = fsutil.wavs(tmpdir)
        assert result == []
