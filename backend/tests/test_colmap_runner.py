"""COLMAP executable の解決順序を検証する。"""

from pathlib import Path

from sphere_reconstruct.colmap import runner


def test_auto_installer_record_is_reused(tmp_path: Path, monkeypatch):
    executable = tmp_path / "colmap.exe"
    executable.write_bytes(b"binary")
    record = tmp_path / "colmap-path.txt"
    record.write_text(str(executable), encoding="utf-8")
    monkeypatch.setattr(runner.shutil, "which", lambda _name: None)
    monkeypatch.setattr(runner, "_auto_installed_colmap_record", lambda: record)
    assert runner.resolve_colmap_bin(None) == str(executable)
