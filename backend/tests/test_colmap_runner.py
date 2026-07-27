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


def test_global_positioning_gpu_is_independent_from_ceres_gpu(tmp_path: Path, monkeypatch):
    captured = {}

    def fake_run_command(_binary, args, **_kwargs):
        captured["args"] = args
        return object()

    monkeypatch.setattr(runner, "run_command", fake_run_command)
    runner.global_mapper(
        "colmap",
        database_path=tmp_path / "database.db",
        image_path=tmp_path / "images",
        output_path=tmp_path / "sparse",
        ba_use_gpu=False,
        global_positioning_use_gpu=True,
    )

    args = captured["args"]
    assert args[args.index("--GlobalMapper.ba_ceres_use_gpu") + 1] == "0"
    assert args[args.index("--GlobalMapper.gp_use_gpu") + 1] == "1"
