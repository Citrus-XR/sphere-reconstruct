"""Hardware decoder の選択は実 source probe の成功を根拠にし、必須設定を守る。"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sphere_reconstruct.imaging import ffmpeg


def test_explicit_hardware_decoder_is_selected_after_source_probe(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.touch()
    commands = []
    monkeypatch.setattr(ffmpeg, "_resolve_bin", lambda _explicit: "ffmpeg")
    monkeypatch.setattr(ffmpeg, "_hardware_acceleration_methods", lambda _binary: {"cuda"})

    def run(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(ffmpeg.subprocess, "run", run)
    result = ffmpeg.resolve_hardware_decode(
        source,
        stream_index=1,
        preference="cuda",
        required=True,
    )

    assert result.method == "cuda"
    assert commands[0][commands[0].index("-hwaccel") + 1] == "cuda"
    assert "0:v:1" in commands[0]


def test_auto_hardware_decoder_tries_candidates_in_priority_order(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.touch()
    attempted = []
    monkeypatch.setattr(ffmpeg, "_resolve_bin", lambda _explicit: "ffmpeg")
    monkeypatch.setattr(
        ffmpeg,
        "_hardware_acceleration_methods",
        lambda _binary: {"cuda", "qsv"},
    )

    def run(command, **_kwargs):
        method = command[command.index("-hwaccel") + 1]
        attempted.append(method)
        return SimpleNamespace(
            returncode=0 if method == "qsv" else 1,
            stdout="",
            stderr="CUDA probe failed",
        )

    monkeypatch.setattr(ffmpeg.subprocess, "run", run)
    result = ffmpeg.resolve_hardware_decode(
        source,
        stream_index=0,
        preference="auto",
        required=True,
    )

    assert result.method == "qsv"
    assert attempted == ["cuda", "qsv"]


def test_required_hardware_decoder_exposes_probe_failure(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.touch()
    monkeypatch.setattr(ffmpeg, "_resolve_bin", lambda _explicit: "ffmpeg")
    monkeypatch.setattr(ffmpeg, "_hardware_acceleration_methods", lambda _binary: set())

    with pytest.raises(RuntimeError, match="hardware decode unavailable"):
        ffmpeg.resolve_hardware_decode(
            source,
            stream_index=0,
            preference="cuda",
            required=True,
        )


def test_optional_hardware_decoder_reports_software_fallback(tmp_path, monkeypatch):
    source = tmp_path / "source.mp4"
    source.touch()
    monkeypatch.setattr(ffmpeg, "_resolve_bin", lambda _explicit: "ffmpeg")
    monkeypatch.setattr(ffmpeg, "_hardware_acceleration_methods", lambda _binary: set())

    result = ffmpeg.resolve_hardware_decode(
        source,
        stream_index=0,
        preference="auto",
        required=False,
    )

    assert result.method is None
    assert "using software" in result.detail
