"""Stage directory の atomic promotion を検証する."""

from sphere_reconstruct.infrastructure.filesystem import (
    _atomic_replace_windows,
    sha256_bytes,
    sha256_file,
)


def test_windows_ready_directory_promotion_replaces_existing_output(tmp_path):
    temporary = tmp_path / ".stage.tmp"
    final = tmp_path / "stage"
    temporary.mkdir()
    final.mkdir()
    (temporary / "new.txt").write_text("new")
    (final / "old.txt").write_text("old")

    _atomic_replace_windows(temporary, final)

    assert (final / "new.txt").read_text() == "new"
    assert not (final / "old.txt").exists()
    assert not temporary.exists()
    assert not list(tmp_path.glob("stage.ready-*"))
    assert not list(tmp_path.glob("stage.old-*"))


def test_sha256_file_reports_monotonic_progress(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"abcdefghij")
    progress = []

    digest = sha256_file(
        path, chunk_size=3, progress=lambda current, total: progress.append((current, total))
    )

    assert digest == sha256_bytes(b"abcdefghij")
    assert progress[-1] == (10, 10)
    assert [current for current, _total in progress] == sorted(
        current for current, _total in progress
    )
