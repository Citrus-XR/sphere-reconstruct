"""Stage directory の atomic promotion を検証する."""

from sphere_reconstruct.infrastructure.filesystem import _atomic_replace_windows


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
