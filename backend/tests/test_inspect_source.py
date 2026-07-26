"""Source inspection input fingerprint contract."""

from types import SimpleNamespace

from sphere_reconstruct.stages.inspect_source import InspectSource


def test_erp_image_directory_is_fingerprinted_as_a_manifest(tmp_path):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.jpg").write_bytes(b"bb")
    (tmp_path / "ignored.txt").write_text("ignored")
    context = SimpleNamespace(source_path=tmp_path)
    first = InspectSource().collect_inputs(context)
    assert len(first) == 1
    assert first[0].size == 3

    (tmp_path / "b.jpg").write_bytes(b"changed")
    second = InspectSource().collect_inputs(context)
    assert first[0].sha256 != second[0].sha256
