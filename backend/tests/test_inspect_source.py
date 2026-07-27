"""Source inspection input fingerprint contract."""

from pathlib import Path

from sphere_reconstruct.domain.source import MediaKind, Projection, SourceAdapter, SourceRole
from sphere_reconstruct.pipeline.stage import ProgressReporter, SourceContext
from sphere_reconstruct.stages.inspect_source import InspectSource


def test_erp_image_directory_is_fingerprinted_as_a_manifest(tmp_path):
    (tmp_path / "a.png").write_bytes(b"a")
    (tmp_path / "b.jpg").write_bytes(b"bb")
    (tmp_path / "ignored.txt").write_text("ignored")
    context = type(
        "Context",
        (),
        {
            "sources": (
                SourceContext(
                    id="source-a",
                    label="images",
                    role=SourceRole.PRIMARY,
                    adapter=SourceAdapter.GENERIC_IMAGES,
                    media_kind=MediaKind.IMAGES,
                    projection=Projection.PERSPECTIVE,
                    path=Path(tmp_path),
                    ordinal=0,
                    enabled=True,
                ),
            ),
            "progress": ProgressReporter(lambda *_args: None),
        },
    )()
    first = InspectSource().collect_inputs(context)
    assert len(first) == 2
    assert sum(reference.size for reference in first) == 3

    (tmp_path / "b.jpg").write_bytes(b"changed")
    second = InspectSource().collect_inputs(context)
    assert [reference.sha256 for reference in first] != [reference.sha256 for reference in second]
