"""Project source の型制約と primary role 遷移を検証する。"""

import pytest

from sphere_reconstruct.domain import project as project_domain
from sphere_reconstruct.domain import source as source_domain
from sphere_reconstruct.infrastructure.database import Database


async def test_first_source_becomes_primary_and_role_can_be_swapped(tmp_path):
    database = Database(tmp_path / "state.db")
    await database.connect()
    project = await project_domain.create_project(database, "mixed")
    primary = await source_domain.add_source(
        database,
        project.id,
        label="360",
        role=source_domain.SourceRole.SUPPLEMENTAL,
        adapter=source_domain.SourceAdapter.INSTA360_INSV,
        media_kind=source_domain.MediaKind.VIDEO,
        projection=source_domain.Projection.DUAL_FISHEYE,
        path=str(tmp_path / "primary.insv"),
    )
    phone = await source_domain.add_source(
        database,
        project.id,
        label="phone",
        role=source_domain.SourceRole.SUPPLEMENTAL,
        adapter=source_domain.SourceAdapter.GENERIC_IMAGES,
        media_kind=source_domain.MediaKind.IMAGES,
        projection=source_domain.Projection.PERSPECTIVE,
        path=str(tmp_path / "phone"),
    )

    assert primary.role == source_domain.SourceRole.PRIMARY
    assert phone.role == source_domain.SourceRole.SUPPLEMENTAL
    await source_domain.make_primary(database, project.id, phone.id)
    sources = await source_domain.list_sources(database, project.id)
    assert (
        next(source for source in sources if source.id == phone.id).role == source_domain.SourceRole.PRIMARY
    )
    assert (
        next(source for source in sources if source.id == primary.id).role
        == source_domain.SourceRole.SUPPLEMENTAL
    )
    with pytest.raises(ValueError, match="primary source"):
        await source_domain.remove_source(database, project.id, phone.id)
    await database.close()


def test_invalid_adapter_projection_combination_is_rejected():
    with pytest.raises(ValueError, match="dual_fisheye"):
        source_domain.ProjectSource(
            id="invalid",
            project_id="project",
            label="invalid",
            role=source_domain.SourceRole.PRIMARY,
            adapter=source_domain.SourceAdapter.GENERIC_IMAGES,
            media_kind=source_domain.MediaKind.IMAGES,
            projection=source_domain.Projection.DUAL_FISHEYE,
            path="/images",
            ordinal=0,
            enabled=True,
        )


def test_unregistered_adapter_is_rejected_with_its_id():
    with pytest.raises(ValueError, match="vendor.camera_v1"):
        source_domain.ProjectSource(
            id="unknown",
            project_id="project",
            label="unknown",
            role=source_domain.SourceRole.PRIMARY,
            adapter="vendor.camera_v1",
            media_kind=source_domain.MediaKind.VIDEO,
            projection=source_domain.Projection.DUAL_FISHEYE,
            path="/capture.bin",
            ordinal=0,
            enabled=True,
        )
