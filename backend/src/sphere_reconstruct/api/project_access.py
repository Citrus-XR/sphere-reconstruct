"""成果物変更 request の所有権を response 完了まで保持する。"""

from collections.abc import AsyncIterator
from typing import Annotated

from fastapi import Depends, HTTPException

from ..domain import project as project_domain
from ..infrastructure.database import get_db
from ..infrastructure.project_lock import (
    ProjectBusyError,
    ProjectNotFoundError,
    project_lock,
    require_idle_project,
)


def project_access_error(error: ProjectBusyError | ProjectNotFoundError) -> HTTPException:
    return HTTPException(status_code=404 if isinstance(error, ProjectNotFoundError) else 409, detail=str(error))


async def idle_project(project_id: str) -> AsyncIterator[project_domain.Project]:
    db = get_db()
    async with project_lock(project_id):
        try:
            await require_idle_project(db, project_id)
        except (ProjectBusyError, ProjectNotFoundError) as error:
            raise project_access_error(error) from error
        project = await project_domain.get_project(db, project_id)
        assert project is not None
        yield project


MutableProject = Annotated[project_domain.Project, Depends(idle_project)]
