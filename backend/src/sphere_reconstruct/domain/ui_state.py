"""サーバーに保存する workspace / project の UI 設定。"""
# ruff: noqa: N815 -- UI JSON keys match the TypeScript state model.

from __future__ import annotations

import json
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator


class UiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ViewerPreferences(UiModel):
    showPoints: bool = True
    showCams: bool = True
    pointSize: float = Field(default=2.5, ge=1, le=8)
    showGrid: bool = True
    showCenter: bool = True
    background: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")


class ConsolePreferences(UiModel):
    info: bool = True
    warn: bool = True
    error: bool = True
    debug: bool = False
    search: str = Field(default="", max_length=1000)


class ViewerCameraPose(UiModel):
    position: tuple[FiniteFloat, FiniteFloat, FiniteFloat]
    quaternion: tuple[FiniteFloat, FiniteFloat, FiniteFloat, FiniteFloat]

    @field_validator("quaternion")
    @classmethod
    def unit_quaternion(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if abs(sum(component * component for component in value) - 1) > 1e-4:
            raise ValueError("camera quaternion must have unit length")
        return value


class WorkspacePreferences(UiModel):
    theme: Literal["auto", "light", "dark"] = "auto"
    lang: Literal["ja", "zh", "en"] | None = None
    lastProjectId: str | None = Field(default=None, max_length=100)
    layout: dict | None = None
    compactLayout: dict | None = None
    viewer: ViewerPreferences = Field(default_factory=ViewerPreferences)
    console: ConsolePreferences = Field(default_factory=ConsolePreferences)

    @field_validator("layout", "compactLayout")
    @classmethod
    def valid_layout(cls, value: dict | None) -> dict | None:
        if value is None:
            return value
        if len(json.dumps(value)) > 50_000:
            raise ValueError("layout exceeds 50 KB")
        if value.get("subLayouts") or value.get("popouts") or value.get("borders"):
            raise ValueError("only in-page docking is supported")
        tabs: list[str] = []

        def visit(node: dict, depth: int, parent: str | None) -> None:
            if not isinstance(node, dict) or depth > 16:
                raise ValueError("invalid layout node or nesting depth")
            kind = node.get("type")
            allowed = {None: {"row"}, "row": {"row", "tabset"}, "tabset": {"tab"}}
            if not isinstance(kind, str) or kind not in allowed.get(parent, set()):
                raise ValueError("invalid docking hierarchy")
            if kind == "tab":
                component = node.get("component")
                if not isinstance(component, str) or component not in {"console", "inspector", "scene", "sceneHier", "steps"}:
                    raise ValueError("unknown docking component")
                if node.get("id") != component:
                    raise ValueError("tab ID must match its component")
                tabs.append(component)
            else:
                children = node.get("children")
                if not isinstance(children, list) or len(children) > 16:
                    raise ValueError("invalid docking children")
                for child in children:
                    visit(child, depth + 1, kind)

        visit(value.get("layout"), 0, None)
        if sorted(tabs) != ["console", "inspector", "scene", "sceneHier", "steps"]:
            raise ValueError("layout must contain each application tab exactly once")
        return value


class ProjectUiState(UiModel):
    params: dict = Field(default_factory=dict)
    reconMode: Literal["native_fisheye", "pinhole_rig", "equirectangular"] | None = None
    clearOutputStages: list[str] = Field(default_factory=list)
    selectedStage: str | None = Field(default=None, max_length=100)
    selectedCameraId: int | None = Field(default=None, ge=0)
    selectedFrameIndex: int | None = Field(default=None, ge=0)
    cameraPose: ViewerCameraPose | None = None


def merge_ui_patch(current: dict, patch: dict) -> dict:
    """独立したパネルの部分更新で他のパネルの設定を上書きしない。"""
    result = {**current, **patch}
    for key in ("params", "viewer", "console"):
        if key in patch:
            result[key] = {**current.get(key, {}), **patch[key]}
    return result
