"""Stage 実装のレジストリ.

Worker サブプロセス起動時に import して実装を登録する. FastAPI プロセスから
Stage 実体を import してはならない (Torch などが引きずり込まれるため).
"""

from __future__ import annotations

from ..domain.pipeline_state import StageName
from .stage import Stage

_REGISTRY: dict[StageName, type[Stage]] = {}


def register(stage_cls: type[Stage]) -> type[Stage]:
    """decorator. `@register` で Stage 実装を登録."""
    _REGISTRY[stage_cls.name] = stage_cls
    return stage_cls


def get(name: StageName) -> type[Stage]:
    if name not in _REGISTRY:
        raise KeyError(f"Stage {name} is not registered. did the worker import stages package?")
    return _REGISTRY[name]


def known() -> list[StageName]:
    return list(_REGISTRY.keys())
