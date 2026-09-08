"""pytest 用共有 fixture."""

from __future__ import annotations

import sys
from pathlib import Path

# `src` レイアウトなので, テストからは src/ を sys.path に追加しておく (uv sync 済みなら不要だが保険).
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
# Repository-level CLI scripts も小さい pure function を test 対象として import する。
_REPOSITORY_ROOT = _ROOT.parent
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))
