"""UI で現在の言語へ変換できる domain error。"""

from __future__ import annotations


class LocalizedError(RuntimeError):
    def __init__(self, key: str, fallback: str, args: dict | None = None) -> None:
        super().__init__(fallback)
        self.key = key
        self.formatting_args = args
