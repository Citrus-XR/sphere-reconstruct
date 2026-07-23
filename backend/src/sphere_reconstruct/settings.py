"""アプリ設定. runtime/config.toml を読んで pydantic-settings に載せる.

優先度 (下が強い):
  1. runtime/config.toml
  2. .env / OS 環境変数 (SPHERE_SERVER__PORT のような double underscore)
  3. AppSettings() コンストラクタに渡した引数

環境変数で config.toml の値を上書きできる (テストや CI で使う).
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)


class ServerConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8787


class WorkspaceConfig(BaseModel):
    root: Path = Path("./workspace")


class FilesystemConfig(BaseModel):
    # Source ページのサーバサイド file browser が列挙してよいディレクトリ.
    # 空配列だと file browser API は 403 で弾く.
    allowed_roots: list[Path] = Field(default_factory=list)


class BinariesConfig(BaseModel):
    ffmpeg: str = ""
    ffprobe: str = ""
    colmap: str = ""


class Sam3Config(BaseModel):
    repo_path: str = ""
    checkpoint_path: str = ""
    config_path: str = ""
    device: str = "cuda:0"
    dtype: str = "bfloat16"


class LogConfig(BaseModel):
    level: str = "INFO"


def _resolve_config_path() -> Path:
    # 明示指定 (SPHERE_CONFIG=/path/to.toml) を最優先, なければ repo ルート/runtime/config.toml.
    import os

    override = os.environ.get("SPHERE_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    # backend/src/sphere_reconstruct/settings.py -> ../../.. で repo ルート
    here = Path(__file__).resolve()
    repo_root = here.parents[3]
    return repo_root / "runtime" / "config.toml"


class _TomlSource(PydanticBaseSettingsSource):
    """pydantic-settings に config.toml を差し込む source. env より低い優先度."""

    def __init__(self, settings_cls: type[BaseSettings], path: Path):
        super().__init__(settings_cls)
        self._path = path
        self._data: dict[str, Any] | None = None

    def _load(self) -> dict[str, Any]:
        if self._data is not None:
            return self._data
        if not self._path.exists():
            self._data = {}
            return self._data
        with self._path.open("rb") as f:
            self._data = tomllib.load(f)
        return self._data

    def get_field_value(self, field, field_name):  # type: ignore[override]
        data = self._load()
        if field_name in data:
            return data[field_name], field_name, False
        return None, field_name, False

    def __call__(self) -> dict[str, Any]:
        return self._load()


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SPHERE_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    server: ServerConfig = Field(default_factory=ServerConfig)
    workspace: WorkspaceConfig = Field(default_factory=WorkspaceConfig)
    filesystem: FilesystemConfig = Field(default_factory=FilesystemConfig)
    binaries: BinariesConfig = Field(default_factory=BinariesConfig)
    sam3: Sam3Config = Field(default_factory=Sam3Config)
    log: LogConfig = Field(default_factory=LogConfig)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        toml_source = _TomlSource(settings_cls, _resolve_config_path())
        # 優先度: init > env > dotenv > toml > file_secret
        return (init_settings, env_settings, dotenv_settings, toml_source, file_secret_settings)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    return AppSettings()


def workspace_root() -> Path:
    """workspace ルートを絶対化して返す. 存在しなければ作る."""
    root = get_settings().workspace.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    return root
