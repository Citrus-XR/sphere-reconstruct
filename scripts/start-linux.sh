#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "$0")/.." && pwd)"
port="${SPHERE_PORT:-8787}"
cd "$repository_dir"

command -v uv >/dev/null || { echo "uv が必要です: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v pnpm >/dev/null || { echo "pnpm が必要です: npm install -g pnpm@9.15.0" >&2; exit 1; }

echo "[1/4] Frontend 依存関係"
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build

echo "[2/4] Backend 依存関係"
backend_extras=(--extra imaging --extra aliked --extra denoise)
if [[ "${SPHERE_WITH_SAM3:-0}" == "1" ]]; then backend_extras+=(--extra sam3); fi
(cd backend && uv sync "${backend_extras[@]}")

if [[ -z "${SPHERE_FILESYSTEM__ALLOWED_ROOTS:-}" ]]; then
  configured_roots="$(backend/.venv/bin/python -c 'import json; from sphere_reconstruct.settings import get_settings; print(json.dumps([str(path) for path in get_settings().filesystem.allowed_roots]))')"
  if [[ "$configured_roots" == "[]" ]]; then
    export SPHERE_FILESYSTEM__ALLOWED_ROOTS="[\"${HOME}\"]"
  fi
fi

echo "[3/4] 環境診断"
(cd backend && uv run sphere-doctor)

echo "[4/4] http://127.0.0.1:${port} で起動"
cd backend
exec .venv/bin/python -m uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port "$port"
