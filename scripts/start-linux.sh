#!/usr/bin/env bash
set -euo pipefail

repository_dir="$(cd "$(dirname "$0")/.." && pwd)"
port="${SPHERE_PORT:-8787}"
cd "$repository_dir"

command -v uv >/dev/null || { echo "uv が必要です: https://docs.astral.sh/uv/" >&2; exit 1; }
command -v pnpm >/dev/null || { echo "pnpm が必要です: npm install -g pnpm@9.15.0" >&2; exit 1; }

echo "[1/5] Frontend 依存関係"
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build

echo "[2/5] Backend 依存関係"
backend_extras=(--extra imaging --extra aliked)
if [[ "${SPHERE_WITH_SAM3:-0}" == "1" ]]; then backend_extras+=(--extra sam3); fi
if [[ "${SPHERE_WITH_DENSE:-0}" == "1" ]]; then backend_extras+=(--extra dense); fi
(cd backend && uv sync "${backend_extras[@]}")

if [[ "${SPHERE_WITH_DENSE:-0}" == "1" ]]; then
  backend/.venv/bin/python scripts/install_romav2.py
fi

if [[ -z "${SPHERE_FILESYSTEM__ALLOWED_ROOTS:-}" ]]; then
  configured_roots="$(backend/.venv/bin/python -c 'import json; from sphere_reconstruct.settings import get_settings; print(json.dumps([str(path) for path in get_settings().filesystem.allowed_roots]))')"
  if [[ "$configured_roots" == "[]" ]]; then
    export SPHERE_FILESYSTEM__ALLOWED_ROOTS="[\"${HOME}\"]"
  fi
fi

echo "[3/5] Vocabulary tree"
configured_vocab_tree="${SPHERE_BINARIES__VOCAB_TREE:-}"
if [[ -z "$configured_vocab_tree" ]]; then
  configured_vocab_tree="$(backend/.venv/bin/python -c 'from sphere_reconstruct.settings import get_settings; print(get_settings().binaries.vocab_tree)')"
fi
if [[ -z "$configured_vocab_tree" ]]; then
  backend/.venv/bin/python scripts/install_vocab_tree.py
  export SPHERE_BINARIES__VOCAB_TREE="$(<.runtime/vocab-tree-path.txt)"
fi

echo "[4/5] 環境診断"
(cd backend && uv run sphere-doctor)

echo "[5/5] http://127.0.0.1:${port} で起動"
cd backend
exec .venv/bin/python -m uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port "$port"
