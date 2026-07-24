#!/usr/bin/env bash
# sphere-reconstruct 起動スクリプト (Linux / macOS).
#
# frontend をビルドして backend (uvicorn) を起動する. backend が dist を静的配信
# するので, ブラウザで http://127.0.0.1:<port> を開けば UI が使える (Electron 不要).
#
# 依存: uv, node/pnpm(またはnpm), ffmpeg, colmap, (SAM3 は手動配置).

set -euo pipefail
cd "$(dirname "$0")/.."

PORT="${SPHERE_PORT:-8787}"

echo "[1/3] frontend build"
pushd frontend >/dev/null
if command -v pnpm >/dev/null 2>&1; then
    pnpm install --frozen-lockfile || pnpm install
    pnpm build
else
    npm install
    npm run build
fi
popd >/dev/null

echo "[2/3] backend deps (uv sync)"
pushd backend >/dev/null
# CUDA 機なら SPHERE_WITH_SAM3=1 で SAM3 extra も入れる.
if [ "${SPHERE_WITH_SAM3:-0}" = "1" ]; then
    uv sync --extra sam3 --extra imaging
else
    uv sync --extra imaging
fi

echo "[3/3] starting backend on http://127.0.0.1:${PORT}"
exec uv run uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port "${PORT}"
