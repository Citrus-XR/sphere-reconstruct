# sphere-reconstruct 起動スクリプト (Windows PowerShell).
#
# frontend をビルドして backend (uvicorn) を起動する. backend が dist を静的配信
# するので, ブラウザで http://127.0.0.1:<port> を開けば UI が使える.
#
# 依存: uv, node/npm, ffmpeg, colmap, (SAM3 は手動配置).
# CUDA 機は環境変数 SPHERE_WITH_SAM3=1 で SAM3 extra も同期する.

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$port = if ($env:SPHERE_PORT) { $env:SPHERE_PORT } else { "8787" }

Write-Host "[1/3] frontend build"
Push-Location frontend
npm install
npm run build
Pop-Location

Write-Host "[2/3] backend deps (uv sync)"
Push-Location backend
if ($env:SPHERE_WITH_SAM3 -eq "1") {
    uv sync --extra sam3 --extra imaging
} else {
    uv sync --extra imaging
}

Write-Host "[3/3] starting backend on http://127.0.0.1:$port"
uv run uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port $port
Pop-Location
