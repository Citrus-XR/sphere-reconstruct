$ErrorActionPreference = "Stop"
$repositoryDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$port = if ($env:SPHERE_PORT) { $env:SPHERE_PORT } else { "8787" }
Set-Location $repositoryDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv が必要です: https://docs.astral.sh/uv/" }
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) { throw "pnpm が必要です: npm install -g pnpm@9.15.0" }

Write-Host "[1/5] Frontend 依存関係"
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build

Write-Host "[2/5] Backend 依存関係"
Push-Location backend
$extras = @("--extra", "imaging", "--extra", "aliked")
if ($env:SPHERE_WITH_SAM3 -eq "1") { $extras += @("--extra", "sam3") }
uv sync @extras
Pop-Location

Write-Host "[3/5] COLMAP"
$configuredColmap = $env:SPHERE_BINARIES__COLMAP
if (-not $configuredColmap -and -not (Get-Command colmap -ErrorAction SilentlyContinue)) {
    if ($env:SPHERE_SKIP_AUTO_INSTALL_COLMAP -eq "1") {
        throw "COLMAP がありません"
    }
    & backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant cuda
    $env:SPHERE_BINARIES__COLMAP = (Get-Content .runtime\colmap-path.txt -Raw).Trim()
}

Write-Host "[4/5] 環境診断"
Push-Location backend
uv run sphere-doctor

Write-Host "[5/5] http://127.0.0.1:$port で起動"
uv run uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port $port
Pop-Location
