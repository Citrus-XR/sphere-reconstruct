$ErrorActionPreference = "Stop"
$repositoryDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$port = if ($env:SPHERE_PORT) { $env:SPHERE_PORT } else { "8787" }
Set-Location $repositoryDir

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv が必要です: https://docs.astral.sh/uv/" }
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) { throw "pnpm が必要です: npm install -g pnpm@9.15.0" }

Write-Host "[1/6] Frontend 依存関係"
pnpm --dir frontend install --frozen-lockfile
pnpm --dir frontend build

Write-Host "[2/6] Backend 依存関係"
Push-Location backend
$extras = @("--extra", "imaging", "--extra", "aliked")
if ($env:SPHERE_WITH_SAM3 -eq "1") { $extras += @("--extra", "sam3") }
uv sync @extras
Pop-Location

if (-not $env:SPHERE_FILESYSTEM__ALLOWED_ROOTS) {
    $configuredRoots = & backend\.venv\Scripts\python.exe -c "import json; from sphere_reconstruct.settings import get_settings; print(json.dumps([str(path) for path in get_settings().filesystem.allowed_roots]))"
    if ($configuredRoots.Trim() -eq "[]") {
        $roots = @(Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root } | ForEach-Object { $_.Root })
        $env:SPHERE_FILESYSTEM__ALLOWED_ROOTS = ConvertTo-Json -Compress -InputObject $roots
    }
}

Write-Host "[3/6] COLMAP"
$configuredColmap = $env:SPHERE_BINARIES__COLMAP
if (-not $configuredColmap -and -not (Get-Command colmap -ErrorAction SilentlyContinue)) {
    if ($env:SPHERE_SKIP_AUTO_INSTALL_COLMAP -eq "1") {
        throw "COLMAP がありません"
    }
    $colmapVariant = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { "cuda" } else { "cpu" }
    Write-Host "NVIDIA GPU 検出結果に基づき COLMAP $colmapVariant package を導入します"
    & backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant $colmapVariant
    $env:SPHERE_BINARIES__COLMAP = (Get-Content .runtime\colmap-path.txt -Raw).Trim()
}

Write-Host "[4/6] jpegtran"
if (-not $env:SPHERE_BINARIES__JPEGTRAN -and -not (Get-Command jpegtran -ErrorAction SilentlyContinue)) {
    if ($env:SPHERE_SKIP_AUTO_INSTALL_JPEGTRAN -ne "1") {
        & backend\.venv\Scripts\python.exe scripts\install_jpegtran.py
        $env:SPHERE_BINARIES__JPEGTRAN = (Get-Content .runtime\jpegtran-path.txt -Raw).Trim()
    }
}

Write-Host "[5/6] 環境診断"
Push-Location backend
uv run sphere-doctor

Write-Host "[6/6] http://127.0.0.1:$port で起動"
& .\.venv\Scripts\python.exe -m uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port $port
Pop-Location
