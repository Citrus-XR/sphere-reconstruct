$ErrorActionPreference = "Stop"
$repositoryDir = Resolve-Path (Join-Path $PSScriptRoot "..")
$port = if ($env:SPHERE_PORT) { $env:SPHERE_PORT } else { "8787" }
Set-Location $repositoryDir

function Assert-NativeSuccess {
    param([Parameter(Mandatory = $true)][string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation が exit code $LASTEXITCODE で失敗しました"
    }
}

function Select-ColmapVariant {
    if ($env:SPHERE_COLMAP_VARIANT) {
        if ($env:SPHERE_COLMAP_VARIANT -notin @("cpu", "cuda", "cuda-ba")) {
            throw "SPHERE_COLMAP_VARIANT は cpu / cuda / cuda-ba のいずれかです"
        }
        return $env:SPHERE_COLMAP_VARIANT
    }
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        return "cpu"
    }
    $gpuLine = @(& $nvidiaSmi.Source `
        --query-gpu=name,driver_version,compute_cap `
        --format=csv,noheader,nounits 2>$null) | Select-Object -First 1
    if (-not $gpuLine -or $LASTEXITCODE -ne 0) {
        Write-Warning "NVIDIA GPU は検出しましたが driver / compute capability を取得できないため official CUDA runtime を使います"
        return "cuda"
    }
    $fields = @($gpuLine -split "," | ForEach-Object { $_.Trim() })
    try {
        $driverMajor = [int]($fields[1].Split(".", 2)[0])
        $computeCapability = [double]::Parse(
            $fields[2],
            [Globalization.CultureInfo]::InvariantCulture
        )
    }
    catch {
        Write-Warning "NVIDIA capability の形式を解釈できないため official CUDA runtime を使います: $gpuLine"
        return "cuda"
    }
    if ($driverMajor -ge 580 -and $computeCapability -ge 7.5) {
        Write-Host "CUDA BA 対応 GPU: $($fields[0]) / driver $($fields[1]) / compute $computeCapability"
        return "cuda-ba"
    }
    Write-Warning "CUDA BA runtime は driver 580+ / compute 7.5+ が必要です。GPU feature 対応の official CUDA runtime を使います: $gpuLine"
    return "cuda"
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv が必要です: https://docs.astral.sh/uv/" }
if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) { throw "pnpm が必要です: npm install -g pnpm@9.15.0" }

Write-Host "[1/7] Frontend 依存関係"
pnpm --dir frontend install --frozen-lockfile
Assert-NativeSuccess "Frontend dependency installation"
pnpm --dir frontend build
Assert-NativeSuccess "Frontend build"

Write-Host "[2/7] Backend 依存関係"
Push-Location backend
$extras = @("--extra", "imaging", "--extra", "aliked")
if ($env:SPHERE_WITH_SAM3 -eq "1") { $extras += @("--extra", "sam3") }
uv sync @extras
Assert-NativeSuccess "Backend dependency synchronization"
Pop-Location

if (-not $env:SPHERE_FILESYSTEM__ALLOWED_ROOTS) {
    $configuredRoots = & backend\.venv\Scripts\python.exe -c "import json; from sphere_reconstruct.settings import get_settings; print(json.dumps([str(path) for path in get_settings().filesystem.allowed_roots]))"
    Assert-NativeSuccess "Filesystem root diagnosis"
    if ($configuredRoots.Trim() -eq "[]") {
        $roots = @(Get-PSDrive -PSProvider FileSystem | Where-Object { $_.Root } | ForEach-Object { $_.Root })
        $env:SPHERE_FILESYSTEM__ALLOWED_ROOTS = ConvertTo-Json -Compress -InputObject $roots
    }
}

Write-Host "[3/7] COLMAP"
$configuredColmap = $env:SPHERE_BINARIES__COLMAP
if (-not $configuredColmap) {
    $configuredColmap = & backend\.venv\Scripts\python.exe -c "from sphere_reconstruct.settings import get_settings; print(get_settings().binaries.colmap)"
    Assert-NativeSuccess "COLMAP configuration read"
}
if (-not $configuredColmap.Trim()) {
    if ($env:SPHERE_SKIP_AUTO_INSTALL_COLMAP -eq "1") {
        $systemColmap = Get-Command colmap -ErrorAction SilentlyContinue
        $installedRecord = ".runtime\colmap-path.txt"
        $installedColmap = if (Test-Path $installedRecord) {
            (Get-Content $installedRecord -Raw).Trim()
        }
        else {
            ""
        }
        if (-not $systemColmap -and (-not $installedColmap -or -not (Test-Path $installedColmap))) {
            throw "COLMAP がありません"
        }
    }
    else {
        $colmapVariant = Select-ColmapVariant
        Write-Host "NVIDIA GPU 検出結果に基づき COLMAP $colmapVariant package を導入します"
        & backend\.venv\Scripts\python.exe scripts\install_colmap.py --variant $colmapVariant
        Assert-NativeSuccess "COLMAP installation"
        $env:SPHERE_BINARIES__COLMAP = (Get-Content .runtime\colmap-path.txt -Raw).Trim()
    }
}

Write-Host "[4/7] jpegtran"
if (-not $env:SPHERE_BINARIES__JPEGTRAN -and -not (Get-Command jpegtran -ErrorAction SilentlyContinue)) {
    if ($env:SPHERE_SKIP_AUTO_INSTALL_JPEGTRAN -ne "1") {
        & backend\.venv\Scripts\python.exe scripts\install_jpegtran.py
        Assert-NativeSuccess "jpegtran installation"
        $env:SPHERE_BINARIES__JPEGTRAN = (Get-Content .runtime\jpegtran-path.txt -Raw).Trim()
    }
}

Write-Host "[5/7] Vocabulary tree"
$configuredVocabTree = $env:SPHERE_BINARIES__VOCAB_TREE
if (-not $configuredVocabTree) {
    $configuredVocabTree = & backend\.venv\Scripts\python.exe -c "from sphere_reconstruct.settings import get_settings; print(get_settings().binaries.vocab_tree)"
    Assert-NativeSuccess "Vocabulary tree configuration read"
}
if (-not $configuredVocabTree.Trim()) {
    & backend\.venv\Scripts\python.exe scripts\install_vocab_tree.py
    Assert-NativeSuccess "Vocabulary tree installation"
    $env:SPHERE_BINARIES__VOCAB_TREE = (Get-Content .runtime\vocab-tree-path.txt -Raw).Trim()
}

Write-Host "[6/7] 環境診断"
Push-Location backend
uv run sphere-doctor
Assert-NativeSuccess "Environment diagnosis"

Write-Host "[7/7] http://127.0.0.1:$port で起動"
& .\.venv\Scripts\python.exe -m uvicorn sphere_reconstruct.main:app --host 127.0.0.1 --port $port
Assert-NativeSuccess "Backend server"
Pop-Location
