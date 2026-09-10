[CmdletBinding()]
param(
    [switch]$SkipSetup,
    [switch]$NoBrowser
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false
$repositoryDir = $PSScriptRoot
$python = Join-Path $repositoryDir "backend\.venv\Scripts\python.exe"
$serviceScript = Join-Path $repositoryDir "scripts\server_service.py"
$runtimeDir = if ($env:SPHERE_SERVICE_RUNTIME) {
    [IO.Path]::GetFullPath($env:SPHERE_SERVICE_RUNTIME)
} else {
    Join-Path $repositoryDir "runtime"
}
$env:SPHERE_SERVICE_RUNTIME = $runtimeDir
if ($env:SPHERE_CONFIG) { $env:SPHERE_CONFIG = [IO.Path]::GetFullPath($env:SPHERE_CONFIG) }
$env:PYTHONUTF8 = "1"
$env:PYTHONUNBUFFERED = "1"
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = New-Object Text.UTF8Encoding $false
$OutputEncoding = [Console]::OutputEncoding
$exitCode = 0
$phase = "Launcher initialization"
$transcribing = $false
$launcherLock = $null
$logPath = Join-Path $runtimeDir ("logs\launcher-{0}-{1}.log" -f (Get-Date -Format "yyyyMMdd-HHmmss-fff"), $PID)

function Assert-NativeSuccess {
    param([string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        $script:exitCode = $LASTEXITCODE
        throw "$Operation が exit code $LASTEXITCODE で失敗しました"
    }
}

function Invoke-Native {
    param([string]$Operation, [string]$File, [string[]]$Arguments)
    $script:phase = $Operation
    Write-Host "[$(Get-Date -Format HH:mm:ss)] $Operation"
    $command = Get-Command $File -ErrorAction Stop
    # Windows PowerShell 5.1 の native stderr は ErrorRecord になる。終了判定は exit code で行う。
    $ErrorActionPreference = "Continue"
    & $command.Source @Arguments 2>&1 | ForEach-Object { Write-Host $_.ToString() }
    Assert-NativeSuccess $Operation
}

function Read-RuntimeSettings {
    $script:phase = "Runtime configuration"
    $json = & $python -c "from sphere_reconstruct.settings import get_settings; print(get_settings().model_dump_json())"
    Assert-NativeSuccess $script:phase
    return $json | ConvertFrom-Json
}

function Select-ColmapVariant {
    if ($env:SPHERE_COLMAP_VARIANT) {
        if ($env:SPHERE_COLMAP_VARIANT -notin @("cpu", "cuda")) {
            throw "SPHERE_COLMAP_VARIANT は cpu / cuda を指定してください。独自 CUDA BA build は binaries.colmap に設定します"
        }
        return $env:SPHERE_COLMAP_VARIANT
    }
    if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { return "cuda" }
    return "cpu"
}

function Show-Ready {
    param([string]$Url)
    Write-Host ""
    Write-Host "準備完了: $Url" -ForegroundColor Green
    Write-Host "Server は background で稼働します。この window を閉じても停止しません。"
    Write-Host "停止: backend\.venv\Scripts\python.exe scripts\server_service.py stop"
    Write-Host "Server log: $(Join-Path $runtimeDir 'logs\backend.stderr.log')"
    if (-not $NoBrowser) {
        try { Start-Process $Url }
        catch { Write-Warning "Browser を開けませんでした。上記 URL を開いてください: $_" }
    }
}

Push-Location $repositoryDir
try {
    New-Item -ItemType Directory -Force (Split-Path $logPath) | Out-Null
    Start-Transcript -Path $logPath | Out-Null
    $transcribing = $true
    Write-Host "Sphere Reconstruct / $repositoryDir"
    Write-Host "Startup log: $logPath"
    try {
        $launcherLock = [IO.File]::Open((Join-Path $runtimeDir "launcher.lock"), 'OpenOrCreate', 'ReadWrite', 'None')
    }
    catch [IO.IOException] {
        throw "別の launcher が起動処理中です。完了を待ってください: $_"
    }

    if (Test-Path $python) {
        $phase = "Existing server status"
        $statusJson = & $python $serviceScript status --json
        $statusCode = $LASTEXITCODE
        if ($statusCode -notin @(0, 1, 2)) { Assert-NativeSuccess $phase }
        $status = $statusJson | ConvertFrom-Json
        if ($status.state -eq "unhealthy") { throw "既存 backend が応答しません。Server log を確認してから stop / start してください" }
        if ($status.state -eq "healthy") {
            $settings = Read-RuntimeSettings
            $requestedPort = if ($env:SPHERE_PORT) { [int]$env:SPHERE_PORT } else { [int]$settings.server.port }
            if ($status.port -ne $requestedPort) { throw "既存 backend は port=$($status.port) です。Port 変更前に stop してください" }
            Write-Host "既存 backend を使用します: pid=$($status.pid)"
            Show-Ready $status.url
            return
        }
    }

    if (-not $SkipSetup) {
        if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv が必要です: https://docs.astral.sh/uv/" }
        if (-not (Get-Command pnpm -ErrorAction SilentlyContinue)) { throw "pnpm が必要です: https://pnpm.io/installation" }
        Invoke-Native "Frontend dependencies" "pnpm" @("--dir", "frontend", "install", "--frozen-lockfile")
        Invoke-Native "Frontend build" "pnpm" @("--dir", "frontend", "build")

        $extras = @("--extra", "imaging", "--extra", "aliked")
        if ($env:SPHERE_WITH_SAM3 -eq "1") { $extras += @("--extra", "sam3") }
        if ($env:SPHERE_WITH_DENSE -eq "1") { $extras += @("--extra", "dense") }
        Push-Location backend
        try { Invoke-Native "Backend dependencies" "uv" (@("sync", "--locked", "--inexact") + $extras) }
        finally { Pop-Location }
        if ($env:SPHERE_WITH_DENSE -eq "1") {
            Invoke-Native "RoMaV2 model" $python @("scripts\install_romav2.py")
        }
    }
    if (-not (Test-Path $python) -or -not (Test-Path "frontend\dist\index.html")) {
        throw "Backend environment または frontend build がありません。-SkipSetup を外して実行してください"
    }

    $settings = Read-RuntimeSettings
    $port = if ($env:SPHERE_PORT) { [int]$env:SPHERE_PORT } else { [int]$settings.server.port }
    if ($port -lt 1 -or $port -gt 65535) { throw "Port は 1–65535 で指定してください" }
    if (-not $env:SPHERE_FILESYSTEM__ALLOWED_ROOTS -and $settings.filesystem.allowed_roots.Count -eq 0) {
        $env:SPHERE_FILESYSTEM__ALLOWED_ROOTS = ConvertTo-Json -Compress -InputObject @((Split-Path $repositoryDir))
        Write-Host "Source browser root: $(Split-Path $repositoryDir)"
    }

    if (-not $SkipSetup) {
        if (-not $settings.binaries.colmap) {
            if ($env:SPHERE_SKIP_AUTO_INSTALL_COLMAP -ne "1") {
                $variant = Select-ColmapVariant
                Invoke-Native "COLMAP ($variant)" $python @("scripts\install_colmap.py", "--variant", $variant)
                $env:SPHERE_BINARIES__COLMAP = (Get-Content .runtime\colmap-path.txt -Raw).Trim()
            }
        }
        if (-not $settings.binaries.jpegtran -and -not (Get-Command jpegtran -ErrorAction SilentlyContinue)) {
            if ($env:SPHERE_SKIP_AUTO_INSTALL_JPEGTRAN -ne "1") {
                Invoke-Native "jpegtran" $python @("scripts\install_jpegtran.py")
                $env:SPHERE_BINARIES__JPEGTRAN = (Get-Content .runtime\jpegtran-path.txt -Raw).Trim()
            }
        }
        if (-not $settings.binaries.vocab_tree) {
            Invoke-Native "Vocabulary tree" $python @("scripts\install_vocab_tree.py")
            $env:SPHERE_BINARIES__VOCAB_TREE = (Get-Content .runtime\vocab-tree-path.txt -Raw).Trim()
        }
    }

    Invoke-Native "Environment diagnosis" $python @("-m", "sphere_reconstruct.cli")
    Invoke-Native "Backend startup / health check" $python @($serviceScript, "start", "--port", "$port", "--wait", "30")
    Show-Ready "http://127.0.0.1:$port"
}
catch {
    if ($exitCode -eq 0) { $exitCode = 1 }
    Write-Host ""
    Write-Host "起動失敗: $phase" -ForegroundColor Red
    Write-Host ($_ | Out-String) -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
    Write-Host "Startup log: $logPath"
    Write-Host "Server log: $(Join-Path $runtimeDir 'logs\backend.stderr.log')"
}
finally {
    if ($launcherLock) { $launcherLock.Dispose() }
    if ($transcribing) { Stop-Transcript | Out-Null }
    Pop-Location
}
exit $exitCode
