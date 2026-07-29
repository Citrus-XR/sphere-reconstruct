param(
    [Parameter(Mandatory = $true)]
    [string]$InstallRoot
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$BinDirectory = Join-Path $InstallRoot "bin"
if (-not (Test-Path (Join-Path $BinDirectory "colmap.exe"))) {
    throw "COLMAP runtime の bin directory が不正です: $BinDirectory"
}
if (-not (Get-Command dumpbin -ErrorAction SilentlyContinue)) {
    throw "Runtime dependency 検証には Visual Studio の dumpbin が必要です"
}

$RequiredRuntimePatterns = @(
    "ceres.dll",
    "cudss64_*.dll",
    "cudss_mtlayer_*.dll",
    "onnxruntime_providers_cuda.dll",
    "cufft64_*.dll",
    "cudnn64_9.dll",
    "nvJitLink_*.dll",
    "nvrtc64_*.dll",
    "nvrtc-builtins64_*.dll"
)
foreach ($Pattern in $RequiredRuntimePatterns) {
    if (-not (Get-ChildItem $BinDirectory -Filter $Pattern -File -ErrorAction SilentlyContinue)) {
        throw "必須 runtime が package にありません: $Pattern"
    }
}

# dumpbin /IMPORTS は通常 import と delay-load import の両方を出す。全 PE を seed にすることで、
# COLMAP が実行時に選ぶ ONNX provider も起動前に検証する。
# https://github.com/MicrosoftDocs/cpp-docs/blob/c98cb7524abb5c241c81b494be159d08a84d308d/docs/build/reference/imports-dumpbin.md#L9-L16
$Unresolved = [System.Collections.Generic.List[string]]::new()
$Binaries = Get-ChildItem $BinDirectory -File | Where-Object {
    $_.Extension -in @(".exe", ".dll")
}
foreach ($Binary in $Binaries) {
    $ImportOutput = & dumpbin /NOLOGO /IMPORTS $Binary.FullName | Out-String
    if ($LASTEXITCODE -ne 0) {
        throw "dumpbin が失敗しました: $($Binary.FullName)"
    }
    $Dependencies = [regex]::Matches(
        $ImportOutput,
        '(?im)^\s*(?<name>[A-Za-z0-9_.+\-]+\.dll)\s*$'
    ) | ForEach-Object { $_.Groups['name'].Value } | Sort-Object -Unique
    foreach ($Dependency in $Dependencies) {
        if (Test-Path (Join-Path $BinDirectory $Dependency)) {
            continue
        }
        if (Test-Path (Join-Path "$env:SystemRoot\System32" $Dependency)) {
            continue
        }
        if ($Dependency -match '^(api|ext)-ms-win-' -or $Dependency -ieq "nvcuda.dll") {
            continue
        }
        $Unresolved.Add("$($Binary.Name) -> $Dependency")
    }
}
if ($Unresolved.Count -gt 0) {
    throw "未解決の app-local DLL import があります:`n$($Unresolved -join [Environment]::NewLine)"
}

Add-Type -TypeDefinition @'
using System;
using System.ComponentModel;
using System.Runtime.InteropServices;

public static class SphereRuntimeLoader
{
    private const uint LoadLibrarySearchDllLoadDir = 0x00000100;
    private const uint LoadLibrarySearchDefaultDirs = 0x00001000;

    [DllImport("kernel32.dll", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr LoadLibraryExW(string path, IntPtr file, uint flags);

    [DllImport("kernel32.dll")]
    private static extern bool FreeLibrary(IntPtr module);

    public static void Validate(string path)
    {
        var module = LoadLibraryExW(
            path,
            IntPtr.Zero,
            LoadLibrarySearchDllLoadDir | LoadLibrarySearchDefaultDirs);
        if (module == IntPtr.Zero)
            throw new Win32Exception(Marshal.GetLastWin32Error(), path);
        FreeLibrary(module);
    }
}
'@

$OriginalPath = $env:PATH
try {
    $env:PATH = "$BinDirectory;$env:SystemRoot\System32;$env:SystemRoot"
    foreach ($Library in @("ceres.dll", "cudss64_0.dll")) {
        [SphereRuntimeLoader]::Validate((Join-Path $BinDirectory $Library))
    }
    $DriverLibrary = Join-Path "$env:SystemRoot\System32" "nvcuda.dll"
    if (Test-Path $DriverLibrary) {
        [SphereRuntimeLoader]::Validate(
            (Join-Path $BinDirectory "onnxruntime_providers_cuda.dll")
        )
    }
    else {
        # GitHub hosted Windows runner は NVIDIA driver を持たない。Provider 自体の全 static / delay
        # import は上の PE closure で検査済みなので、driver が無い場合だけ DllMain smoke を省く。
        Write-Output "ONNX CUDA provider LoadLibrary: skipped (nvcuda.dll unavailable)"
    }
    & (Join-Path $BinDirectory "colmap.exe") version
    if ($LASTEXITCODE -ne 0) {
        throw "standalone COLMAP runtime smoke test が失敗しました: $LASTEXITCODE"
    }
}
finally {
    $env:PATH = $OriginalPath
}

Write-Output "COLMAP runtime dependency validation: ok"
