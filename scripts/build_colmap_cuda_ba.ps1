param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$CudaArchitectures = "",
    [string]$CudssWheelPath = "",
    [switch]$ReuseBuildRoot
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ColmapTag = "4.1.1"
$CeresCommit = "bac1127f9ef672405bd0d2d9c84e809ae89bd239"
$VcpkgCommit = "6d9d7df564a1ccdaa994e4ad39ccd4a32360867b"
$CudaVersionMatch = [regex]::Match($env:CUDA_PATH, 'v(?<major>\d+)\.(?<minor>\d+)')
if (-not $CudaVersionMatch.Success) {
    throw "CUDA_PATH から CUDA version を判定できません: $env:CUDA_PATH"
}
$CudaMajor = [int]$CudaVersionMatch.Groups['major'].Value
$CudssWheel = switch ($CudaMajor) {
    12 {
        @{
            FileName = "nvidia_cudss_cu12-0.8.0.10-py3-none-win_amd64.whl"
            Url = "https://files.pythonhosted.org/packages/49/4c/85bfa40b863e1d2ec8629964283ba8b4b3f85b2c01b8ab6fe5ba60d5eed0/nvidia_cudss_cu12-0.8.0.10-py3-none-win_amd64.whl"
            Sha256 = "89bf57d4d05d25c7f6e1288acbc94c5eed519763c55d7ec1f6d73099ace8554e"
        }
    }
    13 {
        @{
            FileName = "nvidia_cudss_cu13-0.8.0.10-py3-none-win_amd64.whl"
            Url = "https://files.pythonhosted.org/packages/04/83/42ca016cd77181354147edb9c03587fc6d517a706d042256ad290feb7923/nvidia_cudss_cu13-0.8.0.10-py3-none-win_amd64.whl"
            Sha256 = "c084261bdc9cf3468d0fda5a20c56aa29b54b888d143bbdd39205c19fe3639a3"
        }
    }
    default { throw "CUDA $CudaMajor 用 cuDSS 0.8.0.10 Windows runtime は提供されていません" }
}
$TemporaryRoot = if ($env:RUNNER_TEMP) { $env:RUNNER_TEMP } else { $env:TEMP }
$BuildRoot = Join-Path $TemporaryRoot "sphere-colmap-cuda-ba"
$Triplet = "x64-windows-release"
$CompilerLauncherArgs = @()
if (Get-Command sccache -ErrorAction SilentlyContinue) {
    $CompilerLauncherArgs = @(
        "-DCMAKE_C_COMPILER_LAUNCHER=sccache",
        "-DCMAKE_CXX_COMPILER_LAUNCHER=sccache"
    )
}

function Initialize-PinnedRepository {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url,
        [Parameter(Mandatory = $true)]
        [string]$Revision,
        [Parameter(Mandatory = $true)]
        [string]$Destination
    )

    New-Item $Destination -ItemType Directory | Out-Null
    git -C $Destination init --quiet
    Assert-NativeSuccess "git init: $Destination"
    git -C $Destination remote add origin $Url
    Assert-NativeSuccess "git remote add: $Url"
    git -C $Destination fetch --depth 1 origin $Revision
    Assert-NativeSuccess "git fetch: $Revision"
    git -C $Destination checkout --detach FETCH_HEAD
    Assert-NativeSuccess "git checkout: $Revision"
}

function Assert-NativeSuccess {
    param([Parameter(Mandatory = $true)][string]$Operation)
    if ($LASTEXITCODE -ne 0) {
        throw "$Operation が exit code $LASTEXITCODE で失敗しました"
    }
}

function Ensure-GitPatch {
    param(
        [Parameter(Mandatory = $true)][string]$Repository,
        [Parameter(Mandatory = $true)][string]$Patch,
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][string]$SentinelFile,
        [Parameter(Mandatory = $true)][string]$SentinelText
    )
    if (Select-String `
        -Path (Join-Path $Repository $SentinelFile) `
        -Pattern $SentinelText `
        -SimpleMatch `
        -Quiet) {
        return
    }
    git -C $Repository apply --check $Patch
    Assert-NativeSuccess "$Name patch check"
    git -C $Repository apply $Patch
    Assert-NativeSuccess "$Name patch"
}

function Resolve-CmakeExecutable {
    $Candidates = @(
        Get-Command cmake -All -ErrorAction SilentlyContinue |
            Select-Object -ExpandProperty Source
    )
    $CommonCmake = Join-Path $env:ProgramFiles "CMake/bin/cmake.exe"
    if (Test-Path $CommonCmake) {
        $Candidates += $CommonCmake
    }
    $Resolved = @(
        $Candidates | Select-Object -Unique | ForEach-Object {
            $Output = & $_ --version | Select-Object -First 1
            if ($Output -match '^cmake version (?<version>\d+\.\d+\.\d+)') {
                [PSCustomObject]@{ Path = $_; Version = [version]$Matches['version'] }
            }
        }
    ) | Sort-Object Version -Descending | Select-Object -First 1
    if ($null -eq $Resolved -or $Resolved.Version -lt [version]"3.30.0") {
        throw "CMake 3.30+ が必要です"
    }
    Write-Host "CMake $($Resolved.Version): $($Resolved.Path)"
    return $Resolved.Path
}

$CmakeExecutable = Resolve-CmakeExecutable

if ((Test-Path $BuildRoot) -and -not $ReuseBuildRoot) {
    Remove-Item $BuildRoot -Recurse -Force
}
New-Item $BuildRoot -ItemType Directory -Force | Out-Null
New-Item $OutputDirectory -ItemType Directory -Force | Out-Null
if ($env:VCPKG_DEFAULT_BINARY_CACHE) {
    New-Item $env:VCPKG_DEFAULT_BINARY_CACHE -ItemType Directory -Force | Out-Null
}

$ColmapSource = Join-Path $BuildRoot "colmap"
$CeresSource = Join-Path $BuildRoot "ceres"
$VcpkgRoot = Join-Path $BuildRoot "vcpkg"
$CudssRoot = Join-Path $BuildRoot "cudss"
$CeresBuild = Join-Path $BuildRoot "ceres-build"
$CeresInstall = Join-Path $BuildRoot "ceres-install"
$CeresConfigDirectory = Join-Path $CeresInstall "lib/cmake/Ceres"
$ColmapBuild = Join-Path $BuildRoot "colmap-build"
$ColmapInstall = Join-Path $BuildRoot "colmap-install"

if (-not (Test-Path (Join-Path $ColmapSource ".git"))) {
    Initialize-PinnedRepository `
        -Url "https://github.com/colmap/colmap.git" `
        -Revision $ColmapTag `
        -Destination $ColmapSource
}
$RigPairPatch = Join-Path $PSScriptRoot "patches/colmap-4.1.1-sequential-rig-pairs.patch"
# 4.1.1 の folder-major rig pairing bug に upstream の最小 semantic fix だけを backport する。
# https://github.com/colmap/colmap/pull/4591
Ensure-GitPatch `
    $ColmapSource `
    $RigPairPatch `
    "COLMAP rig pairing" `
    "src/colmap/controllers/pairing.cc" `
    "IsValidSequentialNeighbor("
if (-not (Test-Path (Join-Path $CeresSource ".git"))) {
    Initialize-PinnedRepository `
        -Url "https://github.com/ceres-solver/ceres-solver.git" `
        -Revision $CeresCommit `
        -Destination $CeresSource
}
$CeresArchitecturePatch = Join-Path $PSScriptRoot "patches/ceres-preserve-cuda-architectures.patch"
# Ceres bac1127 は caller の architecture list を破棄するため、明示値だけを優先させる。
# https://github.com/ceres-solver/ceres-solver/blob/bac1127f9ef672405bd0d2d9c84e809ae89bd239/CMakeLists.txt#L319-L353
Ensure-GitPatch `
    $CeresSource `
    $CeresArchitecturePatch `
    "Ceres CUDA architecture" `
    "CMakeLists.txt" `
    "Using caller-provided CUDA Architecture"
git -C $CeresSource submodule update --init --depth 1 third_party/abseil-cpp
Assert-NativeSuccess "Ceres abseil submodule"
if (-not (Test-Path (Join-Path $VcpkgRoot "vcpkg.exe"))) {
    Initialize-PinnedRepository `
        -Url "https://github.com/microsoft/vcpkg.git" `
        -Revision $VcpkgCommit `
        -Destination $VcpkgRoot
    & (Join-Path $VcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics
    Assert-NativeSuccess "vcpkg bootstrap"
}
Remove-Item Env:VCPKG_ROOT -ErrorAction SilentlyContinue

# COLMAP の manifest Ceres は CPU build で、直後に Ceres_DIR で差し替えても vcpkg が重複 build する。
# Ceres feature が暗黙に供給していた LAPACK / SuiteSparse は COLMAP 自身の CHOLMOD 検出にも必要なため、
# manifest の直接依存へ移して custom Ceres と COLMAP の両方から同じ package を解決する。
$ColmapManifestPath = Join-Path $ColmapSource "vcpkg.json"
$ColmapManifest = Get-Content $ColmapManifestPath -Raw | ConvertFrom-Json
$CeresDependencyCount = @(
    $ColmapManifest.dependencies | Where-Object {
        if ($_ -is [string]) { $_ -eq "ceres" } else { $_.name -eq "ceres" }
    }
).Count
if ($CeresDependencyCount -gt 1) {
    throw "COLMAP vcpkg manifest の Ceres dependency が重複しています"
}
$ColmapManifest.dependencies = @(
    $ColmapManifest.dependencies | Where-Object {
        if ($_ -is [string]) {
            return $_ -ne "ceres"
        }
        return $_.name -ne "ceres"
    }
)
$RequiredColmapDependencies = @(
    "lapack",
    [PSCustomObject]@{
        name = "suitesparse-cholmod"
        "default-features" = $false
        features = @("matrixops")
    },
    "suitesparse-config",
    "suitesparse-spqr"
)
foreach ($Dependency in $RequiredColmapDependencies) {
    $DependencyName = if ($Dependency -is [string]) { $Dependency } else { $Dependency.name }
    $DependencyNames = @(
        $ColmapManifest.dependencies | ForEach-Object {
            if ($_ -is [string]) { $_ } else { $_.name }
        }
    )
    if ($DependencyNames -notcontains $DependencyName) {
        $ColmapManifest.dependencies += $Dependency
    }
}
$ColmapManifest | ConvertTo-Json -Depth 20 | Set-Content $ColmapManifestPath -Encoding utf8

& (Join-Path $VcpkgRoot "vcpkg.exe") install `
    "eigen3:$Triplet" `
    "glog:$Triplet" `
    "suitesparse-cholmod[matrixops]:$Triplet" `
    "suitesparse-config:$Triplet" `
    "suitesparse-spqr:$Triplet" `
    "lapack:$Triplet"
Assert-NativeSuccess "vcpkg Ceres dependencies"

$WheelDirectory = Join-Path $BuildRoot "wheel"
$CudssPackageRoot = Join-Path $CudssRoot "nvidia/cu$CudaMajor"
$CudssDll = Join-Path $CudssPackageRoot "bin/cudss64_0.dll"
$CudssHeader = Join-Path $CudssPackageRoot "include/cudss.h"
if (-not ((Test-Path $CudssDll) -and (Test-Path $CudssHeader))) {
    if (Test-Path $CudssRoot) {
        Remove-Item $CudssRoot -Recurse -Force
    }
    New-Item $WheelDirectory -ItemType Directory -Force | Out-Null
    $WheelPath = Join-Path $WheelDirectory $CudssWheel.FileName
    if ($CudssWheelPath) {
        Copy-Item $CudssWheelPath $WheelPath -Force
    }
    else {
        curl.exe `
            --fail `
            --location `
            --retry 5 `
            --retry-all-errors `
            --retry-delay 2 `
            --connect-timeout 30 `
            --speed-limit 1024 `
            --speed-time 60 `
            --continue-at - `
            --output $WheelPath `
            $CudssWheel.Url
        Assert-NativeSuccess "cuDSS wheel download"
    }
    $WheelHash = (Get-FileHash $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($WheelHash -ne $CudssWheel.Sha256) {
        throw "cuDSS wheel の SHA-256 が一致しません: $WheelHash"
    }
    $WheelZip = Join-Path $WheelDirectory "cudss.zip"
    Copy-Item $WheelPath $WheelZip -Force
    Expand-Archive $WheelZip -DestinationPath $CudssRoot
}

$CudssConfigDirectory = Join-Path $CudssPackageRoot "lib/cmake/cudss"
New-Item $CudssConfigDirectory -ItemType Directory -Force | Out-Null
$CudssConfig = @(
    'set(cudss_VERSION "0.8.0.10")',
    'if(NOT TARGET cudss)',
    '  add_library(cudss SHARED IMPORTED)',
    '  set_target_properties(cudss PROPERTIES',
    '    IMPORTED_LOCATION "${CMAKE_CURRENT_LIST_DIR}/../../../bin/cudss64_0.dll"',
    '    IMPORTED_IMPLIB "${CMAKE_CURRENT_LIST_DIR}/../../cudss.lib"',
    '    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_LIST_DIR}/../../../include")',
    'endif()'
) -join [Environment]::NewLine
Set-Content (Join-Path $CudssConfigDirectory "cudssConfig.cmake") $CudssConfig -Encoding utf8
$CudssVersionConfig = @(
    'set(PACKAGE_VERSION "0.8.0.10")',
    'set(PACKAGE_VERSION_COMPATIBLE TRUE)',
    'set(PACKAGE_VERSION_EXACT TRUE)'
) -join [Environment]::NewLine
Set-Content (Join-Path $CudssConfigDirectory "cudssConfigVersion.cmake") $CudssVersionConfig -Encoding utf8

$Toolchain = Join-Path $VcpkgRoot "scripts/buildsystems/vcpkg.cmake"
$EffectiveCudaArchitectures = if ($CudaArchitectures) {
    $CudaArchitectures
}
elseif ($CudaMajor -ge 13) {
    "75;80-real;86-real;89-real;90-real;100-real;120-real"
}
else {
    "75;80-real;86-real;89-real;90-real"
}
$CeresDll = Join-Path $CeresInstall "bin/ceres.dll"
if (-not ($ReuseBuildRoot -and (Test-Path $CeresDll))) {
    & $CmakeExecutable -S $CeresSource -B $CeresBuild -GNinja `
        "-DCMAKE_BUILD_TYPE=Release" `
        "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release" `
        @CompilerLauncherArgs `
        "-DCMAKE_CUDA_STANDARD=17" `
        "-DCMAKE_CUDA_STANDARD_REQUIRED=ON" `
        "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
        "-DVCPKG_TARGET_TRIPLET=$Triplet" `
        "-DVCPKG_MANIFEST_MODE=OFF" `
        "-DCMAKE_INSTALL_PREFIX=$CeresInstall" `
        "-DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor" `
        "-DCMAKE_PREFIX_PATH=$CudssPackageRoot" `
        "-Dcudss_DIR=$CudssConfigDirectory" `
        "-DBUILD_SHARED_LIBS=ON" `
        "-DBUILD_TESTING=OFF" `
        "-DBUILD_EXAMPLES=OFF" `
        "-DUSE_CUDA=ON" `
        "-DSUITESPARSE=ON" `
        "-DLAPACK=ON" `
        "-DCMAKE_CUDA_ARCHITECTURES:STRING=$EffectiveCudaArchitectures"
    Assert-NativeSuccess "Ceres configure"
    & $CmakeExecutable --build $CeresBuild --target install
    Assert-NativeSuccess "Ceres build"
}

& $CmakeExecutable -S $ColmapSource -B $ColmapBuild -GNinja `
    "-DCMAKE_BUILD_TYPE=Release" `
    "-DCMAKE_TRY_COMPILE_CONFIGURATION=Release" `
    @CompilerLauncherArgs `
    "-DCMAKE_CUDA_STANDARD=17" `
    "-DCMAKE_CUDA_STANDARD_REQUIRED=ON" `
    "-DCMAKE_TOOLCHAIN_FILE=$Toolchain" `
    "-DVCPKG_TARGET_TRIPLET=$Triplet" `
    "-DCMAKE_INSTALL_PREFIX=$ColmapInstall" `
    "-DCMAKE_CUDA_FLAGS=-Xcompiler=/Zc:preprocessor" `
    "-DCMAKE_PREFIX_PATH=$CeresInstall;$CudssPackageRoot" `
    "-DCeres_DIR=$CeresConfigDirectory" `
    "-Dcudss_DIR=$CudssConfigDirectory" `
    "-DCUDA_ENABLED=ON" `
    "-DONNX_ENABLED=ON" `
    "-DFETCH_ONNX=ON" `
    "-DGUI_ENABLED=OFF" `
    "-DMVS_ENABLED=OFF" `
    "-DOPENGL_ENABLED=OFF" `
    "-DCGAL_ENABLED=OFF" `
    "-DTESTS_ENABLED=OFF" `
    "-DCMAKE_CUDA_ARCHITECTURES:STRING=$EffectiveCudaArchitectures"
Assert-NativeSuccess "COLMAP configure"
& $CmakeExecutable --build $ColmapBuild --target install
Assert-NativeSuccess "COLMAP build"

$InstallBin = Join-Path $ColmapInstall "bin"
Copy-Item (Join-Path $CeresInstall "bin/*.dll") $InstallBin -Force
Copy-Item (Join-Path $CudssPackageRoot "bin/cudss64_0.dll") $InstallBin -Force

$VcpkgBinDirectories = @(
    (Join-Path $VcpkgRoot "installed/$Triplet/bin"),
    (Join-Path $ColmapBuild "vcpkg_installed/$Triplet/bin")
)
foreach ($Directory in $VcpkgBinDirectories) {
    if (Test-Path $Directory) {
        Copy-Item (Join-Path $Directory "*.dll") $InstallBin -Force
    }
}

$CudaBin = Join-Path $env:CUDA_PATH "bin"
$CudaRuntimePatterns = @(
    "cudart64_*.dll",
    "cublas64_*.dll",
    "cublasLt64_*.dll",
    "cusolver64_*.dll",
    "cusolverMg64_*.dll",
    "cusparse64_*.dll",
    "nvJitLink_*.dll",
    "curand64_*.dll"
)
foreach ($Pattern in $CudaRuntimePatterns) {
    Get-ChildItem $CudaBin -Filter $Pattern | Copy-Item -Destination $InstallBin -Force
}

$CeresDll = Join-Path $InstallBin "ceres.dll"
$Dependencies = & dumpbin /DEPENDENTS $CeresDll | Out-String
Assert-NativeSuccess "dumpbin Ceres dependency inspection"
if ($Dependencies -notmatch "cudss64_0.dll") {
    throw "The generated Ceres library is not linked to cuDSS"
}
$ColmapExecutable = Join-Path $InstallBin "colmap.exe"
& $ColmapExecutable version
Assert-NativeSuccess "COLMAP runtime smoke test"

$Metadata = @{
    colmap_version = $ColmapTag
    ceres_version = "2.3.0-dev"
    ceres_commit = $CeresCommit
    vcpkg_commit = $VcpkgCommit
    cuda_version = $env:CUDA_PATH -replace '^.*v', ''
    cuda_architectures = $EffectiveCudaArchitectures
    cudss_version = "0.8.0.10"
    ceres_cuda = $true
    cudss = $true
    sequential_rig_pairing_fix = "COLMAP PR 4591 semantic backport"
    gpu_bundle_adjustment_dense = $true
    gpu_bundle_adjustment_sparse = $true
}
$Metadata | ConvertTo-Json | Set-Content (Join-Path $ColmapInstall "sphere-colmap-capabilities.json") -Encoding utf8

$LicenseDirectory = Join-Path $ColmapInstall "licenses"
New-Item $LicenseDirectory -ItemType Directory -Force | Out-Null
Copy-Item (Join-Path $ColmapSource "COPYING.txt") (Join-Path $LicenseDirectory "COLMAP-COPYING.txt")
Copy-Item (Join-Path $CeresSource "LICENSE") (Join-Path $LicenseDirectory "CERES-LICENSE.txt")
$CudssMetadata = Get-ChildItem $CudssRoot -Recurse -Filter "METADATA" | Select-Object -First 1
if ($null -ne $CudssMetadata) {
    Copy-Item $CudssMetadata.FullName (Join-Path $LicenseDirectory "NVIDIA-CUDSS-METADATA.txt")
}
$CudaEula = Join-Path $env:CUDA_PATH "EULA.txt"
if (Test-Path $CudaEula) {
    Copy-Item $CudaEula (Join-Path $LicenseDirectory "NVIDIA-CUDA-EULA.txt")
}
Invoke-WebRequest `
    -Uri "https://docs.nvidia.com/cuda/cudss/license.html" `
    -OutFile (Join-Path $LicenseDirectory "NVIDIA-CUDSS-LICENSE.html") `
    -UseBasicParsing

$RuntimeBundle = Join-Path $BuildRoot "runtime-bundle"
if (Test-Path $RuntimeBundle) {
    Remove-Item $RuntimeBundle -Recurse -Force
}
New-Item $RuntimeBundle -ItemType Directory | Out-Null
Copy-Item $InstallBin $RuntimeBundle -Recurse
Copy-Item $LicenseDirectory $RuntimeBundle -Recurse
Copy-Item (Join-Path $ColmapInstall "sphere-colmap-capabilities.json") $RuntimeBundle

$Archive = Join-Path $OutputDirectory "colmap-4.1.1-x64-windows-cuda-ba.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path (Join-Path $RuntimeBundle "*") -DestinationPath $Archive -CompressionLevel Optimal
$Hash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content "${Archive}.sha256" "$Hash  $(Split-Path $Archive -Leaf)" -Encoding ascii
Write-Output "COLMAP CUDA BA archive: $Archive"
Write-Output "SHA-256: $Hash"
