param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,
    [string]$CudaArchitectures = "",
    [string]$CudssWheelPath = ""
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
    git -C $Destination remote add origin $Url
    git -C $Destination fetch --depth 1 origin $Revision
    git -C $Destination checkout --detach FETCH_HEAD
}

if (Test-Path $BuildRoot) {
    Remove-Item $BuildRoot -Recurse -Force
}
New-Item $BuildRoot -ItemType Directory | Out-Null
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

Initialize-PinnedRepository `
    -Url "https://github.com/colmap/colmap.git" `
    -Revision $ColmapTag `
    -Destination $ColmapSource
$RigPairPatch = Join-Path $PSScriptRoot "patches/colmap-4.1.1-sequential-rig-pairs.patch"
# 4.1.1 の folder-major rig pairing bug に upstream の最小 semantic fix だけを backport する。
# https://github.com/colmap/colmap/pull/4591
git -C $ColmapSource apply --check $RigPairPatch
git -C $ColmapSource apply $RigPairPatch
Initialize-PinnedRepository `
    -Url "https://github.com/ceres-solver/ceres-solver.git" `
    -Revision $CeresCommit `
    -Destination $CeresSource
$CeresArchitecturePatch = Join-Path $PSScriptRoot "patches/ceres-preserve-cuda-architectures.patch"
# Ceres bac1127 は caller の architecture list を破棄するため、明示値だけを優先させる。
# https://github.com/ceres-solver/ceres-solver/blob/bac1127f9ef672405bd0d2d9c84e809ae89bd239/CMakeLists.txt#L319-L353
git -C $CeresSource apply --check $CeresArchitecturePatch
git -C $CeresSource apply $CeresArchitecturePatch
git -C $CeresSource submodule update --init --depth 1 third_party/abseil-cpp
Initialize-PinnedRepository `
    -Url "https://github.com/microsoft/vcpkg.git" `
    -Revision $VcpkgCommit `
    -Destination $VcpkgRoot
Remove-Item Env:VCPKG_ROOT -ErrorAction SilentlyContinue
& (Join-Path $VcpkgRoot "bootstrap-vcpkg.bat") -disableMetrics

# COLMAP の manifest Ceres は CPU build で、直後に Ceres_DIR で差し替えても vcpkg が重複 build する。
# Ceres feature が暗黙に供給していた LAPACK / SuiteSparse は COLMAP 自身の CHOLMOD 検出にも必要なため、
# manifest の直接依存へ移して custom Ceres と COLMAP の両方から同じ package を解決する。
$ColmapManifestPath = Join-Path $ColmapSource "vcpkg.json"
$ColmapManifest = Get-Content $ColmapManifestPath -Raw | ConvertFrom-Json
$OriginalDependencyCount = $ColmapManifest.dependencies.Count
$ColmapManifest.dependencies = @(
    $ColmapManifest.dependencies | Where-Object {
        if ($_ -is [string]) {
            return $_ -ne "ceres"
        }
        return $_.name -ne "ceres"
    }
)
if ($ColmapManifest.dependencies.Count -ne $OriginalDependencyCount - 1) {
    throw "COLMAP vcpkg manifest から Ceres dependency を一意に除外できません"
}
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

$WheelDirectory = Join-Path $BuildRoot "wheel"
New-Item $WheelDirectory -ItemType Directory | Out-Null
$WheelPath = Join-Path $WheelDirectory $CudssWheel.FileName
if ($CudssWheelPath) {
    Copy-Item $CudssWheelPath $WheelPath
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
}
$WheelHash = (Get-FileHash $WheelPath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($WheelHash -ne $CudssWheel.Sha256) {
    throw "cuDSS wheel の SHA-256 が一致しません: $WheelHash"
}
$WheelZip = Join-Path $WheelDirectory "cudss.zip"
Copy-Item $WheelPath $WheelZip
Expand-Archive $WheelZip -DestinationPath $CudssRoot

$CudssPackageRoot = Join-Path $CudssRoot "nvidia/cu$CudaMajor"
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
cmake -S $CeresSource -B $CeresBuild -GNinja `
    "-DCMAKE_BUILD_TYPE=Release" `
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
cmake --build $CeresBuild --target install

cmake -S $ColmapSource -B $ColmapBuild -GNinja `
    "-DCMAKE_BUILD_TYPE=Release" `
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
cmake --build $ColmapBuild --target install

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
if ($Dependencies -notmatch "cudss64_0.dll") {
    throw "The generated Ceres library is not linked to cuDSS"
}
$ColmapExecutable = Join-Path $InstallBin "colmap.exe"
& $ColmapExecutable version

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

$Archive = Join-Path $OutputDirectory "colmap-4.1.1-x64-windows-cuda-ba.zip"
if (Test-Path $Archive) {
    Remove-Item $Archive -Force
}
Compress-Archive -Path (Join-Path $ColmapInstall "*") -DestinationPath $Archive -CompressionLevel Optimal
$Hash = (Get-FileHash $Archive -Algorithm SHA256).Hash.ToLowerInvariant()
Set-Content "${Archive}.sha256" "$Hash  $(Split-Path $Archive -Leaf)" -Encoding ascii
Write-Output "COLMAP CUDA BA archive: $Archive"
Write-Output "SHA-256: $Hash"
