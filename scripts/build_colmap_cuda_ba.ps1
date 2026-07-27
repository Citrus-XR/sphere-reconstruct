param(
    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $true
$ColmapTag = "4.1.1"
$CeresCommit = "bac1127f9ef672405bd0d2d9c84e809ae89bd239"
$VcpkgCommit = "6d9d7df564a1ccdaa994e4ad39ccd4a32360867b"
$CudssPackage = "nvidia-cudss-cu13==0.8.0.10"
$BuildRoot = Join-Path $env:RUNNER_TEMP "sphere-colmap-cuda-ba"
$Triplet = "x64-windows-release"

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

git clone --depth 1 --branch $ColmapTag https://github.com/colmap/colmap.git $ColmapSource
$RigPairPatch = Join-Path $PSScriptRoot "patches/colmap-4.1.1-sequential-rig-pairs.patch"
# 4.1.1 の folder-major rig pairing bug に upstream の最小 semantic fix だけを backport する。
# https://github.com/colmap/colmap/pull/4591
git -C $ColmapSource apply --check $RigPairPatch
git -C $ColmapSource apply $RigPairPatch
git clone https://github.com/ceres-solver/ceres-solver.git $CeresSource
git -C $CeresSource checkout $CeresCommit
git -C $CeresSource submodule update --init --depth 1 third_party/abseil-cpp
git clone https://github.com/microsoft/vcpkg.git $VcpkgRoot
git -C $VcpkgRoot checkout $VcpkgCommit
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
python -m pip download --no-deps $CudssPackage --dest $WheelDirectory
$Wheel = Get-ChildItem $WheelDirectory -Filter "*.whl" | Select-Object -First 1
if ($null -eq $Wheel) {
    throw "cuDSS wheel を取得できません"
}
$WheelZip = Join-Path $WheelDirectory "cudss.zip"
Copy-Item $Wheel.FullName $WheelZip
Expand-Archive $WheelZip -DestinationPath $CudssRoot

$CudssPackageRoot = Join-Path $CudssRoot "nvidia/cu13"
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
$CudaArchitectures = "75;80;86;89;100;120"
cmake -S $CeresSource -B $CeresBuild -GNinja `
    "-DCMAKE_BUILD_TYPE=Release" `
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
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
cmake --build $CeresBuild --target install

cmake -S $ColmapSource -B $ColmapBuild -GNinja `
    "-DCMAKE_BUILD_TYPE=Release" `
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
    "-DCMAKE_CUDA_ARCHITECTURES=$CudaArchitectures"
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
