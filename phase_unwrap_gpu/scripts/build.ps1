$ErrorActionPreference = "Stop"

$Project = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$CMake = "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
$Vcpkg = "D:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake"
$Build = Join-Path $Project "build"

& $CMake -S $Project -B $Build -G "Visual Studio 17 2022" -A x64 `
  -DCMAKE_TOOLCHAIN_FILE=$Vcpkg -DVCPKG_TARGET_TRIPLET=x64-windows
if ($LASTEXITCODE -ne 0) { throw "CMake configure failed: $LASTEXITCODE" }

& $CMake --build $Build --config Release --target phase_unwrap_gpu -j
if ($LASTEXITCODE -ne 0) { throw "Build failed: $LASTEXITCODE" }

Write-Host "Built: $Build\Release\phase_unwrap_gpu.exe"

