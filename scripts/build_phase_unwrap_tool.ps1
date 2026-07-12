$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$VcVars = "D:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
$VcpkgInstalled = "D:\vcpkg\vcpkg\installed\x64-windows"
$OutDir = Join-Path $Root "build\phase_unwrap_tool"
$Src = Join-Path $Root "src\phase_unwrap_tool.cpp"

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$Cmd = @"
call "$VcVars"
cl /nologo /std:c++17 /O2 /EHsc /W4 /D_CRT_SECURE_NO_WARNINGS ^
  /I"$VcpkgInstalled\include" /I"$VcpkgInstalled\include\opencv4" ^
  "$Src" ^
  /Fe:"$OutDir\phase_unwrap_tool.exe" ^
  /link /LIBPATH:"$VcpkgInstalled\lib" opencv_core4.lib opencv_imgcodecs4.lib opencv_imgproc4.lib
"@

$BuildCmd = Join-Path $OutDir "build_phase_unwrap_tool.cmd"
Set-Content -LiteralPath $BuildCmd -Value $Cmd -Encoding ASCII
cmd /c "`"$BuildCmd`""
if ($LASTEXITCODE -ne 0) {
    throw "cl failed with exit code $LASTEXITCODE"
}

Write-Host "Built: $OutDir\phase_unwrap_tool.exe"
Write-Host "Before running, include OpenCV DLLs in PATH:"
Write-Host "  `$env:PATH = `"$VcpkgInstalled\bin;`$env:PATH`""
