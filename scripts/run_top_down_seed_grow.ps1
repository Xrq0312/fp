param(
    [string]$InputPly = ".\data02\data02\top_down.ply",
    [string]$OutputDir = ".\results_seed_grow_top_down",
    [switch]$RenderImages
)

$ErrorActionPreference = "Stop"

$exe = ".\build\Release\edge_aware_plane_extract.exe"
if (-not (Test-Path $exe)) {
    $cmake = "D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
    if (-not (Test-Path $cmake)) {
        throw "edge_aware_plane_extract.exe is missing and CMake was not found at $cmake"
    }
    & $cmake --build build --config Release --target edge_aware_plane_extract
}

$args = @(
    $InputPly,
    "--output-dir", $OutputDir,
    "--profile", "top-down"
)

& $exe @args

if (-not $RenderImages) {
    return
}

$stem = [IO.Path]::GetFileNameWithoutExtension($InputPly)
$sampleDir = Join-Path $OutputDir $stem
$imageDir = Join-Path $sampleDir "images"
python .\scripts\render_ply_views.py (Join-Path $sampleDir "edge_aware_plane_patches.ply") --out-dir $imageDir --prefix seed_grow --rgb --sample-limit 300000 --size 1800
python .\scripts\render_ply_views.py (Join-Path $sampleDir "feature_edges.ply") --out-dir $imageDir --prefix seed_grow_edges --rgb --point-radius 3 --sample-limit 200000 --size 1800
python .\scripts\render_ply_views.py (Join-Path $sampleDir "extracted_corners.ply") --out-dir $imageDir --prefix seed_grow_corners --rgb --point-radius 6 --sample-limit 200000 --size 1800
