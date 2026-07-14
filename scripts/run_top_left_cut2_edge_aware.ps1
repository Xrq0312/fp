param(
    [string]$InputPly = ".\data01\data01\top_left_cut2.ply",
    [string]$OutputDir = ".\results_seed_grow"
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
    "--profile", "top-left-cut2"
)

& $exe @args

$stem = [IO.Path]::GetFileNameWithoutExtension($InputPly)
$ply = Join-Path (Join-Path $OutputDir $stem) "edge_aware_plane_patches.ply"
$imageDir = Join-Path (Join-Path $OutputDir $stem) "images"
python .\scripts\render_ply_views.py $ply --out-dir $imageDir --prefix seed_grow --rgb --sample-limit 300000 --size 1800
python .\scripts\render_ply_views.py (Join-Path (Join-Path $OutputDir $stem) "feature_edges.ply") --out-dir $imageDir --prefix seed_grow_edges --rgb --point-radius 3 --sample-limit 200000 --size 1800
python .\scripts\render_ply_views.py (Join-Path (Join-Path $OutputDir $stem) "extracted_corners.ply") --out-dir $imageDir --prefix seed_grow_corners --rgb --point-radius 6 --sample-limit 200000 --size 1800
