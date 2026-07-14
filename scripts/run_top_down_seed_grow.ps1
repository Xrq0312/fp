param(
    [string]$InputPly = ".\data02\data02\top_down.ply",
    [string]$OutputDir = ".\results_seed_grow_top_down"
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
    "--input-scale", "1",
    "--surfel-size", "0.5",
    "--normal-k", "18",
    "--smooth-angle-deg", "8",
    "--edge-angle-deg", "20",
    "--coplanar-distance", "0.35",
    "--grow-refine-iterations", "2",
    "--merge-angle-deg", "5",
    "--merge-distance", "0.5",
    "--merge-gap", "8.0",
    "--assign-distance", "2.0",
    "--assign-margin", "2.0",
    "--min-surfels", "20",
    "--min-output-points", "5000",
    "--min-extent", "3",
    "--min-aspect-ratio", "0.02",
    "--max-planes", "160",
    "--corner-support-radius-scale", "7.0",
    "--intersection-merge-scale", "4.5",
    "--min-corner-support", "8",
    "--min-plane-local-support", "3",
    "--min-corner-planes", "3"
)

& $exe @args

$stem = [IO.Path]::GetFileNameWithoutExtension($InputPly)
$sampleDir = Join-Path $OutputDir $stem
$imageDir = Join-Path $sampleDir "images"
python .\scripts\render_ply_views.py (Join-Path $sampleDir "edge_aware_plane_patches.ply") --out-dir $imageDir --prefix seed_grow --rgb --sample-limit 300000 --size 1800
python .\scripts\render_ply_views.py (Join-Path $sampleDir "feature_edges.ply") --out-dir $imageDir --prefix seed_grow_edges --rgb --point-radius 3 --sample-limit 200000 --size 1800
python .\scripts\render_ply_views.py (Join-Path $sampleDir "extracted_corners.ply") --out-dir $imageDir --prefix seed_grow_corners --rgb --point-radius 6 --sample-limit 200000 --size 1800
