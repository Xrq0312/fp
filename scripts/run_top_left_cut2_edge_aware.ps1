param(
    [string]$InputPly = ".\data01\data01\top_left_cut2.ply",
    [string]$OutputDir = ".\results_edge_aware_v15_high_recall_d2"
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
    "--input-scale", "1000",
    "--surfel-size", "0.5",
    "--normal-k", "18",
    "--smooth-angle-deg", "8",
    "--edge-angle-deg", "20",
    "--coplanar-distance", "0.35",
    "--merge-angle-deg", "5",
    "--merge-distance", "0.5",
    "--merge-gap", "8.0",
    "--assign-distance", "2.0",
    "--assign-margin", "12.0",
    "--min-surfels", "20",
    "--min-output-points", "5000",
    "--min-extent", "3",
    "--min-aspect-ratio", "0.02",
    "--max-planes", "120",
    "--save-plane-files"
)

& $exe @args

$stem = [IO.Path]::GetFileNameWithoutExtension($InputPly)
$ply = Join-Path (Join-Path $OutputDir $stem) "edge_aware_plane_patches.ply"
$imageDir = Join-Path (Join-Path $OutputDir $stem) "images"
python .\scripts\render_ply_views.py $ply --out-dir $imageDir --prefix edge_aware_plane_support --rgb --sample-limit 300000 --size 1800
