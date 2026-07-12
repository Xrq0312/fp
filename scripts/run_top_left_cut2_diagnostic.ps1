$ErrorActionPreference = "Stop"

$root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$exe = Join-Path $root "build\Release\pointcloud_corner_compare.exe"
$input = Join-Path $root "data01\data01\top_left_cut2.ply"
$outRoot = Join-Path $root "results\top_left_cut2_diagnostic"
$runOut = Join-Path $outRoot "current"
$imageOut = Join-Path $outRoot "images"
$renderer = Join-Path $root "scripts\render_ply_views.py"

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe. Build the project first."
}
if (-not (Test-Path $input)) {
    throw "Input file not found: $input"
}

New-Item -ItemType Directory -Force -Path $runOut, $imageOut | Out-Null

python $renderer $input `
    --out-dir $imageOut `
    --prefix input_scaled_mm `
    --scale 1000 `
    --sample-limit 300000 `
    --size 1800

& $exe $input `
    --output-dir $runOut `
    --input-scale 1000 `
    --preserve-input-points `
    --point-spacing-resolution-estimator `
    --voxel-leaf 0.2 `
    --strict-plane-distance-scale 1.2 `
    --support-conflict-scale 1.0 `
    --core-conflict-scale 0.5 `
    --merge-plane-adjacency-scale 80 `
    --merge-plane-angle-deg 8 `
    --merge-plane-distance-scale 12 `
    --min-visible-support-core-ratio 0.25 `
    --min-visible-support-median-ratio 0.06 `
    --min-visible-density-median-ratio 0.03 `
    --min-corner-planes 3 `
    --max-y 30 `
    --expand-plane-distance-scale 1.5 `
    --expand-plane-margin-scale 120 `
    --atom-cell-scales 48,96,160 `
    --atom-min-points 40 `
    --atom-merge-angle-deg 8 `
    --atom-merge-distance-scale 8 `
    --max-planes 80 `
    --min-plane-extent-scale 20 `
    --min-plane-aspect-ratio 0.03
if ($LASTEXITCODE -ne 0) {
    throw "top_left_cut2 run failed with exit code $LASTEXITCODE"
}

$sample = Join-Path $runOut "top_left_cut2"
python $renderer (Join-Path $sample "plane_patches.ply") `
    --out-dir $imageOut `
    --prefix current_patches `
    --rgb `
    --sample-limit 300000 `
    --size 1800

python $renderer (Join-Path $sample "polyhedron_corners.ply") `
    --out-dir $imageOut `
    --prefix current_corners `
    --rgb `
    --point-radius 5 `
    --sample-limit 100000 `
    --size 1800

Write-Host ""
Write-Host "Diagnostic outputs:"
Write-Host "  Images: $imageOut"
Write-Host "  PLY and plane report: $sample"
