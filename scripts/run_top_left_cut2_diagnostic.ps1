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
    --strict-plane-distance-scale 0.125 `
    --support-conflict-scale 0.5 `
    --core-conflict-scale 0.25 `
    --merge-plane-adjacency-scale 8 `
    --min-visible-support-core-ratio 0.5 `
    --min-visible-support-median-ratio 0.10 `
    --min-visible-density-median-ratio 0.05 `
    --min-corner-planes 3 `
    --max-y 30 `
    --expand-plane-distance-scale 0.25
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
