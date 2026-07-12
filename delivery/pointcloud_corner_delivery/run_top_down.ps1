$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $root "bin\pointcloud_corner_compare.exe"
$out = Join-Path $root ("run_results\top_down_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$input = Join-Path $root "data02\data02\top_down.ply"
if (-not (Test-Path $input)) { $input = Join-Path $root "..\..\data02\data02\top_down.ply" }
if (-not (Test-Path $input)) { throw "Input file not found: data02\data02\top_down.ply" }

& $exe $input `
    --output-dir $out `
    --strict-plane-distance-scale 0.125 `
    --support-conflict-scale 0.5 `
    --core-conflict-scale 0.25 `
    --merge-plane-adjacency-scale 20 `
    --min-visible-support-core-ratio 0.5 `
    --min-visible-support-median-ratio 0.10 `
    --min-visible-density-median-ratio 0.05 `
    --merge-plane-angle-deg 6 `
    --merge-plane-distance-scale 6 `
    --atom-cell-scales 8,16,32 `
    --atom-merge-angle-deg 5 `
    --atom-merge-distance-scale 1.5 `
    --min-corner-planes 3 `
    --corner-edge-radius-scale 7 `
    --corner-edge-multi-scales 2.5,4.5,7 `
    --max-corner-edge-score 0.05 `
    --max-corner-edge-growth-ratio 1.5 `
    --expand-plane-distance-scale 0.25 `
    --min-x -25 `
    --max-x 65 `
    --min-y -30 `
    --max-y 50
if ($LASTEXITCODE -ne 0) { throw "top_down run failed with exit code $LASTEXITCODE" }
