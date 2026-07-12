$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$exe = Join-Path $root "bin\pointcloud_corner_compare.exe"
$out = Join-Path $root ("run_results\top_left_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
$input = Join-Path $root "data01\data01\top_left_cut.ply"
if (-not (Test-Path $input)) { $input = Join-Path $root "..\..\data01\data01\top_left_cut.ply" }
if (-not (Test-Path $input)) { throw "Input file not found: data01\data01\top_left_cut.ply" }

& $exe $input `
    --output-dir $out `
    --strict-plane-distance-scale 0.125 `
    --support-conflict-scale 0.5 `
    --core-conflict-scale 0.25 `
    --merge-plane-adjacency-scale 8 `
    --min-visible-support-core-ratio 0.5 `
    --min-visible-support-median-ratio 0.10 `
    --min-visible-density-median-ratio 0.05 `
    --max-y 30 `
    --expand-plane-distance-scale 0.25
if ($LASTEXITCODE -ne 0) { throw "top_left run failed with exit code $LASTEXITCODE" }
