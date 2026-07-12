$ErrorActionPreference = "Stop"

$exe = Join-Path $PSScriptRoot "..\build\Release\pointcloud_corner_compare.exe"

if (-not (Test-Path $exe)) {
    throw "Executable not found: $exe. Build the project first."
}

& $exe `
    data01\data01\top_left_cut.ply `
    --output-dir results\top_left_y30 `
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

& $exe `
    data02\data02\top_down.ply `
    --output-dir results\top_down `
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
    --sor-mean-k 16 `
    --sor-stddev 1.25 `
    --min-x -25 `
    --max-x 65 `
    --min-y -30 `
    --max-y 50
