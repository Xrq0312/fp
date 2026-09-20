# Point Cloud Plane, Edge, and Corner Extraction

This repository's current point-cloud extraction path is `edge_aware_plane_extract`.
It targets clean model-exported PLY point clouds and produces only the core
geometry outputs:

- `edge_aware_plane_patches.ply`: dense points assigned to extracted planes
- `feature_edges.ply`: surfel centers on plane boundaries
- `extracted_corners.ply`: validated three-plane corner points

After seed growth and coplanar merging, directly adjacent surfels perform one
synchronous plane-competition pass. Distance, local normal agreement, envelope
support, and connectivity all participate; labels do not propagate beyond the
original boundary during this pass.

## Build

```powershell
& 'D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_TOOLCHAIN_FILE='D:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake' -DVCPKG_TARGET_TRIPLET=x64-windows
& 'D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release --target edge_aware_plane_extract
```

## Run

For `top_left_cut2.ply`, whose coordinates are in meters:

```powershell
.\scripts\run_top_left_cut2_seed_grow.ps1
```

For `top_down.ply`, whose coordinates are already in millimeters:

```powershell
.\scripts\run_top_down_seed_grow.ps1
```

Add `-RenderImages` to either script when projection images are needed.

## Direct CLI

```powershell
.\build\Release\edge_aware_plane_extract.exe input.ply --profile model-mm --output-dir results
```

To inspect a questionable location without changing extraction parameters:

```powershell
.\build\Release\edge_aware_plane_extract.exe input.ply --profile top-down --output-dir results --probe 67.5148 -27.3549 -10.9934
```

The probe reports the local surfel owner and normal, nearby plane equations,
point-to-plane distances, envelope decisions, and the final dense assignment.

Profiles:

```text
model-mm       input coordinates are already millimeters
model-m        input coordinates are meters and are scaled by 1000
top-down       profile for data02/data02/top_down.ply
top-left-cut2  profile for data01/data01/top_left_cut2.ply
```

The public interface is intentionally small. Plane-growth thresholds, merge
thresholds, dense assignment margin, and corner support checks are inferred or
kept internal so the tool does not require dataset-specific manual tuning.

## Current Checks

Recent validated results:

```text
top_down:
  Plane patches: 15
  Feature edge points: 517
  Extracted corners: 16
  Output plane points: 5702072

top_left_cut2:
  Plane patches: 54
  Feature edge points: 179
  Extracted corners: 75
  Output plane points: 9888767
```

The full four-dataset regression is in
`results_rerun_competition_20260920` (ignored by Git).
