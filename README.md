# Point Cloud Plane, Edge, and Corner Extraction

This repository's current point-cloud extraction path is `edge_aware_plane_extract`.
It targets clean model-exported PLY point clouds and produces only the core
geometry outputs:

- `edge_aware_plane_patches.ply`: dense points assigned to extracted planes
- `feature_edges.ply`: surfel centers on plane boundaries
- `extracted_corners.ply`: validated three-plane corner points
- `topology_corrections.ply/.csv`: points whose joint label was conservatively
  changed by `--pipeline topology`, including source/target candidate ids

After seed growth and coplanar merging, directly adjacent surfels perform one
synchronous plane-competition pass. Distance, local normal agreement, envelope
support, and connectivity all participate; labels do not propagate beyond the
original boundary during this pass.

Reliable surfel normals are also clustered into global direction families.
These families are used only to repair repeated coplanar fragmentation during
merge; an isolated relaxed merge is rejected. Dense assignment uses nearby
3-D surfel ownership as real support, so a plane cannot win merely because its
2-D bounding rectangle extends through another surface. Edge-like cells may
still use their normal when the competing directions are clearly separated.

## Build

The checked-in preset pins dependency discovery to the requested vcpkg tree:

```powershell
cmake --preset vcpkg-release
cmake --build --preset vcpkg-release
```

The equivalent explicit commands are:

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE='C:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake' -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release --target edge_aware_plane_extract
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

Use `--pipeline baseline` for the preserved original algorithm,
`--pipeline multiscale` for global/local proposals, or `--pipeline joint` to
add iterative multi-model assignment, refitting, outlier handling, and label
cost pruning. Use `--pipeline topology` to add conservative observed-boundary
constraints that correct dense points spilling across an adjacent plane while
leaving the joint decision unchanged elsewhere. Run the deterministic synthetic
comparison with:

```powershell
.\scripts\run_ablation.ps1 -SkipBuild
```

Add `-IncludeLargeDatasets` to include all four supplied real point clouds
(`top_left_cut`, `top_left_cut2`, `top_down`, and `top_down2`). Use
`-DatasetNames top_left_cut,top_down2` to select a subset. Design
and measured ablations are recorded in `docs/multiplane_corner_optimization_plan.md`
and `docs/ablation_results.md`.

To inspect a questionable location without changing extraction parameters:

```powershell
.\build\Release\edge_aware_plane_extract.exe input.ply --profile top-down --output-dir results --probe 67.5148 -27.3549 -10.9934
```

The probe reports the local surfel owner and normal, nearby plane equations,
point-to-plane distances, envelope decisions, and the final dense assignment.

To write a read-only per-plane check of dense assignments:

```powershell
.\build\Release\edge_aware_plane_extract.exe input.ply --profile model-mm --output-dir results --audit-assignments
```

This adds `assignment_audit.csv`. Use `--distance-only` with the audit for an
A/B control run that disables normal-guided dense assignment without changing
plane extraction. The crossing-plane investigation is recorded in
`docs/dense_assignment_crossing_audit.md`.

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
top_left_cut:
  Plane patches: 33
  Feature edge points: 891
  Extracted corners: 23
  Output plane points: 12621846

top_down:
  Plane patches: 15
  Feature edge points: 517
  Extracted corners: 16
  Output plane points: 5702636

top_left_cut2:
  Plane patches: 54
  Feature edge points: 179
  Extracted corners: 75
  Output plane points: 9889388

top_down2:
  Plane patches: 12
  Feature edge points: 880
  Extracted corners: 16
  Output plane points: 10000006
```

The full four-dataset regression is in
`results_logic_fix_final_20260921` (ignored by Git).
