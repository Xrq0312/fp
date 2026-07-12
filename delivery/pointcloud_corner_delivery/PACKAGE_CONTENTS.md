# Package Contents

This package is prepared for customer delivery.

## Included

- Source code:
  - `src/main.cpp`
  - `CMakeLists.txt`
- Runtime:
  - `bin/pointcloud_corner_compare.exe`
  - required runtime DLLs from the Release build directory
- Launch scripts:
  - `run_recommended.ps1`
  - `run_top_left.ps1`
  - `run_top_down.ps1`
- Documentation:
  - `docs/technical_report.docx`
  - `docs/technical_report_source.md`
  - `README_DELIVERY.md`
- Best verified results:
  - `best_results/top_left_cut/plane_patches.ply`
  - `best_results/top_left_cut/polyhedron_corners.ply`
  - `best_results/top_down/plane_patches.ply`
  - `best_results/top_down/polyhedron_corners.ply`

## Not Included

- Raw input point clouds. The source data is large and should be provided separately.
- Historical debug outputs, JSON summaries, bitmap views, intermediate experiments, and old result folders.
- Build directory internals other than the runtime executable and DLLs.

## Verified Result Counts

| Dataset | Plane Patches | Corners |
| --- | ---: | ---: |
| `top_left_cut.ply` | 26 | 30 |
| `top_down.ply` | 17 | 8 |
