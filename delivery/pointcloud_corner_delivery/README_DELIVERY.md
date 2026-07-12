# Point Cloud Corner Detection Delivery

## Package Contents

- `bin/`: Windows Release executable and runtime DLLs.
- `src/main.cpp`: complete C++ implementation.
- `CMakeLists.txt`: build configuration.
- `run_recommended.ps1`: runs the recommended `top_left` and `top_down` configurations.
- `run_top_left.ps1`: runs only `top_left_cut.ply`.
- `run_top_down.ps1`: runs only `top_down.ply`.
- `best_results/`: current best PLY outputs.
- `docs/technical_report.docx`: technical report.
- `docs/technical_report_source.md`: report source text.

## Expected Input Data Layout

The delivery package does not include raw input point clouds because the source data is large.

To run directly, place input files under the package root:

```text
data01/data01/top_left_cut.ply
data02/data02/top_down.ply
```

Then run:

```powershell
.\run_recommended.ps1
```

The scripts also support the original development workspace layout for local verification.
Each run writes to a timestamped directory under `run_results/`. The checked, known-good outputs are kept separately under `best_results/`.

## Current Best Results

```text
best_results/top_left_cut/plane_patches.ply
best_results/top_left_cut/polyhedron_corners.ply
best_results/top_down/plane_patches.ply
best_results/top_down/polyhedron_corners.ply
```

Current verified counts:

| Dataset | Plane Patches | Corners |
| --- | ---: | ---: |
| top_left_cut.ply | 26 | 30 |
| top_down.ply | 17 | 8 |

## Build From Source

The source can be rebuilt with CMake and PCL. The original build used Visual Studio 2022 and vcpkg.

```powershell
cmake -S . -B build -DCMAKE_TOOLCHAIN_FILE=<path-to-vcpkg>/scripts/buildsystems/vcpkg.cmake -DVCPKG_TARGET_TRIPLET=x64-windows
cmake --build build --config Release
```

## Notes

- Output files are PLY point clouds.
- `plane_patches.ply` contains colored plane patches.
- `polyhedron_corners.ply` contains extracted corner points.
- Historical debug outputs, JSON summaries, bitmap views, and experimental result folders are intentionally excluded from this package.
