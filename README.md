# Point Cloud Polyhedron Corner Detection

This project detects planar patches and candidate polyhedron corners from large DLP point clouds.

The reusable entry point is `CornerFeatureDetector`. It exposes `detect(cloud)` for in-memory PCL clouds
and `detectFile(path)` for PLY files; the command-line `main` only parses parameters, calls the detector,
and saves the final PLY outputs.

## Current Pipeline

1. Load PLY as XYZ.
2. Optional crop, for example `--max-y 30` for `top_left_cut.ply`.
3. Build seed plane candidates with one of two modes:
   - default scanned-cloud mode: multi-scale local plane atoms from resolution-scaled 3D cells;
   - CAD/model-cloud mode: PCL normal estimation plus normal/curvature region growing.
4. Merge adjacent compatible seed candidates when they are coplanar.
5. Fit planes with `rspd-lite` robust residual trimming.
6. Expand each plane independently on the full-resolution cloud.
7. Fit each expanded plane strictly with robust residual trimming.
8. Merge coplanar full patches again, but only when projected patch bounds are adjacent.
9. Reject narrow strip-like patches near edges.
10. Trim loose support points that are closer to another non-parallel plane.
11. Trim/refit strict core points that are ambiguous between non-parallel planes.
12. Reject weak visible patches that have little independent support or very low adaptive density.
13. Compute corners from intersections of valid planes and boundary patch corners.

The default resolution estimator keeps the historical calibrated scale used by the
recommended scripts. Use `--point-spacing-resolution-estimator` when you want thresholds
based on true nearest-neighbor point spacing instead.

## Recommended Runs

Use:

```powershell
.\scripts\run_recommended.ps1
```

Outputs are written to:

```text
results\top_left_y30
results\top_down
```

Current recommended counts:

| Dataset | Plane Patches | Corners |
| --- | ---: | ---: |
| top_left_cut.ply | 26 | 19 |
| top_down.ply | 13 | 8 |

For the CAD-exported `top_left_cut2.ply` model cloud, use:

```powershell
.\scripts\run_top_left_cut2_diagnostic.ps1
```

That script reads meter coordinates with `--input-scale 1000`, runs the detector, and exports
diagnostic PNG projections under `results\top_left_cut2_diagnostic\images`.
The model-exported cloud uses true point-spacing scale, `--voxel-leaf 0.2`, `--preserve-input-points`,
and `--region-growing-plane-mode`. Detection runs on the downsampled cloud, but the final plane PLY is
refilled from the scaled full input cloud.

Current `top_left_cut2.ply` diagnostic count:

| Dataset | Plane Patches | Corners | Output Plane Points |
| --- | ---: | ---: | ---: |
| top_left_cut2.ply | 54 | 63 | 9,696,956 / 10,000,014 |

## Build

```powershell
& 'D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' -S . -B build -DCMAKE_TOOLCHAIN_FILE='D:/vcpkg/vcpkg/scripts/buildsystems/vcpkg.cmake' -DVCPKG_TARGET_TRIPLET=x64-windows
& 'D:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe' --build build --config Release
```

## Important Parameters

- `--atom-cell-scales`: cell sizes, expressed as multiples of model resolution.
- `--load-limit`: random cap for large experimental runs.
- `--voxel-leaf` / `--voxel-leaf-scale`: optional voxel downsampling by absolute size or resolution multiple.
- `--sor-mean-k` / `--sor-stddev`: optional statistical outlier removal.
- `--ror-radius-scale` / `--ror-min-neighbors`: optional radius outlier removal.
- `--resolution-sample-cap`: number of points sampled when estimating scale.
- `--point-spacing-resolution-estimator`: use true sampled nearest-neighbor spacing instead of calibrated sparse-sample scale.
- `--input-scale`: multiplier applied to loaded XYZ coordinates, for example `1000` for meter-to-millimeter model exports.
- `--preserve-input-points`: use filtered/downsampled points only for detection, then refill output patches from the scaled/cropped input cloud.
- `--region-growing-plane-mode`: use PCL normal-based Region Growing to produce plane seeds; this is recommended for clean CAD/model point clouds.
- `--normal-k`: KNN count for normal estimation in Region Growing mode.
- `--region-neighbors`: neighbor count used while growing smooth normal-consistent regions.
- `--region-smoothness-deg`: maximum normal-angle difference for Region Growing.
- `--region-curvature-threshold`: curvature threshold for accepting points into a smooth region.
- `--region-min-cluster-size`: minimum Region Growing cluster size before plane fitting.
- `--normal-family-filter`: rejects small isolated normal families after full-cloud refill; useful for removing edge-mixed stripe patches on clean model clouds.
- `--normal-family-angle-deg`: angular tolerance used to group parallel normal families.
- `--min-normal-family-support-ratio`: support ratio below which a small normal family can be rejected.
- `--min-normal-family-members`: minimum same-family patch count before a weak normal family is trusted.
- `--robust-fit-keep-ratio`: fraction of low-residual points used when refitting a plane in `rspd-lite`.
- `--strict-plane-distance-scale`: strict post-fit inlier distance used after plane expansion.
- `--support-conflict-scale`: trims visible support points that sit on ambiguous/non-parallel neighboring planes.
- `--core-conflict-scale`: removes edge/competition points from strict plane cores and refits.
- `--merge-plane-adjacency-scale`: requires projected patch bounds to touch before coplanar merging.
- `--min-visible-support-core-ratio`: rejects patches whose visible support is too small compared with their strict core.
- `--min-visible-support-median-ratio`: scene-adaptive lower bound for visible support, expressed as a fraction of the median patch support.
- `--min-visible-density-median-ratio`: adaptive density false-positive rejection.
- `--min-corner-planes`: rejects merged corner candidates that are supported by too few plane hypotheses.
- `--corner-edge-radius-scale`: local covariance radius used to score whether a corner sits near an edge/corner-like geometry change.
- `--corner-edge-multi-scales`: three local covariance scales used for multi-scale corner filtering.
- `--max-corner-edge-score`: optional rejection for overly strong local scatter/occlusion-boundary corners; `0` disables it.
- `--max-corner-edge-growth-ratio`: rejects candidates whose large-scale covariance score grows too much relative to the mid-scale score.
- `--expand-plane-distance-scale`: full-cloud point-to-plane distance threshold.
- `--min-plane-extent-scale`: rejects patches whose short projected extent is too small.
- `--min-plane-aspect-ratio`: rejects thin strip-like patches.

Each run also writes `plane_report.csv` beside the output PLY files. It lists every plane patch's normal,
centroid, support size, and projected extents for debugging.

## Reference Ideas

- PCL Region Growing Segmentation: normal-angle and curvature constrained connected growth; this is the basis of `--region-growing-plane-mode`.
- CGAL Shape Detection and Region Growing: model-fitting and region-growing plane extraction for point sets.
- Schnabel, Wahl, Klein, "Efficient RANSAC for Point-Cloud Shape Detection", 2007: useful baseline for primitive detection, but less stable here than connected normal growing because the model has many adjacent planar faces and edge residuals.
- Open3D planar patch detection: another open-source reference for normal-driven planar patch extraction and robust patch validation.
- `xiaohulugo/PointCloudSegmentation`: PLinkage-style local linkage and adjacency-aware patch merging.
- `abner-math/PlaneDetection`: recursive planar patch detection, neighbor-limited growth, merge-update cycles, and false-positive rejection.
- `denabazazian/Edge_Extraction`: local covariance eigenvalue contrast for edge-like/high-curvature points; this project scores only final corner candidates so edge evidence can be inspected, or used as a low/high threshold to reject false hull and occlusion-boundary corners without a full-cloud KNN pass.
