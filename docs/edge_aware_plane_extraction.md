# Edge-Aware Plane Extraction

This is a fresh implementation for clean model-exported PLY point clouds. The main target is to keep each real planar face continuous while preserving as many model faces as possible.

## References Checked

- PCL RegionGrowing: normal smoothness and residual based region growth for point cloud segmentation. https://pointclouds.org/documentation/tutorials/region_growing_segmentation.html
- PCL RegionGrowing API notes: implementation of the Rabbani, van den Heuvel, and Vosselman smoothness-constraint method. https://pointclouds.org/documentation/classpcl_1_1_region_growing.html
- CGAL Shape Detection: Efficient RANSAC and Region Growing for primitive/plane detection. https://doc.cgal.org/latest/Shape_detection/index.html
- RSPD / robust statistics plane detection: deterministic split-and-merge style plane detection for unorganized point clouds. https://github.com/plusk01/pointcloud-plane-segmentation
- Original RSPD project page. https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/RE.html

## Current Strategy

1. Load the model-exported PLY and scale meters to millimeters with `--input-scale 1000`.
2. Build a surfel grid instead of operating directly on all 10M points. This keeps the model dense enough while removing sampling-order artifacts.
3. Estimate surfel normals with PCL k-nearest normal estimation.
4. Grow connected planar regions over neighboring surfel cells using normal angle and point-to-plane residual checks.
5. Mark sharp normal discontinuities as edge-like so they do not seed stripe fragments inside a planar face.
6. Merge coplanar neighboring patches after growth.
7. Keep visibility and local-support filters off in the recommended high-recall mode; those filters removed too many model faces on `top_left_cut2.ply`.
8. Assign original dense points back to detected planes with a wider nearest-plane distance threshold. This keeps model-exported faces instead of throwing away internal/back-side patches.
9. Save each plane as an individual PLY for debugging when `--save-plane-files` is enabled.

## Recommended Command

Use the script:

```powershell
.\scripts\run_top_left_cut2_edge_aware.ps1
```

Current best diagnostic output:

```text
results_edge_aware_v15_high_recall_d2\top_left_cut2\edge_aware_plane_patches.ply
results_edge_aware_v15_high_recall_d2\top_left_cut2\planes\
results_edge_aware_v15_high_recall_d2\top_left_cut2\images\
```

Observed on `top_left_cut2.ply`:

```text
Input points: 10000014
Surfels: 27545
Plane patches: 27
Output points: 9265315
```

For `top_left_cut2.ply`, the four long connector planes are represented as four dominant plane patches rather than many stripe-like fragments:

```text
plane 2: normal ~= (-0.985, 0, 0.174), 976151 points
plane 3: normal ~= (-0.985, 0, 0.173), 860160 points
plane 7: normal ~= (0, 1, 0), 699528 points
plane 8: normal ~= (0, 1, 0), 702368 points
```

The earlier v13 result used visibility and UV local-support filtering. It looked cleaner in some projections, but it only kept 5,988,552 / 10,000,014 points and therefore lost too many faces for the current goal.
