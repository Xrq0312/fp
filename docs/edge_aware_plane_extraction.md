# Seed-Grown Plane Extraction

This implementation targets clean model-exported PLY point clouds. The current direction is seed-fitted plane extraction with feature-aware plane growth, not post-hoc point filtering.

## References Checked

- PCL RegionGrowing: normal smoothness and residual based region growth for point cloud segmentation. https://pointclouds.org/documentation/tutorials/region_growing_segmentation.html
- PCL RegionGrowing API notes: implementation of the Rabbani, van den Heuvel, and Vosselman smoothness-constraint method. https://pointclouds.org/documentation/classpcl_1_1_region_growing.html
- CGAL Shape Detection: Efficient RANSAC and Region Growing for primitive/plane detection. https://doc.cgal.org/latest/Shape_detection/index.html
- RSPD / robust statistics plane detection: deterministic split-and-merge style plane detection for unorganized point clouds. https://github.com/plusk01/pointcloud-plane-segmentation
- Original RSPD project page. https://www.inf.ufrgs.br/~oliveira/pubs_files/RE/RE.html

## Current Strategy

1. Load the model-exported PLY and scale meters to millimeters with `--input-scale 1000`.
2. Build a 0.5 mm surfel grid as the segmentation graph. The original dense points are preserved for final assignment.
3. Estimate surfel normals with PCL k-nearest normal estimation.
4. Detect sharp normal discontinuities as line features. These edge surfels are allowed to belong to a plane, but they do not act as seeds that grow through an edge.
5. Sort candidate seeds by non-edge status, low curvature, and support count.
6. Fit a local seed plane from each seed neighborhood.
7. Grow the plane by BFS over neighboring surfels, accepting only surfels that match the seed-fitted plane by normal angle and residual distance.
8. Refit and refine each grown region, then merge coplanar adjacent patches.
9. Assign original dense points to the nearest accepted plane. This is a final recovery step, not the main segmentation mechanism.
10. Export plane patches, per-plane PLY files, edge-feature points, and corner-feature candidates for diagnosis.

## Recommended Command

Use the script:

```powershell
.\scripts\run_top_left_cut2_seed_grow.ps1
```

Current best diagnostic output:

```text
results_seed_grow\top_left_cut2\edge_aware_plane_patches.ply
results_seed_grow\top_left_cut2\planes\
results_seed_grow\top_left_cut2\feature_edges.ply
results_seed_grow\top_left_cut2\feature_corners.ply
results_seed_grow\top_left_cut2\images\
```

Observed on `top_left_cut2.ply`:

```text
Input points: 10000014
Surfels: 27545
Plane patches: 54
Output points: 9888831
```

For `top_left_cut2.ply`, the connector's four long planes are represented by the four largest grown patches rather than stripe-like fragments:

```text
plane 0: normal ~= (-0.980, 0, 0.198), 665729 points
plane 1: normal ~= (-0.980, 0, 0.198), 668153 points
plane 2: normal ~= (0, 1, 0), 562251 points
plane 3: normal ~= (0, 1, 0), 544151 points
```

The earlier v13 result used visibility and UV local-support filtering. It looked cleaner in some projections, but it only kept 5,988,552 / 10,000,014 points and therefore lost too many faces. The later v15 result recovered points but still relied too much on wide point assignment. The current seed-grown result moves the segmentation decision back to fitted local planes and growth constraints.
