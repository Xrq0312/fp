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
3. Estimate surfel normals with PCL k-nearest normal estimation, then robustly
   cluster low-curvature surfels into global direction families. Local normals
   remain unchanged; the families are merge evidence rather than replacements.
4. Detect sharp normal discontinuities as line features. These edge surfels are allowed to belong to a plane, but they do not act as seeds that grow through an edge.
5. Sort candidate seeds by non-edge status, low curvature, and support count.
6. Fit a local seed plane from each seed neighborhood.
7. Grow the plane by BFS over neighboring surfels, accepting only surfels that match the seed-fitted plane by normal angle and residual distance.
8. Refit and merge coplanar adjacent patches. Repeated fragments agreeing with
   one stable global direction may merge using that direction's centroid offset;
   an isolated relaxed merge is rejected. Then run one synchronous
   boundary-competition pass. Only labels present in the original 26-neighborhood
   may compete, preventing an inclined plane from propagating across a large face.
9. Refit the corrected regions and merge once more.
10. Assign original dense points to accepted planes. Distance remains the
    default, but ambiguous points consult nearby 3-D surfel ownership and the
    local normal. This distinguishes real local support from a plane's rectangular
    projected envelope. Small patches are removed only after all safely
    transferable points have been moved to established planes.
11. Generate diagnostic edge/corner candidates, then extract stricter corner points from three-plane intersections validated by real point support and per-plane local support.

## Public Parameters

The executable intentionally exposes only a small public interface:

```powershell
edge_aware_plane_extract input.ply --profile model-mm --output-dir results
```

Use one of these profiles:

```text
model-mm       model-exported PLY already in millimeters
model-m        model-exported PLY in meters; coordinates are scaled by 1000
top-down       tuned profile for data02/data02/top_down.ply
top-left-cut2  tuned profile for data01/data01/top_left_cut2.ply
```

The plane-growth thresholds, merge thresholds, dense assignment distance, corner
support thresholds, and dense assignment margin are internal values. The
assignment margin is inferred from the point cloud after plane growth: it starts
with a tight 2 mm envelope and only widens when many dense points are close to a
plane but fall just outside that plane's grown support envelope.

## Recommended Command

Use the script:

```powershell
.\scripts\run_top_left_cut2_seed_grow.ps1
```

For `top_down.ply`, use:

```powershell
.\scripts\run_top_down_seed_grow.ps1
```

The `top_down` script selects `--profile top-down`; this keeps millimeter-scale
input. `top_left_cut2.ply` uses `--profile top-left-cut2`, which applies
meter-to-mm scaling. Both profiles use the same automatic assignment-margin
inference.

Current extraction output:

```text
results_seed_grow\top_left_cut2\edge_aware_plane_patches.ply
results_seed_grow\top_left_cut2\feature_edges.ply
results_seed_grow\top_left_cut2\extracted_corners.ply
```

Run the helper scripts with `-RenderImages` when projection images are needed.

Observed on `top_left_cut2.ply`:

```text
Input points: 10000014
Surfels: 27545
Plane patches: 54
Output points: 9889388
Extracted corners: 75
```

For `top_left_cut2.ply`, the connector's four long planes are represented by the four largest grown patches rather than stripe-like fragments:

```text
plane 0: normal ~= (-0.980, 0, 0.198), 665729 points
plane 1: normal ~= (-0.980, 0, 0.198), 668153 points
plane 2: normal ~= (0, 1, 0), 562251 points
plane 3: normal ~= (0, 1, 0), 544151 points
```

The earlier v13 result used visibility and UV local-support filtering. It looked cleaner in some projections, but it only kept 5,988,552 / 10,000,014 points and therefore lost too many faces. The later v15 result recovered points but still relied too much on wide point assignment. The current seed-grown result moves the segmentation decision back to fitted local planes and growth constraints.
