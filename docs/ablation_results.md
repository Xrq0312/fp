# Plane and corner ablation results

All timings below are Release builds on the same Windows CPU. Dependencies were
resolved from `C:\vcpkg\vcpkg`. Real datasets have no checked-in instance ground
truth, so their rows are regression measurements rather than accuracy claims.

## Stage 0: preserved baseline

| Dataset | Planes | Corners | Point coverage | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| stepped_box | 6 | 8 | 97.41% | 1.23 |
| top_down | 15 | 16 | 99.98% | 36.25 |
| top_left_cut2 | 54 | 75 | 98.89% | 81.20 |

Synthetic quality: plane precision 1.0, plane recall 0.5455, macro IoU 0.5454,
corner precision 1.0, and corner recall 0.5. The baseline correctly recovered
the six large base-box faces but rejected all five faces of the small boss.

## Stage 1: multi-scale global/local candidates

Only the proposal/retention stage changed. A second, smaller normal support is
fused with the original support; global and local proposal passes are generated
independently and duplicate surfels are removed when coincident patches merge.
A boundary must persist at both normal scales before it can suppress a local
seed. Validated small patches use a lower dense-support floor.

| Dataset | Planes | Corners | Point coverage | Time (s) | Delta time |
| --- | ---: | ---: | ---: | ---: | ---: |
| stepped_box | 11 | 16 | 100.00% | 1.37 | +11% |
| top_down | 19 | 16 | 99.99% | 36.83 | +2% |
| top_left_cut2 | 54 | 75 | 98.89% | 80.70 | -1% |

Synthetic quality reached 1.0 plane precision, recall, macro IoU, point
coverage, corner precision, and corner recall. Mean matched-corner error was
0.00032 mm. On `top_down`, the four additionally retained faces are small or
narrow (the two smallest have 510/519 dense points and 0.026/0.009 mm surfel
fit RMS); they need the later joint-model stage and visual/ground-truth review
before being called true positives. `top_left_cut2` retained the exact baseline
plane/corner counts and two additional dense points.
