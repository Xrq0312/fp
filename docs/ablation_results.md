# Plane and corner ablation results

All timings below are Release builds on the same Windows CPU. Dependencies were
resolved from `C:\vcpkg\vcpkg`. Real datasets have no checked-in instance ground
truth, so their rows are regression measurements rather than accuracy claims.

## Stage 0: preserved baseline

| Dataset | Planes | Corners | Point coverage | Time (s) |
| --- | ---: | ---: | ---: | ---: |
| stepped_box | 6 | 8 | 97.41% | 1.23 |
| top_left_cut | 33 | 23 | 99.85% | 39.60 |
| top_left_cut2 | 54 | 75 | 98.89% | 81.20 |
| top_down | 15 | 16 | 99.98% | 36.25 |
| top_down2 | 12 | 16 | 100.00% | 72.24 |

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
| top_left_cut | 59 | 21 | 99.92% | 40.62 | +3% |
| top_left_cut2 | 54 | 75 | 98.89% | 80.70 | -1% |
| top_down | 19 | 16 | 99.99% | 36.83 | +2% |
| top_down2 | 20 | 32 | 100.00% | 72.34 | <1% |

Synthetic quality reached 1.0 plane precision, recall, macro IoU, point
coverage, corner precision, and corner recall. Mean matched-corner error was
0.00032 mm. On `top_down`, the four additionally retained faces are small or
narrow (the two smallest have 510/519 dense points and 0.026/0.009 mm surfel
fit RMS); they need the later joint-model stage and visual/ground-truth review
before being called true positives. `top_left_cut2` retained the exact baseline
plane/corner counts and two additional dense points. The initially omitted
regressions materially change the conclusion: `top_left_cut` grows from 33 to
59 patches, and `top_down2` from 12 to 20. Some are low-RMS narrow faces, but
the `top_left_cut` increase is an over-segmentation risk and prevents enabling
this stage by default without stronger joint validation.

## Stage 2: joint multi-model assignment and refit

This stage replaces the one-pass, neighbor-label-only competition with bounded
coordinate descent over all geometrically feasible hypotheses. The energy has
point-to-plane/normal data cost, local label coherence, an outlier label, and a
model cost that removes a small label only when another model explains at least
90% of its support with negligible loss. Plane parameters are refitted after
each synchronous labeling round.

| Dataset | Planes | Corners | Point coverage | Time (s) | Joint iterations |
| --- | ---: | ---: | ---: | ---: | ---: |
| stepped_box | 11 | 16 | 100.00% | 1.39 | 3 |
| top_left_cut | 49 | 19 | 99.94% | 44.89 | 5 |
| top_left_cut2 | 54 | 62 | 98.89% | 67.66 | 5 |
| top_down | 20 | 16 | 99.99% | 36.72 | 5 |
| top_down2 | 20 | 32 | 100.00% | 72.71 | 3 |

The synthetic quality metrics remain perfect; 1,549 surfel label changes fill
previously unowned boundary support without deleting a true small face. The
real-data runs pruned no model under the conservative label-cost rule. That
rule is too conservative on `top_left_cut`: joint assignment reduces the
multi-scale result from 59 to 49 patches but remains well above the 33-patch
validated baseline. `top_down2` remains at 20 patches and 32 corners; without
ground truth, those eight additional low-RMS faces remain unclassified rather
than being counted as improvements. The additional `top_down` face has 627
dense points and 0.0076 mm fit RMS. On
`top_left_cut2`, plane count and coverage remain stable while the old cubic
corner filter returns 62 rather than 75 intersections. With no corner ground
truth this is ambiguous, so corner quality is deferred to the later
adjacency/boundary-aware stage instead of tuning this optimizer to reproduce a
raw count.
