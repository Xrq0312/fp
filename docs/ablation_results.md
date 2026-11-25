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

## Stage 3: observed-topology boundary correction

This stage targets the small colored wedges and speckles caused by dense points
being assigned across an observed plane intersection. It does not replace the
joint classifier. Plane adjacency is read from the surfel label graph; a
directed half-space constraint is created only for non-parallel neighbors when
the owner patch has decisive support on exactly one side of the adjacent plane.
Support on both sides vetoes clipping, which preserves crossing sheets and large
faces surrounding internal openings. The tolerance is three fitted-noise
standard deviations rather than a dataset-specific distance.

Small proposals remain detectable, but a log-support median/MAD lower-tail
outlier cannot clip an established face. A joint assignment is changed only if
it violates such a constraint, the actually crossed neighbor is also a feasible
candidate, and that neighbor has no weaker observed local support. If no safe
replacement exists, the original joint label is retained.

| Dataset | Planes | Corners | Point coverage | Corrected dense points | Active constraints | Time (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| stepped_box | 11 | 16 | 100.00% | 0 | 44 | 1.98 |
| top_left_cut | 49 | 19 | 99.94% | 1,736 | 119 | 54.82 |
| top_left_cut2 | 54 | 61 | 98.89% | 348 | 348 | 88.75 |
| top_down | 20 | 16 | 99.99% | 44 | 48 | 46.49 |
| top_down2 | 20 | 32 | 100.00% | 0 | 72 | 77.23 |

On the dedicated intersecting-plane fixture, topology changed 233 edge-band
points and improved macro IoU from 0.96918 (`joint`) to 0.98540 while retaining
2/2 planes and 100% coverage. The stepped-box regression remains exactly 1.0
for plane precision, recall, macro IoU, coverage, corner precision, and corner
recall, demonstrating that external boundaries, internal openings, and the
small raised box are not clipped.

All four real PLY files were run with the same rule set. Patch count and coverage
match the joint stage on every file; `top_left_cut2` changes from 62 to 61 raw
corner candidates, which is reported as a regression observation rather than an
accuracy claim. Against `data2.STEP`, all 54 predicted `top_left_cut2` patches
match an analytic STEP plane within 3 degrees and 0.75 coordinate units, and the
matched-point fraction is 1.0. This is only an infinite-plane precision proxy:
the finite-face evaluator below shows that one planar annulus is missed.
Earlier, less conservative topology experiments
created up to three small transition patches; those variants were rejected and
are not part of this stage.

The stage writes `topology_corrections.ply` and `.csv` for direct inspection.
Orthographic overlays show the corrected points as thin line segments and corner
clusters along observed plane intersections, rather than patches scattered over
face interiors. The renderer now has a standard-library PNG fallback, so these
checks do not require Pillow. Timings are representative Release runs and vary
with concurrent desktop load; the zero-correction `top_down2` case is about 6%
slower than the recorded joint run after deferring topology checks until the
ordinary joint candidate has been selected.

## Correctness and performance audit

The dense-assignment candidate buffer was a fixed `std::array<..., 160>` even
though the fine proposal pass permits 240 models and the coarse/fine sets are
concatenated before merging. Candidates beyond index 159 were silently ignored.
It is now a reusable dynamic buffer reserved to the current model count. On the
current fixtures the observed maximum was 2--7 candidates per point, so all
saved synthetic outputs remained byte-for-byte identical; the change removes a
latent correctness limit rather than tuning present data.

Proposal generation had a second silent limit: reaching `max_planes` stopped
the ordered seed traversal immediately, so later disconnected faces could never
enter competition. The value is now only an initial capacity. If one additional
valid proposal is found, the capacity doubles and the deterministic extraction
restarts until the full seed order is exhausted. A forced-capacity regression
started the stepped-box baseline at one proposal, expanded three times, and
produced byte-identical plane, edge, and corner outputs to the normal-capacity
run. No supplied dataset expands under the normal 160/240 initial capacities.

Stage timers show that algorithmic all-model loops are not the dominant wall
time on `top_down`: a representative 50.92 s Release run spent 25.17 s loading
the PLY and 18.96 s writing ASCII diagnostics. Proposal generation took 2.79 s,
dense assignment 2.08 s, corner extraction 1.42 s, and joint competition only
0.27 s. This rejects the initial assumption that accelerating joint labeling
must be the first CPU priority. Output format remains unchanged for evaluator
compatibility. A per-plane KDTree experiment preserved results exactly but made
the real corner stage slower (1.42 s to 1.91 s) because tree construction
dominated the small number of accepted triple queries; it was removed.

The 26-cell surfel neighborhood is now built once in the original deterministic
offset order and reused by edge marking, both proposal passes, label
competition, topology constraints, finite intersections, and feature-edge
output. All synthetic artifacts remained byte-identical. On `top_down`, proposal
time fell from 1.97 s to 1.32 s, joint competition from 0.181 s to 0.128 s, and
dense assignment from 1.573 s to 1.488 s. The cache uses dynamically grown
neighbor arrays instead of reserving 26 entries for every surfel and reports its
allocated storage in the run log; the stepped-box cache is 1.60 MiB for 27,573
surfels and 225,302 directed neighbor references.

## Stage 4: observed finite intersections and adjacency corners

The optional `--corner-mode adjacency` constructs a plane graph from pairs of
differently labelled surfels in neighboring grid cells. For each non-parallel
pair, observed interface samples are projected onto the analytic plane-plane
intersection. Samples inconsistent with that line are rejected and the
remaining parameter range defines a finite observed intersection. Corner
hypotheses come only from graph wedges (two observed pairwise intersections),
not all plane triples. Requiring only two branches preserves corners whose third
edge is occluded. A relative SVD singular-value test rejects ill-conditioned
three-plane systems before solving them.

The first experiment trimmed 5% from both ends of every observed line. It
recalled only 8/16 stepped-box corners because the trim shortened long edges by
2.5--3.5 mm near their vertices. This density- and length-dependent rule was
removed. Interface samples already pass a perpendicular line-distance test;
the existing local point and per-plane support tests remain the outlier guard.
The corrected version recovers 16/16 stepped-box corners with precision and
recall 1.0 and 0.00023 mm mean error. It reduces the considered hypotheses from
165 global triples to 56 graph-local triples.

Two supplied clean PLY files are CAD samples and provide independent STEP
validation rather than mere regression data: `top_down2` corresponds to
`data1.STEP`, and `top_left_cut2` to `data2.STEP`. A new evaluator extracts STEP
vertices incident to at least three distinct analytic planes and performs
one-to-one corner matching.

| Dataset / mode | STEP planar corners | Predicted | Precision | Recall | F1 | Mean error (mm) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| top_down2 / global | 32 | 32 | 1.000 | 1.000 | 1.000 | 0.01026 |
| top_down2 / adjacency | 32 | 32 | 1.000 | 1.000 | 1.000 | 0.01026 |
| top_left_cut2 / global | 48 | 61 | 0.738 | 0.938 | 0.826 | 0.05071 |
| top_left_cut2 / adjacency | 48 | 48 | 1.000 | 1.000 | 1.000 | 0.01111 |

The adjacency method therefore removed 13 false corners and recovered three
missed STEP corners on `top_left_cut2`; it did not trade precision for recall.
Raw corner counts on the two scans without CAD corner truth changed from 19 to
17 (`top_left_cut`) and remained 16 (`top_down`). Those count changes are not
treated as accuracy claims. The option remains independently selectable while
the remaining scan visualizations and boundary metrics are expanded.

## Finite STEP plane-instance evaluation

`evaluate_step_instances.py` follows each planar `ADVANCED_FACE` through its
outer/inner `EDGE_LOOP` boundaries. Ordinary loops use their ordered vertices;
the annular face in `data2.STEP` uses paired circular arcs and is represented by
sampled analytic circles. Clean CAD PLY points and predicted points are assigned
to finite faces, an instance IoU matrix is built, and one-to-one Hungarian
matching reports detection and segmentation quality. This avoids both nearest
infinite-plane double counting and collapsing disconnected coplanar faces.

| Dataset | STEP planar faces | Unique infinite planes | Predicted | Precision | Recall | F1 | Macro IoU | Over / under |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| top_down2 / data1.STEP | 20 | 20 | 20 | 1.000 | 1.000 | 1.000 | 0.9930 | 0 / 0 |
| top_left_cut2 / data2.STEP | 55 | 52 | 54 | 1.000 | 0.9818 | 0.9908 | 0.9566 | 0 / 0 |

The missing `data2.STEP` instance is a planar annulus at `y=22.5475 mm` with
outer radius 3.0 mm and inner radius 1.6 mm. It contains 57,613 clean points but
its best predicted-instance IoU is only 0.0483. The miss is therefore not a
sparse-output artifact. Its 1.4 mm radial width is almost entirely boundary
support at the 0.5 mm surfel scale, explaining why non-edge seed growing fails.
This becomes the next proposal-generation target; the default minimum extent is
not reduced globally because that would reintroduce transition-plane false
positives on scans.

## Stage 5: residual boundary proposals and occupied support

The optional `--edge-residual-proposals` pass targets narrow planes whose
surfels are all persistent multi-scale boundaries. It runs only on boundary
surfels left unexplained by both ordinary proposal passes. A candidate is born
only if a deterministic 200k-point sample estimates at least the existing
`min_output_points` worth of locally occupied points that no established model
can explain. This is a model-birth benefit test, not a new geometry threshold.
Accepted boundary models additionally require a same-label surfel in the local
3-D occupancy grid during dense assignment, preventing their rectangular
envelopes from filling holes or absorbing adjacent curved surfaces.

| Dataset | Generated / accepted | Final planes | Corners | Coverage | Result versus adjacency-only |
| --- | ---: | ---: | ---: | ---: | --- |
| stepped_box | 0 / 0 | 11 | 16 | 100.00% | byte-identical |
| crossing_planes | 0 / 0 | 2 | 0 | 100.00% | unchanged |
| top_left_cut | 5 / 0 | 49 | 17 | 99.94% | byte-identical |
| top_left_cut2 | 2 / 2 | 55 | 48 | 99.19% | recovered annulus |
| top_down | 1 / 0 | 20 | 16 | 99.99% | byte-identical |
| top_down2 | 0 / 0 | 20 | 32 | 100.00% | byte-identical |

Without occupied support, the recovered annulus received 126,361 points and
had IoU 0.456 because the rectangular envelope filled its hole. Occupancy
validation rejects 150,338 infeasible candidate assignments, leaves 79,236
points on the annulus, and raises its IoU to 0.685. Overall `top_left_cut2`
improves from 54/55 to 55/55 matched finite faces: plane precision, recall, and
F1 become 1.0 and instance macro IoU rises from 0.9566 to 0.9713 with no over-
or under-segmentation. Corner F1 remains 1.0.

A rejected BIC experiment is retained in `proposal_diagnostics.csv`: the true
annulus has 15.8 degree normal RMS and positive quadratic BIC gain because both
boundaries contaminate its local normals, while the false `top_down` patch looks
more planar by BIC. Thus curvature/BIC rejection would remove the desired face
before the false one. Novel unexplained support correctly rejects the scan
candidate (estimated novel points: zero) and accepts the CAD annulus (maximum
estimate: 27,550 points).
