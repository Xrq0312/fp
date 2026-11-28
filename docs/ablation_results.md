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

### Stable-owner correction for thin boundary spill

A remaining failure mode occurred inside a surfel that was already classified
as smooth and owned by the correct plane. Near an intersection, a different
infinite plane could still win dense assignment by a slightly smaller point-to-
plane distance, even though its normal strongly disagreed with the observed
surfel normal. This produced the one-cell-wide colored slivers seen in the real
scan.

Topology mode now restores the observed owner only when all of the following
hold: the surfel is not edge-like, the selected plane differs by more than the
20-degree edge angle, the owner agrees within the 8-degree smooth angle, and
the distance disadvantage is within three combined fitted RMS deviations. A
scene-adaptive median/MAD lower-tail test prevents tiny support fragments from
gaining this authority. Thus the rule uses measured fit uncertainty and support
statistics rather than a dataset-specific distance or plane-size threshold.

On `top_down`, recoverable normal-inconsistent assignments decreased from 5,739
to 4,437 (1,302 fewer, 22.69%). The largest affected face decreased from 955 to
183 such assignments. Plane count (20), output coverage, and adjacency corners
(16) were unchanged. The final four-real-cloud regression retained the prior
plane counts. Clean STEP instance checks remained 55/55 and 20/20 with no over-
or under-segmentation; `top_left_cut2` macro IoU changed from 0.971293 to
0.971183, while `top_down2` was 0.993013. The stepped-box regression remained
perfect and the crossing-plane macro IoU remained 0.985402.

Correction PLY files now use the deterministic color of each point's final
plane. Earlier files forced every changed point to red, which made a diagnostic
subset look like an additional classification when loaded over the main PLY.
The CSV retains the stage-local `from_plane` and `to_plane` and adds
`final_plane`, because a later finite-support pass may legitimately supersede
an earlier topology move. The renderer can still draw a red overlay explicitly
when a changed-point diagnostic is wanted.

### Edge-cell owner recovery and picking-list trace (2026-11-28)

Four points in `picking_list1.txt` exposed a second form of thin boundary spill
in `top_left_cut`. In the preceding result all four were assigned to plane 29,
whereas four nearby counterexample points in `picking_list2.txt` belonged to
plane 4. The first list's cells were already owned by plane 4 at surfel level.
For a representative point, plane 29 won infinite-plane distance (0.0345 mm
versus 0.0694 mm), but the observed normal agreed with plane 4 at 0.997 and
with plane 29 at only 0.638; local support was also 19 versus 15. The dense
classifier had excluded this evidence solely because the surfel was marked as
edge-like.

Topology dense assignment now permits an authoritative edge-cell owner under
two conservative evidence paths. Either its smooth normal and neighborhood
support dominate, or its normal is within the existing edge-angle tolerance
and its relative advantage exceeds the same angle-derived cosine budget. Both
paths additionally require tight finite envelopes, measured strict-neighborhood
support, an established scene-adaptive owner, and a distance disadvantage no
larger than three combined fitted RMS deviations. This is a general geometric
rule; no coordinate, plane id, or dataset threshold is encoded.

The exact float32 owner trace changed from
`[29,29,29,29 | 4,4,4,4]` to `[4,4,4,4 | 4,4,4,4]`. On the full scan this
created 8,746 decisive edge-owner moves before the later finite-region stage;
plane count (49), output count (12,632,707), and adjacency corners (19) were
unchanged. On `top_down`, plane count (20), output count (5,703,086), and
corners (16) were also unchanged. The synthetic regressions remained:

| Dataset | Plane precision / recall | Macro IoU | Corner precision / recall |
| --- | --- | ---: | --- |
| stepped_box | 1.0 / 1.0 | 1.000000 | 1.0 / 1.0 |
| crossing_planes | 1.0 / 1.0 | 0.985402 | n/a |
| near_parallel_planes | 1.0 / 1.0 | 1.000000 | n/a |

All correction rows in the two real-data runs resolved to a final owner. Forced
red rows are zero. Stage targets differed from final owners for 740/10,282
topology rows and 370/744 finite-region rows on `top_left_cut`, and for
291/2,061 topology rows on `top_down`; recording `final_plane` is therefore
necessary rather than cosmetic.

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
Adjacency-mode duplicate handling now uses deterministic complete-link
clustering: every pair of candidates in a cluster must lie within the merge
radius, so a chain of intermediate candidates cannot merge two nearby real
corners. Candidates are first sorted lexicographically and support/position
ties are deterministic. The stepped-box coordinate set is unchanged. On
`top_left_cut2`, two cluster representatives move slightly, but all 48 STEP
corners remain matched (F1 1.0), maximum error stays 0.0721 mm, and mean error
changes only from 0.01111 to 0.01122 mm.
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

## Binary result serialization

`--binary-plane-output` changes only the large colored
`edge_aware_plane_patches.ply` serialization to standard little-endian binary
PLY; all vertices, labels/colors, and ordering are identical. PCL successfully
reloads the file, and the NumPy evaluator reads both formats. On stepped_box the
file shrinks from 7.18 MB to 2.61 MB and save time falls from roughly 0.53 s to
0.019 s. On `top_down`, save time falls from 17.14 s to 0.094 s and total time
from 44.09 s to 27.45 s (38% reduction). The default stays ASCII for human
inspection and backward compatibility; binary is the recommended production
setting for multi-million-point results.

## Parallel relation refinement (GlobFit ablation)

`--parallel-relations none|soft|hard` forms deterministic complete-link normal
clusters whose maximum pairwise spread is at most 20% of the existing merge
angle (1 degree with the default configuration). Support-weighted family normals
are applied halfway (`soft`) or exactly (`hard`); each plane keeps its fitted
centroid and recomputes its offset and envelope. No orthogonality constraint is
inferred: a near-90-degree angle alone is not evidence that a structure was
designed orthogonally.

| Dataset / mode | Instance macro IoU | Mean normal error | Mean corner error |
| --- | ---: | ---: | ---: |
| top_left_cut2 / none | 0.97129 | 0.03235 deg | 0.01122 mm |
| top_left_cut2 / soft | 0.97178 | 0.02364 deg | 0.01021 mm |
| top_left_cut2 / hard | 0.97182 | 0.01825 deg | 0.00865 mm |
| top_down2 / none | 0.99300 | 0.08080 deg | 0.01026 mm |
| top_down2 / soft | 0.99396 | 0.04051 deg | 0.00526 mm |
| top_down2 / hard | 0.99412 | 0.00233 deg | 0.00392 mm |

All STEP plane and corner F1 scores remain 1.0. The hard mode also leaves plane
and corner counts unchanged on both scans, but that is not sufficient evidence
that every discovered relation is real. A new synthetic fixture contains two
planes intentionally separated by 0.8 degrees. `none` recovers them with
0.00009-degree mean normal error, while `soft` and `hard` incorrectly increase
the error to 0.200 and 0.400 degrees. Therefore relation refinement is an
explicit domain-prior option only: `none` remains the general default; `hard`
is recommended only when manufacturing semantics independently guarantee the
clustered faces are parallel. This failure also rules out automatic hard
orthogonal snapping without stronger semantic or repeated-structure evidence.

## Stage 6: label-independent finite-boundary correction

`--finite-boundary-postprocess` implements the final-stage architecture
"multi-plane support, pairwise boundary analysis, multi-finite-plane joint
correction." Trusted support cells are rebuilt from raw coordinates by a
best/second-best plane competition and do not use dense labels. Pairwise
intersections are verified and segmented with those cells. Boundary candidates
then compete against every locally plausible finite plane and are updated
synchronously.

Two failed variants explain the safeguards in the retained implementation:

- A dense-label/fine-normal boundary vote moved 2,270 `top_left_cut2` points
  from plane 3 to plane 5. STEP correspondence showed 2,251 belonged to the
  source plane, confirming the circular-label failure. This implementation was
  removed rather than hidden behind another threshold.
- Applying an observed side as a global half-space corrected 84 points on the
  crossing fixture but incorrectly moved 180 points on the perfect stepped box
  (macro IoU 1.0000 to 0.9915). The top face contains an opening: a constraint
  from one side of the hole cannot clip the opposite side. Restricting each
  segmented constraint to its observed local cross-section restores all 180
  points and preserves the crossing improvement.

The finite-distance energy uses
`plane_residual^2 + in_plane_support_distance^2` in a common observation-noise
scale. The fitted-plane RMS is kept separately for intersection uncertainty;
using that very small RMS as point noise rejected all 233 noisy crossing
candidates even though their boundary was correctly found.

| Dataset | Baseline macro IoU | Enabled macro IoU | Candidates / corrections | Added boundary time | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| crossing_planes | 0.96918 | 0.97499 | 233 / 84 | 4.8 ms | improved noisy edge bands |
| stepped_box | 1.00000 | 1.00000 | 0 / 0 | 46.9 ms | exact regression preserved |
| top_left_cut2 / data2.STEP | 0.97118 | 0.97120 | 223 / 88 | 2.13 s | 55/55 faces, F1 1.0 |
| top_down2 / data1.STEP | 0.99301 | 0.99301 | 3 / 0 | 2.05 s | 20/20 faces, F1 1.0 |

Both remaining scan files were also run, rather than testing only the clean CAD
samples. `top_left_cut` produced 2,760 local candidates and 847 synchronous
corrections in 2.73 s over 12.64 million points; `top_down` produced 404
candidates and 86 corrections in 1.00 s over 5.70 million points. Those scans
are not aligned with the supplied STEP coordinate frames, so the counts are
reported only as regression diagnostics and not as accuracy claims.

On `top_left_cut2`, the older in-assignment topology rule made 293 changes. The
new final-stage option disables that rule and makes 88 independently supported
changes; the former 2,270-point 3-to-5 failure is absent. On `top_down2`, three
points crossed a proposed boundary but no finite target achieved lower energy,
so all labels were retained. This conservative no-change behavior is intentional
for occlusion and missing-data cases. The method remains independently enabled
for ablation rather than becoming the default pipeline until aligned boundary
ground truth is available for the two real scans.

## Stage 7: connected finite-support region optimization

The reported `top_left_cut` failures were connected triangular/scan-line
regions, not only sub-millimetre crossing bands. An experimental
`--finite-support-region-optimization` switch now builds raw-cloud finite
support domains, finds connected local plane competitions, and applies a
synchronous graph proposal followed by a pointwise residual/normal validation.
It is mutually exclusive with `--finite-boundary-postprocess`.

Several intermediate variants were rejected:

- Treating an ambiguous raw observation as occupied support for every plausible
  plane moved 18,908 `top_left_cut2` points. STEP point auditing classified only
  1,466 as beneficial and 17,338 as harmful; instance macro IoU fell from the
  historical 0.97118 baseline to 0.96797.
- Allowing pairwise smoothness to override a locally worse unary score created
  coherent but geometrically wrong regions. The graph now only resolves a
  connected proposal; a final residual/noise or confident-normal test has veto
  authority.
- Letting a weak sub-threshold source model create outgoing graph components
  conflated region optimization with the existing sparse-patch absorber. Such
  candidates remain valid destinations but their original points are left to
  the baseline finalizer.

The following table is a paired run from the same executable and extraction
result. It must not be mixed with older rows above whose proposal stage produced
55 rather than 54 `top_left_cut2` patches.

| Dataset | Same-build baseline | Region optimization | Corrections | Added wall time | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| stepped_box | macro IoU 1.000000 | 1.000000 | 0 | about 0.26 s | exact planes and 16/16 corners preserved |
| crossing_planes | macro IoU 0.969178 | 0.969178 | 0 | about 0.03 s | independent region ablation; narrow-band method remains better here |
| near_parallel_planes | macro IoU 1.000000 | 1.000000 | 0 | about 0.05 s | no cross-plane propagation |
| top_left_cut2 / data2.STEP | macro IoU 0.956463 | 0.956508 | 290 | about 9.5 s | 54/55 face recall unchanged; corner F1 0.82569 to 0.85185 |
| top_down2 / data1.STEP | macro IoU 0.993013 | 0.993119 | 16 | about 11.5 s | 20/20 faces and corner F1 1.0 preserved |
| top_left_cut | no aligned instance GT | no aligned instance GT | 889 | about 9.1 s | target multi-face pairs corrected; accuracy not claimed |
| top_down | no aligned instance GT | no aligned instance GT | 180 | about 12.0 s | conservative scan regression only |

For `top_left_cut2`, an exact-coordinate correction audit maps the baseline and
optimized final patch owners independently through the STEP one-to-one instance
matching. Of 290 synchronous moves, 239 are beneficial, 35 harmful, 8 keep the
same STEP-instance interpretation, and 8 fall outside the evaluator tolerance.
The decisive correction precision is therefore 87.2%. The audit utility writes
a category-colored PLY (green beneficial, red harmful, yellow unchanged, grey
unresolved) and reports distributions of residual, finite-support distance,
normal confidence, and energy gain.

On `top_left_cut`, the final 889 changes include the previously reported local
competitions 29-to-4 (89 points), 11-to-29 (10), 11-to-4 (48), and 21-to-4
(21), without any plane-id-specific condition. These counts are diagnostics,
not ground-truth accuracy. Binary PLY support was added to the projection
renderer so the baseline, corrected labels, and red correction overlay can be
inspected without producing a second multi-hundred-megabyte ASCII result.

This table is retained as the historical Stage 7 result. The follow-up below
found two configuration confounders: the run omitted the Stage 5 residual-edge
proposal, and enabling either postprocessor disabled the embedded topology
assignment. The final recommendation therefore uses the unified additive
ablation in Stage 8 rather than this replacement-mode result.

## Stage 8: unified attribution and CGAL operator comparison

### Configuration reconciliation

All final comparisons use the same executable, `topology` pipeline,
`--edge-residual-proposals`, binary output for large clouds, and unchanged
profile parameters. The missing 55th `top_left_cut2` face was STEP face 24, the
planar annulus already identified in Stage 5. Omitting the residual-edge
proposal produced 54 planes before either postprocessor ran; this, rather than
Stage 6/7, explains the apparent loss of the historical 55/55 result.

For completeness, the same-build 54-plane check was:

| Configuration | Planes | Instance macro IoU | Corner F1 |
| --- | ---: | ---: | ---: |
| no postprocess | 54 | 0.95646304 | 0.82569 |
| Stage 6 finite-boundary replacement | 54 | 0.95648712 | 0.81818 |
| Stage 7 region replacement | 54 | 0.95650756 | 0.85185 |

Thus Stage 7 did exceed the same-build Stage 6 in both IoU and corner F1, but
neither result is comparable with the old 55-plane Stage 6 row. With the
residual-edge proposal restored, all tested variants recover 55/55 faces and
corner F1 1.0.

The second confounder was control flow. At `a3f5363`, enabling Stage 7 disabled
the embedded topology guidance during dense assignment. On `crossing_planes`,
that removed 233 already-correct topology assignments; Stage 7 then made zero
new changes, so IoU fell from 0.985402 to 0.969178 without the region optimizer
actually rejecting or relabeling a point. Stage 7 now runs as a true additive
final postprocess over the same dense labels as the no-postprocess baseline.
Stage 6 retains its historical replacement semantics for independent ablation.

| Dataset | No postprocess | Stage 6 replacement | Stage 7 additive PCA | Stage 7 additive PCA+VCM |
| --- | ---: | ---: | ---: | ---: |
| top_left_cut2 macro IoU | 0.97118257 | 0.97120066 | **0.97121635** | 0.97120910 |
| top_left_cut2 corner F1 | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| crossing_planes macro IoU | **0.98540206** | 0.97499040 | **0.98540206** | **0.98540206** |
| stepped_box macro IoU | 1.00000 | 1.00000 | 1.00000 | 1.00000 |
| near_parallel_planes macro IoU | 1.00000 | not required | 1.00000 | 1.00000 |
| top_down2 macro IoU | 0.993013 historical | 0.993013 historical | 0.993119 historical | 0.99309268 |

The final PCA+VCM run keeps 20/20 `top_down2` faces and corner F1 1.0. Its
`top_left_cut2` wall time was 57.40 s versus 42.81 s for extraction without a
region postprocess; the ordinary PCA region run was 56.06 s on the same host.
VCM adds about 1.3 s in this run and is therefore retained only as an opt-in
high-precision validator.

### Pointwise attribution of the 35 harmful moves

The correction CSV now records node retain/target unary energy, whether a point
entered through missing finite support or normal conflict, whether its owner
cell was raw-geometry ambiguous, and whether the final acceptance came from
residual or normal evidence. `evaluate_step_corrections.py --csv` joins these
fields with the exact STEP truth assignment. The original 54-plane Stage 7
audit gives the following complete accounting:

- 16 harmful points are pair 1-to-4 in component 12 and 15 are pair 2-to-4 in
  component 24. The other four are one point each in 8-to-0, 1-to-17, 0-to-53,
  and 1-to-21. This concentration rules out traversal-order noise.
- All 35 were created and finally accepted by the PCA normal path; none had a
  target residual that beat the source by fitted-noise uncertainty. Median
  source/target residuals were about 0.0008/0.0168 mm, while median normal dots
  were about 0.627/0.992.
- 34/35 lie in an owner cell marked ambiguous by raw multi-plane competition.
  They were admitted through normal conflict, not missing-support evidence. The
  remaining point had both missing-support and normal-conflict evidence.
- The two large connected components already preferred the wrong target in
  their unary terms (13 points: 381.82 versus 89.06; 15 points: 177.69 versus
  88.85). Four isolated harmful cases had infinite retain energy. Therefore
  changing ICM traversal cannot repair the principal error: the mixed local PCA
  normal has already biased candidate generation and unary construction.
- The final geometry check used `residual_decisive OR normal_decisive`, so the
  same contaminated normal was able to override a much better source residual.
  This is the direct failure mechanism. Treating every ambiguous cell as
  immutable was rejected because the same audit contains 138 valid normal-only
  corrections in ambiguous cells.

The fault is consequently shared by fine-scale normal evidence and the final
geometric veto, not by a specific plane id or a half-space width. The finite
support grid correctly represented uncertainty for 34 points, but uncertainty
alone could not stop a separate normal-only proposal. The added VCM consensus
requires two geometrically different normal estimators to pass the existing
smooth/edge tests; it introduces no new correction distance or angle threshold.

### Independent CGAL comparisons

Each CGAL operator was first substituted independently on the same 55-plane
replacement-mode Stage 7 input. Output PLY and correction-CSV hashes were also
compared, so an equal aggregate score is not mistaken for a different solution.

| Normal/support/solver | IoU | Corrections | Beneficial / harmful / same / unresolved | Conclusion |
| --- | ---: | ---: | ---: | --- |
| PCA / grid / ICM | 0.97121851 | 275 | 225 / 34 / 8 / 8 | reference |
| CGAL VCM / grid / ICM | 0.97106850 | 857 | 102 / 745 / 6 / 4 | reject standalone VCM |
| PCA / Alpha_shape_2 / ICM | 0.97121851 | 275 | 225 / 34 / 8 / 8 | byte-identical; no gain |
| PCA / grid / alpha-expansion | 0.97121851 | 275 | 225 / 34 / 8 / 8 | byte-identical; no gain |
| PCA+VCM consensus / grid / ICM | 0.97119900 | 37 | 30 / 1 / 2 / 4 | conservative validation |

`Alpha_shape_2` classified no queried point differently from the existing
occupied-cell distance in this experiment and cost additional time. CGAL
alpha-expansion selected exactly the same labels as monotone ICM in all 26
eligible components (the remaining components were outside the binary
homogeneous comparison), showing that solver suboptimality was not the cause.
Standalone VCM at the extraction surfel scale mixed boundary evidence more
aggressively and was clearly harmful. These three modes were removed after the
comparison. Only PCA+VCM consensus is retained.

After the additive control-flow fix, the retained consensus makes 167
`top_left_cut2` changes: 144 beneficial, 1 harmful, 18 instance-neutral, and 4
unresolved. Its decisive correction precision is 99.31%, compared with 87.73%
for additive PCA (243 beneficial and 34 harmful among 306 changes). It preserves
the best synthetic results and improves over both the no-postprocess and Stage
6 `top_left_cut2` IoU, although ordinary PCA retains the highest aggregate IoU
by 0.00000725. The recommended default remains the topology pipeline with
residual-edge proposals and no region postprocess; use PCA+VCM consensus when
minimizing harmful relabels is more important than recovering every marginal
boundary point. Ordinary PCA region optimization remains an explicit research
mode, not the default.

Reproducible outputs are under `results/final_additive_*`; the original 35-row
point audit is under
`results/unified_stage7_diagnostic/topology/top_left_cut2/`.
