# Plane ownership logic fix — 2026-09-21

## Root causes

The projected patch envelope is an axis-aligned rectangle in the plane basis.
It is useful as a coarse bound, but it is not proof that the patch has local
support. A long inclined patch can therefore extend through a large face and
win dense points by a slightly smaller point-to-plane distance.

Two secondary effects amplified the problem:

- fitted fragments of one large face could fail the ordinary offset test because
  their slightly different normals made the global plane `d` values origin-dependent;
- patches below `min_output_points` were erased after dense assignment, even when
  all their points could be transferred safely to established faces.

## Logic changes

1. Estimate stable global normal families from low-curvature surfels using a
   median/MAD curvature cutoff. Local surfel normals are not overwritten.
2. Run ordinary coplanar merge first. Try direction-family merge on a copy and
   accept it only when at least two relaxed merges provide repeated-fragmentation
   evidence. This merged the three `top_left_cut` strips but rejected the single
   speculative merge on `top_left_cut2`.
3. During ambiguous dense assignment, count plane-owned surfels in a local 3-D
   neighborhood. A normal-consistent plane may replace the distance winner when
   the winner is loose-envelope-only or lacks real local support.
4. For `edge_like` cells, use the existing edge angle instead of discarding the
   normal completely. This fixes mixed edge normals when one plane is still
   clearly closer in orientation.
5. Keep the whole-cloud evidence gate. Only stable, non-edge, tight-alternative
   conflicts count toward enabling normal-guided assignment for the dataset;
   local and edge evidence cannot enable it by themselves.
6. Before filtering a small output patch, transfer all its points when every
   point has a valid established destination. Partially unsupported sparse
   patches still follow the existing minimum-output filter.

## Four-dataset regression

| Dataset | Planes | Plane points | Feature-edge points | Corners |
|---|---:|---:|---:|---:|
| `top_left_cut` | 33 | 12,621,846 | 891 | 23 |
| `top_left_cut2` | 54 | 9,889,388 | 179 | 75 |
| `top_down` | 15 | 5,702,636 | 517 | 16 |
| `top_down2` | 12 | 10,000,006 | 880 | 16 |

Compared with the preceding `top_left_cut` result, retained plane points increase
from 12,613,605 to 12,621,846 and feature-edge points fall from 1,538 to 891.
The other three datasets retain their established plane, edge, and corner counts;
their retained point counts are unchanged or higher.

All eleven user-reported coordinates around the crossing planes have the same
final face color. An enlarged XZ crop over `x=[16,20]`, `y=[-0.5,0.5]`,
`z=[-1,4.5]` contains 11,341 points and a single plane color.

Final outputs are in `results_logic_fix_final_20260921`; enlarged visual checks
are in `compare_logic_fix_final_20260921`. Both directories are ignored by Git.
