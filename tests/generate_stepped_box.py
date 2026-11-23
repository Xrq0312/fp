#!/usr/bin/env python3
"""Generate a deterministic labeled multi-plane fixture for ablation tests."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def sample_axis(lo: float, hi: float, spacing: float) -> list[float]:
    count = max(1, int(round((hi - lo) / spacing)))
    return [lo + (i + 0.5) * (hi - lo) / count for i in range(count)]


def add_face(
    points: list[tuple[float, float, float, int]],
    label: int,
    fixed_axis: int,
    fixed_value: float,
    first_axis: int,
    first_range: tuple[float, float],
    second_axis: int,
    second_range: tuple[float, float],
    spacing: float,
    rng: random.Random,
    exclude_center: bool = False,
) -> None:
    for a in sample_axis(*first_range, spacing):
        for b in sample_axis(*second_range, spacing):
            xyz = [0.0, 0.0, 0.0]
            xyz[fixed_axis] = fixed_value
            xyz[first_axis] = a
            xyz[second_axis] = b
            if exclude_center and abs(xyz[0]) < 5.0 and abs(xyz[1]) < 5.0:
                continue
            tangent_jitter = 0.01
            xyz[first_axis] += rng.uniform(-tangent_jitter, tangent_jitter)
            xyz[second_axis] += rng.uniform(-tangent_jitter, tangent_jitter)
            xyz[fixed_axis] += rng.uniform(-0.006, 0.006)
            points.append((xyz[0], xyz[1], xyz[2], label))


def generate(output: Path, spacing: float) -> None:
    rng = random.Random(0x5EED)
    points: list[tuple[float, float, float, int]] = []

    # 60 x 40 x 10 base box. Plane 1 has a 10 x 10 opening occupied by the boss.
    add_face(points, 0, 2, 0.0, 0, (-30, 30), 1, (-20, 20), spacing, rng)
    add_face(points, 1, 2, 10.0, 0, (-30, 30), 1, (-20, 20), spacing, rng, True)
    add_face(points, 2, 0, -30.0, 1, (-20, 20), 2, (0, 10), spacing, rng)
    add_face(points, 3, 0, 30.0, 1, (-20, 20), 2, (0, 10), spacing, rng)
    add_face(points, 4, 1, -20.0, 0, (-30, 30), 2, (0, 10), spacing, rng)
    add_face(points, 5, 1, 20.0, 0, (-30, 30), 2, (0, 10), spacing, rng)

    # A deliberately small raised box tests local/fine-scale plane recovery.
    add_face(points, 6, 2, 14.0, 0, (-5, 5), 1, (-5, 5), spacing, rng)
    add_face(points, 7, 0, -5.0, 1, (-5, 5), 2, (10, 14), spacing, rng)
    add_face(points, 8, 0, 5.0, 1, (-5, 5), 2, (10, 14), spacing, rng)
    add_face(points, 9, 1, -5.0, 0, (-5, 5), 2, (10, 14), spacing, rng)
    add_face(points, 10, 1, 5.0, 0, (-5, 5), 2, (10, 14), spacing, rng)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment deterministic stepped-box ablation fixture\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z, _ in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")

    with output.with_suffix(".labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "plane_id"])
        writer.writerows(points)

    corners = []
    for x in (-30.0, 30.0):
        for y in (-20.0, 20.0):
            for z in (0.0, 10.0):
                corners.append([x, y, z])
    for x in (-5.0, 5.0):
        for y in (-5.0, 5.0):
            for z in (10.0, 14.0):
                corners.append([x, y, z])
    metadata = {
        "plane_count": 11,
        "corner_tolerance": 1.0,
        "corners": corners,
    }
    output.with_suffix(".ground_truth.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, default=0.20)
    args = parser.parse_args()
    generate(args.output, args.spacing)


if __name__ == "__main__":
    main()
