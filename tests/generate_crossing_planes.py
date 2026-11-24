#!/usr/bin/env python3
"""Generate two intersecting planes that expose dense-assignment edge spill."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path


def samples(lo: float, hi: float, spacing: float) -> list[float]:
    count = int(round((hi - lo) / spacing))
    return [lo + (index + 0.5) * (hi - lo) / count for index in range(count)]


def generate(output: Path, spacing: float) -> None:
    rng = random.Random(0xC2055)
    points: list[tuple[float, float, float, int]] = []
    for x in samples(-20.0, 0.0, spacing):
        for y in samples(-20.0, 20.0, spacing):
            points.append(
                (
                    x + rng.uniform(-0.008, 0.008),
                    y + rng.uniform(-0.008, 0.008),
                    rng.uniform(-0.006, 0.006),
                    0,
                )
            )
    for y in samples(-20.0, 20.0, spacing):
        for z in samples(0.0, 10.0, spacing):
            points.append(
                (
                    rng.uniform(-0.006, 0.006),
                    y + rng.uniform(-0.008, 0.008),
                    z + rng.uniform(-0.008, 0.008),
                    1,
                )
            )

    # Dense, noisy samples near the intersection reproduce the practical case:
    # their wrong-plane distance can be smaller than their own measurement
    # residual even though surrounding surfels clearly identify the owner.
    edge_bands = (0.010, 0.025, 0.050)
    for x_distance in edge_bands:
        x = -x_distance
        for y in samples(-20.0, 20.0, spacing):
            points.append(
                (
                    x + rng.uniform(-0.002, 0.002),
                    y + rng.uniform(-0.008, 0.008),
                    rng.uniform(-0.040, 0.040),
                    0,
                )
            )
    for z in edge_bands:
        for y in samples(-20.0, 20.0, spacing):
            points.append(
                (
                    rng.uniform(-0.040, 0.040),
                    y + rng.uniform(-0.008, 0.008),
                    z + rng.uniform(-0.002, 0.002),
                    1,
                )
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment intersecting-plane dense-assignment fixture\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z, _ in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    with output.with_suffix(".labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "plane_id"])
        writer.writerows(points)
    output.with_suffix(".ground_truth.json").write_text(
        json.dumps({"plane_count": 2, "corner_tolerance": 1.0, "corners": []}, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, default=0.20)
    args = parser.parse_args()
    generate(args.output, args.spacing)


if __name__ == "__main__":
    main()
