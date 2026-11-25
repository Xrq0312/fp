#!/usr/bin/env python3
"""Generate two intentionally non-parallel planes separated by only 0.8 degrees."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path


def samples(lo: float, hi: float, spacing: float) -> list[float]:
    count = int(round((hi - lo) / spacing))
    return [lo + (index + 0.5) * (hi - lo) / count for index in range(count)]


def generate(output: Path, spacing: float, angle_deg: float) -> None:
    rng = random.Random(0x8A11CE)
    angle = math.radians(angle_deg)
    normal_a = (0.0, 0.0, 1.0)
    normal_b = (0.0, -math.sin(angle), math.cos(angle))
    point_b = (0.0, 15.0, 10.0)
    offset_b = -sum(normal_b[i] * point_b[i] for i in range(3))
    points: list[tuple[float, float, float, int]] = []
    for x in samples(-20.0, 20.0, spacing):
        for y in samples(-25.0, -5.0, spacing):
            points.append((x, y, rng.uniform(-0.003, 0.003), 0))
    for x in samples(-20.0, 20.0, spacing):
        for y in samples(5.0, 25.0, spacing):
            z = -(normal_b[1] * y + offset_b) / normal_b[2]
            points.append((x, y, z + rng.uniform(-0.003, 0.003), 1))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="ascii", newline="\n") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write("comment intentionally near-parallel plane fixture\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\nend_header\n")
        for x, y, z, _ in points:
            handle.write(f"{x:.6f} {y:.6f} {z:.6f}\n")
    with output.with_suffix(".labels.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["x", "y", "z", "plane_id"])
        writer.writerows(points)
    output.with_suffix(".ground_truth.json").write_text(
        json.dumps(
            {
                "plane_count": 2,
                "corner_tolerance": 1.0,
                "corners": [],
                "planes": [
                    {"normal": normal_a, "d": 0.0},
                    {"normal": normal_b, "d": offset_b},
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--spacing", type=float, default=0.25)
    parser.add_argument("--angle", type=float, default=0.8)
    args = parser.parse_args()
    generate(args.output, args.spacing, args.angle)


if __name__ == "__main__":
    main()
