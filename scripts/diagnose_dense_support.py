#!/usr/bin/env python3
"""Read-only point diagnostics for a selected set of predicted planes.

The plane PLY stores patches consecutively, so ``dense_points`` in
``plane_parameters.csv`` provides an unambiguous label without relying on the
display colour.  This script intentionally ignores feature-edge overlays.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np


def read_ply_header(handle) -> str:
    ply_format = ""
    while True:
        raw = handle.readline()
        if not raw:
            raise RuntimeError("PLY end_header was not found")
        line = raw.decode("ascii").strip()
        if line.startswith("format "):
            ply_format = line.split()[1]
        if line == "end_header":
            return ply_format


def read_planes(path: Path) -> list[dict[str, float]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({key: float(value) for key, value in row.items()})
    return rows


def read_selected_points(
    ply_path: Path, planes: list[dict[str, float]], selected: set[int]
) -> dict[int, np.ndarray]:
    points: dict[int, list[tuple[float, float, float]]] = {plane: [] for plane in selected}
    with ply_path.open("rb") as handle:
        ply_format = read_ply_header(handle)
        if ply_format == "binary_little_endian":
            dtype = np.dtype(
                [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "u1", (3,))]
            )
            for plane_id, plane in enumerate(planes):
                count = int(plane["dense_points"])
                values = np.fromfile(handle, dtype=dtype, count=count)
                if len(values) != count:
                    raise RuntimeError(f"truncated plane PLY while reading plane {plane_id}")
                if plane_id in selected:
                    xyz = np.column_stack((values["x"], values["y"], values["z"]))
                    points[plane_id].extend(map(tuple, xyz))
        elif ply_format == "ascii":
            for plane_id, plane in enumerate(planes):
                count = int(plane["dense_points"])
                for _ in range(count):
                    fields = handle.readline().split()
                    if len(fields) < 3:
                        raise RuntimeError(f"truncated plane PLY while reading plane {plane_id}")
                    if plane_id in selected:
                        points[plane_id].append(tuple(map(float, fields[:3])))
        else:
            raise RuntimeError(f"unsupported PLY format: {ply_format}")
    return {
        plane: np.asarray(values, dtype=np.float64).reshape((-1, 3))
        for plane, values in points.items()
    }


def connected_sizes(points: np.ndarray, cell_size: float) -> list[int]:
    if len(points) == 0:
        return []
    keys = {tuple(value) for value in np.floor(points / cell_size).astype(np.int64)}
    sizes: list[int] = []
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if dx or dy or dz
    ]
    while keys:
        seed = keys.pop()
        queue = deque([seed])
        size = 0
        while queue:
            key = queue.popleft()
            size += 1
            for dx, dy, dz in offsets:
                neighbor = key[0] + dx, key[1] + dy, key[2] + dz
                if neighbor in keys:
                    keys.remove(neighbor)
                    queue.append(neighbor)
        sizes.append(size)
    return sorted(sizes, reverse=True)


def write_selected_ply(path: Path, points: dict[int, np.ndarray]) -> None:
    palette = [(244, 67, 54), (33, 150, 243), (76, 175, 80), (255, 193, 7)]
    count = sum(len(values) for values in points.values())
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {count}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for color_id, (plane_id, values) in enumerate(sorted(points.items())):
            color = palette[color_id % len(palette)]
            for point in values:
                handle.write(
                    f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                    f"{color[0]} {color[1]} {color[2]}\n"
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--planes", type=int, nargs="+", required=True)
    parser.add_argument("--cell-size", type=float, default=0.25)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--selected-ply", type=Path)
    args = parser.parse_args()

    planes = read_planes(args.result_dir / "plane_parameters.csv")
    selected = set(args.planes)
    if not selected or min(selected) < 0 or max(selected) >= len(planes):
        raise RuntimeError("selected plane id is out of range")
    points = read_selected_points(
        args.result_dir / "edge_aware_plane_patches.ply", planes, selected
    )
    candidate_ids = sorted(selected)
    normals = np.asarray(
        [[planes[i]["nx"], planes[i]["ny"], planes[i]["nz"]] for i in candidate_ids]
    )
    offsets = np.asarray([planes[i]["d"] for i in candidate_ids])

    report: dict[str, object] = {"planes": candidate_ids, "per_plane": {}}
    pair_counts: Counter[str] = Counter()
    for owner in candidate_ids:
        values = points[owner]
        distances = np.abs(values @ normals.T + offsets) if len(values) else np.empty((0, len(selected)))
        owner_column = candidate_ids.index(owner)
        best_columns = np.argmin(distances, axis=1) if len(values) else np.empty(0, dtype=np.int64)
        owner_distance = distances[:, owner_column] if len(values) else np.empty(0)
        uncertainty = np.asarray(
            [
                3.0
                * math.hypot(planes[owner]["rms_distance"], planes[target]["rms_distance"])
                for target in candidate_ids
            ]
        )
        competitive = distances + uncertainty < owner_distance[:, None]
        competitive[:, owner_column] = False
        suspect = np.any(competitive, axis=1)
        for column, target in enumerate(candidate_ids):
            count = int(np.count_nonzero(competitive[:, column]))
            if count:
                pair_counts[f"{owner}->{target}"] += count
        suspect_points = values[suspect]
        report["per_plane"][str(owner)] = {
            "points": int(len(values)),
            "bbox_min": values.min(axis=0).tolist() if len(values) else None,
            "bbox_max": values.max(axis=0).tolist() if len(values) else None,
            "median_owner_residual": float(np.median(owner_distance)) if len(values) else None,
            "better_selected_plane_points": int(np.count_nonzero(best_columns != owner_column)),
            "decisive_competition_points": int(np.count_nonzero(suspect)),
            "competition_component_cells": connected_sizes(suspect_points, args.cell_size)[:20],
        }
    report["decisive_pairs"] = dict(pair_counts.most_common())

    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    if args.selected_ply:
        args.selected_ply.parent.mkdir(parents=True, exist_ok=True)
        write_selected_ply(args.selected_ply, points)
    print(payload, end="")


if __name__ == "__main__":
    main()
