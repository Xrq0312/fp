#!/usr/bin/env python3
"""Audit dense-label corrections against finite STEP face ground truth.

STEP labels are evaluation-only and never participate in extraction. Optional
exact-coordinate owner lookup avoids assuming that plane ids stay unchanged
when sparse candidates are absorbed during finalization.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from evaluate_step_instances import (
    group_coplanar_faces,
    label_points,
    load_ascii_ply_points,
    load_patch_parameters,
    parse_step_planar_faces,
)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def write_audit_ply(path: Path, points: np.ndarray, categories: list[str]) -> None:
    colors = {
        "beneficial": (46, 204, 113),
        "harmful": (231, 76, 60),
        "unchanged": (241, 196, 15),
        "unresolved": (149, 165, 166),
    }
    with path.open("w", encoding="ascii", newline="\n") as handle:
        handle.write(
            "ply\nformat ascii 1.0\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point, category in zip(points, categories):
            color = colors[category]
            handle.write(
                f"{point[0]:.9g} {point[1]:.9g} {point[2]:.9g} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def point_owners(result_dir: Path, selected_points: np.ndarray) -> np.ndarray:
    """Find final patch owners by exact float32 coordinates."""
    sizes, _ = load_patch_parameters(result_dir / "plane_parameters.csv")
    points = load_ascii_ply_points(result_dir / "edge_aware_plane_patches.ply", 1.0)
    selected = np.ascontiguousarray(selected_points, dtype=np.float32)
    selected_bits = selected.view(np.uint32).reshape((-1, 3)).astype(np.uint64)
    point_bits = (
        np.ascontiguousarray(points, dtype=np.float32)
        .view(np.uint32)
        .reshape((-1, 3))
        .astype(np.uint64)
    )
    constants = np.asarray(
        [0x9E3779B185EBCA87, 0xC2B2AE3D27D4EB4F, 0x165667B19E3779F9],
        dtype=np.uint64,
    )
    selected_hash = np.bitwise_xor.reduce(selected_bits * constants, axis=1)
    point_hash = np.bitwise_xor.reduce(point_bits * constants, axis=1)
    unique_hash = np.unique(selected_hash)
    positions = np.searchsorted(unique_hash, point_hash)
    in_range = positions < len(unique_hash)
    matched = np.zeros(len(points), dtype=bool)
    matched[in_range] = unique_hash[positions[in_range]] == point_hash[in_range]
    point_indices = np.flatnonzero(matched)
    cumulative_sizes = np.cumsum(np.asarray(sizes, dtype=np.int64))
    owners = np.full(len(selected), -1, dtype=np.int32)
    by_hash: dict[int, list[int]] = defaultdict(list)
    for selected_id, value in enumerate(selected_hash):
        by_hash[int(value)].append(selected_id)
    for point_id in point_indices:
        owner = int(np.searchsorted(cumulative_sizes, point_id, side="right"))
        for selected_id in by_hash[int(point_hash[point_id])]:
            if np.array_equal(points[point_id].astype(np.float32), selected[selected_id]):
                owners[selected_id] = owner
    return owners


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("corrections_csv", type=Path)
    parser.add_argument("instance_metrics", type=Path)
    parser.add_argument("--distance", type=float, default=0.75)
    parser.add_argument("--source-result", type=Path)
    parser.add_argument("--source-metrics", type=Path)
    parser.add_argument("--target-result", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--ply", type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    with args.corrections_csv.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    points = np.asarray(
        [[float(row[axis]) for axis in ("x", "y", "z")] for row in rows],
        dtype=np.float64,
    ).reshape((-1, 3))
    faces = parse_step_planar_faces(args.step)
    truth = label_points(points, faces, group_coplanar_faces(faces), args.distance)

    metrics = json.loads(args.instance_metrics.read_text(encoding="utf-8"))
    target_to_truth = {
        int(match["prediction"]): int(match["truth_face"])
        for match in metrics["matches"]
    }
    if bool(args.source_result) != bool(args.source_metrics):
        parser.error("--source-result and --source-metrics must be supplied together")
    source_to_truth = target_to_truth
    source_owners = None
    target_owners = None
    if args.source_result:
        source_metrics = json.loads(args.source_metrics.read_text(encoding="utf-8"))
        source_to_truth = {
            int(match["prediction"]): int(match["truth_face"])
            for match in source_metrics["matches"]
        }
        source_owners = point_owners(args.source_result, points)
    if args.target_result:
        target_owners = point_owners(args.target_result, points)

    counts: Counter[str] = Counter()
    pair_counts: dict[str, Counter[str]] = defaultdict(Counter)
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    threshold_sweep: dict[str, Counter[str]] = defaultdict(Counter)
    id_mismatches: Counter[str] = Counter()
    categories: list[str] = []
    audited_rows: list[dict[str, object]] = []
    for row_id, (row, truth_face) in enumerate(zip(rows, truth)):
        csv_source = int(row["from_plane"])
        csv_target = int(row["to_plane"])
        source = int(source_owners[row_id]) if source_owners is not None else csv_source
        target = int(target_owners[row_id]) if target_owners is not None else csv_target
        if source != csv_source:
            id_mismatches["source"] += 1
        if target != csv_target:
            id_mismatches["target"] += 1
        if source < 0:
            id_mismatches["source_not_found"] += 1
        if target < 0:
            id_mismatches["target_not_found"] += 1
        source_truth = source_to_truth.get(source, -1)
        target_truth = target_to_truth.get(target, -1)
        if truth_face < 0 or source_truth < 0 or target_truth < 0:
            category = "unresolved"
        elif source_truth == truth_face and target_truth != truth_face:
            category = "harmful"
        elif source_truth != truth_face and target_truth == truth_face:
            category = "beneficial"
        else:
            category = "unchanged"
        categories.append(category)
        audited_rows.append({
            **row,
            "truth_face": int(truth_face),
            "source_final_plane": source,
            "target_final_plane": target,
            "source_truth_face": source_truth,
            "target_truth_face": target_truth,
            "audit_category": category,
        })
        counts[category] += 1
        pair_counts[f"{csv_source}->{csv_target}"][category] += 1
        for field in (
            "from_energy",
            "to_energy",
            "from_support_distance",
            "to_support_distance",
            "normal_confidence",
            "from_residual",
            "to_residual",
            "from_normal_dot",
            "to_normal_dot",
        ):
            if field in row and row[field] != "":
                value = float(row[field])
                if np.isfinite(value):
                    values[category][field].append(value)
        if "from_energy" in row and "to_energy" in row:
            gain = float(row["from_energy"]) - float(row["to_energy"])
            if np.isfinite(gain):
                values[category]["energy_gain"].append(gain)
                for threshold in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0):
                    if gain > threshold:
                        threshold_sweep[f"gain_gt_{threshold:g}"][category] += 1

    decisive = counts["beneficial"] + counts["harmful"]
    summary: dict[str, object] = {
        "corrections": len(rows),
        "resolved": len(rows) - counts["unresolved"],
        "counts": dict(counts),
        "final_owner_vs_csv_id": dict(id_mismatches),
        "beneficial_fraction_of_decisive": counts["beneficial"] / decisive if decisive else None,
        "category_distributions": {},
        "energy_gain_sweep": {},
    }
    for category in ("beneficial", "harmful", "unchanged", "unresolved"):
        summary["category_distributions"][category] = {
            field: {
                "mean": float(np.mean(samples)) if samples else None,
                "p10": percentile(samples, 10),
                "median": percentile(samples, 50),
                "p90": percentile(samples, 90),
            }
            for field, samples in values[category].items()
        }
    summary["energy_gain_sweep"] = {
        threshold: {
            "total": sum(counter.values()),
            **dict(counter),
            "beneficial_fraction_of_decisive": (
                counter["beneficial"] / (counter["beneficial"] + counter["harmful"])
                if counter["beneficial"] + counter["harmful"]
                else None
            ),
        }
        for threshold, counter in threshold_sweep.items()
    }
    summary["pairs"] = [
        {"pair": pair, "total": sum(counter.values()), **dict(counter)}
        for pair, counter in sorted(
            pair_counts.items(), key=lambda item: sum(item[1].values()), reverse=True
        )
    ]

    payload = json.dumps(summary, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(payload, encoding="utf-8")
    if args.ply:
        args.ply.parent.mkdir(parents=True, exist_ok=True)
        write_audit_ply(args.ply, points, categories)
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = list(audited_rows[0].keys()) if audited_rows else ["audit_category"]
        with args.csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(audited_rows)
    print(payload, end="")


if __name__ == "__main__":
    main()
