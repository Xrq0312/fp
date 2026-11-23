#!/usr/bin/env python3
"""Evaluate plane instances and corners for the stepped-box fixture."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional


def key(x: float, y: float, z: float) -> tuple[int, int, int]:
    # PCL's ASCII writer uses fewer significant digits than the fixture CSV.
    # A millimetre-grid fixture with 0.2 mm spacing remains unique at 1e-3.
    scale = 1_000.0
    return round(x * scale), round(y * scale), round(z * scale)


def read_ply(path: Path) -> list[list[str]]:
    rows: list[list[str]] = []
    with path.open("r", encoding="ascii") as handle:
        for line in handle:
            if line.strip() == "end_header":
                break
        for line in handle:
            values = line.split()
            if values:
                rows.append(values)
    return rows


def evaluate(labels_path: Path, truth_path: Path, result_dir: Path) -> dict[str, object]:
    labels: dict[tuple[int, int, int], int] = {}
    gt_sizes: Counter[int] = Counter()
    with labels_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            label = int(row["plane_id"])
            labels[key(float(row["x"]), float(row["y"]), float(row["z"]))] = label
            gt_sizes[label] += 1

    predicted_sizes: Counter[tuple[int, int, int]] = Counter()
    contingency: dict[tuple[int, int, int], Counter[int]] = defaultdict(Counter)
    unmatched_output = 0

    def lookup_label(point_key: tuple[int, int, int]) -> Optional[int]:
        label = labels.get(point_key)
        if label is not None:
            return label
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    label = labels.get((point_key[0] + dx, point_key[1] + dy, point_key[2] + dz))
                    if label is not None:
                        return label
        return None

    plane_rows = read_ply(result_dir / "edge_aware_plane_patches.ply")
    for row in plane_rows:
        xyz_key = key(float(row[0]), float(row[1]), float(row[2]))
        predicted = int(row[3]), int(row[4]), int(row[5])
        predicted_sizes[predicted] += 1
        label = lookup_label(xyz_key)
        if label is None:
            unmatched_output += 1
        else:
            contingency[predicted][label] += 1


    best_iou_by_gt: dict[int, float] = {label: 0.0 for label in gt_sizes}
    best_gt_by_prediction: dict[tuple[int, int, int], tuple[int, float]] = {}
    for predicted, predicted_size in predicted_sizes.items():
        best_label = -1
        best_iou = 0.0
        for label, intersection in contingency[predicted].items():
            union = predicted_size + gt_sizes[label] - intersection
            iou = intersection / union if union else 0.0
            if iou > best_iou:
                best_label, best_iou = label, iou
            best_iou_by_gt[label] = max(best_iou_by_gt[label], iou)
        best_gt_by_prediction[predicted] = best_label, best_iou

    detected = sum(iou >= 0.5 for iou in best_iou_by_gt.values())
    valid_predictions = sum(iou >= 0.5 for _, iou in best_gt_by_prediction.values())
    fragments_per_gt: Counter[int] = Counter()
    for predicted, counts in contingency.items():
        for label, intersection in counts.items():
            if intersection / gt_sizes[label] >= 0.1:
                fragments_per_gt[label] += 1
    oversegmentation = sum(max(0, count - 1) for count in fragments_per_gt.values())

    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    gt_corners = [tuple(map(float, corner)) for corner in truth["corners"]]
    predicted_corners = [tuple(map(float, row[:3])) for row in read_ply(result_dir / "extracted_corners.ply")]
    tolerance = float(truth["corner_tolerance"])
    available = set(range(len(predicted_corners)))
    corner_errors: list[float] = []
    for gt in gt_corners:
        best_index = -1
        best_distance = math.inf
        for index in available:
            point = predicted_corners[index]
            distance = math.dist(gt, point)
            if distance < best_distance:
                best_index, best_distance = index, distance
        if best_index >= 0 and best_distance <= tolerance:
            available.remove(best_index)
            corner_errors.append(best_distance)

    matched_points = sum(sum(counts.values()) for counts in contingency.values())
    metrics: dict[str, object] = {
        "plane_precision": valid_predictions / len(predicted_sizes) if predicted_sizes else 0.0,
        "plane_recall": detected / len(gt_sizes) if gt_sizes else 0.0,
        "plane_macro_iou": sum(best_iou_by_gt.values()) / len(gt_sizes) if gt_sizes else 0.0,
        "plane_instances": len(predicted_sizes),
        "plane_oversegmentation": oversegmentation,
        "point_coverage": matched_points / len(labels) if labels else 0.0,
        "unmatched_output_points": unmatched_output,
        "corner_precision": len(corner_errors) / len(predicted_corners) if predicted_corners else 0.0,
        "corner_recall": len(corner_errors) / len(gt_corners) if gt_corners else 0.0,
        "corner_mean_error": sum(corner_errors) / len(corner_errors) if corner_errors else None,
        "corner_instances": len(predicted_corners),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("labels", type=Path)
    parser.add_argument("ground_truth", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    metrics = evaluate(args.labels, args.ground_truth, args.result_dir)
    payload = json.dumps(metrics, indent=2, allow_nan=False) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
