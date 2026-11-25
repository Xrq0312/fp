#!/usr/bin/env python3
"""One-to-one geometric evaluation of fitted planes against synthetic truth."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from evaluate_step_planes import normalize


def evaluate(truth_path: Path, parameters_path: Path, angle_limit: float, distance_limit: float) -> dict[str, object]:
    truth_payload = json.loads(truth_path.read_text(encoding="utf-8"))
    truth = [(normalize(tuple(row["normal"])), float(row["d"])) for row in truth_payload["planes"]]
    with parameters_path.open("r", encoding="utf-8", newline="") as handle:
        predictions = [
            (
                int(row["plane_id"]),
                normalize((float(row["nx"]), float(row["ny"]), float(row["nz"]))),
                float(row["d"]),
            )
            for row in csv.DictReader(handle)
        ]
    candidates = []
    for prediction_id, normal, offset in predictions:
        for truth_id, (truth_normal, truth_offset) in enumerate(truth):
            dot = max(-1.0, min(1.0, sum(normal[i] * truth_normal[i] for i in range(3))))
            sign = 1.0 if dot >= 0.0 else -1.0
            angle = math.degrees(math.acos(abs(dot)))
            distance = abs(offset - sign * truth_offset)
            score = angle / angle_limit + distance / distance_limit
            candidates.append((score, prediction_id, truth_id, angle, distance))
    used_predictions = set()
    used_truth = set()
    matches = []
    for _, prediction_id, truth_id, angle, distance in sorted(candidates):
        if prediction_id in used_predictions or truth_id in used_truth:
            continue
        used_predictions.add(prediction_id)
        used_truth.add(truth_id)
        matches.append((prediction_id, truth_id, angle, distance))
    accepted = [match for match in matches if match[2] <= angle_limit and match[3] <= distance_limit]
    angles = [match[2] for match in accepted]
    distances = [match[3] for match in accepted]
    precision = len(accepted) / len(predictions) if predictions else 0.0
    recall = len(accepted) / len(truth) if truth else 0.0
    return {
        "truth_planes": len(truth),
        "predicted_planes": len(predictions),
        "matched_planes": len(accepted),
        "plane_precision": precision,
        "plane_recall": recall,
        "plane_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "normal_mean_error_deg": sum(angles) / len(angles) if angles else None,
        "normal_max_error_deg": max(angles) if angles else None,
        "offset_mean_error": sum(distances) / len(distances) if distances else None,
        "offset_max_error": max(distances) if distances else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("truth", type=Path)
    parser.add_argument("plane_parameters", type=Path)
    parser.add_argument("--angle", type=float, default=1.0)
    parser.add_argument("--distance", type=float, default=0.75)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = evaluate(args.truth, args.plane_parameters, args.angle, args.distance)
    payload = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
