#!/usr/bin/env python3
"""Compare fitted plane equations with analytic PLANE entities in a STEP file."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


ENTITY_RE = re.compile(r"#(\d+)\s*=\s*(.*?);", re.DOTALL)
REF_RE = re.compile(r"#(\d+)")
NUMBER_RE = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:E[-+]?\d+)?", re.IGNORECASE)


def vector(values: list[float]) -> tuple[float, float, float]:
    if len(values) < 3:
        raise ValueError("Expected a 3-D tuple in STEP entity")
    return values[-3], values[-2], values[-1]


def normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(component * component for component in value))
    if length <= 0.0:
        raise ValueError("STEP direction has zero length")
    return tuple(component / length for component in value)  # type: ignore[return-value]


def parse_step_planes(path: Path) -> list[tuple[tuple[float, float, float], float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entities = {int(identifier): body.strip() for identifier, body in ENTITY_RE.findall(text)}
    planes: list[tuple[tuple[float, float, float], float]] = []
    for body in entities.values():
        if not body.startswith("PLANE"):
            continue
        references = [int(value) for value in REF_RE.findall(body)]
        if not references:
            continue
        placement = entities[references[-1]]
        if not placement.startswith("AXIS2_PLACEMENT_3D"):
            continue
        placement_refs = [int(value) for value in REF_RE.findall(placement)]
        if len(placement_refs) < 2:
            continue
        point_body = entities[placement_refs[0]]
        normal_body = entities[placement_refs[1]]
        point = vector([float(value) for value in NUMBER_RE.findall(point_body)])
        normal = normalize(vector([float(value) for value in NUMBER_RE.findall(normal_body)]))
        offset = -sum(normal[i] * point[i] for i in range(3))
        planes.append((normal, offset))
    return planes


def evaluate(
    step_path: Path,
    parameters_path: Path,
    angle_tolerance: float,
    distance_tolerance: float,
) -> dict[str, object]:
    truth = parse_step_planes(step_path)
    predictions = []
    with parameters_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            predictions.append(
                (
                    int(row["plane_id"]),
                    normalize((float(row["nx"]), float(row["ny"]), float(row["nz"]))),
                    float(row["d"]),
                    int(row["dense_points"]),
                )
            )

    rows = []
    matched_predictions = 0
    matched_points = 0
    total_points = sum(prediction[3] for prediction in predictions)
    for plane_id, normal, offset, dense_points in predictions:
        best_angle = math.inf
        best_distance = math.inf
        best_truth = -1
        for truth_id, (truth_normal, truth_offset) in enumerate(truth):
            dot = max(-1.0, min(1.0, sum(normal[i] * truth_normal[i] for i in range(3))))
            sign = 1.0 if dot >= 0.0 else -1.0
            angle = math.degrees(math.acos(abs(dot)))
            distance = abs(offset - sign * truth_offset)
            score = angle / max(angle_tolerance, 1e-9) + distance / max(distance_tolerance, 1e-9)
            best_score = best_angle / max(angle_tolerance, 1e-9) + best_distance / max(distance_tolerance, 1e-9)
            if score < best_score:
                best_angle, best_distance, best_truth = angle, distance, truth_id
        matched = best_angle <= angle_tolerance and best_distance <= distance_tolerance
        if matched:
            matched_predictions += 1
            matched_points += dense_points
        rows.append(
            {
                "plane_id": plane_id,
                "dense_points": dense_points,
                "matched": matched,
                "step_plane": best_truth,
                "angle_deg": best_angle,
                "offset_error": best_distance,
            }
        )

    return {
        "step_plane_entities": len(truth),
        "predicted_patches": len(predictions),
        "matched_patches": matched_predictions,
        "patch_precision_proxy": matched_predictions / len(predictions) if predictions else 0.0,
        "matched_point_fraction": matched_points / total_points if total_points else 0.0,
        "angle_tolerance_deg": angle_tolerance,
        "offset_tolerance": distance_tolerance,
        "patches": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("plane_parameters", type=Path)
    parser.add_argument("--angle", type=float, default=3.0)
    parser.add_argument("--distance", type=float, default=0.75)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = evaluate(args.step, args.plane_parameters, args.angle, args.distance)
    payload = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
