#!/usr/bin/env python3
"""Evaluate extracted planar corners against analytic STEP vertices."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from evaluate_step_planes import ENTITY_RE, NUMBER_RE, REF_RE, parse_step_planes


def parse_step_vertices(path: Path) -> list[tuple[float, float, float]]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entities = {int(identifier): body.strip() for identifier, body in ENTITY_RE.findall(text)}
    vertices: list[tuple[float, float, float]] = []
    for body in entities.values():
        if not body.startswith("VERTEX_POINT"):
            continue
        references = [int(value) for value in REF_RE.findall(body)]
        if not references:
            continue
        values = [float(value) for value in NUMBER_RE.findall(entities[references[-1]])]
        point = tuple(values[-3:])
        if not any(math.dist(point, existing) <= 1e-7 for existing in vertices):
            vertices.append(point)  # type: ignore[arg-type]
    return vertices


def unique_planes(
    planes: list[tuple[tuple[float, float, float], float]],
) -> list[tuple[tuple[float, float, float], float]]:
    unique: list[tuple[tuple[float, float, float], float]] = []
    for normal, offset in planes:
        duplicate = False
        for other_normal, other_offset in unique:
            dot = sum(normal[i] * other_normal[i] for i in range(3))
            sign = 1.0 if dot >= 0.0 else -1.0
            if abs(abs(dot) - 1.0) <= 1e-7 and abs(offset - sign * other_offset) <= 1e-6:
                duplicate = True
                break
        if not duplicate:
            unique.append((normal, offset))
    return unique


def planar_step_corners(path: Path, plane_tolerance: float) -> list[tuple[float, float, float]]:
    planes = unique_planes(parse_step_planes(path))
    corners = []
    for point in parse_step_vertices(path):
        incident_planes = sum(
            abs(sum(normal[i] * point[i] for i in range(3)) + offset) <= plane_tolerance
            for normal, offset in planes
        )
        if incident_planes >= 3:
            corners.append(point)
    return corners


def load_corner_ply(path: Path) -> list[tuple[float, float, float]]:
    with path.open("r", encoding="ascii") as handle:
        vertex_count = None
        for line in handle:
            fields = line.strip().split()
            if fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            if fields == ["end_header"]:
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        corners = []
        for _ in range(vertex_count):
            fields = handle.readline().split()
            if len(fields) < 3:
                raise ValueError(f"Truncated corner PLY: {path}")
            corners.append(tuple(float(value) for value in fields[:3]))
    return corners  # type: ignore[return-value]


def maximum_matching(
    predictions: list[tuple[float, float, float]],
    truth: list[tuple[float, float, float]],
    tolerance: float,
) -> list[tuple[int, int, float]]:
    adjacency = []
    distances: dict[tuple[int, int], float] = {}
    for prediction_id, prediction in enumerate(predictions):
        candidates = []
        for truth_id, target in enumerate(truth):
            distance = math.dist(prediction, target)
            if distance <= tolerance:
                candidates.append((distance, truth_id))
                distances[(prediction_id, truth_id)] = distance
        adjacency.append([truth_id for _, truth_id in sorted(candidates)])

    truth_to_prediction = [-1] * len(truth)

    def augment(prediction_id: int, visited: list[bool]) -> bool:
        for truth_id in adjacency[prediction_id]:
            if visited[truth_id]:
                continue
            visited[truth_id] = True
            owner = truth_to_prediction[truth_id]
            if owner < 0 or augment(owner, visited):
                truth_to_prediction[truth_id] = prediction_id
                return True
        return False

    prediction_order = sorted(range(len(predictions)), key=lambda index: len(adjacency[index]))
    for prediction_id in prediction_order:
        augment(prediction_id, [False] * len(truth))

    return [
        (prediction_id, truth_id, distances[(prediction_id, truth_id)])
        for truth_id, prediction_id in enumerate(truth_to_prediction)
        if prediction_id >= 0
    ]


def evaluate(step: Path, corners_path: Path, tolerance: float, plane_tolerance: float) -> dict[str, object]:
    truth = planar_step_corners(step, plane_tolerance)
    predictions = load_corner_ply(corners_path)
    matches = maximum_matching(predictions, truth, tolerance)
    precision = len(matches) / len(predictions) if predictions else 0.0
    recall = len(matches) / len(truth) if truth else 0.0
    errors = [match[2] for match in matches]
    return {
        "step_planar_corners": len(truth),
        "predicted_corners": len(predictions),
        "matched_corners": len(matches),
        "corner_precision": precision,
        "corner_recall": recall,
        "corner_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "corner_mean_error": sum(errors) / len(errors) if errors else None,
        "corner_max_error": max(errors) if errors else None,
        "corner_tolerance": tolerance,
        "step_plane_incidence_tolerance": plane_tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("corners", type=Path)
    parser.add_argument("--tolerance", type=float, default=1.0)
    parser.add_argument("--plane-tolerance", type=float, default=1e-5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = evaluate(args.step, args.corners, args.tolerance, args.plane_tolerance)
    payload = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
