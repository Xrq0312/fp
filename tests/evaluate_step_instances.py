#!/usr/bin/env python3
"""Evaluate plane instances against finite planar STEP faces and a clean CAD PLY."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from evaluate_step_planes import ENTITY_RE, NUMBER_RE, REF_RE, normalize


ORIENTATION_RE = re.compile(r"\.(T|F)\.\s*\)\s*$", re.IGNORECASE)


@dataclass
class StepFace:
    normal: np.ndarray
    offset: float
    loops: list[tuple[bool, np.ndarray]]
    centroid: np.ndarray


def entity_orientation(body: str) -> bool:
    match = ORIENTATION_RE.search(body)
    return not match or match.group(1).upper() == "T"


def entity_point(entities: dict[int, str], vertex_id: int) -> np.ndarray:
    vertex_refs = [int(value) for value in REF_RE.findall(entities[vertex_id])]
    values = [float(value) for value in NUMBER_RE.findall(entities[vertex_refs[-1]])]
    return np.asarray(values[-3:], dtype=np.float64)


def entity_plane(entities: dict[int, str], plane_id: int) -> tuple[np.ndarray, float]:
    plane_refs = [int(value) for value in REF_RE.findall(entities[plane_id])]
    placement = entities[plane_refs[-1]]
    placement_refs = [int(value) for value in REF_RE.findall(placement)]
    point_values = [float(value) for value in NUMBER_RE.findall(entities[placement_refs[0]])]
    normal_values = [float(value) for value in NUMBER_RE.findall(entities[placement_refs[1]])]
    point = np.asarray(point_values[-3:], dtype=np.float64)
    normal = np.asarray(normalize(tuple(normal_values[-3:])), dtype=np.float64)
    return normal, -float(normal @ point)


def parse_step_planar_faces(path: Path) -> list[StepFace]:
    text = path.read_text(encoding="utf-8", errors="replace")
    entities = {int(identifier): body.strip() for identifier, body in ENTITY_RE.findall(text)}
    faces: list[StepFace] = []
    for body in entities.values():
        if not body.startswith("ADVANCED_FACE"):
            continue
        references = [int(value) for value in REF_RE.findall(body)]
        if not references or not entities[references[-1]].startswith("PLANE"):
            continue
        normal, offset = entity_plane(entities, references[-1])
        loops: list[tuple[bool, np.ndarray]] = []
        all_vertices = []
        for bound_id in references[:-1]:
            bound = entities[bound_id]
            if not (bound.startswith("FACE_OUTER_BOUND") or bound.startswith("FACE_BOUND")):
                continue
            bound_refs = [int(value) for value in REF_RE.findall(bound)]
            if not bound_refs:
                continue
            loop = entities[bound_refs[-1]]
            ordered_vertices = []
            circle_curves = []
            for oriented_id in [int(value) for value in REF_RE.findall(loop)]:
                oriented = entities[oriented_id]
                edge_refs = [int(value) for value in REF_RE.findall(oriented)]
                if not edge_refs:
                    continue
                edge = entities[edge_refs[-1]]
                vertex_refs = [int(value) for value in REF_RE.findall(edge)]
                if len(vertex_refs) < 2:
                    continue
                start_id = vertex_refs[0] if entity_orientation(oriented) else vertex_refs[1]
                ordered_vertices.append(entity_point(entities, start_id))
                if len(vertex_refs) >= 3 and entities[vertex_refs[2]].startswith("CIRCLE"):
                    circle_curves.append(vertex_refs[2])
            if len(ordered_vertices) < 3 and circle_curves:
                circle = entities[circle_curves[0]]
                circle_refs = [int(value) for value in REF_RE.findall(circle)]
                placement_refs = [int(value) for value in REF_RE.findall(entities[circle_refs[-1]])]
                center_values = [
                    float(value) for value in NUMBER_RE.findall(entities[placement_refs[0]])
                ]
                center = np.asarray(center_values[-3:], dtype=np.float64)
                radius = float(NUMBER_RE.findall(circle)[-1])
                axis_u, axis_v = plane_basis(normal)
                angles = np.linspace(0.0, 2.0 * math.pi, 128, endpoint=False)
                ordered_vertices = [
                    center + radius * (math.cos(angle) * axis_u + math.sin(angle) * axis_v)
                    for angle in angles
                ]
            if len(ordered_vertices) >= 3:
                vertices = np.asarray(ordered_vertices, dtype=np.float64)
                loops.append((bound.startswith("FACE_OUTER_BOUND"), vertices))
                all_vertices.extend(ordered_vertices)
        if not loops:
            continue
        faces.append(StepFace(normal, offset, loops, np.mean(all_vertices, axis=0)))
    return faces


def group_coplanar_faces(faces: list[StepFace]) -> list[list[int]]:
    groups: list[list[int]] = []
    for face_id, face in enumerate(faces):
        for group in groups:
            other = faces[group[0]]
            dot = float(face.normal @ other.normal)
            sign = 1.0 if dot >= 0.0 else -1.0
            if abs(abs(dot) - 1.0) <= 1e-7 and abs(face.offset - sign * other.offset) <= 1e-6:
                group.append(face_id)
                break
        else:
            groups.append([face_id])
    return groups


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = np.asarray((0.0, 0.0, 1.0))
    axis_u = np.cross(normal, reference)
    if np.linalg.norm(axis_u) < 1e-8:
        axis_u = np.cross(normal, np.asarray((1.0, 0.0, 0.0)))
    axis_u /= np.linalg.norm(axis_u)
    return axis_u, np.cross(normal, axis_u)


def points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    inside = np.zeros(points.shape[0], dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        crosses = (current[1] > points[:, 1]) != (previous[1] > points[:, 1])
        denominator = previous[1] - current[1]
        denominator = denominator if abs(denominator) > 1e-15 else math.copysign(1e-15, denominator or 1.0)
        x_cross = (previous[0] - current[0]) * (points[:, 1] - current[1]) / denominator + current[0]
        inside ^= crosses & (points[:, 0] < x_cross)
        previous = current
    return inside


def points_in_face(points: np.ndarray, face: StepFace) -> np.ndarray:
    axis_u, axis_v = plane_basis(face.normal)
    projected_points = np.column_stack((points @ axis_u, points @ axis_v))
    result = np.zeros(points.shape[0], dtype=bool)
    for outer, loop in face.loops:
        polygon = np.column_stack((loop @ axis_u, loop @ axis_v))
        membership = points_in_polygon(projected_points, polygon)
        if outer:
            result |= membership
        else:
            result &= ~membership
    return result


def label_points(
    points: np.ndarray,
    faces: list[StepFace],
    groups: list[list[int]],
    distance_tolerance: float,
    chunk_size: int = 200_000,
) -> np.ndarray:
    normals = np.asarray([faces[group[0]].normal for group in groups])
    offsets = np.asarray([faces[group[0]].offset for group in groups])
    labels = np.full(points.shape[0], -1, dtype=np.int32)
    for begin in range(0, points.shape[0], chunk_size):
        end = min(begin + chunk_size, points.shape[0])
        chunk = points[begin:end].astype(np.float64, copy=False)
        distances = np.abs(chunk @ normals.T + offsets)
        group_labels = np.argmin(distances, axis=1)
        minimum_distances = distances[np.arange(chunk.shape[0]), group_labels]
        for group_id in np.unique(group_labels):
            local = np.flatnonzero((group_labels == group_id) & (minimum_distances <= distance_tolerance))
            if local.size == 0:
                continue
            face_ids = groups[int(group_id)]
            if len(face_ids) == 1:
                labels[begin + local] = face_ids[0]
                continue
            group_points = chunk[local]
            assigned = np.full(local.size, -1, dtype=np.int32)
            for face_id in face_ids:
                mask = points_in_face(group_points, faces[face_id])
                assigned[(assigned < 0) & mask] = face_id
            unresolved = np.flatnonzero(assigned < 0)
            if unresolved.size:
                centroids = np.asarray([faces[face_id].centroid for face_id in face_ids])
                nearest = np.argmin(
                    np.linalg.norm(group_points[unresolved, None, :] - centroids[None, :, :], axis=2),
                    axis=1,
                )
                assigned[unresolved] = np.asarray(face_ids, dtype=np.int32)[nearest]
            labels[begin + local] = assigned
    return labels


def load_ascii_ply_points(path: Path, scale: float) -> np.ndarray:
    with path.open("rb") as handle:
        vertex_count = None
        ply_format = None
        properties = []
        for line in handle:
            fields = line.decode("ascii").strip().split()
            if fields[:1] == ["format"]:
                ply_format = fields[1]
            elif fields[:2] == ["element", "vertex"]:
                vertex_count = int(fields[2])
            elif fields[:1] == ["property"] and vertex_count is not None:
                properties.append((fields[1], fields[2]))
            if fields == ["end_header"]:
                break
        if vertex_count is None:
            raise ValueError(f"PLY has no vertex count: {path}")
        if ply_format == "ascii":
            points = np.loadtxt(handle, dtype=np.float32, usecols=(0, 1, 2), max_rows=vertex_count)
        elif ply_format == "binary_little_endian":
            type_map = {
                "float": "<f4",
                "float32": "<f4",
                "uchar": "u1",
                "uint8": "u1",
            }
            try:
                dtype = np.dtype([(name, type_map[data_type]) for data_type, name in properties])
            except KeyError as error:
                raise ValueError(f"Unsupported binary PLY property type: {error.args[0]}") from error
            records = np.fromfile(handle, dtype=dtype, count=vertex_count)
            points = np.column_stack((records["x"], records["y"], records["z"]))
        else:
            raise ValueError(f"Unsupported PLY format {ply_format!r}: {path}")
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.shape[0] != vertex_count:
        raise ValueError(f"Expected {vertex_count} points in {path}, read {points.shape[0]}")
    if scale != 1.0:
        points *= scale
    return points


def load_patch_parameters(
    path: Path,
) -> tuple[list[int], list[tuple[tuple[float, float, float], float]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    sizes = [int(row["dense_points"]) for row in rows]
    planes = [
        (
            normalize((float(row["nx"]), float(row["ny"]), float(row["nz"]))),
            float(row["d"]),
        )
        for row in rows
    ]
    return sizes, planes


def hungarian_maximize(values: np.ndarray) -> list[tuple[int, int]]:
    transposed = values.shape[0] > values.shape[1]
    matrix = values.T if transposed else values
    row_count, column_count = matrix.shape
    costs = 1.0 - matrix
    u = np.zeros(row_count + 1)
    v = np.zeros(column_count + 1)
    p = np.zeros(column_count + 1, dtype=np.int32)
    way = np.zeros(column_count + 1, dtype=np.int32)
    for row in range(1, row_count + 1):
        p[0] = row
        minimum = np.full(column_count + 1, np.inf)
        used = np.zeros(column_count + 1, dtype=bool)
        column = 0
        while True:
            used[column] = True
            current_row = p[column]
            delta = np.inf
            next_column = 0
            for candidate in range(1, column_count + 1):
                if used[candidate]:
                    continue
                reduced = costs[current_row - 1, candidate - 1] - u[current_row] - v[candidate]
                if reduced < minimum[candidate]:
                    minimum[candidate] = reduced
                    way[candidate] = column
                if minimum[candidate] < delta:
                    delta = minimum[candidate]
                    next_column = candidate
            for candidate in range(column_count + 1):
                if used[candidate]:
                    u[p[candidate]] += delta
                    v[candidate] -= delta
                else:
                    minimum[candidate] -= delta
            column = next_column
            if p[column] == 0:
                break
        while True:
            previous = way[column]
            p[column] = p[previous]
            column = previous
            if column == 0:
                break
    pairs = []
    for column in range(1, column_count + 1):
        if p[column] == 0:
            continue
        pair = (p[column] - 1, column - 1)
        pairs.append((pair[1], pair[0]) if transposed else pair)
    return pairs


def evaluate(
    step_path: Path,
    clean_ply: Path,
    result_dir: Path,
    input_scale: float,
    distance_tolerance: float,
    iou_threshold: float,
) -> dict[str, object]:
    faces = parse_step_planar_faces(step_path)
    groups = group_coplanar_faces(faces)
    clean_points = load_ascii_ply_points(clean_ply, input_scale)
    truth_labels = label_points(clean_points, faces, groups, distance_tolerance)
    truth_counts = np.bincount(truth_labels[truth_labels >= 0], minlength=len(faces))
    unmatched_truth_points = int(np.count_nonzero(truth_labels < 0))
    del clean_points, truth_labels

    patch_sizes, predicted_planes = load_patch_parameters(result_dir / "plane_parameters.csv")
    predicted_points = load_ascii_ply_points(result_dir / "edge_aware_plane_patches.ply", 1.0)
    if sum(patch_sizes) != predicted_points.shape[0]:
        raise ValueError("plane_parameters.csv dense counts do not match the result PLY")
    predicted_truth = label_points(predicted_points, faces, groups, distance_tolerance)
    prediction_labels = np.repeat(np.arange(len(patch_sizes), dtype=np.int32), patch_sizes)
    valid = predicted_truth >= 0
    overlap = np.bincount(
        prediction_labels[valid] * len(faces) + predicted_truth[valid],
        minlength=len(patch_sizes) * len(faces),
    ).reshape(len(patch_sizes), len(faces))
    prediction_counts = np.asarray(patch_sizes, dtype=np.int64)
    union = prediction_counts[:, None] + truth_counts[None, :] - overlap
    iou = np.divide(overlap, union, out=np.zeros_like(overlap, dtype=np.float64), where=union > 0)
    assignment = hungarian_maximize(iou)
    accepted = [(prediction, truth) for prediction, truth in assignment if iou[prediction, truth] >= iou_threshold]
    precision = len(accepted) / len(patch_sizes) if patch_sizes else 0.0
    recall = len(accepted) / len(faces) if faces else 0.0

    prediction_purity = np.divide(
        overlap,
        prediction_counts[:, None],
        out=np.zeros_like(overlap, dtype=np.float64),
        where=prediction_counts[:, None] > 0,
    )
    truth_coverage = np.divide(
        overlap,
        truth_counts[None, :],
        out=np.zeros_like(overlap, dtype=np.float64),
        where=truth_counts[None, :] > 0,
    )
    fragments_per_truth = np.count_nonzero(prediction_purity >= 0.5, axis=0)
    faces_per_prediction = np.count_nonzero(truth_coverage >= 0.5, axis=1)
    matched_ious = [float(iou[prediction, truth]) for prediction, truth in accepted]
    normal_errors = []
    offset_errors = []
    residual_errors = []
    for prediction, truth in accepted:
        predicted_normal, predicted_offset = predicted_planes[prediction]
        truth_normal = faces[truth].normal
        dot = max(-1.0, min(1.0, sum(predicted_normal[i] * truth_normal[i] for i in range(3))))
        sign = 1.0 if dot >= 0.0 else -1.0
        normal_errors.append(math.degrees(math.acos(abs(dot))))
        offset_errors.append(abs(predicted_offset - sign * faces[truth].offset))
        residual_errors.append(
            abs(sum(predicted_normal[i] * faces[truth].centroid[i] for i in range(3)) + predicted_offset)
        )
    matched_truth = {truth for _, truth in accepted}
    unmatched_truth = [
        {
            "truth_face": truth,
            "clean_points": int(truth_counts[truth]),
            "best_iou": float(np.max(iou[:, truth])) if len(patch_sizes) else 0.0,
        }
        for truth in range(len(faces))
        if truth not in matched_truth
    ]
    return {
        "step_planar_faces": len(faces),
        "step_unique_plane_groups": len(groups),
        "predicted_patches": len(patch_sizes),
        "matched_instances": len(accepted),
        "plane_precision": precision,
        "plane_recall": recall,
        "plane_f1": 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "instance_macro_iou": sum(matched_ious) / len(faces) if faces else 0.0,
        "matched_instance_mean_iou": sum(matched_ious) / len(matched_ious) if matched_ious else 0.0,
        "normal_mean_error_deg": sum(normal_errors) / len(normal_errors) if normal_errors else None,
        "normal_max_error_deg": max(normal_errors) if normal_errors else None,
        "offset_mean_error": sum(offset_errors) / len(offset_errors) if offset_errors else None,
        "offset_max_error": max(offset_errors) if offset_errors else None,
        "face_centroid_residual_mean": sum(residual_errors) / len(residual_errors) if residual_errors else None,
        "face_centroid_residual_max": max(residual_errors) if residual_errors else None,
        "oversegmented_truth_instances": int(np.count_nonzero(fragments_per_truth > 1)),
        "undersegmented_predictions": int(np.count_nonzero(faces_per_prediction > 1)),
        "unmatched_clean_points": unmatched_truth_points,
        "unmatched_predicted_points": int(np.count_nonzero(~valid)),
        "unmatched_truth_instances": unmatched_truth,
        "iou_threshold": iou_threshold,
        "distance_tolerance": distance_tolerance,
        "matches": [
            {
                "prediction": int(prediction),
                "truth_face": int(truth),
                "iou": float(iou[prediction, truth]),
            }
            for prediction, truth in accepted
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", type=Path)
    parser.add_argument("clean_ply", type=Path)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--input-scale", type=float, default=1.0)
    parser.add_argument("--distance", type=float, default=0.75)
    parser.add_argument("--iou", type=float, default=0.5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = evaluate(
        args.step,
        args.clean_ply,
        args.result_dir,
        args.input_scale,
        args.distance,
        args.iou,
    )
    payload = json.dumps(result, indent=2) + "\n"
    if args.json:
        args.json.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()
