#!/usr/bin/env python3
"""Register a meter-scale model PLY to a millimeter-scale measured PLY.

The implementation deliberately only depends on NumPy/SciPy/Matplotlib.  It
uses deterministic uniform sampling, PCA-based global hypotheses, and trimmed
point-to-point ICP so that a partially observed measured cloud can be handled.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy.spatial import cKDTree


def ply_header(path: Path) -> tuple[int, int, str, list[tuple[str, str]]]:
    vertex_count = 0
    header_bytes = 0
    file_format = ""
    properties: list[tuple[str, str]] = []
    in_vertices = False
    with path.open("rb") as stream:
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: incomplete PLY header")
            header_bytes += len(line)
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("format "):
                file_format = text.split()[1]
            if text.startswith("element vertex "):
                vertex_count = int(text.split()[-1])
                in_vertices = True
            elif text.startswith("element "):
                in_vertices = False
            elif in_vertices and text.startswith("property "):
                fields = text.split()
                if len(fields) == 3:
                    properties.append((fields[2], fields[1]))
            if text == "end_header":
                break
    if vertex_count <= 0:
        raise ValueError(f"{path}: missing vertex count")
    return vertex_count, header_bytes, file_format, properties


def load_uniform_sample(path: Path, limit: int) -> tuple[np.ndarray, int]:
    count, header_bytes, file_format, properties = ply_header(path)
    stride = max(1, math.ceil(count / limit))
    if file_format == "binary_little_endian":
        type_map = {
            "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
            "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
            "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
            "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
        }
        dtype = np.dtype([(name, type_map[data_type]) for name, data_type in properties])
        mapped = np.memmap(path, mode="r", offset=header_bytes, dtype=dtype, shape=(count,))
        selected = mapped[::stride]
        cloud = np.column_stack((selected["x"], selected["y"], selected["z"])).astype(
            np.float64, copy=False
        )
        cloud = np.asarray(cloud)
        return cloud[np.isfinite(cloud).all(axis=1)], count
    if file_format != "ascii":
        raise ValueError(f"{path}: unsupported PLY format {file_format!r}")
    points: list[tuple[float, float, float]] = []
    with path.open("rb") as stream:
        stream.seek(header_bytes)
        for index, raw in enumerate(stream):
            if index >= count:
                break
            if index % stride:
                continue
            fields = raw.split(maxsplit=3)
            if len(fields) >= 3:
                points.append((float(fields[0]), float(fields[1]), float(fields[2])))
    cloud = np.asarray(points, dtype=np.float64)
    cloud = cloud[np.isfinite(cloud).all(axis=1)]
    return cloud, count


def robust_crop(points: np.ndarray, tail_percent: float = 0.05) -> np.ndarray:
    lo = np.percentile(points, tail_percent, axis=0)
    hi = np.percentile(points, 100.0 - tail_percent, axis=0)
    keep = np.logical_and(points >= lo, points <= hi).all(axis=1)
    return points[keep]


def voxel_sample(points: np.ndarray, leaf: float) -> np.ndarray:
    keys = np.floor(points / leaf).astype(np.int64)
    _, indices = np.unique(keys, axis=0, return_index=True)
    return points[np.sort(indices)]


def pca_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.median(points, axis=0)
    centered = points - center
    covariance = centered.T @ centered / len(centered)
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    return center, vectors[:, order], values[order]


def proper_signed_permutations() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    from itertools import permutations, product

    for perm in permutations(range(3)):
        base = np.eye(3)[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            candidate = base @ np.diag(signs)
            if np.linalg.det(candidate) > 0.5:
                matrices.append(candidate)
    return matrices


def transform(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    return scale * (points @ rotation.T) + translation


def trimmed_score(
    source_transformed: np.ndarray,
    target: np.ndarray,
    target_tree: cKDTree,
    trim: float = 0.72,
) -> float:
    distances = target_tree.query(source_transformed, workers=-1)[0]
    cutoff = np.quantile(distances, trim)
    a = distances[distances <= cutoff]
    # Reverse distance prevents a small matching fragment from winning.
    reverse = cKDTree(source_transformed).query(target, workers=-1)[0]
    reverse_cutoff = np.quantile(reverse, trim)
    b = reverse[reverse <= reverse_cutoff]
    return float(0.5 * (np.sqrt(np.mean(a * a)) + np.sqrt(np.mean(b * b))))


def rigid_fit(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    covariance = (source - source_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def icp(
    source_original: np.ndarray,
    target: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    max_distance: float,
    iterations: int,
    trim: float,
) -> tuple[float, np.ndarray, np.ndarray, list[dict[str, float]]]:
    tree = cKDTree(target)
    history: list[dict[str, float]] = []
    previous = float("inf")
    for iteration in range(iterations):
        moved = transform(source_original, scale, rotation, translation)
        distances, indices = tree.query(moved, workers=-1)
        eligible = distances < max_distance
        if eligible.sum() < 30:
            break
        cutoff = min(max_distance, float(np.quantile(distances[eligible], trim)))
        keep = distances <= cutoff
        delta_r, delta_t = rigid_fit(moved[keep], target[indices[keep]])
        rotation = delta_r @ rotation
        translation = translation @ delta_r.T + delta_t
        rmse = float(np.sqrt(np.mean(distances[keep] ** 2)))
        history.append({"iteration": iteration + 1, "rmse_mm": rmse, "pairs": int(keep.sum())})
        if abs(previous - rmse) < 1e-5:
            break
        previous = rmse
    return scale, rotation, translation, history


def estimate_scale(model: np.ndarray, measured: np.ndarray) -> float:
    _, _, model_values = pca_frame(model)
    _, _, measured_values = pca_frame(measured)
    ratios = np.sqrt(measured_values / np.maximum(model_values, 1e-18))
    raw = float(np.median(ratios))
    common = np.asarray([1.0, 10.0, 100.0, 1000.0, 0.1, 0.01, 0.001])
    return float(common[np.argmin(np.abs(np.log(common) - np.log(raw)))])


def save_ascii_ply(path: Path, points: np.ndarray, color: tuple[int, int, int]) -> None:
    with path.open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\n"
            "comment Generated by register_pointclouds.py\n"
            f"element vertex {len(points)}\n"
            "property float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        for point in points:
            stream.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def plot_comparison(path: Path, model: np.ndarray, measured: np.ndarray) -> None:
    panel = 800
    margin = 35
    canvas = Image.new("RGB", (panel * 3, panel), "white")
    draw = ImageDraw.Draw(canvas, "RGBA")
    views = ((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))
    take_model = model[:: max(1, len(model) // 50000)]
    take_measured = measured[:: max(1, len(measured) // 50000)]
    for panel_index, (x_axis, y_axis, title) in enumerate(views):
        both = np.vstack(
            (take_model[:, [x_axis, y_axis]], take_measured[:, [x_axis, y_axis]])
        )
        lo = np.percentile(both, 0.2, axis=0)
        hi = np.percentile(both, 99.8, axis=0)
        span = np.maximum(hi - lo, 1e-9)
        scale = (panel - 2 * margin) / max(span)
        origin_x = panel_index * panel + (panel - span[0] * scale) * 0.5
        origin_y = (panel - span[1] * scale) * 0.5

        def xy(point: np.ndarray) -> tuple[int, int]:
            return (
                int(origin_x + (point[x_axis] - lo[0]) * scale),
                int(panel - origin_y - (point[y_axis] - lo[1]) * scale),
            )

        for point in take_measured:
            x, y = xy(point)
            draw.point((x, y), fill=(22, 138, 173, 95))
        for point in take_model:
            x, y = xy(point)
            draw.point((x, y), fill=(230, 57, 70, 95))
        draw.text((panel_index * panel + 15, 12), title, fill=(20, 20, 20, 255))
    draw.rectangle((15, panel - 48, 29, panel - 34), fill=(230, 57, 70, 220))
    draw.text((35, panel - 50), "model", fill=(20, 20, 20, 255))
    draw.rectangle((100, panel - 48, 114, panel - 34), fill=(22, 138, 173, 220))
    draw.text((120, panel - 50), "measured", fill=(20, 20, 20, 255))
    canvas.save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path)
    parser.add_argument("measured", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("registration_results"))
    parser.add_argument("--sample-limit", type=int, default=500_000)
    parser.add_argument("--leaf-mm", type=float, default=0.8)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_raw, model_total = load_uniform_sample(args.model, args.sample_limit)
    measured_raw, measured_total = load_uniform_sample(args.measured, args.sample_limit)
    model_raw = robust_crop(model_raw)
    measured_raw = robust_crop(measured_raw)

    unit_scale = estimate_scale(model_raw, measured_raw)
    model_mm = model_raw * unit_scale
    model = voxel_sample(model_mm, args.leaf_mm)
    measured = voxel_sample(measured_raw, args.leaf_mm)

    model_center, model_axes, _ = pca_frame(model)
    measured_center, measured_axes, _ = pca_frame(measured)
    target_tree = cKDTree(measured)
    hypotheses: list[tuple[float, np.ndarray, np.ndarray]] = []
    for signed_perm in proper_signed_permutations():
        rotation = measured_axes @ signed_perm @ model_axes.T
        translation = measured_center - model_center @ rotation.T
        moved = transform(model_raw, unit_scale, rotation, translation)
        score = trimmed_score(moved[::4], measured[::2], cKDTree(measured[::2]))
        hypotheses.append((score, rotation, translation))
    hypotheses.sort(key=lambda item: item[0])

    best: tuple[float, float, np.ndarray, np.ndarray, list[dict[str, float]]] | None = None
    coarse_leaf = max(args.leaf_mm * 2.5, 2.0)
    coarse_model = voxel_sample(model_mm, coarse_leaf)
    coarse_measured = voxel_sample(measured_raw, coarse_leaf)
    for _, rotation, translation in hypotheses[:8]:
        result = icp(
            source_original=coarse_model / unit_scale,
            target=coarse_measured,
            scale=unit_scale,
            rotation=rotation,
            translation=translation,
            max_distance=12.0,
            iterations=45,
            trim=0.78,
        )
        scale, refined_r, refined_t, history = result
        moved = transform(coarse_model / unit_scale, scale, refined_r, refined_t)
        score = trimmed_score(moved, coarse_measured, cKDTree(coarse_measured), trim=0.75)
        if best is None or score < best[0]:
            best = (score, scale, refined_r, refined_t, history)

    assert best is not None
    _, scale, rotation, translation, coarse_history = best
    scale, rotation, translation, fine_history = icp(
        source_original=model / unit_scale,
        target=measured,
        scale=scale,
        rotation=rotation,
        translation=translation,
        max_distance=4.0,
        iterations=80,
        trim=0.82,
    )

    aligned = transform(model / unit_scale, scale, rotation, translation)
    distances = cKDTree(measured).query(aligned, workers=-1)[0]
    overlap = distances < 2.0
    overlap_1 = distances < 1.0
    metrics = {
        "model_file": str(args.model),
        "measured_file": str(args.measured),
        "model_total_vertices": model_total,
        "measured_total_vertices": measured_total,
        "model_sample_vertices": int(len(model)),
        "measured_sample_vertices": int(len(measured)),
        "model_to_measured_scale": scale,
        "rotation": rotation.tolist(),
        "translation_mm": translation.tolist(),
        "homogeneous_transform_model_native_to_measured_mm": [
            [*(scale * rotation)[0], translation[0]],
            [*(scale * rotation)[1], translation[1]],
            [*(scale * rotation)[2], translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        "model_nn_rmse_trimmed_90_mm": float(
            np.sqrt(np.mean(np.sort(distances)[: int(0.9 * len(distances))] ** 2))
        ),
        "model_fraction_within_1mm": float(overlap_1.mean()),
        "model_fraction_within_2mm": float(overlap.mean()),
        "coarse_icp_history": coarse_history,
        "fine_icp_history": fine_history,
    }
    (args.output_dir / "registration.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_ascii_ply(args.output_dir / "model_aligned_sample.ply", aligned, (230, 57, 70))
    save_ascii_ply(args.output_dir / "measured_sample.ply", measured, (22, 138, 173))
    combined = np.vstack((aligned, measured))
    colors = np.vstack(
        (
            np.tile(np.array([[230, 57, 70]]), (len(aligned), 1)),
            np.tile(np.array([[22, 138, 173]]), (len(measured), 1)),
        )
    )
    with (args.output_dir / "overlay_sample.ply").open("w", encoding="ascii", newline="\n") as stream:
        stream.write(
            "ply\nformat ascii 1.0\ncomment red=model, blue=measured\n"
            f"element vertex {len(combined)}\nproperty float x\nproperty float y\n"
            "property float z\nproperty uchar red\nproperty uchar green\n"
            "property uchar blue\nend_header\n"
        )
        for point, color in zip(combined, colors):
            stream.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )
    plot_comparison(args.output_dir / "overlay_views.png", aligned, measured)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
