#!/usr/bin/env python3
"""Apply a saved registration matrix to every vertex of an ASCII PLY."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def read_header(path: Path) -> tuple[int, int, int]:
    vertex_count = 0
    property_count = 0
    header_bytes = 0
    in_vertices = False
    with path.open("rb") as stream:
        while True:
            raw = stream.readline()
            if not raw:
                raise ValueError(f"{path}: incomplete PLY header")
            header_bytes += len(raw)
            line = raw.decode("ascii").strip()
            if line == "format ascii 1.0":
                pass
            elif line.startswith("format "):
                raise ValueError(f"{path}: input must be an ASCII PLY")
            if line.startswith("element vertex "):
                vertex_count = int(line.split()[-1])
                in_vertices = True
            elif line.startswith("element "):
                in_vertices = False
            elif in_vertices and line.startswith("property "):
                property_count += 1
            if line == "end_header":
                break
    if vertex_count <= 0 or property_count < 3:
        raise ValueError(f"{path}: invalid vertex declaration")
    return vertex_count, property_count, header_bytes


def apply_full_cloud(
    input_path: Path,
    registration_path: Path,
    output_path: Path,
    block_lines: int,
) -> None:
    vertex_count, property_count, header_bytes = read_header(input_path)
    registration = json.loads(registration_path.read_text(encoding="utf-8"))
    matrix = np.asarray(
        registration["homogeneous_transform_model_native_to_measured_mm"],
        dtype=np.float64,
    )
    linear = matrix[:3, :3]
    translation = matrix[:3, 3]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with input_path.open("rb") as source, output_path.open("wb") as target:
        source.seek(header_bytes)
        header = (
            "ply\n"
            "format binary_little_endian 1.0\n"
            "comment Full model transformed into measured point-cloud coordinates\n"
            f"comment Source {input_path.as_posix()}\n"
            f"comment Registration {registration_path.as_posix()}\n"
            f"element vertex {vertex_count}\n"
            "property float x\n"
            "property float y\n"
            "property float z\n"
            "end_header\n"
        )
        target.write(header.encode("ascii"))

        while written < vertex_count:
            lines = list(
                itertools.islice(source, min(block_lines, vertex_count - written))
            )
            if not lines:
                break
            values = np.fromstring(b"".join(lines).decode("ascii"), sep=" ")
            if values.size != len(lines) * property_count:
                raise ValueError(
                    f"{input_path}: malformed vertex block near vertex {written}"
                )
            xyz = values.reshape(-1, property_count)[:, :3]
            transformed = xyz @ linear.T + translation
            target.write(np.asarray(transformed, dtype="<f4").tobytes())
            written += len(lines)
            print(f"\r{output_path.name}: {written:,}/{vertex_count:,}", end="", flush=True)

    print()
    if written != vertex_count:
        raise ValueError(f"{input_path}: expected {vertex_count} vertices, wrote {written}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("registration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--block-lines", type=int, default=100_000)
    args = parser.parse_args()
    apply_full_cloud(args.input, args.registration, args.output, args.block_lines)


if __name__ == "__main__":
    main()
