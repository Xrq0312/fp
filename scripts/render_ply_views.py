#!/usr/bin/env python3
import argparse
import struct
import math
import random
import zlib
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw
except ModuleNotFoundError:
    Image = None
    ImageDraw = None


Point = Tuple[float, float, float, Tuple[int, int, int]]


class RasterImage:
    """Small dependency-free RGB canvas used when Pillow is unavailable."""

    def __init__(self, width: int, height: int, color: Tuple[int, int, int]):
        self.width = width
        self.height = height
        self.pixels = bytearray(color * (width * height))

    def putpixel(self, point: Tuple[int, int], color: Tuple[int, int, int]) -> None:
        x, y = point
        if not (0 <= x < self.width and 0 <= y < self.height):
            return
        offset = 3 * (y * self.width + x)
        self.pixels[offset : offset + 3] = bytes(color)

    @staticmethod
    def _chunk(name: bytes, payload: bytes) -> bytes:
        body = name + payload
        return struct.pack(">I", len(payload)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    def save(self, path: Path) -> None:
        rows = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            rows.append(0)
            rows.extend(self.pixels[y * stride : (y + 1) * stride])
        header = struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)
        payload = b"\x89PNG\r\n\x1a\n"
        payload += self._chunk(b"IHDR", header)
        payload += self._chunk(b"IDAT", zlib.compress(bytes(rows), 6))
        payload += self._chunk(b"IEND", b"")
        path.write_bytes(payload)


class RasterDraw:
    def __init__(self, image: RasterImage):
        self.image = image

    def text(self, _point, _text, fill=None) -> None:
        # Labels are cosmetic; keeping the raster fallback dependency-free is
        # more useful than embedding a bitmap font.
        return

    def ellipse(self, box, fill=None, outline=None, width=1) -> None:
        x0, y0, x1, y1 = map(int, box)
        cx = 0.5 * (x0 + x1)
        cy = 0.5 * (y0 + y1)
        rx = max(0.5, 0.5 * abs(x1 - x0))
        ry = max(0.5, 0.5 * abs(y1 - y0))
        color = fill if fill is not None else outline
        if color is None:
            return
        inner_rx = max(0.0, rx - width)
        inner_ry = max(0.0, ry - width)
        for y in range(max(0, y0), min(self.image.height - 1, y1) + 1):
            for x in range(max(0, x0), min(self.image.width - 1, x1) + 1):
                outer = ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0
                if not outer:
                    continue
                if fill is None and inner_rx > 0.0 and inner_ry > 0.0:
                    inner = ((x - cx) / inner_rx) ** 2 + ((y - cy) / inner_ry) ** 2 < 1.0
                    if inner:
                        continue
                self.image.putpixel((x, y), color)

    def line(self, points, fill, width=1) -> None:
        x0, y0, x1, y1 = map(int, points)
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        error = dx + dy
        radius = max(0, width // 2)
        while True:
            for oy in range(-radius, radius + 1):
                for ox in range(-radius, radius + 1):
                    self.image.putpixel((x0 + ox, y0 + oy), fill)
            if x0 == x1 and y0 == y1:
                break
            twice = 2 * error
            if twice >= dy:
                error += dy
                x0 += sx
            if twice <= dx:
                error += dx
                y0 += sy


def parse_header(path: Path):
    with path.open("rb") as fh:
        header = []
        while True:
            line = fh.readline()
            if not line:
                raise RuntimeError(f"invalid PLY header: {path}")
            header.append(line)
            if line.strip() == b"end_header":
                break
        offset = fh.tell()

    fmt = "ascii"
    vertex_count = 0
    props: List[Tuple[str, str]] = []
    in_vertex = False
    for raw in header:
        line = raw.decode("ascii", errors="ignore").strip()
        parts = line.split()
        if len(parts) >= 3 and parts[0] == "format":
            fmt = parts[1]
        elif len(parts) >= 3 and parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and len(parts) >= 3 and parts[0] == "property":
            props.append((parts[1], parts[2]))
    return fmt, vertex_count, props, offset


def sample_ascii(
    path: Path,
    count: int,
    sample_limit: int,
    scale: float,
    seed: int,
    crop: Optional[Tuple[float, float, float, float, float, float]],
) -> List[Point]:
    rng = random.Random(seed)
    sample: List[Point] = []
    accepted = 0
    with path.open("r", encoding="ascii", errors="ignore") as fh:
        for line in fh:
            if line.strip() == "end_header":
                break
        for line in fh:
            parts = line.split()
            if len(parts) < 3:
                continue
            x = float(parts[0]) * scale
            y = float(parts[1]) * scale
            z = float(parts[2]) * scale
            if crop is not None and not (
                crop[0] <= x <= crop[1]
                and crop[2] <= y <= crop[3]
                and crop[4] <= z <= crop[5]
            ):
                continue
            color = (210, 210, 210)
            if len(parts) >= 6:
                color = tuple(max(0, min(255, int(float(v)))) for v in parts[3:6])
            point = (x, y, z, color)
            if len(sample) < sample_limit:
                sample.append(point)
            else:
                j = rng.randrange(accepted + 1)
                if j < sample_limit:
                    sample[j] = point
            accepted += 1
    return sample


def load_points(
    path: Path,
    sample_limit: int,
    scale: float,
    seed: int,
    crop: Optional[Tuple[float, float, float, float, float, float]],
) -> List[Point]:
    fmt, vertex_count, _props, _offset = parse_header(path)
    if fmt != "ascii":
        raise RuntimeError(f"{path} is {fmt}; this lightweight renderer expects ascii PLY")
    return sample_ascii(path, vertex_count, sample_limit, scale, seed, crop)


def bounds(points: Iterable[Point]):
    xs, ys, zs = [], [], []
    for x, y, z, _c in points:
        xs.append(x)
        ys.append(y)
        zs.append(z)
    return (min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))


def render_projection(
    points: List[Point],
    out_path: Path,
    axes: Tuple[int, int],
    size: int,
    title: str,
    point_radius: int,
    use_rgb: bool,
    marks: List[Tuple[float, float, float]],
    mark_radius: int,
):
    b = bounds(points)
    ax0, ax1 = axes
    min0, max0 = b[ax0]
    min1, max1 = b[ax1]
    span0 = max(max0 - min0, 1e-9)
    span1 = max(max1 - min1, 1e-9)

    margin = 44
    if Image is not None and ImageDraw is not None:
        img = Image.new("RGB", (size, size), (18, 20, 22))
        draw = ImageDraw.Draw(img)
    else:
        img = RasterImage(size, size, (18, 20, 22))
        draw = RasterDraw(img)

    usable = size - margin * 2
    scale = min(usable / span0, usable / span1)
    offset0 = (size - span0 * scale) * 0.5
    offset1 = (size - span1 * scale) * 0.5

    label = f"{title}  {('xyz'[ax0]).upper()}{('xyz'[ax1]).upper()}"
    draw.text((14, 12), label, fill=(235, 235, 235))
    draw.text((14, size - 28), f"{span0:.3f} x {span1:.3f}", fill=(170, 170, 170))

    for p in points:
        vals = p[:3]
        px = int((vals[ax0] - min0) * scale + offset0)
        py = int(size - ((vals[ax1] - min1) * scale + offset1))
        color = p[3] if use_rgb else (185, 195, 205)
        if point_radius <= 1:
            img.putpixel((max(0, min(size - 1, px)), max(0, min(size - 1, py))), color)
        else:
            draw.ellipse((px - point_radius, py - point_radius, px + point_radius, py + point_radius), fill=color)

    for mark in marks:
        px = int((mark[ax0] - min0) * scale + offset0)
        py = int(size - ((mark[ax1] - min1) * scale + offset1))
        draw.ellipse(
            (px - mark_radius, py - mark_radius, px + mark_radius, py + mark_radius),
            outline=(255, 45, 45),
            width=max(2, mark_radius // 3),
        )
        draw.line((px - mark_radius, py, px + mark_radius, py), fill=(255, 45, 45), width=2)
        draw.line((px, py - mark_radius, px, py + mark_radius), fill=(255, 45, 45), width=2)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ply", type=Path)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--sample-limit", type=int, default=250000)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--size", type=int, default=1800)
    parser.add_argument("--rgb", action="store_true")
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--crop", type=float, nargs=6, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument("--mark", type=float, nargs=3, action="append", default=[], metavar=("X", "Y", "Z"))
    parser.add_argument("--overlay-ply", type=Path)
    parser.add_argument("--overlay-sample-limit", type=int, default=50000)
    parser.add_argument("--mark-radius", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    crop = tuple(args.crop) if args.crop is not None else None
    points = load_points(args.ply, args.sample_limit, args.scale, args.seed, crop)
    if not points:
        raise RuntimeError("no points loaded")

    marks = [tuple(mark) for mark in args.mark]
    if args.overlay_ply is not None:
        overlay = load_points(
            args.overlay_ply,
            args.overlay_sample_limit,
            args.scale,
            args.seed,
            crop,
        )
        marks.extend((point[0], point[1], point[2]) for point in overlay)

    prefix = args.prefix or args.ply.stem
    for name, axes in [("xy", (0, 1)), ("xz", (0, 2)), ("yz", (1, 2))]:
        render_projection(
            points,
            args.out_dir / f"{prefix}_{name}.png",
            axes,
            args.size,
            prefix,
            args.point_radius,
            args.rgb,
            marks,
            args.mark_radius,
        )


if __name__ == "__main__":
    main()
