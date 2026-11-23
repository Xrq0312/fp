from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data_phase_unwrap" / "data2" / "RAW"
OUT = ROOT / "phase_unwrap_task" / "results_data2_simple_trend_k"
TWO_PI = 2.0 * np.pi

# Existing hand-labelled absolute fringe orders.  They are used only to fit
# one plane; there is no quality-guided unwrap or connected-component merge.
ANCHORS = [
    (4110, 1493, 383, 2099, 3),
    (3472, 1405, 566, 2143, 4),
    (2849, 1427, 566, 2165, 5),
    (2247, 1434, 551, 2150, 6),
    (1631, 1361, 537, 940, 7),
    (1044, 1310, 427, 155, 8),
    (406, 1302, 456, 133, 9),
]
ROI = (100, 1200, 4701, 2401)
ROW_Y = 1410


def decode_phase() -> tuple[np.ndarray, np.ndarray]:
    frames = [np.asarray(Image.open(RAW / f"{i}.jpg").convert("L"), dtype=np.float32)
              for i in range(1, 5)]
    c = (frames[0] - frames[2]) * 0.5
    s = (frames[1] - frames[3]) * 0.5
    # Matches data2_current.cfg: invert=true and offset=pi/2.
    phase = -np.arctan2(s, c) + np.pi / 2.0
    phase = (phase + np.pi) % TWO_PI - np.pi
    modulation = 0.5 * np.sqrt(c * c + s * s)
    return phase.astype(np.float32), modulation.astype(np.float32)


def fit_plane(wrapped: np.ndarray, modulation: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = wrapped.shape
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    scale = float(max(width, height))
    lhs = np.zeros((3, 3), dtype=np.float64)
    rhs = np.zeros(3, dtype=np.float64)
    count = 0
    limit = 0.6 * np.pi
    for x, y, w, h, k in ANCHORS:
        yy, xx = np.mgrid[y:y + h, x:x + w]
        phi = wrapped[y:y + h, x:x + w]
        mod = modulation[y:y + h, x:x + w]
        valid = (np.abs(phi) < limit) & (mod >= 4.0)
        basis = np.column_stack(((xx[valid] - cx) / scale,
                                 (yy[valid] - cy) / scale,
                                 np.ones(np.count_nonzero(valid))))
        target = phi[valid].astype(np.float64) + k * TWO_PI
        lhs += basis.T @ basis
        rhs += basis.T @ target
        count += target.size
    return np.linalg.solve(lhs, rhs), count


def fit_quadratic_surface(wrapped: np.ndarray, modulation: np.ndarray) -> tuple[np.ndarray, int]:
    height, width = wrapped.shape
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    scale = float(max(width, height))
    lhs = np.zeros((6, 6), dtype=np.float64)
    rhs = np.zeros(6, dtype=np.float64)
    count = 0
    limit = 0.6 * np.pi
    for x, y, w, h, k in ANCHORS:
        yy, xx = np.mgrid[y:y + h, x:x + w]
        phi = wrapped[y:y + h, x:x + w]
        mod = modulation[y:y + h, x:x + w]
        valid = (np.abs(phi) < limit) & (mod >= 4.0)
        xn = (xx[valid] - cx) / scale
        yn = (yy[valid] - cy) / scale
        basis = np.column_stack((xn*xn, xn*yn, yn*yn, xn, yn,
                                 np.ones(np.count_nonzero(valid))))
        target = phi[valid].astype(np.float64) + k * TWO_PI
        lhs += basis.T @ basis
        rhs += basis.T @ target
        count += target.size
    return np.linalg.solve(lhs, rhs), count


def evaluate_quadratic(coeff: np.ndarray, xx: np.ndarray, yy: np.ndarray,
                       cx: float, cy: float, scale: float) -> np.ndarray:
    xn = (xx - cx) / scale
    yn = (yy - cy) / scale
    return (coeff[0]*xn*xn + coeff[1]*xn*yn + coeff[2]*yn*yn +
            coeff[3]*xn + coeff[4]*yn + coeff[5]).astype(np.float32)


def percentile_preview(values: np.ndarray, low: float, high: float) -> Image.Image:
    lo, hi = np.percentile(values[np.isfinite(values)], [low, high])
    scaled = np.clip((values - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return Image.fromarray(np.round(scaled * 255.0).astype(np.uint8), mode="L")


def valid_roi_preview(values: np.ndarray, modulation: np.ndarray,
                      path: Path, crop: bool = False) -> None:
    x0, y0, w, h = ROI
    roi_values = values[y0:y0+h, x0:x0+w]
    roi_mod = modulation[y0:y0+h, x0:x0+w]
    valid = roi_mod >= 4.0
    lo, hi = np.percentile(roi_values[valid], [0.5, 99.5])
    scaled = np.clip((values - lo) / max(hi-lo, 1e-9), 0.0, 1.0)
    gray = np.round(scaled * 255.0).astype(np.uint8)
    gray[modulation < 4.0] = 0
    image = Image.fromarray(gray, mode="L")
    if crop:
        image = image.crop((x0, y0, x0+w, y0+h))
    image.save(path)


def draw_distribution(wrapped: np.ndarray, unwrapped: np.ndarray,
                      residual: np.ndarray, k_order: np.ndarray, path: Path) -> None:
    x0, y0, w, h = ROI
    step = 4
    arrays = [wrapped[y0:y0+h:step, x0:x0+w:step],
              unwrapped[y0:y0+h:step, x0:x0+w:step],
              residual[y0:y0+h:step, x0:x0+w:step]]
    titles = ["wrapped phase (rad)", "unwrapped = wrapped + 2pi*k (rad)",
              "residual = unwrapped - fitted trend (rad)"]
    canvas = Image.new("RGB", (1500, 900), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    colors = [(45, 95, 190), (25, 145, 90), (205, 75, 45)]
    for index, (values, title, color) in enumerate(zip(arrays, titles, colors)):
        left, right = 80, 1420
        top = 45 + index * 245
        bottom = top + 180
        counts, edges = np.histogram(values.ravel(), bins=180)
        counts = np.log1p(counts)
        points = []
        for i, count in enumerate(counts):
            px = left + int(i * (right - left) / max(len(counts) - 1, 1))
            py = bottom - int(count * (bottom - top) / max(float(counts.max()), 1.0))
            points.append((px, py))
        draw.rectangle((left, top, right, bottom), outline=(80, 80, 80))
        draw.line(points, fill=color, width=3)
        draw.text((left, top - 22), title, fill="black", font=font)
        draw.text((left, bottom + 4), f"{edges[0]:.3f}", fill="black", font=font)
        draw.text((right - 70, bottom + 4), f"{edges[-1]:.3f}", fill="black", font=font)

    values, counts = np.unique(k_order[y0:y0+h, x0:x0+w], return_counts=True)
    lines = ["K distribution in ROI:"]
    lines.extend(f"k={int(v):2d}: {c:9d} ({100*c/counts.sum():5.2f}%)"
                 for v, c in zip(values, counts))
    draw.multiline_text((80, 790), "    ".join(lines), fill="black", font=font, spacing=4)
    canvas.save(path)


def draw_row_profile(wrapped: np.ndarray, trend: np.ndarray,
                     modulation: np.ndarray, y: int, path: Path) -> None:
    width = wrapped.shape[1]
    model = (trend[y] + np.pi) % TWO_PI - np.pi
    residual = (wrapped[y] - model + np.pi) % TWO_PI - np.pi
    series = [
        (wrapped[y], model, -np.pi, np.pi, "wrapped phase: measured (blue), fitted trend (red)"),
        (modulation[y], None, 0.0, max(4.5, float(np.percentile(modulation[y], 99.9))),
         "four-step modulation; red line = validity threshold 4"),
        (residual, None, -np.pi, np.pi, "circular residual: measured - fitted trend"),
    ]
    canvas = Image.new("RGB", (1500, 850), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (primary, secondary, lo, hi, title) in enumerate(series):
        left, right = 80, 1430
        top, bottom = 50 + index * 255, 235 + index * 255
        draw.rectangle((left, top, right, bottom), outline=(80, 80, 80))
        def points(values):
            return [(left + int(x * (right-left) / (width-1)),
                     bottom - int(np.clip((v-lo) / (hi-lo), 0, 1) * (bottom-top)))
                    for x, v in enumerate(values)]
        draw.line(points(primary), fill=(35, 90, 205), width=2)
        if secondary is not None:
            draw.line(points(secondary), fill=(210, 65, 40), width=2)
        if index == 1:
            threshold_y = bottom - int((4.0-lo) / (hi-lo) * (bottom-top))
            draw.line((left, threshold_y, right, threshold_y), fill=(210, 65, 40), width=2)
        draw.text((left, top-22), title, fill="black", font=font)
        draw.text((left, bottom+4), "x=0", fill="black", font=font)
        draw.text((right-55, bottom+4), f"x={width-1}", fill="black", font=font)
    draw.text((80, 815), f"data2 RAW decoded row y={y}", fill="black", font=font)
    canvas.save(path)


def draw_marked_wrapped_phase(wrapped: np.ndarray, y: int, path: Path) -> None:
    gray = np.clip((wrapped + np.pi) / TWO_PI, 0.0, 1.0)
    image = Image.fromarray(np.round(gray * 255.0).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    line_width = 7
    draw.line((0, y, image.width - 1, y), fill=(255, 0, 0), width=line_width)
    label = f"y={y}"
    box = (18, max(0, y - 48), 155, y - 10)
    draw.rectangle(box, fill=(255, 255, 0), outline=(255, 0, 0), width=3)
    draw.text((30, max(0, y - 42)), label, fill=(0, 0, 0), font=ImageFont.load_default())
    image.save(path)


def stats(values: np.ndarray) -> str:
    p = np.percentile(values, [0.1, 1, 5, 50, 95, 99, 99.9])
    return (f"mean={values.mean():.6f}\nstd={values.std():.6f}\n"
            f"rms={np.sqrt(np.mean(values * values)):.6f}\n"
            f"p0.1,p1,p5,p50,p95,p99,p99.9=" + ",".join(f"{v:.6f}" for v in p))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wrapped, modulation = decode_phase()
    coeff, sample_count = fit_plane(wrapped, modulation)
    quadratic_coeff, quadratic_sample_count = fit_quadratic_surface(wrapped, modulation)
    height, width = wrapped.shape
    cx = (width - 1) * 0.5
    cy = (height - 1) * 0.5
    scale = float(max(width, height))
    yy, xx = np.mgrid[0:height, 0:width]
    trend = (coeff[0] * (xx - cx) / scale + coeff[1] * (yy - cy) / scale + coeff[2]).astype(np.float32)
    quadratic_trend = evaluate_quadratic(quadratic_coeff, xx, yy, cx, cy, scale)
    k_order = np.rint((trend - wrapped) / TWO_PI).astype(np.int16)
    unwrapped = wrapped + k_order.astype(np.float32) * TWO_PI
    residual = unwrapped - trend
    quadratic_k = np.rint((quadratic_trend - wrapped) / TWO_PI).astype(np.int16)
    quadratic_unwrapped = wrapped + quadratic_k.astype(np.float32) * TWO_PI
    quadratic_residual = quadratic_unwrapped - quadratic_trend

    Image.fromarray(unwrapped.astype(np.float32), mode="F").save(OUT / "unwrapped_simple.tiff")
    Image.fromarray(residual.astype(np.float32), mode="F").save(OUT / "residual_simple.tiff")
    Image.fromarray(k_order.astype(np.float32), mode="F").save(OUT / "k_order_simple.tiff")
    percentile_preview(unwrapped, 0.5, 99.5).save(OUT / "unwrapped_simple.png")
    valid_roi_preview(unwrapped, modulation, OUT / "unwrapped_simple_valid.png")
    valid_roi_preview(unwrapped, modulation, OUT / "unwrapped_simple_valid_roi.png", crop=True)
    fixed = np.clip((residual + np.pi) / TWO_PI, 0.0, 1.0)
    Image.fromarray(np.round(fixed * 255).astype(np.uint8), mode="L").save(OUT / "residual_simple.png")
    percentile_preview(k_order.astype(np.float32), 0, 100).save(OUT / "k_order_simple.png")
    Image.fromarray(quadratic_unwrapped.astype(np.float32), mode="F").save(OUT / "unwrapped_quadratic.tiff")
    Image.fromarray(quadratic_residual.astype(np.float32), mode="F").save(OUT / "residual_quadratic.tiff")
    Image.fromarray(quadratic_k.astype(np.float32), mode="F").save(OUT / "k_order_quadratic.tiff")
    valid_roi_preview(quadratic_unwrapped, modulation, OUT / "unwrapped_quadratic_valid.png")
    valid_roi_preview(quadratic_unwrapped, modulation, OUT / "unwrapped_quadratic_valid_roi.png", crop=True)
    quadratic_fixed = np.clip((quadratic_residual + np.pi) / TWO_PI, 0.0, 1.0)
    Image.fromarray(np.round(quadratic_fixed * 255).astype(np.uint8), mode="L").save(
        OUT / "residual_quadratic.png")
    draw_distribution(wrapped, unwrapped, residual, k_order, OUT / "phase_distribution.png")
    draw_row_profile(wrapped, trend, modulation, ROW_Y, OUT / "wrapped_phase_row_y1410.png")
    draw_row_profile(wrapped, quadratic_trend, modulation, ROW_Y,
                     OUT / "wrapped_phase_row_y1410_quadratic.png")
    draw_marked_wrapped_phase(wrapped, ROW_Y, OUT / "wrapped_phase_y1410_marked.png")
    row_model = (trend[ROW_Y] + np.pi) % TWO_PI - np.pi
    row_residual = (wrapped[ROW_Y] - row_model + np.pi) % TWO_PI - np.pi
    np.savetxt(
        OUT / "wrapped_phase_row_y1410.csv",
        np.column_stack((np.arange(width), wrapped[ROW_Y], row_model,
                         row_residual, modulation[ROW_Y])),
        delimiter=",",
        header="x,wrapped_phase_rad,fitted_trend_wrapped_rad,circular_residual_rad,modulation",
        comments="",
    )

    x0, y0, w, h = ROI
    roi_residual = residual[y0:y0+h, x0:x0+w]
    roi_quadratic_residual = quadratic_residual[y0:y0+h, x0:x0+w]
    roi_valid = modulation[y0:y0+h, x0:x0+w] >= 4.0
    dx = np.abs(np.diff(roi_residual, axis=1))
    dy = np.abs(np.diff(roi_residual, axis=0))
    report = (
        "strategy=fit one plane from labelled K anchors; per-pixel k=round((trend-wrapped)/2pi)\n"
        "quality_guided_unwrap=false\nconnected_component_merge=false\n"
        f"width={width}\nheight={height}\nfit_samples={sample_count}\n"
        f"normalized_plane_coeff={coeff[0]:.9f},{coeff[1]:.9f},{coeff[2]:.9f}\n"
        f"quadratic_fit_samples={quadratic_sample_count}\n"
        "normalized_quadratic_coeff_x2,xy,y2,x,y,c=" +
        ",".join(f"{v:.9f}" for v in quadratic_coeff) + "\n"
        "\n[residual_full]\n" + stats(residual) +
        "\n\n[residual_roi]\n" + stats(roi_residual) +
        "\n\n[residual_roi_valid_linear]\n" + stats(roi_residual[roi_valid]) +
        "\n\n[residual_roi_valid_quadratic]\n" + stats(roi_quadratic_residual[roi_valid]) +
        f"\nneighbor_jump_gt_pi_x={np.count_nonzero(dx > np.pi)}"
        f"\nneighbor_jump_gt_pi_y={np.count_nonzero(dy > np.pi)}"
        f"\nneighbor_jump_gt_pi_ratio={((np.count_nonzero(dx > np.pi) + np.count_nonzero(dy > np.pi)) / (dx.size + dy.size)):.9f}\n"
        f"\n[row_y1410]\nmodulation_min={modulation[ROW_Y].min():.6f}"
        f"\nmodulation_median={np.median(modulation[ROW_Y]):.6f}"
        f"\nmodulation_max={modulation[ROW_Y].max():.6f}"
        f"\nvalid_modulation_ge_4={np.count_nonzero(modulation[ROW_Y] >= 4.0)}"
        f"\ncircular_residual_rms_all={np.sqrt(np.mean(row_residual * row_residual)):.6f}"
        f"\ncircular_residual_rms_valid={np.sqrt(np.mean(row_residual[modulation[ROW_Y] >= 4.0] ** 2)):.6f}\n"
    )
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
