from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_data2_simple_trend_k import decode_phase


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase_unwrap_task" / "results_data2_txt_method_roi"
TWO_PI = 2.0 * np.pi
# Half-open bounds, interpreted from the requested two corners.
X0, Y0, X1, Y1 = 100, 1200, 4701, 3648


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % TWO_PI - np.pi


def basis_grid(x: np.ndarray, y: np.ndarray, cx: float, cy: float, scale: float) -> np.ndarray:
    xn = (x - cx) / scale
    yn = (y - cy) / scale
    # Same term order as UnwrapPhaseWithROIPlaneCurveOpenCV in the supplied text.
    return np.column_stack((xn*xn, yn*yn, xn*yn, xn, yn, np.ones(xn.size)))


def infer_initial_plane(phase: np.ndarray, valid: np.ndarray) -> tuple[float, float, float]:
    # Wrapped first differences contain the carrier slope without requiring K.
    dx_ok = valid[:, 1:] & valid[:, :-1]
    dy_ok = valid[1:, :] & valid[:-1, :]
    dx = wrap_pi(phase[:, 1:] - phase[:, :-1])[dx_ok]
    dy = wrap_pi(phase[1:, :] - phase[:-1, :])[dy_ok]
    slope_x = float(np.median(dx))
    slope_y = float(np.median(dy))
    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    local_phase = phase[Y0:Y1, X0:X1]
    local_valid = valid[Y0:Y1, X0:X1]
    reduced = local_phase[local_valid] - slope_x*xx[local_valid] - slope_y*yy[local_valid]
    intercept = float(np.angle(np.mean(np.exp(1j*reduced))))
    return slope_x, slope_y, intercept


def robust_quadratic_fit(wrapped: np.ndarray, valid: np.ndarray,
                         initial: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = wrapped.shape
    cx = (X0 + X1 - 1) * 0.5
    cy = (Y0 + Y1 - 1) * 0.5
    scale = float(max(X1-X0, Y1-Y0))
    yy, xx = np.mgrid[0:height, 0:width]
    sx, sy, offset = initial
    trend = (sx*xx + sy*yy + offset).astype(np.float32)
    coeff = None
    # Two rounds: infer integer K from the current smooth trend, then apply the
    # supplied code's |wrapped| < 0.6*pi filter and SVD-equivalent least squares.
    # No absolute K label or anchor is used.
    for iteration in range(2):
        k = np.floor((trend - wrapped) / TWO_PI + 0.5).astype(np.int16)
        absolute = wrapped + k.astype(np.float32)*TWO_PI
        sample_valid = valid & (np.abs(wrapped) < 0.6*np.pi)
        sample_valid[:Y0] = False
        sample_valid[Y1:] = False
        sample_valid[:, :X0] = False
        sample_valid[:, X1:] = False
        # Sampling every other pixel is ample for a six-parameter surface.
        syy, sxx = np.where(sample_valid[::2, ::2])
        syy = syy*2
        sxx = sxx*2
        design = basis_grid(sxx.astype(np.float64), syy.astype(np.float64), cx, cy, scale)
        target = absolute[syy, sxx].astype(np.float64)
        coeff = np.linalg.lstsq(design, target, rcond=None)[0]
        xn = (xx-cx)/scale
        yn = (yy-cy)/scale
        trend = (coeff[0]*xn*xn + coeff[1]*yn*yn + coeff[2]*xn*yn +
                 coeff[3]*xn + coeff[4]*yn + coeff[5]).astype(np.float32)
    k = np.floor((trend-wrapped)/TWO_PI + 0.5).astype(np.int16)
    absolute = wrapped + k.astype(np.float32)*TWO_PI
    return coeff, trend, absolute


def save_gray(values: np.ndarray, path: Path, lo: float, hi: float) -> None:
    gray = np.round(np.clip((values-lo)/max(hi-lo, 1e-9), 0, 1)*255).astype(np.uint8)
    Image.fromarray(gray, mode="L").save(path)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wrapped, modulation = decode_phase()
    valid = modulation >= 4.0
    initial = infer_initial_plane(wrapped, valid)
    coeff, trend, absolute = robust_quadratic_fit(wrapped, valid, initial)
    # Exact residual formula used by the supplied implementation.
    residual = (absolute-trend).astype(np.float32)

    roi = np.zeros(wrapped.shape, dtype=bool)
    roi[Y0:Y1, X0:X1] = True
    combined_unwrapped = wrapped.copy()
    combined_unwrapped[roi] = absolute[roi]
    combined_residual = wrapped.copy()
    combined_residual[roi] = residual[roi]
    k_roi = np.zeros(wrapped.shape, dtype=np.int16)
    k_roi[roi] = np.rint((absolute[roi]-wrapped[roi])/TWO_PI).astype(np.int16)

    Image.fromarray(combined_unwrapped, mode="F").save(OUT / "phase_roi_unwrapped_outside_original.tiff")
    Image.fromarray(combined_residual, mode="F").save(OUT / "phase_roi_residual_outside_original.tiff")
    Image.fromarray(k_roi.astype(np.float32), mode="F").save(OUT / "k_inferred_roi.tiff")

    roi_valid = valid[Y0:Y1, X0:X1]
    roi_absolute = absolute[Y0:Y1, X0:X1]
    lo, hi = np.percentile(roi_absolute[roi_valid], [0.5, 99.5])
    save_gray(roi_absolute, OUT / "unwrapped_roi.png", lo, hi)
    save_gray(residual[Y0:Y1, X0:X1], OUT / "residual_roi.png", -np.pi, np.pi)
    roi_residual_normalized = residual[Y0:Y1, X0:X1]
    rlo, rhi = np.percentile(roi_residual_normalized[roi_valid], [0.5, 99.5])
    save_gray(roi_residual_normalized, OUT / "residual_roi_normalized.png", rlo, rhi)
    kb = np.clip(residual[Y0:Y1, X0:X1] * -200.0 + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(kb, mode="L").save(OUT / "residual_roi_k-200_b128.png")
    k_values = k_roi[Y0:Y1, X0:X1]
    save_gray(k_values, OUT / "k_inferred_roi.png", float(k_values.min()), float(k_values.max()))

    # Overview: preserve the original wrapped-phase visualization outside;
    # independently stretch the inferred absolute phase inside the ROI.
    overview = np.round((wrapped+np.pi)/TWO_PI*255).astype(np.uint8)
    overview_roi = np.round(np.clip((roi_absolute-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
    overview[Y0:Y1, X0:X1] = overview_roi
    image = Image.fromarray(overview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((X0, Y0, X1-1, Y1-1), outline=(255, 255, 0), width=7)
    draw.text((X0+15, Y0+15), "inferred-K ROI", fill=(255, 255, 0), font=ImageFont.load_default())
    image.save(OUT / "unwrapped_roi_outside_original_overview.png")

    residual_overview = np.round((combined_residual+np.pi)/TWO_PI*255).astype(np.uint8)
    image = Image.fromarray(residual_overview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rectangle((X0, Y0, X1-1, Y1-1), outline=(255, 255, 0), width=7)
    image.save(OUT / "residual_roi_outside_original_overview.png")

    values, counts = np.unique(k_values[roi_valid], return_counts=True)
    roi_residual = residual[Y0:Y1, X0:X1][roi_valid]
    report = (
        "method=supplied normalized quadratic-surface guided K unwrap\n"
        "anchors_used=0 (initial K inferred from wrapped gradients in requested ROI)\n"
        f"roi_half_open={X0},{Y0},{X1},{Y1}\n"
        f"valid_pixels={roi_valid.sum()}\n"
        f"initial_wrapped_gradient_x={initial[0]:.9f}\n"
        f"initial_wrapped_gradient_y={initial[1]:.9f}\n"
        "quadratic_coeff_x2,y2,xy,x,y,c=" + ",".join(f"{v:.9f}" for v in coeff) + "\n"
        f"residual_mean={roi_residual.mean():.9f}\n"
        f"residual_std={roi_residual.std():.9f}\n"
        f"residual_rms={np.sqrt(np.mean(roi_residual**2)):.9f}\n"
        "residual_p1,p5,p50,p95,p99=" +
        ",".join(f"{v:.9f}" for v in np.percentile(roi_residual, [1,5,50,95,99])) + "\n"
        "k_distribution=" + ",".join(f"{int(v)}:{int(c)}" for v,c in zip(values, counts)) + "\n"
    )
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
