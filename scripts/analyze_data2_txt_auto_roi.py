from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analyze_data2_simple_trend_k import decode_phase


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "phase_unwrap_task" / "results_data2_txt_auto_roi"
TWO_PI = 2.0 * np.pi
X0, Y0, X1, Y1 = 100, 1200, 4701, 3648  # half-open
PHASE_FIT_THRESHOLD = 0.6


def wrap_pi(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % TWO_PI - np.pi


def infer_sample_orders(wrapped: np.ndarray, valid: np.ndarray):
    """Replace the TXT's manually drawn rectangles and manually entered K.

    A smooth carrier plane is estimated only from wrapped neighbor differences.
    It is used solely to assign an integer order to candidate calibration pixels;
    the final fit below remains the TXT's single SVD plane fit.
    """
    local = wrapped[Y0:Y1, X0:X1]
    local_valid = valid[Y0:Y1, X0:X1]

    dx_ok = local_valid[:, 1:] & local_valid[:, :-1]
    dy_ok = local_valid[1:, :] & local_valid[:-1, :]
    sx = float(np.median(wrap_pi(local[:, 1:] - local[:, :-1])[dx_ok]))
    sy = float(np.median(wrap_pi(local[1:, :] - local[:-1, :])[dy_ok]))

    yy, xx = np.mgrid[Y0:Y1, X0:X1]
    reduced = local[local_valid] - sx * xx[local_valid] - sy * yy[local_valid]
    c = float(np.angle(np.mean(np.exp(1j * reduced))))
    guide = sx * xx + sy * yy + c
    orders = np.floor((guide - local) / TWO_PI + 0.5).astype(np.int16)

    # This is exactly the phase-center filter used by the TXT. Modulation is
    # used only to prevent background/no-signal pixels becoming automatic ROIs.
    samples = local_valid & (np.abs(local) < PHASE_FIT_THRESHOLD * np.pi)
    return sx, sy, c, orders, samples


def fit_txt_plane(wrapped: np.ndarray, orders: np.ndarray, samples: np.ndarray):
    """The one-pass normalized plane fit from UnwrapPhaseWithROIPlaneOpenCV."""
    height, width = wrapped.shape
    cx = width / 2.0
    cy = height / 2.0
    scale = float(max(width, height))

    # Keep every selected pixel, matching the TXT's ROI traversal exactly.
    sy_local, sx_local = np.where(samples)
    ys = sy_local + Y0
    xs = sx_local + X0
    ks = orders[sy_local, sx_local].astype(np.float64)
    phi_abs = wrapped[ys, xs].astype(np.float64) + ks * TWO_PI

    xn = (xs.astype(np.float64) - cx) / scale
    yn = (ys.astype(np.float64) - cy) / scale
    design = np.column_stack((xn, yn, np.ones(xs.size)))
    coeff = np.linalg.lstsq(design, phi_abs, rcond=None)[0]

    yy, xx = np.mgrid[0:height, 0:width]
    plane = (coeff[0] * (xx - cx) / scale +
             coeff[1] * (yy - cy) / scale + coeff[2]).astype(np.float32)
    k = np.floor((plane - wrapped) / TWO_PI + 0.5).astype(np.int16)
    unwrapped = wrapped + k.astype(np.float32) * TWO_PI
    residual = (unwrapped - plane).astype(np.float32)
    return coeff, plane, k, unwrapped, residual, xs.size


def gray(values: np.ndarray, lo: float, hi: float) -> Image.Image:
    pixels = np.round(np.clip((values - lo) / max(hi - lo, 1e-12), 0, 1) * 255).astype(np.uint8)
    return Image.fromarray(pixels, mode="L")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    wrapped, modulation = decode_phase()
    valid = modulation >= 4.0
    sx, sy, c, auto_orders, samples = infer_sample_orders(wrapped, valid)
    coeff, plane, k, unwrapped, residual, fit_count = fit_txt_plane(
        wrapped, auto_orders, samples
    )

    roi = np.zeros(wrapped.shape, dtype=bool)
    roi[Y0:Y1, X0:X1] = True
    combined_unwrapped = wrapped.copy()
    combined_unwrapped[roi] = unwrapped[roi]
    combined_residual = wrapped.copy()
    combined_residual[roi] = residual[roi]

    Image.fromarray(combined_unwrapped.astype(np.float32), mode="F").save(
        OUT / "phase_unwrapped_roi_outside_original.tiff"
    )
    Image.fromarray(combined_residual.astype(np.float32), mode="F").save(
        OUT / "residual_roi_outside_original.tiff"
    )
    Image.fromarray(k.astype(np.float32), mode="F").save(OUT / "k_txt_plane.tiff")

    local_valid = valid[Y0:Y1, X0:X1]
    local_unwrapped = unwrapped[Y0:Y1, X0:X1]
    local_residual = residual[Y0:Y1, X0:X1]
    ulo, uhi = np.percentile(local_unwrapped[local_valid], [0.5, 99.5])
    rlo, rhi = np.percentile(local_residual[local_valid], [0.5, 99.5])
    gray(local_unwrapped, ulo, uhi).save(OUT / "unwrapped_roi.png")
    gray(local_residual, -np.pi, np.pi).save(OUT / "residual_roi_pm_pi.png")
    gray(local_residual, rlo, rhi).save(OUT / "residual_roi_normalized.png")
    kb = np.clip(local_residual * -200.0 + 128.0, 0, 255).astype(np.uint8)
    Image.fromarray(kb, mode="L").save(OUT / "residual_roi_k-200_b128.png")

    # Show exactly which pixels replaced the manually drawn ROI boxes.
    sample_vis = np.round((wrapped[Y0:Y1, X0:X1] + np.pi) / TWO_PI * 255).astype(np.uint8)
    sample_vis = np.repeat(sample_vis[:, :, None], 3, axis=2)
    sample_vis[samples] = (255, 40, 40)
    Image.fromarray(sample_vis, mode="RGB").save(OUT / "auto_roi_samples.png")

    overview = np.round((wrapped + np.pi) / TWO_PI * 255).astype(np.uint8)
    overview[Y0:Y1, X0:X1] = np.asarray(gray(local_unwrapped, ulo, uhi))
    overview_img = Image.fromarray(overview, mode="L").convert("RGB")
    draw = ImageDraw.Draw(overview_img)
    draw.rectangle((X0, Y0, X1 - 1, Y1 - 1), outline=(255, 255, 0), width=7)
    draw.text((X0 + 15, Y0 + 15), "TXT plane / auto ROI", fill=(255, 255, 0), font=ImageFont.load_default())
    overview_img.save(OUT / "unwrapped_overview.png")

    rv = local_residual[local_valid]
    kv, kc = np.unique(k[Y0:Y1, X0:X1][local_valid], return_counts=True)
    report = (
        "method=TXT UnwrapPhaseWithROIPlaneOpenCV; manual ROI+K replaced only\n"
        f"requested_roi_half_open={X0},{Y0},{X1},{Y1}\n"
        "manual_anchors=0\n"
        f"automatic_candidate_pixels={int(samples.sum())}\n"
        f"svd_fit_pixels={fit_count}\n"
        f"automatic_order_guide_dx={sx:.9f}\n"
        f"automatic_order_guide_dy={sy:.9f}\n"
        f"automatic_order_guide_c={c:.9f}\n"
        "txt_plane_coeff_a,b,c=" + ",".join(f"{v:.9f}" for v in coeff) + "\n"
        f"residual_mean={rv.mean():.9f}\n"
        f"residual_std={rv.std():.9f}\n"
        f"residual_rms={np.sqrt(np.mean(rv * rv)):.9f}\n"
        "residual_p1,p5,p50,p95,p99=" +
        ",".join(f"{v:.9f}" for v in np.percentile(rv, [1, 5, 50, 95, 99])) + "\n"
        "k_distribution=" + ",".join(f"{int(v)}:{int(n)}" for v, n in zip(kv, kc)) + "\n"
    )
    (OUT / "report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
