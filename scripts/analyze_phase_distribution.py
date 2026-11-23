from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data_phase_unwrap" / "t2"
OUTPUT = ROOT / "phase_unwrap_task" / "phase_distribution_analysis"


def moving_average(values: np.ndarray, width: int = 41) -> np.ndarray:
    pad = width // 2
    padded = np.pad(values, pad, mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def profile_from_wrapped(path: Path, axis: str) -> tuple[np.ndarray, np.ndarray]:
    gray = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    phase = gray * (2.0 * np.pi / 255.0) - np.pi
    if axis == "x":
        # The X carrier is almost constant transversely.  Circular averaging
        # suppresses 8-bit phase noise without averaging across phase wraps.
        strip = phase[1800:3600:4, :]
        wrapped_profile = np.angle(np.mean(np.exp(1j * strip), axis=0))
    else:
        strip = phase[:, 1500:6500:4]
        wrapped_profile = np.angle(np.mean(np.exp(1j * strip), axis=1))
    coordinate = np.arange(wrapped_profile.size, dtype=np.float64)
    return coordinate, moving_average(np.unwrap(wrapped_profile), 41)


def analyze(path: Path, axis: str, fit_start: int, fit_end: int) -> dict:
    coordinate, actual = profile_from_wrapped(path, axis)
    fit_mask = (coordinate >= fit_start) & (coordinate <= fit_end)
    design = np.column_stack([coordinate[fit_mask], np.ones(np.count_nonzero(fit_mask))])
    slope, offset = np.linalg.lstsq(design, actual[fit_mask], rcond=None)[0]
    linear = slope * coordinate + offset
    deviation = actual - linear
    derivative = moving_average(np.gradient(actual), 101)
    valid_deviation = deviation[fit_mask]
    valid_derivative = derivative[fit_mask]
    return {
        "axis": axis,
        "coordinate": coordinate,
        "actual": actual,
        "linear": linear,
        "deviation": deviation,
        "slope": slope,
        "offset": offset,
        "rms": float(np.sqrt(np.mean(valid_deviation**2))),
        "peak_to_peak": float(np.ptp(valid_deviation)),
        "derivative_p10": float(np.percentile(valid_derivative, 10)),
        "derivative_p50": float(np.percentile(valid_derivative, 50)),
        "derivative_p90": float(np.percentile(valid_derivative, 90)),
        "fit_start": fit_start,
        "fit_end": fit_end,
    }


def map_points(x: np.ndarray, y: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    left, top, right, bottom = box
    x_span = max(float(np.ptp(x)), 1e-12)
    y_min = float(np.min(y))
    y_span = max(float(np.ptp(y)), 1e-12)
    px = left + (x - x.min()) * (right - left) / x_span
    py = bottom - (y - y_min) * (bottom - top) / y_span
    return list(zip(px.astype(int), py.astype(int)))


def draw_plot(result: dict, path: Path) -> None:
    width, height = 1400, 820
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    upper = (90, 70, width - 50, 390)
    lower = (90, 470, width - 50, height - 70)
    for box in (upper, lower):
        draw.rectangle(box, outline=(80, 80, 80), width=1)

    coordinate = result["coordinate"]
    actual = result["actual"]
    linear = result["linear"]
    deviation = result["deviation"]
    interior = (coordinate >= result["fit_start"]) & (coordinate <= result["fit_end"])
    draw.line(map_points(coordinate, actual, upper), fill=(20, 80, 210), width=3)
    # Map both upper curves against their common range.
    common_min = min(float(actual.min()), float(linear.min()))
    common_max = max(float(actual.max()), float(linear.max()))
    common_span = max(common_max - common_min, 1e-12)
    left, top, right, bottom = upper
    linear_points = list(zip(
        (left + (coordinate - coordinate.min()) * (right - left) / np.ptp(coordinate)).astype(int),
        (bottom - (linear - common_min) * (bottom - top) / common_span).astype(int),
    ))
    actual_points = list(zip(
        (left + (coordinate - coordinate.min()) * (right - left) / np.ptp(coordinate)).astype(int),
        (bottom - (actual - common_min) * (bottom - top) / common_span).astype(int),
    ))
    draw.rectangle(upper, fill="white", outline=(80, 80, 80), width=1)
    draw.line(actual_points, fill=(20, 80, 210), width=3)
    draw.line(linear_points, fill=(220, 70, 40), width=2)
    draw.line(map_points(coordinate[interior], deviation[interior], lower), fill=(20, 130, 70), width=3)

    axis = result["axis"].upper()
    draw.text((90, 25), f"Set2 {axis}: actual unwrapped phase profile vs best linear fit", fill="black", font=font)
    draw.text((100, 80), "blue: measured profile    red: linear fit", fill="black", font=font)
    draw.text((100, 480), "interior only: measured - linear fit (rad)", fill="black", font=font)
    summary = (
        f"fit range={result['fit_start']}..{result['fit_end']} px; "
        f"linear slope={result['slope']:.7f} rad/px; "
        f"nonlinear RMS={result['rms']:.4f} rad; "
        f"peak-to-peak={result['peak_to_peak']:.4f} rad"
    )
    draw.text((100, height - 45), summary, fill="black", font=font)
    image.save(path)


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = [
        # Fit only the phone's broad interior.  Including its silhouette edges
        # mixes two physically different surfaces and introduces whole-cycle
        # jumps that are not carrier non-linearity.
        (DATA / "DataExport_outputPhaseXImage_2.bmp", "x", 1000, 6800),
        (DATA / "DataExport_outputPhaseYImage_2.bmp", "y", 1100, 3900),
    ]
    rows = []
    for source, axis, start, end in cases:
        result = analyze(source, axis, start, end)
        draw_plot(result, OUTPUT / f"set2_{axis}_actual_phase_profile.png")
        rows.append(result)
        np.savetxt(
            OUTPUT / f"set2_{axis}_actual_phase_profile.csv",
            np.column_stack([
                result["coordinate"], result["actual"],
                result["linear"], result["deviation"],
            ]),
            delimiter=",",
            header="coordinate_px,actual_unwrapped_phase_rad,linear_fit_rad,deviation_rad",
            comments="",
        )
    for result in rows:
        print(
            result["axis"].upper(),
            f"slope={result['slope']:.8f} rad/px",
            f"nonlinear_rms={result['rms']:.6f} rad",
            f"nonlinear_p2p={result['peak_to_peak']:.6f} rad",
            "derivative_p10/p50/p90=",
            f"{result['derivative_p10']:.8f}/",
            f"{result['derivative_p50']:.8f}/",
            f"{result['derivative_p90']:.8f}",
        )


if __name__ == "__main__":
    main()
