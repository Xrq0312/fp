# C++ Quality-Guided Phase Unwrap

## References Used

This implementation follows the same family of methods as:

- Herraez, Burton, Lalor, Gdeisat, "A fast two-dimensional phase-unwrapping algorithm based on sorting by reliability following a noncontinuous path", Applied Optics, 2002. DOI: https://doi.org/10.1364/AO.41.007437
- OpenCV contrib `phase_unwrapping::HistogramPhaseUnwrapping`, which describes reliability-map based phase unwrapping: https://docs.opencv.org/4.x/df/d3a/group__phase__unwrapping.html

The code does not call OpenCV's phase unwrapping implementation. OpenCV is only used for image IO and simple filtering/morphology.

## Why Not Row/Column Integration

Row/column integration makes the unwrap path fixed. Once a noisy pixel is integrated with the wrong `2*pi` jump, every later pixel in that scan line inherits the error.

The C++ tool instead:

1. Computes a valid-pixel mask.
   - Uses four-step modulation plus high local phase stability when RAW phase is decoded.
   - Falls back to circular phase coherence when modulation is unavailable or too sparse.
2. Computes pixel reliability from wrapped second differences.
   - Smooth and locally consistent phase gets high reliability.
   - Random background and noise get low reliability.
3. Sorts neighboring-pixel edges by reliability.
4. Uses union-find with integer `2*pi` offsets to merge reliable regions first.
   - Low-quality edges cannot drag a whole row or column across the image.
   - Invalid background pixels are not connected into the unwrap graph.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_phase_unwrap_tool.ps1
```

This uses:

- Visual Studio: `D:\Program Files\Microsoft Visual Studio\2022\Community`
- vcpkg OpenCV: `D:\vcpkg\vcpkg\installed\x64-windows`

## Run

```powershell
$env:PATH = "D:\vcpkg\vcpkg\installed\x64-windows\bin;$env:PATH"
.\build\phase_unwrap_tool\phase_unwrap_tool.exe --batch-root PMD_DATA --out-dir phase_unwrap_task\results
```

Single image example:

```powershell
$env:PATH = "D:\vcpkg\vcpkg\installed\x64-windows\bin;$env:PATH"
.\build\phase_unwrap_tool\phase_unwrap_tool.exe --phase "PMD_DATA\数据2 - 存在背景区域\我们裁切ROI.bmp" --out-dir phase_unwrap_task\results --prefix data2_roi
```

If the result is globally bright/dark reversed versus a reference phase map, use:

```powershell
.\build\phase_unwrap_tool\phase_unwrap_tool.exe --phase "PMD_DATA\数据2 - 存在背景区域\我们裁切ROI.bmp" --out-dir phase_unwrap_task\results --prefix data2_roi_invert --invert-phase
```

This only changes the phase sign convention. It corresponds to choosing the opposite four-step phase formula/order, for example swapping `atan2(I4 - I2, I1 - I3)` with the opposite sign convention. It is not a change to the unwrap algorithm.

## Output Layout

Each input image is written into its own subdirectory:

```text
phase_unwrap_task/results/
  data1_phasex/
    phase_unwrapped_f32.raw
    phase_unwrapped.png
    mask.png
    defect_delta.png
    defect_score.png
    report.txt
  data2_phasex/
    ...
  aluminum_single_y/
    ...
```

## Output Files

- `phase_unwrapped_f32.raw`: actual unwrapped phase in radians.
  - Format: `uint32 width`, `uint32 height`, then `width * height` little-endian `float32`.
  - Invalid pixels are `NaN`.
- `phase_unwrapped.png`: display-only plane-detrended unwrapped phase. This removes the dominant left-right/top-bottom ramp before 8-bit normalization, so local surface changes are easier to compare visually.
- `phase_unwrapped_absolute.png`: display-only normalized raw unwrapped phase, preserving the low-frequency trend. Use this only when the global phase ramp itself is the thing being inspected.
- `phase_wrapped.png`: original wrapped phase display. For four-step RAW input, this is the decoded wrapped phase rather than the first fringe image.
- `phase_unwrapped_with_background.png`: `phase_unwrapped.png` composited over `phase_wrapped.png` where the unwrap mask is invalid. This keeps the background as the original phase result.
- `modulation.png`: four-step modulation image, written when `--decode-raw-phase` can decode RAW images with the same size as the phase input.
- `mask_modulation.png`: modulation-only RAW mask component.
- `mask_phase_stability.png`: high-coherence RAW mask component. This lets stable low-modulation phase regions enter the final mask.
- `phase_background.png`: robust low-frequency background estimated from the unwrapped phase.
- `phase_residual_flat.png`: background-removed residual, shown around mid-gray.
- `phase_relief.png`: lightly smoothed residual display. This is the closest output to a gray-background defect-relief view.
- `phase_enhanced.png`: robust background plus amplified residual. This keeps the global unwrapped phase appearance while making local anomalies more visible.
- `phase_enhanced_with_background.png`: `phase_enhanced.png` composited over `phase_wrapped.png`.
- `phase_competitor_style.png`: a presentation-oriented view inspired by the competitor full-frame image. It keeps the original wrapped phase background outside the valid unwrap mask and compresses the enhanced phase contrast inside the valid region.
- `phase_competitor_style_detail.png`: `phase_competitor_style.png` with every non-zero `defect_delta.png` pixel inserted directly. Only zero-valued `defect_delta` pixels keep the presentation background.
- `mask.png`: valid pixels used in the unwrap graph.
- `defect_delta.png`: local-bias balanced circular local phase residual; mid-gray means zero. This is the preferred defect-detail display because slow residual bias is removed before 8-bit visualization.
- `defect_delta_raw.png`: raw circular local phase residual before display balancing.
- `defect_score.png`: absolute anomaly strength from the raw circular residual.
- `report.txt`: size, valid coverage, and phase percentiles.

## Notes

- Four-step RAW phase decoding is available with `--decode-raw-phase --raw-phase-start <index>`. This computes float wrapped phase from four fringe frames using an `atan2` four-step formula and uses the modulation image for validity. It only applies when the RAW images have the same size as the requested phase image; cropped ROI phase images need a matching RAW crop or a known crop offset.
- Robust phase-surface modeling is implemented in `phase_background.png`, `phase_relief.png`, and `phase_enhanced.png`. This estimates the low-frequency surface from trustworthy samples, downweights local outliers, and inspects or re-injects the residual rather than judging only the raw unwrapped phase.

For competitor comparison of an unwrapped phase image, use `phase_unwrapped.png` or, preferably, compare the float phase after applying the same display normalization. For defect detection, use `defect_delta.png` or `defect_score.png`. The defect images are computed with circular statistics on the wrapped phase so the original `-pi/pi` period boundaries are not treated as real surface defects.
