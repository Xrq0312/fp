# Phase Unwrap Run Parameters

This file summarizes the current per-sample config files and the shared defaults used by
`phase_unwrap_delivery.exe` / `phase_unwrap_tool.exe` when a key is not written in the cfg.

## Config Files

| Sample | Config | Phase input | RAW input | Output dir | Prefix | Decode RAW phase | RAW series |
| --- | --- | --- | --- | --- | --- | --- | --- |
| data1 | `data1.cfg` | `PMD_DATA/数据1 - 存在噪声/PHASEX.bmp` | `PMD_DATA/数据1 - 存在噪声/RAW` | `phase_unwrap_task/results_rerun_cfg` | `data1_phasex` | `false` | empty |
| data2 | `data2.cfg` | unused | `data_phase_unwrap/data2/RAW` | `phase_unwrap_task/results_delivery` | `data2_raw` | `true` | empty |
| aluminum_x | `aluminum_x.cfg` | `PMD_DATA/铝壳single/MP.X1.bmp` | `PMD_DATA/铝壳single` | `phase_unwrap_task/results_aluminum_cfg` | `aluminum_x` | `true` | `X` |
| aluminum_y | `aluminum_y.cfg` | `PMD_DATA/铝壳single/MP.Y1.bmp` | `PMD_DATA/铝壳single` | `phase_unwrap_task/results_aluminum_cfg` | `aluminum_y` | `true` | `Y` |
| aluminum_x rerun | `aluminum_x_rerun.cfg` | `PMD_DATA/铝壳single/MP.X1.bmp` | `PMD_DATA/铝壳single` | `phase_unwrap_task/results_rerun_cfg` | `aluminum_x` | `true` | `X` |
| aluminum_y rerun | `aluminum_y_rerun.cfg` | `PMD_DATA/铝壳single/MP.Y1.bmp` | `PMD_DATA/铝壳single` | `phase_unwrap_task/results_rerun_cfg` | `aluminum_y` | `true` | `Y` |

The two aluminum rerun configs are the same as `aluminum_x.cfg` / `aluminum_y.cfg` except for
`out_dir`.

## Explicit Mask Parameters In All Current Configs

| Key | Value |
| --- | --- |
| `raw_phase_start` | `1` |
| `modulation_abs_floor` | `4.0` |
| `modulation_min_weak_ratio_for_otsu` | `0.2` |
| `modulation_otsu_cap_pct` | `25.0` |
| `modulation_erode_size` | `0` |
| `trend_fit_phase_threshold` | `0.6` for data2; default `0.6` | Excludes wrapped samples near +/-pi from normalized SVD trend fitting. |

## Shared Defaults

These are used unless overridden in a cfg.

| Key | Default | Notes |
| --- | --- | --- |
| `invert_phase` | `false` | No current cfg overrides this. |
| `coherence_threshold` | `0.45` | Used in confidence scoring. |
| `min_amplitude` | `4.0` | Lower edge for amplitude score. |
| `good_amplitude` | `20.0` | Upper edge for amplitude score. |
| `residual_scale` | `0.35` | Four-step model residual scale. |
| `confidence_threshold` | `0.25` | Used for phase-only valid mask fallback. |
| `max_residue_density` | `0.10` | Local residue-density quality cutoff. |
| `coherence_radius` | `4` | Circular coherence box radius. |
| `residue_density_radius` | `3` | Residue density box radius. |
| `residual_radius` | `61` | Local circular residual radius. |
| `background_radius` | `241` | Robust low-frequency background radius. |
| `background_iterations` | `4` | Robust background iterations. |
| `relief_smooth_radius` | `2` | Relief display smoothing radius. |
| `detail_overlay_threshold` | `18` | Signed-detail overlay threshold. |
| `background_outlier_scale` | `1.25` | Robust background outlier weighting scale. |

## Delivery Output Files

The delivery build writes only:

```text
defect_delta.png
defect_delta_raw.png
phase_relief.png
phase_unwrapped.png
phase_unwrapped_with_background.png
phase_unwrapped_with_background_detail.png
phase_wrapped.png
```
