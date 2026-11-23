# CUDA Quality-Guided Phase Unwrap

This is a separate, GPU-accelerated implementation of the repository's
reliability-guided 2-D phase unwrap. The original sources under `src/` are not
modified.

The result-preserving pipeline is:

1. CUDA computes wrapped second-difference reliability.
2. CUDA compacts valid horizontal/vertical edges.
3. CUB radix-sorts compact edge IDs by a monotonic float reliability key.
4. A cache-friendly weighted union-find applies edges in exactly that order.
5. CUDA emits the unwrapped float image (`NaN` for invalid pixels).

The weighted union-find has a true order dependency. Keeping that short stage
on the CPU preserves the reference algorithm; a parallel approximate union can
be faster but changes ambiguous/noisy cases.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File phase_unwrap_gpu\scripts\build.ps1
```

The build emits Ampere/Ada cubins and forward-compatible PTX. On the installed
RTX 5070 driver, PTX is JIT-compiled and cached on first launch.

## Run

```powershell
$env:PATH = "D:\vcpkg\vcpkg\installed\x64-windows\bin;$env:PATH"
phase_unwrap_gpu\build\Release\phase_unwrap_gpu.exe `
  --phase input.bmp --output unwrapped.raw --warmup 3 --iterations 20
```

Optional `--mask mask.png` uses nonzero pixels. BMP/PNG phase values map from
`[0,255]` to `[-pi,pi]`. The raw output format is `uint32 width`, `uint32
height`, followed by row-major little-endian float32 radians.

Use `--self-test` for deterministic correctness tests, or `--synthetic WxH`
to benchmark without image I/O. Timings exclude image decode and file output.

