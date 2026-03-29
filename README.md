# met-cu

Custom CUDA kernels for meteorological calculations.

`met-cu` is a CuPy-first GPU companion to `metrust` and `MetPy`. The current target is strong `metrust.calc` parity on the public calculation surface, with native CUDA kernels for the hot grid-composite paths and direct GPU array outputs.

## Current status

- `104` tests passing in the current suite.
- Public `metrust.calc` name/signature parity is checked in `tests/test_wrapper_parity.py`.
- Native grid kernels now back:
  - `compute_cape_cin`
  - `compute_srh`
  - `compute_shear`
  - `compute_pw`
  - `composite_reflectivity_from_hydrometeors`
- Optional plotting extras install via `met-cu[plot]`.
- Return type is `cupy.ndarray` on GPU. This is not yet a full Pint/xarray MetPy drop-in.

## What this project is optimizing for

- Large flat-array thermo and wind calculations
- Large 2-D stencil workloads such as vorticity, divergence, frontogenesis, and q-vectors
- Grid-wide column composites where thousands of columns can be processed in parallel

It is less optimized for single tiny sounding calls where kernel launch overhead dominates.

## Installation

Requires an NVIDIA GPU with CUDA 12+.

```bash
pip install met-cu
```

Optional plotting extras:

```bash
pip install "met-cu[plot]"
```

## Usage

For the closest API match, think of `met-cu` as the GPU side of `metrust.calc`:

```python
import metcu

theta = metcu.potential_temperature(pressure_hpa, temperature_c)

# CuPy array on GPU
print(type(theta))

# Pull back to NumPy when needed
theta_np = theta.get()
```

Grid-composite APIs run directly on 3-D model fields:

```python
cape, cin, lcl_h, lfc_h = metcu.compute_cape_cin(
    pressure_3d_pa,
    temperature_3d_c,
    qvapor_3d_kgkg,
    height_agl_3d_m,
    psfc_pa,
    t2_k,
    q2_kgkg,
)
```

## Benchmark snapshot vs metrust

Representative results from `python tests/benchmark_complete.py` on an RTX 5090. These numbers will move somewhat run to run, but this is the current shape of performance.

| Function | GPU ms | CPU ms (`metrust`) | Speedup |
|---|---:|---:|---:|
| `potential_temperature` | 2.07 | 12.02 | 5.8x |
| `dewpoint` | 0.12 | 1.23 | 10.1x |
| `vorticity` | 3.43 | 99.24 | 28.9x |
| `frontogenesis` | 8.21 | 342.28 | 41.7x |
| `q_vector` | 7.83 | 307.92 | 39.3x |
| `compute_cape_cin` | 6.74 | 5.70 | 0.8x |
| `compute_srh` | 0.20 | 0.50 | 2.5x |
| `compute_shear` | 0.29 | 0.62 | 2.1x |
| `compute_pw` | 0.13 | 0.34 | 2.7x |
| `composite_reflectivity_from_hydrometeors` | 0.40 | 0.34 | 0.8x |

The headline is:

- Large per-element and stencil workloads are already much faster than `metrust`.
- The new native grid composite kernels are in place and parity-checked.
- `compute_cape_cin` is native now, but still near break-even against `metrust` on the current benchmark shape.

## Single-sounding latency note

Single 40-level soundings are not where a GPU naturally wins. Current latency against `metrust` is mixed:

| Function | GPU ms | CPU ms (`metrust`) | Speedup |
|---|---:|---:|---:|
| `lcl` | 0.09 | 0.19 | 2.1x |
| `lfc` | 0.36 | 0.30 | 0.8x |
| `el` | 0.41 | 0.30 | 0.7x |
| `parcel_profile` | 3.25 | 0.25 | 0.1x |
| `moist_lapse` | 3.40 | 0.21 | 0.1x |
| `surface_based_cape_cin` | 2.10 | 0.41 | 0.2x |

That is expected for tiny one-column workloads. The GPU value proposition here is throughput and keeping larger pipelines on device, not shaving fractions of a millisecond off one sounding call.

## Representative subset vs MetPy

`metrust` is already much faster than plain Python `MetPy`, so a near break-even result against `metrust` can still be materially faster than `MetPy`.

A reproducible subset benchmark lives in `tests/benchmark_metpy_subset.py`. On the current machine, representative results are:

| Function | GPU ms | CPU ms (`MetPy`) | Speedup |
|---|---:|---:|---:|
| `potential_temperature` | 2.29 | 29.15 | 12.7x |
| `vorticity` | 2.78 | 88.39 | 31.7x |
| `frontogenesis` | 7.55 | 690.71 | 91.4x |
| `cape_cin` | 2.45 | 2.97 | 1.2x |
| `surface_based_cape_cin` | 2.17 | 5.62 | 2.6x |
| `precipitable_water` | 0.16 | 1.66 | 10.4x |
| `bulk_shear` | 0.17 | 2.18 | 12.9x |
| `storm_relative_helicity` | 0.24 | 1.18 | 4.9x |
| `bunkers_storm_motion` | 0.45 | 6.83 | 15.2x |

## Reproducing the benchmarks

Full `metrust` comparison:

```bash
python tests/benchmark_complete.py
```

Representative `MetPy` subset:

```bash
pip install metpy
python tests/benchmark_metpy_subset.py
```

## License

MIT
