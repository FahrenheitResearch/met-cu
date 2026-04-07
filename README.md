# met-cu

`met-cu` is a GPU-accelerated meteorological calculation library with CUDA and Metal backends. It targets the `metrust.calc` surface, runs the hot numeric paths as native GPU kernels, and now ships a direct MetPy conformance suite so users can switch without giving up behavioral checks.

The current shape of the project is intentionally pragmatic:

- Large numeric array and stencil workloads stay on GPU.
- Batched sounding and grid-column workflows stay on GPU.
- Small quantity/xarray-heavy compatibility cases can route through MetPy-compatible wrapper paths when exact behavior matters more than raw speed.

## Current status

- Public `metrust.calc` name/signature parity is checked in [tests/test_wrapper_parity.py](tests/test_wrapper_parity.py).
- Direct MetPy conformance is covered in [tests/test_against_metpy_core.py](tests/test_against_metpy_core.py) and [tests/test_against_metpy_extended.py](tests/test_against_metpy_extended.py).
- Cross-checks against `metrust` remain in [tests/test_against_metrust.py](tests/test_against_metrust.py).
- Latest combined verification run:

```bash
python -m pytest tests/test_against_metpy_core.py tests/test_against_metpy_extended.py tests/test_against_metrust.py tests/test_wrapper_parity.py -q --tb=short
```

Result on the current tree:

```text
231 passed, 3 warnings
```

The warnings are existing dependency/runtime warnings from xarray/SciPy and MetPy/Pint. They do not affect pass/fail.

## What met-cu is good at

- Large per-element thermo and wind calculations
- Large 2-D stencil work such as `vorticity`, `divergence`, `laplacian`, `frontogenesis`, and `q_vector`
- Batched CAPE/CIN and other grid-column workflows
- Keeping data resident on GPU in larger pipelines

## Where it is less impressive

- Tiny one-column sounding helpers where launch overhead dominates
- Small compatibility-oriented utility calls that intentionally favor MetPy behavior over squeezing every last millisecond out of the GPU path

That tradeoff is deliberate. The goal is "fast where GPUs matter, boringly compatible where users expect MetPy behavior."

## Installation

Base install:

```bash
pip install met-cu
```

CUDA:

```bash
pip install "met-cu[cuda]"
```

Metal on macOS:

```bash
pip install "met-cu[metal]"
```

Optional extras:

```bash
pip install "met-cu[plot,test,cpu]"
```

## Usage

Raw numeric arrays use the native GPU kernels:

```python
import cupy as cp
import metcu

pressure = cp.array([1000.0, 900.0, 850.0])
temperature = cp.array([20.0, 14.0, 10.0])

theta = metcu.potential_temperature(pressure, temperature)
print(type(theta))
```

Grid-column kernels work directly on 3-D model fields:

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

Compatibility-oriented wrappers accept the same public function names and signatures exposed by `metrust.calc`, with MetPy conformance tests covering the current public surface.

## Performance snapshot vs MetPy

These are current measurements on an RTX 5090 CUDA machine. Exact numbers will move somewhat run to run, but the shape is stable.

From `python tests/benchmark_metpy_subset.py`:

| Function | Speedup vs MetPy |
| --- | ---: |
| `potential_temperature` | 11.95x |
| `dewpoint` | 21.46x |
| `vorticity` | 38.68x |
| `frontogenesis` | 88.84x |
| `q_vector` | 43.59x |
| `surface_based_cape_cin` | 5.40x |
| `precipitable_water` | 8.46x |
| `bunkers_storm_motion` | 18.41x |

From `python tests/benchmark_three_way.py` on RAP data:

| Function | Speedup vs MetPy |
| --- | ---: |
| `potential_temperature [800k]` | 110.56x |
| `saturation_vapor_pressure [800k]` | 121.53x |
| `wind_speed [800k]` | 108.88x |
| `vorticity [864x864]` | 62.96x |
| `divergence [864x864]` | 72.36x |
| `frontogenesis [864x864]` | 134.49x |
| `cape_cin x50 cols [batched]` | 84.03x |

From `python tests/benchmark_hrrr_full.py`:

- `38` functions tested
- `38` accuracy-verified
- average speedup vs MetPy: `12.2x`
- median speedup vs MetPy: `5.5x`
- max speedup vs MetPy: `79.6x`

Representative full-HRRR wins:

- `heat_index 79.6x`
- `windchill 34.0x`
- `frontogenesis 33.6x`
- `divergence 18.9x`
- `vorticity 16.9x`
- `advection 16.8x`

Small-profile caveats still apply. A few helpers are near tie or slower on small inputs, for example `moist_lapse`, `parcel_profile`, and `smooth_n_point` in the current benchmark setup.

## Verifying locally

Core compatibility gate:

```bash
python -m pytest tests/test_against_metpy_core.py tests/test_against_metpy_extended.py tests/test_against_metrust.py tests/test_wrapper_parity.py -q --tb=short
```

Additional focused verification:

```bash
python tests/verify_thermo.py
python tests/verify_wind.py
python tests/verify_grid.py
python tests/verify_indices.py
python tests/verify_severe.py
python tests/verify_cape.py
python tests/verify_soundings.py
```

Benchmarks:

```bash
python tests/benchmark_metpy_subset.py
python tests/benchmark_three_way.py
python tests/benchmark_hrrr_full.py
python tests/benchmark_complete.py
python tests/benchmark_gpu_vs_cpu.py
```

## Notes

- CUDA PTX is cached under `python/metcu/ptx/` and regenerated when the archived CUDA source changes.
- The project includes fused and graph-oriented helpers for reducing wrapper and launch overhead in larger GPU pipelines.
- Performance claims in this README are CUDA numbers. Metal support exists, but these benchmark figures are not Metal benchmarks.

## License

MIT
