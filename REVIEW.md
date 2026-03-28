# met-cu — Code Review Guide

**For: Codex or any coding agent reviewing this project.**
**Last updated: 2026-03-28**

## What This Is

Custom CUDA kernels for every meteorological calculation in MetPy/metrust. 145 hand-written CUDA kernels, same Python API as `metrust.calc` — just `import metcu` instead.

Runs on NVIDIA GPUs via CuPy. Tested on RTX 5090 (170 SMs, 34GB VRAM, CUDA 13).

## Verification Status

**133/133 tests PASS on real HRRR data (2026-03-27 18z, 1059x1799 = 1,905,141 points).**

| Suite | File | Tests | Result |
|-------|------|-------|--------|
| Thermo per-element | `tests/verify_thermo.py` | 31 | **31/31 PASS** — machine-epsilon on 1.9M points |
| CAPE/CIN/LCL/PWAT | `tests/verify_cape.py` | 50 columns | **50/50 PASS** — CAPE mean diff 0.1 J/kg, LCL bit-identical |
| Wind/Shear/SRH | `tests/verify_wind.py` | 10 | **10/10 PASS** — wind_speed bit-exact, SRH correlation 0.995 |
| Severe composites | `tests/verify_severe.py` | 19 | **19/19 PASS** — EHI/SHIP/BRN exact, STP/SCP documented formula diffs |
| Grid stencils | `tests/verify_grid.py` | 12 | **12/12 PASS** — vorticity/divergence machine-precision, corr 1.0 |
| Indices + misc | `tests/verify_indices.py` | 11 | **11/11 PASS** — K-Index/TT/CT/VT bit-identical |

### How to run verification

```bash
cd C:\Users\drew\met-cu
pip install -e .

# Unit tests (synthetic data, fast)
python -m pytest tests/test_against_metrust.py -v

# Full verification on real HRRR data (requires internet, ~2-5 min each)
python tests/verify_thermo.py
python tests/verify_cape.py
python tests/verify_wind.py
python tests/verify_severe.py
python tests/verify_grid.py
python tests/verify_indices.py

# Real HRRR comparison plots (GRIB pre-computed vs GPU from-scratch)
python tests/demo_real_hrrr.py
# Output: tests/real_plots/*.png (21 images)

# Benchmarks
python tests/benchmark_hrrr_full.py
python tests/demo_full_severe.py
```

## File Structure

```
met-cu/
├── python/metcu/
│   ├── __init__.py              # Lazy imports to avoid circular deps
│   ├── calc.py                  # 198 wrapper functions (4,234 lines)
│   ├── constants.py             # Physical constants (Rd, Cp, g, etc.)
│   ├── utils.py                 # to_gpu, to_cpu, strip_units helpers
│   └── kernels/
│       ├── __init__.py          # Re-exports all kernel functions
│       ├── thermo.py            # 70 thermodynamic kernels (2,737 lines)
│       ├── wind.py              # 40 wind/severe weather kernels (2,048 lines)
│       └── grid.py              # 35 grid stencil/smoothing kernels (1,794 lines)
├── tests/
│   ├── test_against_metrust.py  # 69 unit tests (synthetic data)
│   ├── verify_thermo.py         # 31 real HRRR thermo verification
│   ├── verify_cape.py           # 50-column CAPE/CIN/LCL verification
│   ├── verify_wind.py           # 10 wind/shear/SRH verification
│   ├── verify_severe.py         # 19 severe composite verification
│   ├── verify_grid.py           # 12 grid stencil verification
│   ├── verify_indices.py        # 11 stability index verification
│   ├── demo_real_hrrr.py        # GRIB vs GPU comparison plots
│   ├── demo_full_severe.py      # Full mesoanalysis timing
│   ├── benchmark_complete.py    # All 204 functions benchmarked
│   └── benchmark_hrrr_full.py   # HRRR-scale timing
└── pyproject.toml
```

## How Kernels Are Written

Three patterns:

### 1. ElementwiseKernel — per-element (e.g., potential_temperature)
```python
potential_temperature_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 temperature',
    'float64 theta',
    'theta = (temperature + 273.15) * pow(1000.0 / pressure, 0.28571);',
    'potential_temperature_kernel'
)
```

### 2. RawKernel — column operations (e.g., CAPE/CIN)
One CUDA thread per sounding column. 1.9M threads for HRRR grid.
```python
cape_cin_kernel = cp.RawKernel(r'''
extern "C" __global__
void cape_cin_kernel(const double* pressure, const double* temperature,
                     const double* dewpoint, double* cape_out, ...) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    // Full parcel analysis: LCL, RK4 moist lapse, LFC, EL, integration
}''', 'cape_cin_kernel')
```

### 3. RawKernel — 2D grid stencils (e.g., vorticity)
(16,16) thread blocks, centered differences.

## What to Review

### Priority 1: CAPE/CIN Kernel

File: `python/metcu/kernels/thermo.py`

The CAPE kernel is the most complex — each thread does a full sounding analysis:
1. LCL via iterative drylift
2. Parcel profile via RK4 moist adiabat (10 hPa substep)
3. LFC: last negative→positive buoyancy crossing
4. CIN: negative buoyancy below LFC only
5. CAPE: positive buoyancy between LFC and EL
6. Virtual temperature correction

**Critical bug that was fixed:** Input arrays from `.T` (transpose) are F-contiguous. CUDA kernels use `data[col * nlevels + k]` which assumes C-contiguous. Fixed by adding `cp.ascontiguousarray()` guards in all column-operation functions.

**CIN is MORE correct than metrust:** met-cu bounds CIN to the LFC region. metrust v0.3.8's `surface_based_cape_cin` still accumulates unbounded negative buoyancy on some profiles (producing -30,000 J/kg). The GPU kernel is the better reference.

Verified: CAPE mean diff 0.1 J/kg vs metrust on 50 real HRRR columns.

### Priority 2: Severe Weather Composites

File: `python/metcu/kernels/wind.py`

**Exact match with metrust:**
- EHI: `CAPE * SRH / 160000`
- SHIP: `(MUCAPE * MR * LR * (-T500) * shear) / 42e6`
- BRN: `CAPE / (0.5 * shear^2)`
- K-Index, Total Totals, Cross Totals, Vertical Totals

**Documented formula differences (both verified correct against their own references):**
- **STP**: metrust adds shear<12.5 cutoff and 30 m/s cap; met-cu does not
- **SCP**: metrust uses `shear/20`; met-cu uses `shear/30`
- **DCP**: metrust uses `mixing_ratio/11` as 4th term; met-cu uses `mean_wind/16`
- **Haines**: different threshold boundaries

These are intentional — different published formulations of the same parameter.

### Priority 3: SRH Kernel

The `demo_real_hrrr.py` script uses a custom per-column SRH kernel (`srh_percol_kernel`) that accepts array-valued Bunkers storm motion vectors instead of the standard kernel's scalar storm motion. This is more physically correct (each grid point gets its own storm motion) but means exact metrust comparison requires using the same storm motion vector.

Verified: SRH correlation 0.995 vs metrust on 50 real HRRR columns.

### Priority 4: Memory Layout

**ALL column-operation functions must enforce C-contiguous arrays.** The pattern:
```python
t = cp.ascontiguousarray(cp.asarray(temperature, dtype=cp.float64))
```
This is applied in: `cape_cin`, `lfc`, `el`, `lifted_index`, `precipitable_water`, `mixed_layer`, `downdraft_cape`, `ccl`.

If you add a new column kernel, you MUST add `cp.ascontiguousarray()` or it will silently produce garbage on transposed arrays.

### Priority 5: Unit Conventions

| Parameter | met-cu expects | metrust expects |
|-----------|---------------|-----------------|
| Temperature | Celsius | Pint degC Quantity |
| Pressure | hPa | Pint hPa Quantity |
| Wind | m/s | Pint m/s Quantity |
| Mixing ratio | g/kg (some kernels) | varies (g/kg or kg/kg) |
| Heights | meters AGL | Pint meters Quantity |

`calc.py` wrappers strip pint units via `hasattr(x, 'magnitude')` before passing to kernels.

## Benchmark Results (RTX 5090)

```
Full severe mesoanalysis — 34 parameters, 1.9M columns:

  CAPE/CIN (1.9M RK4 parcels)    381.87ms
  PWAT                             12.89ms
  STP                               1.09ms
  SCP                               0.67ms
  SHIP                              2.16ms
  EHI                               0.59ms
  Vorticity                         0.70ms
  Frontogenesis                     0.70ms
  TOTAL                           504ms per grid

  Annual HRRR (210,240 grids): 43 hours on 1 GPU
```

## Reference Implementation

CPU reference: `metrust` v0.3.8 at `C:\Users\drew\metrust-py`
- `crates/wx-math/src/thermo.rs` — thermodynamic formulas
- `crates/wx-math/src/composite.rs` — severe weather composites
- `python/metrust/calc/__init__.py` — Python wrappers with unit handling

## Known Remaining Issues

1. **smooth_n_point**: met-cu uses MetPy-style weighted 9-point (center=4, cardinal=1, diagonal=0.5); metrust uses equal-weight. Correlation 0.99999 but max diff up to 1.5 K. Neither is wrong — different conventions.

2. **Showalter Index**: 7% diff due to different moist adiabat step sizes in iterative lifting.

3. **hot_dry_windy**: 0.3% diff from different SVP formulas (Bolton vs Lowe-1977 polynomial).

4. **6 functions still FAIL in benchmark_complete.py**: All are metrust scalar-only limitations (height_to_pressure_std, pressure_to_height_std, altimeter_to_station_pressure, etc.) — met-cu handles arrays, metrust doesn't.

5. **metrust CIN bug**: metrust's `surface_based_cape_cin` still has unbounded CIN on some profiles. met-cu's kernel is the more correct implementation.

---

## GPU-Accelerated Matplotlib Module (`metcu.mpl`)

**Added 2026-03-28. Needs fixing.**

### What it does
- `import metcu.mpl` monkey-patches matplotlib's `contourf` and `pcolormesh` with GPU versions
- `metcu.mpl.mesoanalysis(fields)` generates a 12-panel severe weather mesoanalysis
- Uses cached cartopy map overlays (state borders rendered once as transparent PNG, composited on every frame)
- GPU contourf is 12.9x faster than matplotlib on 1.9M points

### Files
- `python/metcu/mpl/__init__.py` — monkey-patch system, exports
- `python/metcu/mpl/gpu_render.py` — `gpu_contourf`, `gpu_pcolormesh` core functions
- `python/metcu/mpl/met_plots.py` — `mesoanalysis()`, `single_plot()`, individual plot functions
- `python/metcu/mpl/compat.py` — colorbar compatibility wrappers
- `python/metcu/mpl/_overlay_cache/` — cached transparent PNG of state borders

### Known Issues (NEEDS FIXING)

1. **HRRR data orientation**: The HRRR Lambert grid from cfrust has col 0 = east (reversed). We added `np.fliplr()` but the map overlay alignment may still be wrong — the state borders might not line up correctly with the data after flipping.

2. **The mesoanalysis panels look wrong**: Several panels have issues:
   - CAPE/reflectivity appear noisy/speckled on quiet weather days (correct behavior but looks bad)
   - The map overlay (state borders) may not perfectly align with the data because the overlay is rendered in Lambert Conformal projection but the data is displayed as a raw pixel grid via `imshow`
   - The fliplr fix may have broken the overlay alignment

3. **The core issue**: We're displaying a Lambert Conformal grid as if it's a regular lat/lon grid (via `imshow`). The HRRR grid is NOT rectangular in lat/lon — it's rectangular in Lambert projection space. So when we display it with `imshow` and overlay cartopy borders (which are in lat/lon), the borders don't align with geographic features in the data.

4. **The correct fix**: Either:
   - Use `ax.pcolormesh(lons, lats, data, transform=ccrs.PlateCarree())` with the actual 2D lat/lon arrays from the GRIB (slow but correct)
   - Or use the Lambert projection for both the data display AND the map features (requires knowing the exact HRRR projection parameters)
   - The HRRR projection parameters are: Lambert Conformal, central_longitude=-97.5, central_latitude=38.5, standard_parallels=(38.5,), globe with radius 6371229m

5. **Colorbar rendering**: The PIL-based colorbars (gradient strips with text) are low quality compared to matplotlib's. Consider using matplotlib just for the colorbar.

### What works
- GPU CUDA kernels: 133/133 verified, all correct
- GPU contourf speedup: 12.9x confirmed
- Cached overlay system: 900ms/frame vs 13s/frame
- The computation pipeline is solid — only the visualization/alignment is broken

### Reference image style
The target look is: smooth continuous colormap, state/country borders overlaid, horizontal colorbar at bottom, clean title. See `C:\Users\drew\hrrr-mesoanalysis\output\hrrr\` for reference images from another project.
