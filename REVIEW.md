# met-cu — Code Review Guide

**For: Codex or any coding agent reviewing this project.**

## What This Is

Custom CUDA kernels for every meteorological calculation in MetPy/metrust. 145 hand-written CUDA kernels, same Python API as `metrust.calc` — just `import metcu` instead.

Runs on NVIDIA GPUs via CuPy. Tested on RTX 5090 (170 SMs, 34GB VRAM, CUDA 13).

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
│   ├── test_against_metrust.py  # 69 tests comparing GPU vs CPU values
│   ├── benchmark_complete.py    # All 204 functions benchmarked
│   └── benchmark_hrrr_full.py   # HRRR-scale (1.9M points) benchmark
└── pyproject.toml
```

## How Kernels Are Written

Three patterns used:

### 1. ElementwiseKernel — per-element operations
```python
potential_temperature_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 temperature',
    'float64 theta',
    'theta = (temperature + 273.15) * pow(1000.0 / pressure, 0.28571);',
    'potential_temperature_kernel'
)
```
Used for: potential_temperature, saturation_vapor_pressure, dewpoint, mixing_ratio, wind_speed, etc.

### 2. RawKernel — complex column operations
```python
cape_cin_kernel = cp.RawKernel(r'''
extern "C" __global__
void cape_cin_kernel(const double* pressure, const double* temperature,
                     const double* dewpoint, double* cape_out, double* cin_out,
                     int ncols, int nlevels) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    // Each thread processes one complete sounding column
    // ... LCL, parcel profile (RK4), LFC, EL, integration ...
}
''', 'cape_cin_kernel')
```
Used for: CAPE/CIN, parcel_profile, moist_lapse, LCL, LFC, EL, SRH, bulk_shear, Bunkers

### 3. RawKernel — 2D grid stencils
```python
vorticity_kernel = cp.RawKernel(r'''
extern "C" __global__
void vorticity(const double* u, const double* v, const double* dx,
               const double* dy, double* out, int ny, int nx) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1) return;
    // Centered differences: dv/dx - du/dy
}
''', 'vorticity')
```
Used for: vorticity, divergence, advection, frontogenesis, smoothing, Q-vectors

## What to Review

### Priority 1: CAPE/CIN Kernel Correctness
File: `python/metcu/kernels/thermo.py` — search for `cape_cin`

The CAPE kernel must:
- Compute LCL via iterative drylift
- Build parcel profile using RK4 moist lapse rate with sub-stepping
- Find LFC as the **last** negative→positive buoyancy crossing (NOT the first)
- Accumulate CIN only below the LFC (the cap)
- Accumulate CAPE between LFC and EL
- Use virtual temperature correction

This was a bug in metrust that was just fixed in v0.3.8. Verify the CUDA kernel has the same fix. On a real sounding (32.55N 89.12W HRRR), expected values:
- SBCAPE: ~1050 J/kg
- SBCIN: ~-8 J/kg (NOT -12000)

### Priority 2: Severe Weather Composites
File: `python/metcu/kernels/wind.py`

Check these formulas match metrust/MetPy exactly:
- `significant_tornado_parameter`: STP = (CAPE/1500) * (SRH/150) * (shear/20) * ((2000-LCL)/1000) * ((200+CIN)/150)
- `supercell_composite_parameter`: SCP = (CAPE/1000) * (SRH/50) * (shear/20)
- `compute_ship`: SHIP = (MUCAPE * mixing_ratio * lapse_rate * -T500 * shear) / 42000000
- `compute_ehi`: EHI = CAPE * SRH / 160000

### Priority 3: calc.py Argument Mismatches
File: `python/metcu/calc.py`

The benchmark found 31 FAILs from argument mismatches between calc.py wrappers and kernel functions. Common issues:
- calc.py passes too many args (includes height arrays the kernel doesn't expect)
- calc.py uses `_kernel` suffix names that don't exist
- Some kernels expect 2D arrays but receive 1D

Check every function call in calc.py matches the actual kernel signature in the kernels/ files.

### Priority 4: Large Accuracy Differences
From benchmark_results.txt, these have DIFF > 100:
- `virtual_potential_temperature`: DIFF=4555 — likely unit issue (K vs C)
- `relative_humidity_from_mixing_ratio`: DIFF=1.97e7 — catastrophically wrong formula
- `relative_humidity_from_specific_humidity`: DIFF=1.95e4
- `lfc`: DIFF=1.09e4
- `advection`: DIFF=237
- `geostrophic_wind`: DIFF=5.45e5 — unit conversion (m vs hPa)
- `parcel_profile_with_lcl`: DIFF=975 — K vs C

### Priority 5: Grid Stencil Boundary Handling
All grid stencils show DIFF ~0.03. This is because CUDA kernels skip boundary cells (i<1, i>=nx-1, j<1, j>=ny-1) while metrust may extrapolate. This is expected behavior — verify that INTERIOR values match within rtol=1e-4.

## Benchmark Results Summary

Tested on RTX 5090, HRRR-scale (1,905,141 grid points):

| Category | Avg Speedup | Max Speedup |
|---|---|---|
| Per-element thermo | 6-9x (vs Rust CPU) | 1,180x (vs Python+Pint) |
| Grid stencils | 22x | 43x (frontogenesis) |
| Batch CAPE (500K cols) | GPU-only: 21ms | CPU can't attempt |

## Reference Implementation

The CPU reference is `metrust` v0.3.8 at `C:\Users\drew\metrust-py`. The key source files:
- `crates/wx-math/src/thermo.rs` — all thermodynamic formulas
- `crates/wx-math/src/composite.rs` — severe weather composites
- `crates/metrust/src/calc/thermo.rs` — stability indices
- `python/metrust/calc/__init__.py` — Python wrappers with unit handling

Every met-cu kernel should produce the same numerical result as the corresponding metrust function when given the same input data (stripped of Pint units).

## How to Run Tests

```bash
cd C:\Users\drew\met-cu
pip install -e .
python -m pytest tests/test_against_metrust.py -v    # 69 correctness tests
python tests/benchmark_complete.py                     # Full benchmark (all 204 functions)
python tests/benchmark_hrrr_full.py                    # HRRR-scale timing
```
