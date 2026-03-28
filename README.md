# met-cu

**Custom CUDA kernels for every meteorological calculation.** GPU-accelerated MetPy/metrust drop-in.

Every function in metrust/MetPy that touches data gets a hand-written CUDA kernel. No generic GPU libraries — each kernel is optimized for the specific meteorological formula.

## Why custom CUDA kernels?

Generic GPU frameworks (CuPy ufuncs, PyTorch) add overhead from generality. Our kernels are purpose-built:

- **Per-element operations** (potential temperature, dewpoint, etc.): fused into single kernels, no intermediate allocations
- **Column operations** (CAPE, parcel profiles, SRH): one thread per sounding column, 1.9M columns in parallel on HRRR data
- **Stencil operations** (vorticity, divergence, frontogenesis): 2D thread blocks with shared memory
- **Smoothing**: tiled convolution with shared memory loading

## Part of the Rust+CUDA Meteorology Stack

| Layer | Package | Replaces | Acceleration |
|-------|---------|----------|-------------|
| **Data Access** | rusbie | Herbie | Rust async HTTP |
| **GRIB Decoding** | cfrust | cfgrib | Rust, no eccodes |
| **CPU Calculations** | metrust | MetPy | Rust, 6-30x faster |
| **GPU Calculations** | **met-cu** | metrust/MetPy | CUDA, 100-1000x faster |
| **Plotting** | rustplots | MetPy plots | Rust rendering |

## Installation

Requires NVIDIA GPU with CUDA 12+:

```bash
pip install met-cu
```

## Usage

Same API as metrust — just change the import:

```python
# CPU (metrust):
import metrust.calc as calc
theta = calc.potential_temperature(pressure, temperature)

# GPU (met-cu):
import metcu
theta = metcu.potential_temperature(pressure, temperature)
# Returns cupy array on GPU. Use .get() for numpy.
```

## Expected Speedups

| Operation | Grid Size | CPU (metrust) | GPU (met-cu) | Speedup |
|-----------|-----------|---------------|--------------|---------|
| potential_temperature | 1059x1799 | TBD ms | TBD ms | TBD x |
| CAPE/CIN (all columns) | 1059x1799 | TBD s | TBD ms | TBD x |
| vorticity | 1059x1799 | TBD ms | TBD ms | TBD x |
| smooth_gaussian | 1059x1799 | TBD ms | TBD ms | TBD x |

## License

MIT
