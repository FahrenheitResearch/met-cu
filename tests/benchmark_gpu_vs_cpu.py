"""Benchmark: met-cu CUDA vs metrust Rust/CPU for meteorological calculations."""
import numpy as np
import time
import sys

try:
    import cupy as cp
    HAS_GPU = True
    print(f"GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
except ImportError:
    HAS_GPU = False
    print("No GPU available")
    sys.exit(1)

import metcu
import metrust.calc as mr
from metrust.units import units

def benchmark(name, gpu_fn, cpu_fn, *args, warmup=3, iterations=10):
    """Benchmark GPU vs CPU, return times and speedup."""
    # Warmup GPU
    for _ in range(warmup):
        gpu_fn(*args)
    cp.cuda.Device().synchronize()

    # GPU timing
    gpu_times = []
    for _ in range(iterations):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        result_gpu = gpu_fn(*args)
        cp.cuda.Device().synchronize()
        gpu_times.append(time.perf_counter() - t0)

    gpu_best = min(gpu_times) * 1000

    if cpu_fn is None:
        print(f"  {name:40s}  GPU: {gpu_best:8.2f}ms  CPU: {'--':>8s}  {'--':>6s}")
        return gpu_best, None, None

    # CPU timing
    cpu_times = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        result_cpu = cpu_fn(*args)
        cpu_times.append(time.perf_counter() - t0)

    cpu_best = min(cpu_times) * 1000
    speedup = cpu_best / gpu_best if gpu_best > 0 else 0

    print(f"  {name:40s}  GPU: {gpu_best:8.2f}ms  CPU: {cpu_best:8.2f}ms  {speedup:6.1f}x")
    return gpu_best, cpu_best, speedup

# ─── Test data at various sizes ───
sizes = [
    ("Small (100x100)", 100, 100),
    ("Medium (500x500)", 500, 500),
    ("HRRR-scale (1059x1799)", 1059, 1799),
    ("Large (2000x4000)", 2000, 4000),
]

for size_name, ny, nx in sizes:
    print(f"\n{'=' * 80}")
    print(f"  {size_name}: {ny}x{nx} = {ny*nx:,} grid points")
    print(f"{'=' * 80}")

    # Generate test data
    np.random.seed(42)
    p_2d = np.full((ny, nx), 1000.0)  # surface pressure
    t_2d = np.random.randn(ny, nx) * 15 + 25  # temperature in C
    td_2d = t_2d - np.abs(np.random.randn(ny, nx) * 8)  # dewpoint
    u_2d = np.random.randn(ny, nx) * 10  # u-wind m/s
    v_2d = np.random.randn(ny, nx) * 10  # v-wind m/s
    rh_2d = np.random.rand(ny, nx) * 100  # RH %

    # ─── Per-element operations ───
    print("\n  Per-element operations:")

    benchmark("potential_temperature",
        lambda: metcu.potential_temperature(p_2d, t_2d),
        lambda: mr.potential_temperature(p_2d * units.hPa, t_2d * units.degC),
        warmup=5)

    benchmark("saturation_vapor_pressure",
        lambda: metcu.saturation_vapor_pressure(t_2d),
        lambda: mr.saturation_vapor_pressure(t_2d * units.degC),
        warmup=5)

    benchmark("dewpoint_from_rh",
        lambda: metcu.dewpoint_from_relative_humidity(t_2d, rh_2d),
        lambda: mr.dewpoint_from_relative_humidity(t_2d * units.degC, rh_2d * units.percent),
        warmup=5)

    benchmark("equivalent_potential_temperature",
        lambda: metcu.equivalent_potential_temperature(p_2d, t_2d, td_2d),
        lambda: mr.equivalent_potential_temperature(p_2d * units.hPa, t_2d * units.degC, td_2d * units.degC),
        warmup=5)

    benchmark("virtual_temperature",
        lambda: metcu.virtual_temperature(t_2d, td_2d, p_2d),
        lambda: mr.virtual_temperature(t_2d * units.degC, mr.mixing_ratio(mr.saturation_vapor_pressure(td_2d * units.degC), p_2d * units.hPa)),
        warmup=5)

    benchmark("heat_index",
        lambda: metcu.heat_index(t_2d, rh_2d),
        lambda: mr.heat_index(t_2d * units.degC, rh_2d * units.percent),
        warmup=5)

    benchmark("wind_speed",
        lambda: metcu.wind_speed(u_2d, v_2d),
        lambda: mr.wind_speed(u_2d * units('m/s'), v_2d * units('m/s')),
        warmup=5)

    benchmark("wind_direction",
        lambda: metcu.wind_direction(u_2d, v_2d),
        lambda: mr.wind_direction(u_2d * units('m/s'), v_2d * units('m/s')),
        warmup=5)

    # ─── Grid/stencil operations ───
    print("\n  Grid/stencil operations:")
    dx = np.full((ny, nx), 3000.0)  # 3km grid spacing
    dy = np.full((ny, nx), 3000.0)

    benchmark("vorticity",
        lambda: metcu.vorticity(u_2d, v_2d, dx=dx, dy=dy),
        lambda: mr.vorticity(u_2d * units('m/s'), v_2d * units('m/s'),
                             dx=3000.0 * units.m, dy=3000.0 * units.m),
        warmup=5)

    benchmark("divergence",
        lambda: metcu.divergence(u_2d, v_2d, dx=dx, dy=dy),
        lambda: mr.divergence(u_2d * units('m/s'), v_2d * units('m/s'),
                              dx=3000.0 * units.m, dy=3000.0 * units.m),
        warmup=5)

    benchmark("advection",
        lambda: metcu.advection(t_2d, u_2d, v_2d, dx=dx, dy=dy),
        lambda: mr.advection(t_2d * units.degC, u_2d * units('m/s'), v_2d * units('m/s'),
                             dx=3000.0 * units.m, dy=3000.0 * units.m),
        warmup=5)

    benchmark("smooth_gaussian (sigma=3)",
        lambda: metcu.smooth_gaussian(t_2d, sigma=3),
        lambda: mr.smooth_gaussian(t_2d, 3),
        warmup=5)

    # ─── Column operations (the big GPU wins) ───
    nlevels = 40
    p_col = np.linspace(1000, 100, nlevels)
    t_3d = np.random.randn(ny, nx, nlevels) * 10 + 20 - np.arange(nlevels) * 2
    td_3d = t_3d - np.abs(np.random.randn(ny, nx, nlevels) * 5)

    if ny * nx <= 100000:  # Only run column ops on manageable sizes
        print("\n  Column operations (per-column vertical):")

        pressure_3d_pa = np.broadcast_to(
            (p_col[:, None, None] * 100.0), (nlevels, ny, nx)
        ).copy()
        temperature_3d_c = np.transpose(t_3d, (2, 0, 1)).copy()
        dewpoint_3d_c = np.transpose(td_3d, (2, 0, 1)).copy()
        es_td = 6.112 * np.exp(17.67 * dewpoint_3d_c / (dewpoint_3d_c + 243.5))
        qvapor_3d = 0.6219569100577033 * es_td / (p_col[:, None, None] - es_td)
        z_std = 44330.0 * (1.0 - (p_col / 1013.25) ** 0.1903)
        height_agl_3d = np.broadcast_to(
            (z_std - z_std[0])[:, None, None], (nlevels, ny, nx)
        ).copy()
        psfc_pa = np.full((ny, nx), p_col[0] * 100.0)
        t2_k = t_3d[:, :, 0] + 273.15
        q2 = qvapor_3d[0]

        benchmark("CAPE/CIN (all columns)",
            lambda: metcu.compute_cape_cin(
                pressure_3d_pa, temperature_3d_c, qvapor_3d, height_agl_3d,
                psfc_pa, t2_k, q2,
            ),
            None,  # CPU version would be very slow, skip for large grids
            warmup=3, iterations=5)

print("\nBenchmark complete.")
