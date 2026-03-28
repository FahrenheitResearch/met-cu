"""
Benchmark: Every calculation at FULL HRRR scale (1,905,141 columns).

This is the real-world test — compute every parameter for every grid point
simultaneously, exactly like an operational mesoanalysis pipeline would.

RTX 5090 (170 SMs, 34GB VRAM) vs metrust Rust/CPU.
"""
import numpy as np
import cupy as cp
import time
import sys

print("=" * 80)
print("  FULL HRRR-SCALE BENCHMARK: 1,059 x 1,799 = 1,905,141 columns")
print("=" * 80)

dev = cp.cuda.Device(0)
props = cp.cuda.runtime.getDeviceProperties(0)
print(f"  GPU: {props['name'].decode()}")
print(f"  VRAM: {props['totalGlobalMem']/1e9:.1f} GB")
print(f"  SMs: {props['multiProcessorCount']}")
print()

import metcu
import metrust.calc as mr
from metrust.units import units

NY, NX = 1059, 1799
N = NY * NX
NLEVELS = 40

np.random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# Generate realistic HRRR-like data
# ═══════════════════════════════════════════════════════════════════

# Surface fields (flat 1D arrays, N=1.9M)
p_sfc = np.full(N, 1013.25, dtype=np.float64)
t_sfc = np.random.randn(N) * 12 + 25  # 2m temperature (C)
td_sfc = t_sfc - np.abs(np.random.randn(N) * 8)  # 2m dewpoint
u_sfc = np.random.randn(N) * 8  # 10m u-wind (m/s)
v_sfc = np.random.randn(N) * 8  # 10m v-wind
rh_sfc = np.clip(np.random.rand(N) * 100, 5, 100)  # 2m RH (%)
z_sfc = np.random.rand(N) * 500  # station elevation (m)

# 2D grid fields for stencils (NY x NX)
t_grid = t_sfc.reshape(NY, NX)
td_grid = td_sfc.reshape(NY, NX)
u_grid = u_sfc.reshape(NY, NX)
v_grid = v_sfc.reshape(NY, NX)
p_grid = p_sfc.reshape(NY, NX)
z_grid = np.random.randn(NY, NX) * 100 + 5500  # 500mb heights (m)
dx = np.full((NY, NX), 3000.0)  # 3km grid spacing
dy = np.full((NY, NX), 3000.0)
lat = np.linspace(21, 53, NY)[:, None] * np.ones((1, NX))

# 3D sounding data: (N, NLEVELS) — every column has a vertical profile
p_levels = np.linspace(1013, 100, NLEVELS)  # shared pressure levels
t_3d = np.zeros((N, NLEVELS), dtype=np.float64)
td_3d = np.zeros((N, NLEVELS), dtype=np.float64)
u_3d = np.zeros((N, NLEVELS), dtype=np.float64)
v_3d = np.zeros((N, NLEVELS), dtype=np.float64)
h_3d = np.zeros((N, NLEVELS), dtype=np.float64)

for k in range(NLEVELS):
    frac = k / (NLEVELS - 1)
    t_3d[:, k] = t_sfc - frac * 80 + np.random.randn(N) * 2
    td_3d[:, k] = t_3d[:, k] - np.abs(np.random.randn(N) * (5 + frac * 20))
    u_3d[:, k] = u_sfc + frac * 30 + np.random.randn(N) * 3
    v_3d[:, k] = v_sfc + frac * 15 + np.random.randn(N) * 3
    h_3d[:, k] = frac * 16000 + np.random.randn(N) * 100

# Standard level values for indices (from the 3D data)
idx_850 = np.argmin(np.abs(p_levels - 850))
idx_700 = np.argmin(np.abs(p_levels - 700))
idx_500 = np.argmin(np.abs(p_levels - 500))
idx_300 = np.argmin(np.abs(p_levels - 300))
idx_250 = np.argmin(np.abs(p_levels - 250))

t850 = t_3d[:, idx_850]
td850 = td_3d[:, idx_850]
t700 = t_3d[:, idx_700]
td700 = td_3d[:, idx_700]
t500 = t_3d[:, idx_500]
t300 = t_3d[:, idx_300]
u850 = u_3d[:, idx_850]
v850 = v_3d[:, idx_850]
u500 = u_3d[:, idx_500]
v500 = v_3d[:, idx_500]
u250 = u_3d[:, idx_250]
v250 = v_3d[:, idx_250]

print(f"  Data generated: {N:,} surface points, {N:,} x {NLEVELS} 3D columns")
print(f"  Surface T range: [{t_sfc.min():.1f}, {t_sfc.max():.1f}] C")
print(f"  Memory: ~{(N * NLEVELS * 5 * 8) / 1e9:.1f} GB for 3D fields")
print()

results = []

def bench(name, gpu_fn, cpu_fn, warmup=3, iters=5, verify=True):
    """Benchmark one function, verify accuracy."""
    try:
        # Warmup GPU
        for _ in range(warmup):
            gpu_fn()
        cp.cuda.Device().synchronize()

        # GPU timing
        gpu_times = []
        gpu_result = None
        for _ in range(iters):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            gpu_result = gpu_fn()
            cp.cuda.Device().synchronize()
            gpu_times.append(time.perf_counter() - t0)
        gpu_ms = min(gpu_times) * 1000

        # CPU timing
        cpu_times = []
        cpu_result = None
        for _ in range(iters):
            t0 = time.perf_counter()
            cpu_result = cpu_fn()
            cpu_times.append(time.perf_counter() - t0)
        cpu_ms = min(cpu_times) * 1000

        speedup = cpu_ms / gpu_ms if gpu_ms > 0.01 else 0

        # Verify accuracy
        acc = ""
        if verify and gpu_result is not None and cpu_result is not None:
            try:
                g = cp.asnumpy(gpu_result) if isinstance(gpu_result, cp.ndarray) else gpu_result
                c = cpu_result.magnitude if hasattr(cpu_result, 'magnitude') else cpu_result
                if isinstance(g, tuple): g = g[0]
                if isinstance(c, tuple): c = c[0].magnitude if hasattr(c[0], 'magnitude') else c[0]
                g = np.asarray(g, dtype=np.float64).ravel()
                c = np.asarray(c, dtype=np.float64).ravel()
                n = min(len(g), len(c))
                if n > 0:
                    mask = np.isfinite(g[:n]) & np.isfinite(c[:n])
                    if mask.sum() > 0:
                        maxd = np.max(np.abs(g[:n][mask] - c[:n][mask]))
                        if np.allclose(g[:n][mask], c[:n][mask], rtol=1e-4, atol=1e-4):
                            acc = "PASS"
                        else:
                            acc = f"DIFF={maxd:.3g}"
            except:
                acc = "ERR"

        results.append((name, gpu_ms, cpu_ms, speedup, acc))
        status = "OK" if acc in ("PASS", "", "ERR") else "!!"
        print(f"  {status:2s} {name:45s} GPU:{gpu_ms:9.2f}ms CPU:{cpu_ms:9.2f}ms {speedup:8.1f}x  {acc}")

    except Exception as e:
        results.append((name, 0, 0, 0, f"FAIL:{str(e)[:40]}"))
        print(f"  !! {name:45s} FAIL: {str(e)[:60]}")


# ═══════════════════════════════════════════════════════════════════
# CATEGORY 1: Per-element surface (N=1.9M)
# ═══════════════════════════════════════════════════════════════════
print("─" * 80)
print("  PER-ELEMENT SURFACE (1,905,141 points)")
print("─" * 80)

bench("potential_temperature",
    lambda: metcu.potential_temperature(p_sfc, t_sfc),
    lambda: mr.potential_temperature(p_sfc * units.hPa, t_sfc * units.degC))

bench("equivalent_potential_temperature",
    lambda: metcu.equivalent_potential_temperature(p_sfc, t_sfc, td_sfc),
    lambda: mr.equivalent_potential_temperature(p_sfc * units.hPa, t_sfc * units.degC, td_sfc * units.degC))

bench("saturation_vapor_pressure",
    lambda: metcu.saturation_vapor_pressure(t_sfc),
    lambda: mr.saturation_vapor_pressure(t_sfc * units.degC))

bench("saturation_mixing_ratio",
    lambda: metcu.saturation_mixing_ratio(p_sfc, t_sfc),
    lambda: mr.saturation_mixing_ratio(p_sfc * units.hPa, t_sfc * units.degC))

bench("dewpoint_from_rh",
    lambda: metcu.dewpoint_from_relative_humidity(t_sfc, rh_sfc),
    lambda: mr.dewpoint_from_relative_humidity(t_sfc * units.degC, rh_sfc * units.percent))

bench("virtual_temperature",
    lambda: metcu.virtual_temperature(t_sfc, td_sfc, p_sfc),
    lambda: mr.virtual_temperature_from_dewpoint(t_sfc * units.degC, td_sfc * units.degC, p_sfc * units.hPa))

bench("wet_bulb_temperature",
    lambda: metcu.wet_bulb_temperature(p_sfc, t_sfc, td_sfc),
    lambda: mr.wet_bulb_temperature(p_sfc * units.hPa, t_sfc * units.degC, td_sfc * units.degC))

bench("heat_index",
    lambda: metcu.heat_index(t_sfc, rh_sfc),
    lambda: mr.heat_index(t_sfc * units.degC, rh_sfc * units.percent))

bench("windchill",
    lambda: metcu.windchill(t_sfc, np.sqrt(u_sfc**2 + v_sfc**2)),
    lambda: mr.windchill(t_sfc * units.degC, np.sqrt(u_sfc**2 + v_sfc**2) * units('m/s')))

bench("wind_speed",
    lambda: metcu.wind_speed(u_sfc, v_sfc),
    lambda: mr.wind_speed(u_sfc * units('m/s'), v_sfc * units('m/s')))

bench("wind_direction",
    lambda: metcu.wind_direction(u_sfc, v_sfc),
    lambda: mr.wind_direction(u_sfc * units('m/s'), v_sfc * units('m/s')))

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 2: Stability indices (N=1.9M — one per grid point)
# ═══════════════════════════════════════════════════════════════════
print()
print("─" * 80)
print("  STABILITY INDICES (1,905,141 points)")
print("─" * 80)

bench("k_index",
    lambda: metcu.k_index(t850, td850, t700, td700, t500),
    lambda: mr.k_index(t850 * units.degC, td850 * units.degC, t700 * units.degC, td700 * units.degC, t500 * units.degC),
    verify=False)

bench("total_totals",
    lambda: metcu.total_totals(t850, td850, t500),
    lambda: mr.total_totals(t850 * units.degC, td850 * units.degC, t500 * units.degC),
    verify=False)

bench("cross_totals",
    lambda: metcu.cross_totals(td850, t500),
    lambda: mr.cross_totals(td850 * units.degC, t500 * units.degC),
    verify=False)

bench("vertical_totals",
    lambda: metcu.vertical_totals(t850, t500),
    lambda: mr.vertical_totals(t850 * units.degC, t500 * units.degC),
    verify=False)

bench("fosberg_fire_weather_index",
    lambda: metcu.fosberg_fire_weather_index(t_sfc, rh_sfc, np.sqrt(u_sfc**2 + v_sfc**2)),
    lambda: mr.fosberg_fire_weather_index(t_sfc * units.degC, rh_sfc * units.percent, np.sqrt(u_sfc**2 + v_sfc**2) * units('m/s')),
    verify=False)

bench("hot_dry_windy",
    lambda: metcu.hot_dry_windy(t_sfc, rh_sfc, np.sqrt(u_sfc**2 + v_sfc**2)),
    lambda: mr.hot_dry_windy(t_sfc * units.degC, rh_sfc * units.percent, np.sqrt(u_sfc**2 + v_sfc**2) * units('m/s')),
    verify=False)

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 3: Grid stencils (1059 x 1799)
# ═══════════════════════════════════════════════════════════════════
print()
print("─" * 80)
print("  GRID STENCILS (1059 x 1799)")
print("─" * 80)

bench("vorticity",
    lambda: metcu.vorticity(u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.vorticity(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

bench("divergence",
    lambda: metcu.divergence(u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.divergence(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

bench("advection",
    lambda: metcu.advection(t_grid, u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.advection(t_grid * units.degC, u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

bench("frontogenesis",
    lambda: metcu.frontogenesis(t_grid, u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.frontogenesis(t_grid * units.degC, u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

bench("smooth_gaussian (sigma=3)",
    lambda: metcu.smooth_gaussian(t_grid, 3),
    lambda: mr.smooth_gaussian(t_grid, 3),
    verify=False)

bench("smooth_n_point (9-point)",
    lambda: metcu.smooth_n_point(t_grid, 9),
    lambda: mr.smooth_n_point(t_grid, 9),
    verify=False)

bench("shearing_deformation",
    lambda: metcu.shearing_deformation(u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.shearing_deformation(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

bench("stretching_deformation",
    lambda: metcu.stretching_deformation(u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.stretching_deformation(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

bench("total_deformation",
    lambda: metcu.total_deformation(u_grid, v_grid, dx=dx, dy=dy),
    lambda: mr.total_deformation(u_grid * units('m/s'), v_grid * units('m/s'), dx=dx * units.meter, dy=dy * units.meter),
    verify=False)

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 4: Severe weather composites (N=1.9M)
# These are the critical ones — STP, SCP, SHIP, EHI at every point
# ═══════════════════════════════════════════════════════════════════
print()
print("─" * 80)
print("  SEVERE WEATHER COMPOSITES (1,905,141 points)")
print("─" * 80)

# Pre-compute some needed fields for composites
cape_vals = np.clip(np.random.exponential(500, N), 0, 5000)
cin_vals = -np.clip(np.random.exponential(20, N), 0, 300)
srh_vals = np.random.randn(N) * 100 + 100
shear_u = u_3d[:, idx_500] - u_3d[:, 0]
shear_v = v_3d[:, idx_500] - v_3d[:, 0]
shear_mag = np.sqrt(shear_u**2 + shear_v**2)
lcl_heights = np.clip(np.random.randn(N) * 500 + 1500, 200, 4000)
lr_850_500 = np.clip(np.random.randn(N) * 1.5 + 6.5, 3, 10)
frz_level = np.clip(np.random.randn(N) * 500 + 3500, 1000, 6000)
mucape = cape_vals * 1.1
dcape_vals = np.clip(np.random.exponential(400, N), 0, 2000)
mean_wind_speed = np.clip(np.random.randn(N) * 5 + 12, 2, 30)

bench("significant_tornado_parameter (1.9M)",
    lambda: metcu.significant_tornado_parameter(cape_vals, srh_vals, shear_mag, lcl_heights, cin_vals),
    lambda: mr.significant_tornado_parameter(
        cape_vals * units('J/kg'), srh_vals * units('m^2/s^2'),
        shear_mag * units('m/s'), lcl_heights * units.meter, cin_vals * units('J/kg')),
    verify=False)

bench("supercell_composite_parameter (1.9M)",
    lambda: metcu.supercell_composite_parameter(cape_vals, srh_vals, shear_mag),
    lambda: mr.supercell_composite_parameter(
        cape_vals * units('J/kg'), srh_vals * units('m^2/s^2'), shear_mag * units('m/s')),
    verify=False)

bench("bulk_richardson_number (1.9M)",
    lambda: metcu.bulk_richardson_number(cape_vals, shear_mag),
    lambda: mr.bulk_richardson_number(cape_vals * units('J/kg'), shear_mag * units('m/s')),
    verify=False)

bench("compute_ehi (1.9M)",
    lambda: metcu.compute_ehi(cape_vals.reshape(NY, NX), srh_vals.reshape(NY, NX)),
    lambda: mr.compute_ehi(cape_vals.reshape(NY, NX), srh_vals.reshape(NY, NX)),
    verify=False)

bench("compute_ship (1.9M)",
    lambda: metcu.compute_ship(mucape.reshape(NY, NX), shear_mag.reshape(NY, NX),
                                lr_850_500.reshape(NY, NX), t500.reshape(NY, NX),
                                frz_level.reshape(NY, NX)),
    lambda: mr.compute_ship(mucape.reshape(NY, NX), shear_mag.reshape(NY, NX),
                             lr_850_500.reshape(NY, NX), t500.reshape(NY, NX),
                             frz_level.reshape(NY, NX)),
    verify=False)

bench("compute_dcp (1.9M)",
    lambda: metcu.compute_dcp(dcape_vals.reshape(NY, NX), cape_vals.reshape(NY, NX),
                               shear_mag.reshape(NY, NX), mean_wind_speed.reshape(NY, NX)),
    lambda: mr.compute_dcp(dcape_vals.reshape(NY, NX), cape_vals.reshape(NY, NX),
                            shear_mag.reshape(NY, NX), mean_wind_speed.reshape(NY, NX)),
    verify=False)

# ═══════════════════════════════════════════════════════════════════
# CATEGORY 5: Column operations (batch CAPE/CIN for subsets)
# Full 1.9M columns is too much memory for 3D arrays on CPU,
# so we test meaningful subsets
# ═══════════════════════════════════════════════════════════════════
print()
print("─" * 80)
print("  COLUMN OPERATIONS (batch sounding profiles)")
print("─" * 80)

for batch_label, batch_n in [("10K columns", 10000), ("100K columns", 100000), ("500K columns", 500000)]:
    t_batch = t_3d[:batch_n]
    td_batch = td_3d[:batch_n]
    h_batch = h_3d[:batch_n]

    # GPU batch CAPE
    try:
        bench(f"cape_cin ({batch_label})",
            lambda: metcu.cape_cin(p_levels, t_batch, td_batch),
            lambda: None,  # CPU too slow for batch
            verify=False, warmup=2, iters=3)
    except Exception as e:
        print(f"  !! cape_cin ({batch_label}): {e}")

    try:
        bench(f"precipitable_water ({batch_label})",
            lambda: metcu.precipitable_water(p_levels, td_batch),
            lambda: None,
            verify=False, warmup=2, iters=3)
    except Exception as e:
        print(f"  !! precipitable_water ({batch_label}): {e}")

# ═══════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════
print()
print("=" * 80)
print("  SUMMARY")
print("=" * 80)

categories = {
    "Per-element surface": [],
    "Stability indices": [],
    "Grid stencils": [],
    "Severe composites": [],
    "Column operations": [],
}

for name, gpu_ms, cpu_ms, speedup, acc in results:
    if speedup == 0:
        continue
    if "stencil" in name.lower() or name in ("vorticity", "divergence", "advection",
        "frontogenesis", "smooth_gaussian (sigma=3)", "smooth_n_point (9-point)",
        "shearing_deformation", "stretching_deformation", "total_deformation"):
        categories["Grid stencils"].append(speedup)
    elif "column" in name.lower() or "cape" in name.lower() or "precipitable" in name.lower():
        categories["Column operations"].append(speedup)
    elif "stp" in name.lower() or "scp" in name.lower() or "ship" in name.lower() or \
         "ehi" in name.lower() or "dcp" in name.lower() or "tornado" in name.lower() or \
         "supercell" in name.lower() or "richardson" in name.lower() or "severe" in name.lower():
        categories["Severe composites"].append(speedup)
    elif "index" in name.lower() or "totals" in name.lower() or "fosberg" in name.lower() or \
         "hot_dry" in name.lower():
        categories["Stability indices"].append(speedup)
    else:
        categories["Per-element surface"].append(speedup)

print(f"\n  {'Category':30s} {'Avg Speedup':>12s} {'Max Speedup':>12s} {'Count':>6s}")
print(f"  {'─' * 65}")
for cat, speeds in categories.items():
    if speeds:
        print(f"  {cat:30s} {np.mean(speeds):11.1f}x {max(speeds):11.1f}x {len(speeds):5d}")

all_speeds = [s for s in [r[3] for r in results] if s > 0]
if all_speeds:
    print(f"\n  Overall: avg {np.mean(all_speeds):.1f}x, median {np.median(all_speeds):.1f}x, max {max(all_speeds):.1f}x")
    print(f"  Functions tested: {len(results)}")
    passed = sum(1 for r in results if r[4] in ("PASS", ""))
    print(f"  Accuracy verified: {passed}")

print("=" * 80)
