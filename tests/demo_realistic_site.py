"""
REALISTIC MODEL SITE WORKFLOW

What Pivotal Weather / College of DuPage / Tropical Tidbits ACTUALLY compute.

Key insight: CAPE, CIN, SRH, LI, PWAT, helicity are PRE-COMPUTED by the model
and included in the GRIB file. Sites just read those fields and combine them
into composites (STP, SCP, SHIP, EHI). They don't recompute CAPE from scratch.

What they DO compute:
  - Severe composites (STP, SCP, SHIP, EHI, DCP, BRN) from GRIB ingredients
  - Derived surface fields (theta-e, mixing ratio, lapse rates, RH)
  - Grid stencils (vorticity, divergence, frontogenesis)
  - Bulk shear from wind profiles (if not in GRIB)
  - Smoothing for display
"""
import numpy as np
import cupy as cp
import time
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

props = cp.cuda.runtime.getDeviceProperties(0)
NY, NX = 1059, 1799
N = NY * NX

print()
print("=" * 70)
print("  REALISTIC MODEL SITE WORKFLOW (Pivotal-style)")
print(f"  GPU: {props['name'].decode()} | Grid: {NY}x{NX} = {N:,}")
print("=" * 70)

# Pre-computed fields (read from GRIB — zero compute cost)
print()
print("  FROM GRIB (pre-computed by HRRR):")
print("    SBCAPE, MLCAPE, MUCAPE, SBCIN, MLCIN")
print("    SRH 0-1km, SRH 0-3km, LCL height, Lifted Index, PWAT")
print("    2m T/Td, 10m U/V, MSLP, REFC, standard level T/Td/U/V/HGT")
print()

np.random.seed(42)
sbcape = cp.clip(cp.asarray(np.random.exponential(800, N)), 0, 6000)
mlcape = sbcape * 0.7
mucape = sbcape * 1.1
sbcin = -cp.clip(cp.asarray(np.random.exponential(30, N)), 0, 300)
srh_01 = cp.asarray(np.random.randn(N) * 80 + 80)
srh_03 = cp.asarray(np.random.randn(N) * 120 + 150)
lcl_hgt = cp.clip(cp.asarray(np.random.randn(N) * 500 + 1200), 100, 4000)
t_sfc = cp.asarray(np.random.randn(N) * 8 + 28)
td_sfc = t_sfc - cp.abs(cp.asarray(np.random.randn(N) * 6))
u_sfc = cp.asarray(np.random.randn(N) * 5 + 3)
v_sfc = cp.asarray(np.random.randn(N) * 5 + 5)
t_850 = cp.asarray(np.random.randn(N) * 4 + 15)
td_850 = t_850 - cp.abs(cp.asarray(np.random.randn(N) * 5))
t_700 = cp.asarray(np.random.randn(N) * 3 + 5)
td_700 = t_700 - cp.abs(cp.asarray(np.random.randn(N) * 10))
t_500 = cp.asarray(np.random.randn(N) * 3 - 15)
u_500 = cp.asarray(np.random.randn(N) * 8 + 25)
v_500 = cp.asarray(np.random.randn(N) * 8 + 10)
u_250 = cp.asarray(np.random.randn(N) * 10 + 40)
v_250 = cp.asarray(np.random.randn(N) * 10 + 5)
u_850 = cp.asarray(np.random.randn(N) * 4 + 8)
v_850 = cp.asarray(np.random.randn(N) * 4 + 12)
dcape = cp.clip(cp.asarray(np.random.exponential(400, N)), 0, 2000)
mean_wind = cp.clip(cp.asarray(np.random.randn(N) * 5 + 12), 2, 30)
rh_sfc = cp.clip(cp.asarray(np.random.rand(N) * 100), 20, 100)

t_grid = t_sfc.reshape(NY, NX)
td_grid = td_sfc.reshape(NY, NX)
u_grid = u_sfc.reshape(NY, NX)
v_grid = v_sfc.reshape(NY, NX)
dx = cp.full((NY, NX), 3000.0)
dy = cp.full((NY, NX), 3000.0)

from metcu.kernels import thermo, wind, grid

total = 0.0
results = []

def timed(name, fn):
    global total
    cp.cuda.Device().synchronize()
    fn()  # warmup
    cp.cuda.Device().synchronize()
    times = []
    for _ in range(5):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        fn()
        cp.cuda.Device().synchronize()
        times.append(time.perf_counter() - t0)
    ms = min(times) * 1000
    total += ms
    results.append((name, ms))

# ================================================================
print("  COMPUTED BY THE SITE:")
print("  " + "-" * 60)
print()
print("  Severe Composites:")

shear_06 = cp.sqrt((u_500 - u_sfc)**2 + (v_500 - v_sfc)**2)
shear_01 = cp.sqrt((u_850 - u_sfc)**2 + (v_850 - v_sfc)**2)

timed("STP (Sig Tornado Parameter)", lambda: wind.significant_tornado_parameter(sbcape, srh_03, shear_06, lcl_hgt))
timed("SCP (Supercell Composite)", lambda: wind.supercell_composite_parameter(sbcape, srh_03, shear_06))
timed("SHIP (Sig Hail Parameter)", lambda: wind.compute_ship(
    mucape.reshape(NY,NX), shear_06.reshape(NY,NX),
    ((t_850 - t_500)/3.5).reshape(NY,NX), t_500.reshape(NY,NX),
    cp.full((NY,NX), 3500.0)))
timed("EHI (Energy-Helicity Index)", lambda: wind.compute_ehi(sbcape.reshape(NY,NX), srh_03.reshape(NY,NX)))
timed("DCP (Derecho Composite)", lambda: wind.compute_dcp(
    dcape.reshape(NY,NX), sbcape.reshape(NY,NX), shear_06.reshape(NY,NX), mean_wind.reshape(NY,NX)))
timed("BRN (Bulk Richardson Number)", lambda: wind.bulk_richardson_number(sbcape, shear_06))
timed("0-6km Bulk Shear", lambda: cp.sqrt((u_500 - u_sfc)**2 + (v_500 - v_sfc)**2))
timed("0-1km Bulk Shear", lambda: cp.sqrt((u_850 - u_sfc)**2 + (v_850 - v_sfc)**2))

print()
print("  Stability Indices:")
timed("K-Index", lambda: wind.k_index(t_850, t_700, t_500, td_850, td_700))
timed("Total Totals", lambda: wind.total_totals(t_850, t_500, td_850))
timed("850-500mb Lapse Rate", lambda: (t_850 - t_500) / 3.5)
timed("700-500mb Lapse Rate", lambda: (t_700 - t_500) / 2.5)
timed("250mb Jet Speed", lambda: cp.sqrt(u_250**2 + v_250**2))

print()
print("  Surface Derived:")
timed("Theta-E", lambda: thermo.equivalent_potential_temperature(cp.full(N, 1013.0), t_sfc, td_sfc))
timed("Mixing Ratio", lambda: thermo.saturation_mixing_ratio(cp.full(N, 1013.0), td_sfc))
timed("Relative Humidity", lambda: thermo.relative_humidity_from_dewpoint(t_sfc, td_sfc))
timed("Wind Speed", lambda: wind.wind_speed(u_sfc, v_sfc))
timed("Wind Direction", lambda: wind.wind_direction(u_sfc, v_sfc))
timed("Heat Index", lambda: thermo.heat_index(t_sfc, rh_sfc))

print()
print("  Fire Weather:")
timed("Fosberg FWI", lambda: wind.fosberg_fire_weather_index(t_sfc, rh_sfc, wind.wind_speed(u_sfc, v_sfc)))
timed("Hot-Dry-Windy", lambda: wind.hot_dry_windy(t_sfc, rh_sfc, wind.wind_speed(u_sfc, v_sfc)))

print()
print("  Grid Analysis:")
timed("Vorticity", lambda: grid.vorticity(u_grid, v_grid, dx, dy))
timed("Divergence", lambda: grid.divergence(u_grid, v_grid, dx, dy))
timed("Frontogenesis", lambda: grid.frontogenesis(t_grid, u_grid, v_grid, dx, dy))
timed("Temp Advection", lambda: grid.advection(t_grid, u_grid, v_grid, dx, dy))
timed("Total Deformation", lambda: grid.total_deformation(u_grid, v_grid, dx, dy))
timed("Gaussian Smoothing", lambda: grid.smooth_gaussian(t_grid, 3))

# ================================================================
print()
print("=" * 70)
print("  RESULTS")
print("=" * 70)
print()
print(f"  {'Product':35s} {'Time':>8s}")
print(f"  " + "-" * 50)
for name, ms in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"  {name:35s} {ms:7.2f}ms")
print(f"  " + "-" * 50)
print(f"  {'TOTAL (' + str(len(results)) + ' products)':35s} {total:7.2f}ms")
print()

grids_year = 576 * 365
annual_s = grids_year * total / 1000
annual_hr = annual_s / 3600
annual_min = annual_s / 60

print(f"  Per HRRR grid:  {total:.1f}ms = {total/1000:.4f}s")
print(f"  Per day (576 grids): {576 * total / 1000:.1f}s")
print(f"  Per year ({grids_year:,} grids): {annual_min:.0f} min = {annual_hr:.1f} hrs")
print()
print(f"  That's {len(results)} Pivotal-equivalent products")
print(f"  for {N:,} grid points in {total:.0f}ms.")
print()

# What a CPU would take
cpu_estimate_s = total / 1000 * 30  # ~30x slower on CPU
print(f"  CPU comparison (estimated):")
print(f"    Per grid:  {cpu_estimate_s*1000:.0f}ms = {cpu_estimate_s:.1f}s")
print(f"    Per year:  {grids_year * cpu_estimate_s / 3600:.0f} hrs")
print(f"    Speedup:   ~30x")
print()
print("=" * 70)
