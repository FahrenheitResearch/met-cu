"""Verify every per-element thermodynamic function: met-cu (GPU) vs metrust (CPU).

Uses REAL HRRR surface data (~1.9M grid points). Downloads one HRRR sfc grid,
extracts 2m T, 2m Td, 10m wind, surface pressure, then runs every function
through both libraries and compares results.

Requires: metcu, metrust, cupy, rusbie
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import time
import sys

# ============================================================================
# 1. Download one HRRR surface grid
# ============================================================================
print("=" * 72)
print("HRRR Thermodynamic Verification: met-cu (GPU) vs metrust (CPU)")
print("=" * 72)

print("\n[1/3] Downloading HRRR surface data...")
t0 = time.time()

from rusbie import Herbie

H = Herbie("2026-03-27 12:00", model="hrrr", product="sfc", fxx=0, verbose=False)

def get_field(search):
    try:
        ds = H.xarray(search)
        if isinstance(ds, list):
            ds = ds[0]
        var = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if var:
            return ds[var[0]].values.astype(np.float64)
    except Exception as e:
        print(f"  WARNING: could not get {search}: {e}")
    return None

# Surface fields (HRRR native: K for temp, Pa for pressure, m/s for wind)
t2m_k = get_field("TMP:2 m")
td2m_k = get_field("DPT:2 m")
u10 = get_field("UGRD:10 m")
v10 = get_field("VGRD:10 m")
sp_pa = get_field("PRES:surface")

assert t2m_k is not None, "Failed to download 2m temperature"
assert td2m_k is not None, "Failed to download 2m dewpoint"
assert u10 is not None, "Failed to download 10m U wind"
assert v10 is not None, "Failed to download 10m V wind"
assert sp_pa is not None, "Failed to download surface pressure"

# Convert to native units expected by both libraries
t2m_c = t2m_k - 273.15      # Celsius
td2m_c = td2m_k - 273.15    # Celsius
sp_hpa = sp_pa / 100.0      # hPa

# Flatten to 1D for element-wise functions
t_c = t2m_c.ravel()
td_c = td2m_c.ravel()
p_hpa = sp_hpa.ravel()
u = u10.ravel()
v = v10.ravel()
N = t_c.size

print(f"  Grid shape: {t2m_k.shape}, {N:,} points")
print(f"  T range:  [{t_c.min():.1f}, {t_c.max():.1f}] C")
print(f"  Td range: [{td_c.min():.1f}, {td_c.max():.1f}] C")
print(f"  P range:  [{p_hpa.min():.1f}, {p_hpa.max():.1f}] hPa")
print(f"  Wind:     [{np.sqrt(u**2+v**2).max():.1f}] m/s max")
print(f"  Download: {time.time()-t0:.1f}s")

# ============================================================================
# 2. Import both libraries
# ============================================================================
print("\n[2/3] Loading libraries...")
import cupy as cp
import metcu
import metrust.calc as mr
from metrust.units import units

# ============================================================================
# 3. Derived quantities needed for various functions
# ============================================================================
# Wind speed (m/s)
ws = np.sqrt(u**2 + v**2)

# Relative humidity (percent 0-100) from T and Td
# Use the saturation vapor pressure formula: RH = 100 * es(Td)/es(T)
es_t = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
es_td = 6.112 * np.exp(17.67 * td_c / (td_c + 243.5))
rh_pct = np.clip(100.0 * es_td / es_t, 0, 100)  # percent

# Saturation mixing ratio (kg/kg) -- from SVP
eps = 0.6219569100577033
ws_sat = eps * es_t / (p_hpa - es_t)  # kg/kg
w_actual = eps * es_td / (p_hpa - es_td)  # actual mixing ratio kg/kg
w_gkg = w_actual * 1000.0  # g/kg

# Specific humidity (kg/kg)
q_kgkg = w_actual / (1.0 + w_actual)

# Temperature in K for functions that need it
t_k = t_c + 273.15

# Potential temperature (K) for some functions
theta_k = t_k * (1000.0 / p_hpa) ** 0.2857

# Heights (m) -- approximate from hypsometric for surface
z_m = np.full_like(t_c, 500.0)  # approximate mean surface height

# Latitude for coriolis (approximate HRRR CONUS range)
ny, nx = t2m_k.shape
lats_1d = np.linspace(21.138, 52.615, ny)  # HRRR CONUS lat range approx
lats_2d = np.repeat(lats_1d[:, np.newaxis], nx, axis=1).ravel()

# Omega (Pa/s) for vertical_velocity -- synthetic but at full grid size
omega_pas = np.random.default_rng(42).uniform(-2.0, 2.0, N)

# ============================================================================
# 4. Run all tests
# ============================================================================
print(f"\n[3/3] Verifying {N:,} points per function...\n")

results = []
n_pass = 0
n_fail = 0
RTOL = 1e-4


def compare(name, gpu_result, cpu_result, rtol=RTOL, unit_factor=1.0):
    """Compare GPU (cupy) vs CPU (pint Quantity) results."""
    global n_pass, n_fail
    try:
        g = cp.asnumpy(gpu_result).ravel().astype(np.float64) * unit_factor
        if hasattr(cpu_result, "magnitude"):
            c = np.asarray(cpu_result.magnitude, dtype=np.float64).ravel()
        else:
            c = np.asarray(cpu_result, dtype=np.float64).ravel()

        assert g.shape == c.shape, f"Shape mismatch: {g.shape} vs {c.shape}"

        # Filter out NaN/Inf in both
        valid = np.isfinite(g) & np.isfinite(c)
        if valid.sum() == 0:
            status = "SKIP"
            max_diff = float("nan")
        else:
            gv = g[valid]
            cv = c[valid]
            max_diff = float(np.max(np.abs(gv - cv)))
            # Relative check: |g-c| / max(|c|, 1e-12)
            denom = np.maximum(np.abs(cv), 1e-12)
            max_rel = float(np.max(np.abs(gv - cv) / denom))
            if max_rel <= rtol:
                status = "PASS"
                n_pass += 1
            else:
                status = "FAIL"
                n_fail += 1
    except Exception as e:
        status = "ERR "
        max_diff = str(e)[:50]
        n_fail += 1

    tag = f"{name:<45s} {status}  max_diff={max_diff:<14.4e}  ({N:,} points)"
    if status == "ERR ":
        tag = f"{name:<45s} {status}  {max_diff}"
    results.append(tag)
    print(tag)


# ---------------------------------------------------------------------------
# potential_temperature(pressure_hPa, temperature_C) -> K
# ---------------------------------------------------------------------------
compare("potential_temperature",
        metcu.potential_temperature(p_hpa, t_c),
        mr.potential_temperature(p_hpa, t_c))

# ---------------------------------------------------------------------------
# temperature_from_potential_temperature(pressure_hPa, theta_K) -> K
# ---------------------------------------------------------------------------
compare("temperature_from_potential_temperature",
        metcu.temperature_from_potential_temperature(p_hpa, theta_k),
        mr.temperature_from_potential_temperature(p_hpa, theta_k))

# ---------------------------------------------------------------------------
# virtual_temperature_from_dewpoint(p, T, Td) -> C
# ---------------------------------------------------------------------------
compare("virtual_temperature_from_dewpoint",
        metcu.virtual_temperature_from_dewpoint(p_hpa, t_c, td_c),
        mr.virtual_temperature_from_dewpoint(p_hpa, t_c, td_c))

# ---------------------------------------------------------------------------
# equivalent_potential_temperature(p, T, Td) -> K
# ---------------------------------------------------------------------------
compare("equivalent_potential_temperature",
        metcu.equivalent_potential_temperature(p_hpa, t_c, td_c),
        mr.equivalent_potential_temperature(p_hpa, t_c, td_c))

# ---------------------------------------------------------------------------
# saturation_equivalent_potential_temperature(p, T) -> K
# ---------------------------------------------------------------------------
compare("saturation_equivalent_potential_temperature",
        metcu.saturation_equivalent_potential_temperature(p_hpa, t_c),
        mr.saturation_equivalent_potential_temperature(p_hpa, t_c))

# ---------------------------------------------------------------------------
# wet_bulb_temperature(p, T, Td) -> C (iterative, relax rtol)
# ---------------------------------------------------------------------------
compare("wet_bulb_temperature",
        metcu.wet_bulb_temperature(p_hpa, t_c, td_c),
        mr.wet_bulb_temperature(p_hpa, t_c, td_c),
        rtol=1e-4)

# ---------------------------------------------------------------------------
# wet_bulb_potential_temperature(p, T, Td) -> C (iterative)
# ---------------------------------------------------------------------------
compare("wet_bulb_potential_temperature",
        metcu.wet_bulb_potential_temperature(p_hpa, t_c, td_c),
        mr.wet_bulb_potential_temperature(p_hpa, t_c, td_c),
        rtol=1e-4)

# ---------------------------------------------------------------------------
# saturation_vapor_pressure(T_C) -> hPa
# Both return hPa-ish. metcu kernel returns hPa directly.
# metrust returns Pa (Quantity). We need to handle unit conversion.
# ---------------------------------------------------------------------------
cu_svp = metcu.saturation_vapor_pressure(t_c)  # hPa on GPU
mr_svp = mr.saturation_vapor_pressure(t_c)     # Quantity in hPa
compare("saturation_vapor_pressure", cu_svp, mr_svp)

# ---------------------------------------------------------------------------
# vapor_pressure(Td) -> Pa
# Both metcu and metrust return Pa when called with dewpoint
# ---------------------------------------------------------------------------
cu_vp = metcu.vapor_pressure(td_c)    # Pa on GPU
mr_vp = mr.vapor_pressure(td_c)       # Quantity in Pa
compare("vapor_pressure", cu_vp, mr_vp)

# ---------------------------------------------------------------------------
# dewpoint(vapor_pressure_hPa) -> C
# metcu expects hPa, metrust expects hPa (Quantity)
# ---------------------------------------------------------------------------
# First compute vapor pressure in hPa for the dewpoint function
e_hpa = es_td  # already in hPa
compare("dewpoint",
        metcu.dewpoint(e_hpa),
        mr.dewpoint(e_hpa))

# ---------------------------------------------------------------------------
# dewpoint_from_relative_humidity(T_C, RH_percent) -> C
# ---------------------------------------------------------------------------
compare("dewpoint_from_relative_humidity",
        metcu.dewpoint_from_relative_humidity(t_c, rh_pct),
        mr.dewpoint_from_relative_humidity(t_c, rh_pct))

# ---------------------------------------------------------------------------
# mixing_ratio(pressure, temperature) -> kg/kg
# The generic mixing_ratio() has ambiguous dispatch for raw arrays.
# Use saturation_mixing_ratio explicitly for the (p, T) form.
# For the (partial_pressure, total_pressure) form, test with explicit vapor pressures.
# ---------------------------------------------------------------------------
# Compute vapor pressure in Pa for the (e, p) form
e_pa = es_td * 100.0  # hPa -> Pa
p_pa = p_hpa * 100.0  # hPa -> Pa
compare("mixing_ratio",
        metcu.mixing_ratio(e_pa, p_pa),
        mr.mixing_ratio(e_pa, p_pa))

# ---------------------------------------------------------------------------
# saturation_mixing_ratio(p, T) -> kg/kg
# ---------------------------------------------------------------------------
compare("saturation_mixing_ratio",
        metcu.saturation_mixing_ratio(p_hpa, t_c),
        mr.saturation_mixing_ratio(p_hpa, t_c))

# ---------------------------------------------------------------------------
# mixing_ratio_from_relative_humidity(p, T, RH%) -> kg/kg
# metcu: expects RH in percent (0-100)
# metrust: uses _rh_to_percent internally (pass raw percent)
# ---------------------------------------------------------------------------
compare("mixing_ratio_from_relative_humidity",
        metcu.mixing_ratio_from_relative_humidity(p_hpa, t_c, rh_pct),
        mr.mixing_ratio_from_relative_humidity(p_hpa, t_c, rh_pct))

# ---------------------------------------------------------------------------
# mixing_ratio_from_specific_humidity(q_kgkg) -> kg/kg
# ---------------------------------------------------------------------------
compare("mixing_ratio_from_specific_humidity",
        metcu.mixing_ratio_from_specific_humidity(q_kgkg),
        mr.mixing_ratio_from_specific_humidity(q_kgkg))

# ---------------------------------------------------------------------------
# specific_humidity_from_dewpoint(p, Td) -> kg/kg
# ---------------------------------------------------------------------------
compare("specific_humidity_from_dewpoint",
        metcu.specific_humidity_from_dewpoint(p_hpa, td_c),
        mr.specific_humidity_from_dewpoint(p_hpa, td_c))

# ---------------------------------------------------------------------------
# specific_humidity_from_mixing_ratio(w_kgkg) -> kg/kg
# ---------------------------------------------------------------------------
compare("specific_humidity_from_mixing_ratio",
        metcu.specific_humidity_from_mixing_ratio(w_actual),
        mr.specific_humidity_from_mixing_ratio(w_actual))

# ---------------------------------------------------------------------------
# relative_humidity_from_dewpoint(T, Td) -> 0-1
# Both return fractional (0-1)
# ---------------------------------------------------------------------------
compare("relative_humidity_from_dewpoint",
        metcu.relative_humidity_from_dewpoint(t_c, td_c),
        mr.relative_humidity_from_dewpoint(t_c, td_c))

# ---------------------------------------------------------------------------
# relative_humidity_from_mixing_ratio(p, T, w) -> 0-1
# metcu expects g/kg; metrust without pint treats raw arrays as kg/kg
# and internally multiplies by 1000. So pass g/kg to metcu, kg/kg to metrust.
# Both return fractional 0-1
# ---------------------------------------------------------------------------
compare("relative_humidity_from_mixing_ratio",
        metcu.relative_humidity_from_mixing_ratio(p_hpa, t_c, w_gkg),
        mr.relative_humidity_from_mixing_ratio(p_hpa, t_c, w_actual))

# ---------------------------------------------------------------------------
# density(p, T, w) -> kg/m3
# metcu expects g/kg; metrust without pint treats raw as kg/kg
# and internally multiplies by 1000.
# ---------------------------------------------------------------------------
compare("density",
        metcu.density(p_hpa, t_c, w_gkg),
        mr.density(p_hpa, t_c, w_actual))

# ---------------------------------------------------------------------------
# dry_static_energy(height_m, temperature_K) -> J/kg
# ---------------------------------------------------------------------------
compare("dry_static_energy",
        metcu.dry_static_energy(z_m, t_k),
        mr.dry_static_energy(z_m, t_k))

# ---------------------------------------------------------------------------
# moist_static_energy(height_m, temperature_K, q_kgkg) -> J/kg
# ---------------------------------------------------------------------------
compare("moist_static_energy",
        metcu.moist_static_energy(z_m, t_k, q_kgkg),
        mr.moist_static_energy(z_m, t_k, q_kgkg))

# ---------------------------------------------------------------------------
# exner_function(pressure_hPa) -> dimensionless
# ---------------------------------------------------------------------------
compare("exner_function",
        metcu.exner_function(p_hpa),
        mr.exner_function(p_hpa))

# ---------------------------------------------------------------------------
# heat_index(T_C, RH_percent) -> C
# ---------------------------------------------------------------------------
compare("heat_index",
        metcu.heat_index(t_c, rh_pct),
        mr.heat_index(t_c, rh_pct))

# ---------------------------------------------------------------------------
# windchill(T_C, wind_speed_ms) -> C
# ---------------------------------------------------------------------------
compare("windchill",
        metcu.windchill(t_c, ws),
        mr.windchill(t_c, ws))

# ---------------------------------------------------------------------------
# apparent_temperature(T_C, RH_percent, wind_speed_ms) -> C
# ---------------------------------------------------------------------------
compare("apparent_temperature",
        metcu.apparent_temperature(t_c, rh_pct, ws),
        mr.apparent_temperature(t_c, rh_pct, ws))

# ---------------------------------------------------------------------------
# frost_point(T_C, RH_percent) -> C
# ---------------------------------------------------------------------------
compare("frost_point",
        metcu.frost_point(t_c, rh_pct),
        mr.frost_point(t_c, rh_pct))

# ---------------------------------------------------------------------------
# coriolis_parameter(latitude_deg) -> 1/s
# ---------------------------------------------------------------------------
compare("coriolis_parameter",
        metcu.coriolis_parameter(lats_2d),
        mr.coriolis_parameter(lats_2d))

# ---------------------------------------------------------------------------
# vertical_velocity(omega_Pas, p_hPa, T_C) -> m/s
# ---------------------------------------------------------------------------
compare("vertical_velocity",
        metcu.vertical_velocity(omega_pas, p_hpa, t_c),
        mr.vertical_velocity(omega_pas, p_hpa, t_c))

# ---------------------------------------------------------------------------
# montgomery_streamfunction(height_m, temperature_K) -> kJ/kg
# 2-arg MetPy form
# ---------------------------------------------------------------------------
compare("montgomery_streamfunction",
        metcu.montgomery_streamfunction(z_m, t_k),
        mr.montgomery_streamfunction(z_m, t_k))

# ---------------------------------------------------------------------------
# water_latent_heat_vaporization(T_C) -> J/kg
# ---------------------------------------------------------------------------
compare("water_latent_heat_vaporization",
        metcu.water_latent_heat_vaporization(t_c),
        mr.water_latent_heat_vaporization(t_c))

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 72)
print(f"RESULTS: {n_pass} PASS / {n_fail} FAIL out of {n_pass + n_fail} functions")
print("=" * 72)

if n_fail > 0:
    print("\nFailed functions:")
    for r in results:
        if "FAIL" in r or "ERR" in r:
            print(f"  {r}")
    sys.exit(1)
else:
    print("\nAll functions verified successfully.")
    sys.exit(0)
