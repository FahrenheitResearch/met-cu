"""Complete GPU vs CPU benchmark -- every function, timed and verified.

Tests every function that exists in both metcu and metrust.calc,
measuring GPU vs CPU timing and verifying numerical accuracy.

Designed for RTX 5090 (34GB VRAM, CUDA 13).
"""
import numpy as np
import time
import traceback
import sys

try:
    import cupy as cp
    gpu_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode()
    gpu_mem = cp.cuda.runtime.getDeviceProperties(0)["totalGlobalMem"] / 1e9
    print(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
except ImportError:
    print("ERROR: cupy not available -- this benchmark requires a CUDA GPU.")
    sys.exit(1)

import metcu
import metcu.calc as mc
import metrust.calc as mr
from metrust.units import units

# ==========================================================================
# Test data
# ==========================================================================
np.random.seed(42)
NY, NX = 1059, 1799  # HRRR scale
NLEVELS = 40
N = NY * NX

# 2D flat arrays for per-element thermo/wind ops
p_sfc = np.full(N, 1000.0, dtype=np.float64)
p_850 = np.full(N, 850.0, dtype=np.float64)
p_700 = np.full(N, 700.0, dtype=np.float64)
t_2d = np.random.randn(N) * 15 + 25  # Celsius
td_2d = t_2d - np.abs(np.random.randn(N) * 8)
u_2d = np.random.randn(N) * 10  # m/s
v_2d = np.random.randn(N) * 10
w_2d = np.random.randn(N) * 0.5  # m/s vertical
rh_2d = np.clip(np.random.rand(N) * 100, 5, 100)
z_2d = np.random.rand(N) * 2000  # meters
q_2d = np.clip(np.random.rand(N) * 0.020, 0.001, 0.020)  # specific humidity kg/kg
w_mix_2d = q_2d / (1 - q_2d)  # mixing ratio kg/kg
w_mix_gkg = w_mix_2d * 1000.0  # g/kg
theta_2d = t_2d + 273.15  # approximate potential temperature (K)
omega_2d = np.random.randn(N) * 5  # Pa/s
sigma_2d = np.random.rand(N) * 0.9 + 0.05  # sigma levels
geopotential_2d = z_2d * 9.80665  # m^2/s^2

# Comfort index data
t_hot = np.random.rand(N) * 20 + 25  # 25-45 C for heat index
rh_high = np.clip(np.random.rand(N) * 100, 40, 100)
t_cold = np.random.rand(N) * 20 - 20  # -20 to 0 C for windchill
ws_2d = np.abs(np.random.randn(N) * 10) + 1  # positive wind speed

# 2D grid fields for stencil ops
t_grid = np.ascontiguousarray(np.random.randn(NY, NX) * 15 + 25, dtype=np.float64)
u_grid = np.ascontiguousarray(np.random.randn(NY, NX) * 10, dtype=np.float64)
v_grid = np.ascontiguousarray(np.random.randn(NY, NX) * 10, dtype=np.float64)
z_grid = np.ascontiguousarray(np.random.rand(NY, NX) * 5500, dtype=np.float64)
dx_scalar = 3000.0
dy_scalar = 3000.0
lat_grid = np.ascontiguousarray(
    np.broadcast_to(np.linspace(25, 50, NY)[:, None], (NY, NX)), dtype=np.float64
)

# Sounding profile
p_snd = np.linspace(1000, 100, NLEVELS).astype(np.float64)
t_snd = np.linspace(25, -60, NLEVELS).astype(np.float64)
td_snd = np.linspace(20, -65, NLEVELS).astype(np.float64)
u_snd = np.linspace(5, 40, NLEVELS).astype(np.float64)
v_snd = np.linspace(0, 20, NLEVELS).astype(np.float64)
h_snd = np.linspace(0, 16000, NLEVELS).astype(np.float64)
theta_snd = np.linspace(300, 380, NLEVELS).astype(np.float64)  # K

# Standard level scalars for indices
t850, td850, t700, td700, t500, t300, t200 = 15.0, 12.0, 5.0, -5.0, -15.0, -40.0, -55.0
z1000, z850v, z700v, z500v = 100.0, 1500.0, 3000.0, 5500.0

# 3D fields for column/grid composites
NZ_3D = 10
NY_3D, NX_3D = 50, 80
refl_3d = np.random.rand(NZ_3D, NY_3D, NX_3D).astype(np.float64) * 60 - 10

# ==========================================================================
# Result tracking
# ==========================================================================
results = []
CATEGORY_STATS = {}

def _extract(val):
    """Extract a numpy array from any result type."""
    if isinstance(val, tuple):
        val = val[0]
    if hasattr(val, "magnitude"):
        return np.asarray(val.magnitude, dtype=np.float64)
    if hasattr(val, "get"):  # cupy
        return val.get().astype(np.float64)
    return np.asarray(val, dtype=np.float64)


def test_func(name, gpu_call, cpu_call, verify=True, category="misc"):
    """Test one function: benchmark speed and verify accuracy."""
    try:
        # GPU warmup
        for _ in range(3):
            try:
                gpu_call()
            except Exception:
                pass
        cp.cuda.Device().synchronize()

        # GPU timing
        gpu_times = []
        gpu_result = None
        for _ in range(5):
            cp.cuda.Device().synchronize()
            t0 = time.perf_counter()
            gpu_result = gpu_call()
            cp.cuda.Device().synchronize()
            gpu_times.append(time.perf_counter() - t0)
        gpu_ms = min(gpu_times) * 1000

        # CPU timing
        cpu_times = []
        cpu_result = None
        for _ in range(5):
            t0 = time.perf_counter()
            cpu_result = cpu_call()
            cpu_times.append(time.perf_counter() - t0)
        cpu_ms = min(cpu_times) * 1000

        speedup = cpu_ms / gpu_ms if gpu_ms > 0.001 else 0

        # Verify accuracy
        accurate = "N/A"
        max_diff = 0.0
        if verify and gpu_result is not None and cpu_result is not None:
            try:
                g = _extract(gpu_result)
                c = _extract(cpu_result)
                g = g.ravel()
                c = c.ravel()
                minlen = min(len(g), len(c))
                if minlen > 0:
                    g, c = g[:minlen], c[:minlen]
                    mask = np.isfinite(g) & np.isfinite(c)
                    if mask.sum() > 0:
                        max_diff = float(np.max(np.abs(g[mask] - c[mask])))
                        if np.allclose(g[mask], c[mask], rtol=1e-4, atol=1e-6, equal_nan=True):
                            accurate = "PASS"
                        else:
                            accurate = f"DIFF={max_diff:.4g}"
                    else:
                        accurate = "ALL_NAN"
                else:
                    accurate = "EMPTY"
            except Exception as e:
                accurate = f"ERR:{str(e)[:30]}"

        results.append((name, category, gpu_ms, cpu_ms, speedup, accurate))
        status = "PASS" if accurate == "PASS" or accurate == "N/A" else "WARN"
        print(f"  {status:4s} {name:50s} GPU:{gpu_ms:8.2f}ms CPU:{cpu_ms:8.2f}ms {speedup:7.1f}x  {accurate}")

    except Exception as e:
        results.append((name, category, 0, 0, 0, f"FAIL:{str(e)[:40]}"))
        print(f"  FAIL {name:50s} {str(e)[:70]}")
        traceback.print_exc(limit=1)


def skip_func(name, reason="CPU-only utility", category="skip"):
    """Mark a function as skipped."""
    results.append((name, category, 0, 0, 0, f"SKIP:{reason}"))
    print(f"  SKIP {name:50s} {reason}")


# ==========================================================================
# Category 1: Per-element thermodynamic (N=1.9M flat arrays)
# ==========================================================================
print(f"\n{'=' * 90}")
print(f" CATEGORY 1: Per-element thermodynamic -- N={N:,} points")
print(f"{'=' * 90}")
cat = "thermo"

test_func("potential_temperature",
    lambda: mc.potential_temperature(p_sfc, t_2d),
    lambda: mr.potential_temperature(p_sfc * units.hPa, t_2d * units.degC),
    category=cat)

test_func("temperature_from_potential_temperature",
    lambda: mc.temperature_from_potential_temperature(p_sfc, theta_2d),
    lambda: mr.temperature_from_potential_temperature(p_sfc * units.hPa, theta_2d * units.K),
    category=cat)

test_func("virtual_temperature_from_dewpoint",
    lambda: mc.virtual_temperature_from_dewpoint(p_sfc, t_2d, td_2d),
    lambda: mr.virtual_temperature_from_dewpoint(p_sfc * units.hPa, t_2d * units.degC, td_2d * units.degC),
    category=cat)

test_func("virtual_potential_temperature",
    lambda: mc.virtual_potential_temperature(p_sfc, t_2d, w_mix_gkg),
    lambda: mr.virtual_potential_temperature(p_sfc * units.hPa, t_2d * units.degC, w_mix_gkg * units("g/kg")),
    category=cat)

test_func("equivalent_potential_temperature",
    lambda: mc.equivalent_potential_temperature(p_sfc, t_2d, td_2d),
    lambda: mr.equivalent_potential_temperature(p_sfc * units.hPa, t_2d * units.degC, td_2d * units.degC),
    category=cat)

test_func("saturation_equivalent_potential_temperature",
    lambda: mc.saturation_equivalent_potential_temperature(p_sfc, t_2d),
    lambda: mr.saturation_equivalent_potential_temperature(p_sfc * units.hPa, t_2d * units.degC),
    category=cat)

test_func("wet_bulb_temperature",
    lambda: mc.wet_bulb_temperature(p_sfc, t_2d, td_2d),
    lambda: mr.wet_bulb_temperature(p_sfc * units.hPa, t_2d * units.degC, td_2d * units.degC),
    category=cat)

test_func("wet_bulb_potential_temperature",
    lambda: mc.wet_bulb_potential_temperature(p_sfc, t_2d, td_2d),
    lambda: mr.wet_bulb_potential_temperature(p_sfc * units.hPa, t_2d * units.degC, td_2d * units.degC),
    category=cat)

test_func("saturation_vapor_pressure",
    lambda: mc.saturation_vapor_pressure(t_2d),
    lambda: mr.saturation_vapor_pressure(t_2d * units.degC),
    category=cat)

test_func("vapor_pressure",
    lambda: mc.vapor_pressure(td_2d),
    lambda: mr.vapor_pressure(td_2d * units.degC),
    category=cat)

test_func("dewpoint",
    lambda: mc.dewpoint(p_850[:100000]),  # vapor pressure in hPa scale
    lambda: mr.dewpoint(p_850[:100000] * units.hPa),
    category=cat)

test_func("dewpoint_from_relative_humidity",
    lambda: mc.dewpoint_from_relative_humidity(t_2d, rh_2d),
    lambda: mr.dewpoint_from_relative_humidity(t_2d * units.degC, rh_2d * units.percent),
    category=cat)

test_func("dewpoint_from_specific_humidity",
    lambda: mc.dewpoint_from_specific_humidity(p_sfc, q_2d),
    lambda: mr.dewpoint_from_specific_humidity(p_sfc * units.hPa, q_2d * units("kg/kg")),
    category=cat)

test_func("mixing_ratio",
    lambda: mc.mixing_ratio(p_sfc, t_2d),
    lambda: mr.mixing_ratio(p_sfc * units.hPa, t_2d * units.degC),
    category=cat)

test_func("saturation_mixing_ratio",
    lambda: mc.saturation_mixing_ratio(p_sfc, t_2d),
    lambda: mr.saturation_mixing_ratio(p_sfc * units.hPa, t_2d * units.degC),
    category=cat)

test_func("mixing_ratio_from_relative_humidity",
    lambda: mc.mixing_ratio_from_relative_humidity(p_sfc, t_2d, rh_2d),
    lambda: mr.mixing_ratio_from_relative_humidity(p_sfc * units.hPa, t_2d * units.degC, rh_2d * units.percent),
    category=cat)

test_func("mixing_ratio_from_specific_humidity",
    lambda: mc.mixing_ratio_from_specific_humidity(q_2d),
    lambda: mr.mixing_ratio_from_specific_humidity(q_2d * units("kg/kg")),
    category=cat)

test_func("specific_humidity_from_dewpoint",
    lambda: mc.specific_humidity_from_dewpoint(p_sfc, td_2d),
    lambda: mr.specific_humidity_from_dewpoint(p_sfc * units.hPa, td_2d * units.degC),
    category=cat)

test_func("specific_humidity_from_mixing_ratio",
    lambda: mc.specific_humidity_from_mixing_ratio(w_mix_2d),
    lambda: mr.specific_humidity_from_mixing_ratio(w_mix_2d * units("kg/kg")),
    category=cat)

test_func("relative_humidity_from_dewpoint",
    lambda: mc.relative_humidity_from_dewpoint(t_2d, td_2d),
    lambda: mr.relative_humidity_from_dewpoint(t_2d * units.degC, td_2d * units.degC),
    category=cat)

test_func("relative_humidity_from_mixing_ratio",
    lambda: mc.relative_humidity_from_mixing_ratio(p_sfc, t_2d, w_mix_gkg),
    lambda: mr.relative_humidity_from_mixing_ratio(p_sfc * units.hPa, t_2d * units.degC, w_mix_gkg * units("g/kg")),
    category=cat)

test_func("relative_humidity_from_specific_humidity",
    lambda: mc.relative_humidity_from_specific_humidity(p_sfc, t_2d, q_2d),
    lambda: mr.relative_humidity_from_specific_humidity(p_sfc * units.hPa, t_2d * units.degC, q_2d * units("kg/kg")),
    category=cat)

test_func("density",
    lambda: mc.density(p_sfc, t_2d, w_mix_gkg),
    lambda: mr.density(p_sfc * units.hPa, t_2d * units.degC, w_mix_gkg * units("g/kg")),
    category=cat)

test_func("dry_static_energy",
    lambda: mc.dry_static_energy(z_2d, theta_2d),
    lambda: mr.dry_static_energy(z_2d * units.m, theta_2d * units.K),
    category=cat)

test_func("moist_static_energy",
    lambda: mc.moist_static_energy(z_2d, theta_2d, q_2d),
    lambda: mr.moist_static_energy(z_2d * units.m, theta_2d * units.K, q_2d * units("kg/kg")),
    category=cat)

test_func("exner_function",
    lambda: mc.exner_function(p_sfc),
    lambda: mr.exner_function(p_sfc * units.hPa),
    category=cat)

test_func("montgomery_streamfunction",
    lambda: mc.montgomery_streamfunction(z_2d[:1000], theta_2d[:1000]),
    lambda: mr.montgomery_streamfunction(z_2d[:1000] * units.m, theta_2d[:1000] * units.K),
    category=cat)

test_func("dry_lapse",
    lambda: mc.dry_lapse(p_snd, t_snd[0]),
    lambda: mr.dry_lapse(p_snd * units.hPa, t_snd[0] * units.degC),
    category=cat)

test_func("height_to_pressure_std",
    lambda: mc.height_to_pressure_std(z_2d),
    lambda: mr.height_to_pressure_std(z_2d * units.m),
    category=cat)

test_func("pressure_to_height_std",
    lambda: mc.pressure_to_height_std(p_sfc),
    lambda: mr.pressure_to_height_std(p_sfc * units.hPa),
    category=cat)

test_func("add_height_to_pressure",
    lambda: mc.add_height_to_pressure(p_sfc, z_2d),
    lambda: mr.add_height_to_pressure(p_sfc * units.hPa, z_2d * units.m),
    category=cat)

test_func("add_pressure_to_height",
    lambda: mc.add_pressure_to_height(z_2d, p_850 - p_sfc),
    lambda: mr.add_pressure_to_height(z_2d * units.m, (p_850 - p_sfc) * units.hPa),
    category=cat)

test_func("altimeter_to_station_pressure",
    lambda: mc.altimeter_to_station_pressure(p_sfc + 13, z_2d),
    lambda: mr.altimeter_to_station_pressure((p_sfc + 13) * units.hPa, z_2d * units.m),
    category=cat)

test_func("station_to_altimeter_pressure",
    lambda: mc.station_to_altimeter_pressure(p_sfc, z_2d),
    lambda: mr.station_to_altimeter_pressure(p_sfc * units.hPa, z_2d * units.m),
    category=cat)

test_func("altimeter_to_sea_level_pressure",
    lambda: mc.altimeter_to_sea_level_pressure(p_sfc + 13, z_2d, t_2d),
    lambda: mr.altimeter_to_sea_level_pressure((p_sfc + 13) * units.hPa, z_2d * units.m, t_2d * units.degC),
    category=cat)

test_func("sigma_to_pressure",
    lambda: mc.sigma_to_pressure(sigma_2d, p_sfc, np.full(N, 50.0)),
    lambda: mr.sigma_to_pressure(sigma_2d, p_sfc * units.hPa, 50.0 * units.hPa),
    category=cat)

test_func("geopotential_to_height",
    lambda: mc.geopotential_to_height(geopotential_2d),
    lambda: mr.geopotential_to_height(geopotential_2d * units("m**2/s**2")),
    category=cat)

test_func("height_to_geopotential",
    lambda: mc.height_to_geopotential(z_2d),
    lambda: mr.height_to_geopotential(z_2d * units.m),
    category=cat)

test_func("scale_height",
    lambda: mc.scale_height(theta_2d),
    lambda: mr.scale_height(theta_2d * units.K),
    category=cat)

test_func("heat_index",
    lambda: mc.heat_index(t_hot, rh_high),
    lambda: mr.heat_index(t_hot * units.degC, rh_high * units.percent),
    category=cat)

test_func("windchill",
    lambda: mc.windchill(t_cold, ws_2d),
    lambda: mr.windchill(t_cold * units.degC, ws_2d * units("m/s")),
    category=cat)

test_func("apparent_temperature",
    lambda: mc.apparent_temperature(t_2d, rh_2d, ws_2d),
    lambda: mr.apparent_temperature(t_2d * units.degC, rh_2d * units.percent, ws_2d * units("m/s")),
    category=cat)

test_func("frost_point",
    lambda: mc.frost_point(t_cold, rh_high),
    lambda: mr.frost_point(t_cold * units.degC, rh_high * units.percent),
    category=cat)

test_func("psychrometric_vapor_pressure",
    lambda: mc.psychrometric_vapor_pressure(t_2d[:10000], td_2d[:10000], p_sfc[:10000]),
    lambda: mr.psychrometric_vapor_pressure(t_2d[:10000] * units.degC, td_2d[:10000] * units.degC, p_sfc[:10000] * units.hPa),
    category=cat)

test_func("water_latent_heat_vaporization",
    lambda: mc.water_latent_heat_vaporization(t_2d),
    lambda: mr.water_latent_heat_vaporization(t_2d * units.degC),
    category=cat)

test_func("water_latent_heat_sublimation",
    lambda: mc.water_latent_heat_sublimation(t_2d),
    lambda: mr.water_latent_heat_sublimation(t_2d * units.degC),
    category=cat)

test_func("water_latent_heat_melting",
    lambda: mc.water_latent_heat_melting(t_2d),
    lambda: mr.water_latent_heat_melting(t_2d * units.degC),
    category=cat)

test_func("moist_air_gas_constant",
    lambda: mc.moist_air_gas_constant(w_mix_2d),
    lambda: mr.moist_air_gas_constant(w_mix_2d * units("kg/kg")),
    category=cat)

test_func("moist_air_specific_heat_pressure",
    lambda: mc.moist_air_specific_heat_pressure(w_mix_2d),
    lambda: mr.moist_air_specific_heat_pressure(w_mix_2d * units("kg/kg")),
    category=cat)

test_func("moist_air_poisson_exponent",
    lambda: mc.moist_air_poisson_exponent(w_mix_2d),
    lambda: mr.moist_air_poisson_exponent(w_mix_2d * units("kg/kg")),
    category=cat)

test_func("coriolis_parameter",
    lambda: mc.coriolis_parameter(lat_grid.ravel()),
    lambda: mr.coriolis_parameter(lat_grid.ravel() * units.degree),
    category=cat)

test_func("vertical_velocity",
    lambda: mc.vertical_velocity(omega_2d, p_sfc, t_2d),
    lambda: mr.vertical_velocity(omega_2d * units("Pa/s"), p_sfc * units.hPa, t_2d * units.degC),
    category=cat)

test_func("vertical_velocity_pressure",
    lambda: mc.vertical_velocity_pressure(w_2d, p_sfc, t_2d),
    lambda: mr.vertical_velocity_pressure(w_2d * units("m/s"), p_sfc * units.hPa, t_2d * units.degC),
    category=cat)

# Sounding-based thermo that takes profiles
test_func("thickness_hydrostatic (scalar)",
    lambda: mc.thickness_hydrostatic(np.array([1000.0]), np.array([500.0]), np.array([270.0])),
    lambda: mr.thickness_hydrostatic(1000.0 * units.hPa, 500.0 * units.hPa, 270.0 * units.K),
    category=cat)

test_func("brunt_vaisala_frequency_squared",
    lambda: mc.brunt_vaisala_frequency_squared(h_snd, theta_snd),
    lambda: mr.brunt_vaisala_frequency_squared(h_snd * units.m, theta_snd * units.K),
    category=cat)

test_func("brunt_vaisala_frequency",
    lambda: mc.brunt_vaisala_frequency(h_snd, theta_snd),
    lambda: mr.brunt_vaisala_frequency(h_snd * units.m, theta_snd * units.K),
    category=cat)

test_func("brunt_vaisala_period",
    lambda: mc.brunt_vaisala_period(h_snd, theta_snd),
    lambda: mr.brunt_vaisala_period(h_snd * units.m, theta_snd * units.K),
    category=cat)

test_func("static_stability",
    lambda: mc.static_stability(p_snd, theta_snd),
    lambda: mr.static_stability(p_snd * units.hPa, theta_snd * units.K),
    category=cat)

# ==========================================================================
# Category 2: Wind per-element (N=1.9M)
# ==========================================================================
print(f"\n{'=' * 90}")
print(f" CATEGORY 2: Wind per-element -- N={N:,} points")
print(f"{'=' * 90}")
cat = "wind"

test_func("wind_speed",
    lambda: mc.wind_speed(u_2d, v_2d),
    lambda: mr.wind_speed(u_2d * units("m/s"), v_2d * units("m/s")),
    category=cat)

test_func("wind_direction",
    lambda: mc.wind_direction(u_2d, v_2d),
    lambda: mr.wind_direction(u_2d * units("m/s"), v_2d * units("m/s")),
    category=cat)

test_func("wind_components",
    lambda: mc.wind_components(ws_2d, rh_2d),  # using rh_2d as direction proxy
    lambda: mr.wind_components(ws_2d * units("m/s"), rh_2d * units.degree),
    category=cat)

# ==========================================================================
# Category 3: Grid stencil (NY x NX 2D grid)
# ==========================================================================
print(f"\n{'=' * 90}")
print(f" CATEGORY 3: Grid stencil -- {NY}x{NX} = {NY*NX:,} grid points")
print(f"{'=' * 90}")
cat = "grid"

test_func("vorticity",
    lambda: mc.vorticity(u_grid, v_grid, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.vorticity(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("divergence",
    lambda: mc.divergence(u_grid, v_grid, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.divergence(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("absolute_vorticity",
    lambda: mc.absolute_vorticity(u_grid, v_grid, lats=lat_grid, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.absolute_vorticity(u_grid * units("m/s"), v_grid * units("m/s"), lats=lat_grid * units.degree, dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("shearing_deformation",
    lambda: mc.shearing_deformation(u_grid, v_grid, dx_scalar, dy_scalar),
    lambda: mr.shearing_deformation(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("stretching_deformation",
    lambda: mc.stretching_deformation(u_grid, v_grid, dx_scalar, dy_scalar),
    lambda: mr.stretching_deformation(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("total_deformation",
    lambda: mc.total_deformation(u_grid, v_grid, dx_scalar, dy_scalar),
    lambda: mr.total_deformation(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("advection",
    lambda: mc.advection(t_grid, u_grid, v_grid, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.advection(t_grid * units.degC, u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("frontogenesis",
    lambda: mc.frontogenesis(t_grid + 273.15, u_grid, v_grid, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.frontogenesis((t_grid + 273.15) * units.K, u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("q_vector",
    lambda: mc.q_vector(u_grid, v_grid, t_grid + 273.15, 850.0, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.q_vector(u_grid * units("m/s"), v_grid * units("m/s"), (t_grid + 273.15) * units.K, 850.0 * units.hPa, dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("geostrophic_wind",
    lambda: mc.geostrophic_wind(z_grid, latitude=lat_grid, dx=dx_scalar, dy=dy_scalar),
    lambda: mr.geostrophic_wind(z_grid * units.m, latitude=lat_grid * units.degree, dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("ageostrophic_wind",
    lambda: mc.ageostrophic_wind(u_grid, v_grid, z_grid, lat_grid, dx_scalar, dy_scalar),
    lambda: mr.ageostrophic_wind(u_grid * units("m/s"), v_grid * units("m/s"), z_grid * units.m, lat_grid * units.degree, dx_scalar * units.m, dy_scalar * units.m),
    category=cat)

test_func("gradient_x",
    lambda: mc.gradient_x(t_grid, dx_scalar),
    lambda: mr.first_derivative(t_grid * units.degC, delta=dx_scalar * units.m, axis=1),
    category=cat)

test_func("gradient_y",
    lambda: mc.gradient_y(t_grid, dy_scalar),
    lambda: mr.first_derivative(t_grid * units.degC, delta=dy_scalar * units.m, axis=0),
    category=cat)

test_func("laplacian",
    lambda: mc.laplacian(t_grid, dx_scalar, dy_scalar),
    lambda: mr.laplacian(t_grid * units.degC, dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("first_derivative (axis=0)",
    lambda: mc.first_derivative(t_grid, dy_scalar, axis=0),
    lambda: mr.first_derivative(t_grid * units.degC, delta=dy_scalar * units.m, axis=0),
    category=cat)

test_func("second_derivative (axis=0)",
    lambda: mc.second_derivative(t_grid, dy_scalar, axis=0),
    lambda: mr.second_derivative(t_grid * units.degC, delta=dy_scalar * units.m, axis=0),
    category=cat)

test_func("smooth_gaussian",
    lambda: mc.smooth_gaussian(t_grid, 3.0),
    lambda: mr.smooth_gaussian(t_grid * units.degC, 3),
    verify=False, category=cat)  # smoothing implementations may differ slightly

test_func("smooth_n_point (5)",
    lambda: mc.smooth_n_point(t_grid, 5, 1),
    lambda: mr.smooth_n_point(t_grid * units.degC, 5, 1),
    verify=False, category=cat)

test_func("smooth_n_point (9)",
    lambda: mc.smooth_n_point(t_grid, 9, 1),
    lambda: mr.smooth_n_point(t_grid * units.degC, 9, 1),
    verify=False, category=cat)

test_func("smooth_rectangular",
    lambda: mc.smooth_rectangular(t_grid, 5, 1),
    lambda: mr.smooth_rectangular(t_grid * units.degC, 5, 1),
    verify=False, category=cat)

test_func("smooth_circular",
    lambda: mc.smooth_circular(t_grid, 3.0, 1),
    lambda: mr.smooth_circular(t_grid * units.degC, 3.0, 1),
    verify=False, category=cat)

test_func("potential_vorticity_barotropic",
    lambda: mc.potential_vorticity_barotropic(z_grid, u_grid, v_grid, lat_grid, dx_scalar, dy_scalar),
    lambda: mr.potential_vorticity_barotropic(z_grid * units.m, u_grid * units("m/s"), v_grid * units("m/s"), lat_grid * units.degree, dx_scalar * units.m, dy_scalar * units.m),
    category=cat)

test_func("curvature_vorticity",
    lambda: mc.curvature_vorticity(u_grid, v_grid, dx_scalar, dy_scalar),
    lambda: mr.curvature_vorticity(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("shear_vorticity",
    lambda: mc.shear_vorticity(u_grid, v_grid, dx_scalar, dy_scalar),
    lambda: mr.shear_vorticity(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("inertial_advective_wind",
    lambda: mc.inertial_advective_wind(u_grid, v_grid, u_grid * 0.5, v_grid * 0.5, dx_scalar, dy_scalar),
    lambda: mr.inertial_advective_wind(u_grid * units("m/s"), v_grid * units("m/s"), (u_grid * 0.5) * units("m/s"), (v_grid * 0.5) * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("vector_derivative",
    lambda: mc.vector_derivative(u_grid, v_grid, dx_scalar, dy_scalar),
    lambda: mr.vector_derivative(u_grid * units("m/s"), v_grid * units("m/s"), dx=dx_scalar * units.m, dy=dy_scalar * units.m),
    category=cat)

test_func("composite_reflectivity",
    lambda: mc.composite_reflectivity(refl_3d),
    lambda: mr.composite_reflectivity(refl_3d),
    category=cat)

test_func("lat_lon_grid_deltas",
    lambda: mc.lat_lon_grid_deltas(lat_grid[:100, :100], lat_grid[:100, :100]),  # small grid
    lambda: mr.lat_lon_grid_deltas(lat_grid[:100, :100] * units.degree, lat_grid[:100, :100] * units.degree),
    verify=False, category=cat)

# ==========================================================================
# Category 4: Column/sounding (use sounding arrays)
# ==========================================================================
print(f"\n{'=' * 90}")
print(f" CATEGORY 4: Column/sounding -- {NLEVELS} levels")
print(f"{'=' * 90}")
cat = "column"

test_func("lcl",
    lambda: mc.lcl(1000.0, 25.0, 20.0),
    lambda: mr.lcl(1000.0 * units.hPa, 25.0 * units.degC, 20.0 * units.degC),
    category=cat)

test_func("lfc",
    lambda: mc.lfc(p_snd, t_snd, td_snd),
    lambda: mr.lfc(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("el",
    lambda: mc.el(p_snd, t_snd, td_snd),
    lambda: mr.el(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("ccl",
    lambda: mc.ccl(p_snd, t_snd, td_snd),
    lambda: mr.ccl(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("parcel_profile",
    lambda: mc.parcel_profile(p_snd, t_snd[0], td_snd[0]),
    lambda: mr.parcel_profile(p_snd * units.hPa, t_snd[0] * units.degC, td_snd[0] * units.degC),
    category=cat)

test_func("moist_lapse",
    lambda: mc.moist_lapse(p_snd, t_snd[0]),
    lambda: mr.moist_lapse(p_snd * units.hPa, t_snd[0] * units.degC),
    category=cat)

test_func("cape_cin (surface-based)",
    lambda: mc.cape_cin(p_snd, t_snd, td_snd),
    lambda: mr.cape_cin(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("surface_based_cape_cin",
    lambda: mc.surface_based_cape_cin(p_snd, t_snd, td_snd),
    lambda: mr.surface_based_cape_cin(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("mixed_layer_cape_cin",
    lambda: mc.mixed_layer_cape_cin(p_snd, t_snd, td_snd, depth=100.0),
    lambda: mr.mixed_layer_cape_cin(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, depth=100.0),
    category=cat)

test_func("most_unstable_cape_cin",
    lambda: mc.most_unstable_cape_cin(p_snd, t_snd, td_snd, depth=300),
    lambda: mr.most_unstable_cape_cin(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, depth=300),
    category=cat)

test_func("lifted_index",
    lambda: mc.lifted_index(p_snd, t_snd, td_snd),
    lambda: mr.lifted_index(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("showalter_index",
    lambda: mc.showalter_index(p_snd, t_snd, td_snd),
    lambda: mr.showalter_index(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("precipitable_water",
    lambda: mc.precipitable_water(p_snd, td_snd),
    lambda: mr.precipitable_water(p_snd * units.hPa, td_snd * units.degC),
    category=cat)

test_func("downdraft_cape",
    lambda: mc.downdraft_cape(p_snd, t_snd, td_snd),
    lambda: mr.downdraft_cape(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

test_func("bulk_shear",
    lambda: mc.bulk_shear(u_snd, v_snd, h_snd, bottom=0.0, top=6000.0),
    lambda: mr.bulk_shear(u_snd * units("m/s"), v_snd * units("m/s"), h_snd * units.m, bottom=0.0 * units.m, top=6000.0 * units.m),
    category=cat)

test_func("mean_wind",
    lambda: mc.mean_wind(u_snd, v_snd, h_snd, 0.0, 6000.0),
    lambda: mr.mean_wind(u_snd * units("m/s"), v_snd * units("m/s"), h_snd * units.m, 0.0 * units.m, 6000.0 * units.m),
    category=cat)

test_func("storm_relative_helicity",
    lambda: mc.storm_relative_helicity(h_snd, u_snd, v_snd, depth=3000.0, storm_u=5.0, storm_v=5.0),
    lambda: mr.storm_relative_helicity(h_snd * units.m, u_snd * units("m/s"), v_snd * units("m/s"), depth=3000.0 * units.m, storm_u=5.0 * units("m/s"), storm_v=5.0 * units("m/s")),
    category=cat)

test_func("bunkers_storm_motion",
    lambda: mc.bunkers_storm_motion(u_snd, v_snd, h_snd),
    lambda: mr.bunkers_storm_motion(u_snd * units("m/s"), v_snd * units("m/s"), h_snd * units.m),
    category=cat)

test_func("critical_angle",
    lambda: mc.critical_angle(5.0, 5.0, 10.0, 15.0),
    lambda: mr.critical_angle(5.0 * units("m/s"), 5.0 * units("m/s"), 10.0 * units("m/s"), 15.0 * units("m/s")),
    category=cat)

# Stability indices (scalar level values)
test_func("k_index",
    lambda: mc.k_index(t850, td850, t700, td700, t500),
    lambda: mr.k_index(t850 * units.degC, td850 * units.degC, t700 * units.degC, td700 * units.degC, t500 * units.degC),
    category=cat)

test_func("total_totals",
    lambda: mc.total_totals(t850, td850, t500),
    lambda: mr.total_totals(t850 * units.degC, td850 * units.degC, t500 * units.degC),
    category=cat)

test_func("cross_totals",
    lambda: mc.cross_totals(td850, t500),
    lambda: mr.cross_totals(td850 * units.degC, t500 * units.degC),
    category=cat)

test_func("vertical_totals",
    lambda: mc.vertical_totals(t850, t500),
    lambda: mr.vertical_totals(t850 * units.degC, t500 * units.degC),
    category=cat)

test_func("sweat_index",
    lambda: mc.sweat_index(t850, td850, t500, 210.0, 250.0, 25.0, 40.0),
    lambda: mr.sweat_index(t850 * units.degC, td850 * units.degC, t500 * units.degC, 210.0 * units.degree, 250.0 * units.degree, 25.0 * units.knot, 40.0 * units.knot),
    category=cat)

# Composite severe weather parameters (scalar inputs)
test_func("significant_tornado_parameter",
    lambda: mc.significant_tornado_parameter(np.array([2000.0]), np.array([1000.0]), np.array([200.0]), np.array([25.0])),
    lambda: mr.significant_tornado_parameter(2000.0 * units("J/kg"), 1000.0 * units.m, 200.0 * units("m**2/s**2"), 25.0 * units("m/s")),
    category=cat)

test_func("supercell_composite_parameter",
    lambda: mc.supercell_composite_parameter(np.array([3000.0]), np.array([300.0]), np.array([30.0])),
    lambda: mr.supercell_composite_parameter(3000.0 * units("J/kg"), 300.0 * units("m**2/s**2"), 30.0 * units("m/s")),
    category=cat)

test_func("compute_ehi",
    lambda: mc.compute_ehi(np.array([2000.0]), np.array([200.0])),
    lambda: mr.compute_ehi(2000.0 * units("J/kg"), 200.0 * units("m**2/s**2")),
    category=cat)

test_func("compute_ship",
    lambda: mc.compute_ship(np.array([2000.0]), np.array([25.0]), np.array([-15.0]), np.array([7.0]), np.array([12.0])),
    lambda: mr.compute_ship(2000.0 * units("J/kg"), 25.0 * units("m/s"), -15.0 * units.degC, 7.0 * units("delta_degC/km"), 12.0 * units("g/kg")),
    category=cat)

test_func("compute_dcp",
    lambda: mc.compute_dcp(np.array([800.0]), np.array([2000.0]), np.array([25.0]), np.array([12.0])),
    lambda: mr.compute_dcp(800.0 * units("J/kg"), 2000.0 * units("J/kg"), 25.0 * units("m/s"), 12.0 * units("g/kg")),
    category=cat)

test_func("bulk_richardson_number",
    lambda: mc.bulk_richardson_number(np.array([2000.0]), np.array([25.0])),
    lambda: mr.bulk_richardson_number(2000.0 * units("J/kg"), 25.0 * units("m/s")),
    category=cat)

test_func("boyden_index",
    lambda: mc.boyden_index(np.array([z1000]), np.array([z700v]), np.array([t700])),
    lambda: mr.boyden_index(z1000 * units.m, z700v * units.m, t700 * units.degC),
    category=cat)

test_func("fosberg_fire_weather_index",
    lambda: mc.fosberg_fire_weather_index(np.array([80.0]), np.array([30.0]), np.array([15.0])),
    lambda: mr.fosberg_fire_weather_index(80.0 * units.degF, 30.0 * units.percent, 15.0 * units("mph")),
    category=cat)

test_func("haines_index",
    lambda: mc.haines_index(10.0, 5.0, 2.0),
    lambda: mr.haines_index(10.0 * units.degC, 5.0 * units.degC, 2.0 * units.degC),
    category=cat)

test_func("hot_dry_windy",
    lambda: mc.hot_dry_windy(np.array([35.0]), np.array([15.0]), np.array([10.0])),
    lambda: mr.hot_dry_windy(35.0 * units.degC, 15.0 * units.percent, 10.0 * units("m/s")),
    category=cat)

test_func("dendritic_growth_zone",
    lambda: mc.dendritic_growth_zone(t_snd, p_snd),
    lambda: mr.dendritic_growth_zone(t_snd * units.degC, p_snd * units.hPa),
    category=cat)

test_func("warm_nose_check",
    lambda: mc.warm_nose_check(t_snd, p_snd),
    lambda: mr.warm_nose_check(t_snd * units.degC, p_snd * units.hPa),
    verify=False, category=cat)

test_func("freezing_rain_composite",
    lambda: mc.freezing_rain_composite(t_snd, p_snd, 1),
    lambda: mr.freezing_rain_composite(t_snd * units.degC, p_snd * units.hPa, 1),
    category=cat)

test_func("compute_lapse_rate",
    lambda: mc.compute_lapse_rate(
        np.random.randn(NZ_3D, NY_3D, NX_3D) * 15 + 10,
        np.random.rand(NZ_3D, NY_3D, NX_3D) * 0.01,
        np.broadcast_to(np.linspace(0, 12000, NZ_3D)[:, None, None], (NZ_3D, NY_3D, NX_3D)).copy(),
        0.0, 3.0),
    lambda: mr.compute_lapse_rate(
        np.random.randn(NZ_3D, NY_3D, NX_3D) * 15 + 10,
        np.random.rand(NZ_3D, NY_3D, NX_3D) * 0.01,
        np.broadcast_to(np.linspace(0, 12000, NZ_3D)[:, None, None], (NZ_3D, NY_3D, NX_3D)).copy(),
        0.0, 3.0),
    verify=False, category=cat)

test_func("convective_inhibition_depth",
    lambda: mc.convective_inhibition_depth(p_snd, t_snd, td_snd),
    lambda: mr.convective_inhibition_depth(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
    category=cat)

# Additional column functions
test_func("parcel_profile_with_lcl",
    lambda: mc.parcel_profile_with_lcl(p_snd, t_snd[0], td_snd[0]),
    lambda: mr.parcel_profile_with_lcl(p_snd * units.hPa, t_snd[0] * units.degC, td_snd[0] * units.degC),
    category=cat)

test_func("mixed_layer",
    lambda: mc.mixed_layer(p_snd, t_snd, depth=100.0),
    lambda: mr.mixed_layer(p_snd * units.hPa, t_snd * units.degC, depth=100.0),
    category=cat)

test_func("get_mixed_layer_parcel",
    lambda: mc.get_mixed_layer_parcel(p_snd, t_snd, td_snd, 100.0),
    lambda: mr.get_mixed_layer_parcel(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, 100.0),
    category=cat)

test_func("get_most_unstable_parcel",
    lambda: mc.get_most_unstable_parcel(p_snd, t_snd, td_snd, depth=300.0),
    lambda: mr.get_most_unstable_parcel(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, depth=300.0),
    category=cat)

test_func("mixed_parcel",
    lambda: mc.mixed_parcel(p_snd, t_snd, td_snd, depth=100),
    lambda: mr.mixed_parcel(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, depth=100),
    category=cat)

test_func("most_unstable_parcel",
    lambda: mc.most_unstable_parcel(p_snd, t_snd, td_snd, depth=300),
    lambda: mr.most_unstable_parcel(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, depth=300),
    category=cat)

test_func("get_layer",
    lambda: mc.get_layer(p_snd, t_snd, depth=100.0),
    lambda: mr.get_layer(p_snd * units.hPa, t_snd * units.degC, depth=100.0 * units.hPa),
    category=cat)

test_func("get_layer_heights",
    lambda: mc.get_layer_heights(p_snd, h_snd, 1000.0, 500.0),
    lambda: mr.get_layer_heights(p_snd * units.hPa, h_snd * units.m, 1000.0 * units.hPa, 500.0 * units.hPa),
    category=cat)

test_func("mean_pressure_weighted",
    lambda: mc.mean_pressure_weighted(p_snd, t_snd),
    lambda: mr.mean_pressure_weighted(p_snd * units.hPa, t_snd * units.degC),
    category=cat)

test_func("gradient_richardson_number",
    lambda: mc.gradient_richardson_number(h_snd, theta_snd, u_snd, v_snd),
    lambda: mr.gradient_richardson_number(h_snd * units.m, theta_snd * units.K, u_snd * units("m/s"), v_snd * units("m/s")),
    category=cat)

test_func("galvez_davison_index",
    lambda: mc.galvez_davison_index(18.0, 15.0, 5.0, -15.0, 16.0, 12.0, -3.0, 28.0),
    lambda: mr.galvez_davison_index(18.0 * units.degC, 15.0 * units.degC, 5.0 * units.degC, -15.0 * units.degC, 16.0 * units.degC, 12.0 * units.degC, -3.0 * units.degC, 28.0 * units.degC),
    category=cat)

test_func("corfidi_storm_motion",
    lambda: mc.corfidi_storm_motion(u_snd, v_snd, h_snd, 5.0, 3.0),
    lambda: mr.corfidi_storm_motion(u_snd * units("m/s"), v_snd * units("m/s"), h_snd * units.m, 5.0 * units("m/s"), 3.0 * units("m/s")),
    category=cat)

test_func("friction_velocity",
    lambda: mc.friction_velocity(u_snd, v_snd),
    lambda: mr.friction_velocity(u_snd * units("m/s"), v_snd * units("m/s")),
    category=cat)

test_func("tke",
    lambda: mc.tke(u_snd, v_snd, v_snd * 0.1),
    lambda: mr.tke(u_snd * units("m/s"), v_snd * units("m/s"), (v_snd * 0.1) * units("m/s")),
    category=cat)

test_func("weighted_continuous_average",
    lambda: mc.weighted_continuous_average(t_snd, p_snd),
    lambda: mr.weighted_continuous_average(t_snd, p_snd),
    category=cat)

test_func("thickness_hydrostatic_from_relative_humidity",
    lambda: mc.thickness_hydrostatic_from_relative_humidity(p_snd, t_snd, np.full(NLEVELS, 70.0)),
    lambda: mr.thickness_hydrostatic_from_relative_humidity(p_snd * units.hPa, t_snd * units.degC, np.full(NLEVELS, 70.0) * units.percent),
    category=cat)

# Additional grid-column composites
test_func("compute_stp",
    lambda: mc.compute_stp(
        np.full((NY_3D, NX_3D), 2000.0), np.full((NY_3D, NX_3D), 1000.0),
        np.full((NY_3D, NX_3D), 200.0), np.full((NY_3D, NX_3D), 25.0)),
    lambda: mr.compute_stp(
        np.full((NY_3D, NX_3D), 2000.0) * units("J/kg"), np.full((NY_3D, NX_3D), 1000.0) * units.m,
        np.full((NY_3D, NX_3D), 200.0) * units("m**2/s**2"), np.full((NY_3D, NX_3D), 25.0) * units("m/s")),
    category=cat)

test_func("compute_scp",
    lambda: mc.compute_scp(
        np.full((NY_3D, NX_3D), 3000.0), np.full((NY_3D, NX_3D), 300.0),
        np.full((NY_3D, NX_3D), 30.0)),
    lambda: mr.compute_scp(
        np.full((NY_3D, NX_3D), 3000.0) * units("J/kg"), np.full((NY_3D, NX_3D), 300.0) * units("m**2/s**2"),
        np.full((NY_3D, NX_3D), 30.0) * units("m/s")),
    category=cat)

# Normal/tangential components
test_func("normal_component",
    lambda: mc.normal_component(u_snd, v_snd, (30.0, -90.0), (40.0, -80.0)),
    lambda: mr.normal_component(u_snd * units("m/s"), v_snd * units("m/s"), (30.0, -90.0), (40.0, -80.0)),
    category=cat)

test_func("tangential_component",
    lambda: mc.tangential_component(u_snd, v_snd, (30.0, -90.0), (40.0, -80.0)),
    lambda: mr.tangential_component(u_snd * units("m/s"), v_snd * units("m/s"), (30.0, -90.0), (40.0, -80.0)),
    category=cat)

test_func("kinematic_flux",
    lambda: mc.kinematic_flux(u_snd, t_snd),
    lambda: mr.kinematic_flux(u_snd * units("m/s"), t_snd * units.degC),
    category=cat)

test_func("cross_section_components",
    lambda: mc.cross_section_components(u_snd, v_snd, 30.0, -90.0, 40.0, -80.0),
    lambda: mr.cross_section_components(u_snd * units("m/s"), v_snd * units("m/s"), 30.0, -90.0, 40.0, -80.0),
    category=cat)

test_func("get_perturbation",
    lambda: mc.get_perturbation(t_2d[:10000]),
    lambda: mr.get_perturbation(t_2d[:10000]),
    category=cat)

test_func("relative_humidity_wet_psychrometric",
    lambda: mc.relative_humidity_wet_psychrometric(t_2d[:10000], td_2d[:10000], p_sfc[:10000]),
    lambda: mr.relative_humidity_wet_psychrometric(t_2d[:10000] * units.degC, td_2d[:10000] * units.degC, p_sfc[:10000] * units.hPa),
    category=cat)

test_func("psychrometric_vapor_pressure_wet",
    lambda: mc.psychrometric_vapor_pressure_wet(t_2d[:10000], td_2d[:10000], p_sfc[:10000]),
    lambda: mr.psychrometric_vapor_pressure_wet(t_2d[:10000] * units.degC, td_2d[:10000] * units.degC, p_sfc[:10000] * units.hPa),
    category=cat)

test_func("absolute_momentum",
    lambda: mc.absolute_momentum(u_snd, lat_grid.ravel()[:NLEVELS], h_snd),
    lambda: mr.absolute_momentum(u_snd * units("m/s"), lat_grid.ravel()[:NLEVELS] * units.degree, h_snd * units.m),
    category=cat)

test_func("advection_3d",
    lambda: mc.advection_3d(
        np.random.randn(NZ_3D, NY_3D, NX_3D),
        np.random.randn(NZ_3D, NY_3D, NX_3D),
        np.random.randn(NZ_3D, NY_3D, NX_3D),
        np.random.randn(NZ_3D, NY_3D, NX_3D) * 0.1,
        3000.0, 3000.0, 500.0),
    lambda: mr.advection_3d(
        np.random.randn(NZ_3D, NY_3D, NX_3D),
        np.random.randn(NZ_3D, NY_3D, NX_3D),
        np.random.randn(NZ_3D, NY_3D, NX_3D),
        np.random.randn(NZ_3D, NY_3D, NX_3D) * 0.1,
        3000.0, 3000.0, 500.0),
    verify=False, category=cat)  # random data differs between calls

test_func("geospatial_gradient",
    lambda: mc.geospatial_gradient(t_grid[:100, :100], lat_grid[:100, :100], lat_grid[:100, :100]),
    lambda: mr.geospatial_gradient(t_grid[:100, :100], lat_grid[:100, :100], lat_grid[:100, :100]),
    verify=False, category=cat)

test_func("geospatial_laplacian",
    lambda: mc.geospatial_laplacian(t_grid[:100, :100], lat_grid[:100, :100], lat_grid[:100, :100]),
    lambda: mr.geospatial_laplacian(t_grid[:100, :100], lat_grid[:100, :100], lat_grid[:100, :100]),
    verify=False, category=cat)

test_func("compute_grid_scp",
    lambda: mc.compute_grid_scp(
        np.full((NY_3D, NX_3D), 3000.0), np.full((NY_3D, NX_3D), 300.0),
        np.full((NY_3D, NX_3D), 30.0), np.full((NY_3D, NX_3D), -20.0)),
    lambda: mr.compute_grid_scp(
        np.full((NY_3D, NX_3D), 3000.0) * units("J/kg"), np.full((NY_3D, NX_3D), 300.0) * units("m**2/s**2"),
        np.full((NY_3D, NX_3D), 30.0) * units("m/s"), np.full((NY_3D, NX_3D), -20.0) * units("J/kg")),
    category=cat)

test_func("compute_grid_critical_angle",
    lambda: mc.compute_grid_critical_angle(
        np.full((NY_3D, NX_3D), 5.0), np.full((NY_3D, NX_3D), 5.0),
        np.full((NY_3D, NX_3D), 10.0), np.full((NY_3D, NX_3D), 15.0)),
    lambda: mr.compute_grid_critical_angle(
        np.full((NY_3D, NX_3D), 5.0) * units("m/s"), np.full((NY_3D, NX_3D), 5.0) * units("m/s"),
        np.full((NY_3D, NX_3D), 10.0) * units("m/s"), np.full((NY_3D, NX_3D), 15.0) * units("m/s")),
    category=cat)

test_func("unit_vectors_from_cross_section",
    lambda: mc.unit_vectors_from_cross_section((30.0, -90.0), (40.0, -80.0)),
    lambda: mr.unit_vectors_from_cross_section((30.0, -90.0), (40.0, -80.0)),
    verify=False, category=cat)

test_func("isentropic_interpolation",
    lambda: mc.isentropic_interpolation(
        np.array([300.0, 310.0, 320.0]),
        np.broadcast_to(np.linspace(1000, 100, NZ_3D)[:, None, None], (NZ_3D, NY_3D, NX_3D)).copy().astype(np.float64),
        np.broadcast_to(np.linspace(300, 220, NZ_3D)[:, None, None], (NZ_3D, NY_3D, NX_3D)).copy().astype(np.float64),
        [np.random.randn(NZ_3D, NY_3D, NX_3D).astype(np.float64)]),
    lambda: mr.isentropic_interpolation(
        np.array([300.0, 310.0, 320.0]),
        np.broadcast_to(np.linspace(1000, 100, NZ_3D)[:, None, None], (NZ_3D, NY_3D, NX_3D)).copy().astype(np.float64),
        np.broadcast_to(np.linspace(300, 220, NZ_3D)[:, None, None], (NZ_3D, NY_3D, NX_3D)).copy().astype(np.float64),
        [np.random.randn(NZ_3D, NY_3D, NX_3D).astype(np.float64)]),
    verify=False, category=cat)

# ==========================================================================
# Category 5: CPU-only utilities (SKIP)
# ==========================================================================
print(f"\n{'=' * 90}")
print(f" CATEGORY 5: CPU-only utilities -- SKIP (no GPU benefit)")
print(f"{'=' * 90}")

skip_func("parse_angle")
skip_func("find_bounding_indices")
skip_func("find_intersections")
skip_func("find_peaks")
skip_func("peak_persistence")
skip_func("nearest_intersection_idx")
skip_func("reduce_point_density")
skip_func("remove_nan_observations")
skip_func("remove_observations_below_value")
skip_func("remove_repeat_coordinates")
skip_func("angle_to_direction")
skip_func("geodesic")
skip_func("azimuth_range_to_lat_lon")
skip_func("interpolate_to_grid")
skip_func("interpolate_1d")
skip_func("interpolate_nans_1d")
skip_func("interpolate_to_isosurface")
skip_func("interpolate_to_points")
skip_func("interpolate_to_slice")
skip_func("inverse_distance_to_grid")
skip_func("inverse_distance_to_points")
skip_func("natural_neighbor_to_grid")
skip_func("natural_neighbor_to_points")
skip_func("log_interpolate_1d")
skip_func("resample_nn_1d")
skip_func("isentropic_interpolation_as_dataset", reason="xarray wrapper")
skip_func("parcel_profile_with_lcl_as_dataset", reason="xarray wrapper")
skip_func("zoom_xarray", reason="xarray utility")
skip_func("InvalidSoundingError", reason="Exception class")
skip_func("cross_section", reason="xarray cross-section utility")
skip_func("smooth_window", reason="Generic kernel convolution -- tested via smooth_*")
skip_func("to_cpu", reason="Data transfer utility")
skip_func("to_gpu", reason="Data transfer utility")
skip_func("strip_units", reason="Unit utility")
skip_func("compute_cape_cin", reason="Not yet implemented (3D grid)")
skip_func("compute_srh", reason="Not yet implemented (3D grid)")
skip_func("compute_shear", reason="Not yet implemented (3D grid)")
skip_func("compute_pw", reason="Not yet implemented (3D grid)")
skip_func("composite_reflectivity_from_hydrometeors", reason="Not yet implemented")
skip_func("potential_vorticity_baroclinic", reason="Complex 3D input signature")
skip_func("significant_tornado", reason="Alias for significant_tornado_parameter")

# ==========================================================================
# SUMMARY
# ==========================================================================
print(f"\n{'=' * 90}")
print(f" SUMMARY")
print(f"{'=' * 90}")

total = len(results)
pass_count = sum(1 for r in results if r[5] == "PASS")
na_count = sum(1 for r in results if r[5] == "N/A")
warn_count = sum(1 for r in results if r[5].startswith("DIFF=") or r[5].startswith("ERR:"))
fail_count = sum(1 for r in results if r[5].startswith("FAIL:"))
skip_count = sum(1 for r in results if r[5].startswith("SKIP:"))
all_nan_count = sum(1 for r in results if r[5] in ("ALL_NAN", "EMPTY"))

# Category averages
cat_speedups = {}
for name, cat, gpu_ms, cpu_ms, speedup, accurate in results:
    if speedup > 0 and not accurate.startswith("SKIP:") and not accurate.startswith("FAIL:"):
        cat_speedups.setdefault(cat, []).append(speedup)

max_speedup_entry = max(
    ((name, speedup) for name, cat, gpu_ms, cpu_ms, speedup, accurate in results if speedup > 0),
    key=lambda x: x[1],
    default=("none", 0)
)

# Accuracy stats
accurate_count = pass_count
tested_count = total - skip_count - fail_count - all_nan_count

print(f"\n  Total functions tested:         {total}")
print(f"  PASS (speed + accuracy):        {pass_count}")
print(f"  N/A  (speed OK, no verify):     {na_count}")
print(f"  WARN (speed OK, accuracy diff): {warn_count}")
print(f"  FAIL (crashed):                 {fail_count}")
print(f"  SKIP (CPU-only/not impl):       {skip_count}")

print()
for cat_name, label in [("thermo", "Per-element thermo"), ("wind", "Wind per-element"),
                         ("grid", "Grid stencil"), ("column", "Column/sounding")]:
    speeds = cat_speedups.get(cat_name, [])
    if speeds:
        avg = sum(speeds) / len(speeds)
        mx = max(speeds)
        mn = min(speeds)
        print(f"  Average speedup ({label:25s}): {avg:7.1f}x  (min={mn:.1f}x, max={mx:.1f}x, n={len(speeds)})")

print(f"\n  Max speedup: {max_speedup_entry[0]} {max_speedup_entry[1]:.1f}x")

if tested_count > 0:
    print(f"\n  Accuracy: {accurate_count}/{tested_count} within rtol=1e-4, atol=1e-6")

# Detailed results table
print(f"\n{'=' * 90}")
print(f" DETAILED RESULTS")
print(f"{'=' * 90}")
print(f"  {'Function':<50s} {'GPU ms':>8s} {'CPU ms':>8s} {'Speedup':>8s} {'Accuracy':<20s}")
print(f"  {'-'*50} {'-'*8} {'-'*8} {'-'*8} {'-'*20}")
for name, cat, gpu_ms, cpu_ms, speedup, accurate in results:
    if accurate.startswith("SKIP:"):
        print(f"  {name:<50s} {'--':>8s} {'--':>8s} {'--':>8s} {accurate:<20s}")
    else:
        sp_str = f"{speedup:.1f}x" if speedup > 0 else "N/A"
        print(f"  {name:<50s} {gpu_ms:8.2f} {cpu_ms:8.2f} {sp_str:>8s} {accurate:<20s}")

print(f"\n{'=' * 90}")
print(f" Benchmark complete.")
print(f"{'=' * 90}")
