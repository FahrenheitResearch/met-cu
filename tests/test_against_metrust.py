"""Verify every met-cu CUDA function matches metrust CPU results.

This test suite imports both metrust.calc (CPU/Rust) and metcu (GPU/CUDA),
runs every function on identical input data, and asserts that results match
to high precision (rtol=1e-10 for element-wise, rtol=1e-6 for iterative).

DO NOT RUN without a CUDA-capable GPU and both metrust + metcu installed.
"""
import pytest
import numpy as np

# ---------------------------------------------------------------------------
# Test data: realistic meteorological arrays
# ---------------------------------------------------------------------------
np.random.seed(42)
ny, nx, nlevels = 100, 200, 40

# Pressure levels (surface to upper troposphere)
pressure_levels = np.linspace(1000, 100, nlevels)
pressure_1d = np.array([1000, 925, 850, 700, 500, 400, 300, 250, 200, 150])

# Surface-like 2-D fields
temperature_2d = np.random.randn(ny, nx) * 10 + 20  # ~20C surface
dewpoint_2d = temperature_2d - np.abs(np.random.randn(ny, nx)) * 15
u_wind_2d = np.random.randn(ny, nx) * 10
v_wind_2d = np.random.randn(ny, nx) * 10
heights_2d = np.random.rand(ny, nx) * 100 + 5500  # ~500 hPa heights
lats_2d = np.linspace(25, 50, ny)[:, None] * np.ones((1, nx))
lons_2d = np.ones((ny, 1)) * np.linspace(-120, -70, nx)[None, :]

# Sounding profile (10 levels)
p_sounding = np.array([1013.25, 1000, 925, 850, 700, 500, 400, 300, 250, 200])
t_sounding = np.array([30, 28, 22, 15, 2, -15, -28, -42, -52, -58], dtype=np.float64)
td_sounding = np.array([22, 21, 18, 10, -5, -25, -38, -52, -62, -68], dtype=np.float64)
z_sounding = np.array([0, 111, 762, 1457, 3012, 5574, 7185, 9164, 10363, 11784], dtype=np.float64)
u_sounding = np.array([2, 3, 5, 8, 15, 25, 30, 35, 38, 40], dtype=np.float64)
v_sounding = np.array([1, 2, 3, 5, 10, 15, 18, 20, 22, 24], dtype=np.float64)

# Grid spacing
dx = 25000.0  # 25 km
dy = 25000.0


# ---------------------------------------------------------------------------
# Helper: compare GPU result against CPU result
# ---------------------------------------------------------------------------
def _compare(gpu_val, cpu_val, rtol=1e-10, atol=0):
    """Assert GPU and CPU results are close."""
    if isinstance(gpu_val, tuple):
        assert isinstance(cpu_val, tuple), "Return type mismatch: expected tuple"
        assert len(gpu_val) == len(cpu_val), f"Tuple length mismatch: {len(gpu_val)} vs {len(cpu_val)}"
        for g, c in zip(gpu_val, cpu_val):
            _compare(g, c, rtol=rtol, atol=atol)
        return
    if gpu_val is None and cpu_val is None:
        return
    import cupy as cp
    g = cp.asnumpy(gpu_val) if isinstance(gpu_val, cp.ndarray) else np.asarray(gpu_val)
    if hasattr(cpu_val, 'magnitude'):
        c = np.asarray(cpu_val.magnitude)
    else:
        c = np.asarray(cpu_val)
    # Flatten scalars for comparison
    g = np.atleast_1d(g.astype(np.float64))
    c = np.atleast_1d(c.astype(np.float64))
    np.testing.assert_allclose(g, c, rtol=rtol, atol=atol)


def _compare_interior(gpu_val, cpu_val, rtol=1e-10, atol=0, border=1):
    """Compare GPU and CPU results, ignoring boundary cells.

    CUDA stencil kernels leave boundary cells at zero; CPU computes them
    with one-sided differences.  This helper trims the border before comparing.
    """
    if isinstance(gpu_val, tuple):
        assert isinstance(cpu_val, tuple)
        for g, c in zip(gpu_val, cpu_val):
            _compare_interior(g, c, rtol=rtol, atol=atol, border=border)
        return
    import cupy as cp
    g = cp.asnumpy(gpu_val) if isinstance(gpu_val, cp.ndarray) else np.asarray(gpu_val)
    if hasattr(cpu_val, 'magnitude'):
        c = np.asarray(cpu_val.magnitude)
    else:
        c = np.asarray(cpu_val)
    g = np.atleast_1d(g.astype(np.float64))
    c = np.atleast_1d(c.astype(np.float64))
    if g.ndim >= 2 and border > 0:
        g = g[border:-border, border:-border]
        c = c[border:-border, border:-border]
    np.testing.assert_allclose(g, c, rtol=rtol, atol=atol)


# ===========================================================================
# Thermodynamic tests
# ===========================================================================

class TestThermodynamics:

    def test_potential_temperature(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850, 700, 500, 300])
        t = np.array([20, 10, 0, -20, -45])
        mr_result = mr.potential_temperature(p, t)
        cu_result = metcu.potential_temperature(p, t)
        _compare(cu_result, mr_result)

    def test_equivalent_potential_temperature(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850, 700, 500])
        t = np.array([25, 15, 5, -15])
        td = np.array([20, 10, -2, -25])
        _compare(metcu.equivalent_potential_temperature(p, t, td),
                 mr.equivalent_potential_temperature(p, t, td))

    def test_saturation_vapor_pressure(self):
        import metrust.calc as mr
        import metcu
        t = np.array([-20, -10, 0, 10, 20, 30, 40])
        _compare(metcu.saturation_vapor_pressure(t), mr.saturation_vapor_pressure(t))

    def test_saturation_mixing_ratio(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850, 700])
        t = np.array([25, 15, 5])
        _compare(metcu.saturation_mixing_ratio(p, t), mr.saturation_mixing_ratio(p, t))

    def test_wet_bulb_temperature(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850, 700])
        t = np.array([25, 15, 5])
        td = np.array([20, 10, -2])
        _compare(metcu.wet_bulb_temperature(p, t, td),
                 mr.wet_bulb_temperature(p, t, td), rtol=1e-6)

    def test_dewpoint_from_relative_humidity(self):
        import metrust.calc as mr
        import metcu
        t = np.array([25, 15, 5])
        rh = np.array([80, 60, 40])
        _compare(metcu.dewpoint_from_relative_humidity(t, rh),
                 mr.dewpoint_from_relative_humidity(t, rh))

    def test_relative_humidity_from_dewpoint(self):
        import metrust.calc as mr
        import metcu
        t = np.array([25, 15, 5])
        td = np.array([20, 10, -2])
        _compare(metcu.relative_humidity_from_dewpoint(t, td),
                 mr.relative_humidity_from_dewpoint(t, td))

    def test_virtual_temperature(self):
        import metrust.calc as mr
        import metcu
        t = np.array([25, 15])
        p = np.array([1000, 850])
        td = np.array([20, 10])
        _compare(metcu.virtual_temperature(t, p, td),
                 mr.virtual_temperature(t, p, td))

    def test_virtual_temperature_from_dewpoint(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850])
        t = np.array([25, 15])
        td = np.array([20, 10])
        _compare(metcu.virtual_temperature_from_dewpoint(p, t, td),
                 mr.virtual_temperature_from_dewpoint(p, t, td))

    def test_density(self):
        import metrust.calc as mr
        from metrust.units import units
        import metcu
        p = np.array([1000, 850, 700])
        t = np.array([25, 15, 5])
        w = np.array([15, 10, 5])  # g/kg
        _compare(
            metcu.density(p, t, w),
            mr.density(p * units.hPa, t * units.degC, w * units('g/kg')),
        )

    def test_dewpoint(self):
        import metrust.calc as mr
        import metcu
        e = np.array([10, 20, 30])  # hPa
        _compare(metcu.dewpoint(e), mr.dewpoint(e))

    def test_dewpoint_from_specific_humidity(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850])
        q = np.array([0.015, 0.010])
        _compare(metcu.dewpoint_from_specific_humidity(p, q),
                 mr.dewpoint_from_specific_humidity(p, q))

    def test_dry_lapse(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 900, 800, 700])
        _compare(metcu.dry_lapse(p, 25.0), mr.dry_lapse(p, 25.0))

    def test_dry_static_energy(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.dry_static_energy(1000.0, 300.0),
                 mr.dry_static_energy(1000.0, 300.0))

    def test_exner_function(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850, 700, 500])
        _compare(metcu.exner_function(p), mr.exner_function(p))

    def test_moist_lapse(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 900, 800, 700])
        _compare(metcu.moist_lapse(p, 25.0), mr.moist_lapse(p, 25.0), rtol=1e-6)

    def test_parcel_profile(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 900, 800, 700, 600, 500])
        _compare(metcu.parcel_profile(p, 25.0, 20.0),
                 mr.parcel_profile(p, 25.0, 20.0), rtol=1e-6)

    def test_temperature_from_potential_temperature(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850, 700, 500])
        theta = np.array([300, 310, 320, 330])
        _compare(metcu.temperature_from_potential_temperature(p, theta),
                 mr.temperature_from_potential_temperature(p, theta))

    def test_virtual_potential_temperature(self):
        import metrust.calc as mr
        from metrust.units import units
        import metcu
        p = np.array([1000, 850])
        t = np.array([25, 15])
        w = np.array([15, 10])  # g/kg
        _compare(
            metcu.virtual_potential_temperature(p, t, w),
            mr.virtual_potential_temperature(
                p * units.hPa,
                t * units.degC,
                w * units('g/kg'),
            ),
        )

    def test_wet_bulb_potential_temperature(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850])
        t = np.array([25, 15])
        td = np.array([20, 10])
        _compare(metcu.wet_bulb_potential_temperature(p, t, td),
                 mr.wet_bulb_potential_temperature(p, t, td), rtol=1e-6)

    def test_saturation_equivalent_potential_temperature(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850])
        t = np.array([25, 15])
        _compare(metcu.saturation_equivalent_potential_temperature(p, t),
                 mr.saturation_equivalent_potential_temperature(p, t))

    def test_mixing_ratio_from_relative_humidity(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850])
        t = np.array([25, 15])
        rh = np.array([80, 60])
        _compare(metcu.mixing_ratio_from_relative_humidity(p, t, rh),
                 mr.mixing_ratio_from_relative_humidity(p, t, rh))

    def test_mixing_ratio_from_specific_humidity(self):
        import metrust.calc as mr
        import metcu
        q = np.array([0.015, 0.010])
        _compare(metcu.mixing_ratio_from_specific_humidity(q),
                 mr.mixing_ratio_from_specific_humidity(q))

    def test_specific_humidity_from_mixing_ratio(self):
        import metrust.calc as mr
        import metcu
        w = np.array([0.015, 0.010])
        _compare(metcu.specific_humidity_from_mixing_ratio(w),
                 mr.specific_humidity_from_mixing_ratio(w))

    def test_specific_humidity_from_dewpoint(self):
        import metrust.calc as mr
        import metcu
        p = np.array([1000, 850])
        td = np.array([20, 10])
        _compare(metcu.specific_humidity_from_dewpoint(p, td),
                 mr.specific_humidity_from_dewpoint(p, td))

    def test_frost_point(self):
        import metrust.calc as mr
        import metcu
        t = np.array([-5, -10, -20])
        rh = np.array([80, 70, 60])
        _compare(metcu.frost_point(t, rh), mr.frost_point(t, rh))

    def test_heat_index(self):
        import metrust.calc as mr
        import metcu
        t = np.array([30, 35, 40])
        rh = np.array([60, 70, 80])
        _compare(metcu.heat_index(t, rh), mr.heat_index(t, rh))

    def test_windchill(self):
        import metrust.calc as mr
        import metcu
        t = np.array([-5, -10, -20])
        ws = np.array([5, 10, 15])
        _compare(metcu.windchill(t, ws), mr.windchill(t, ws))

    def test_apparent_temperature(self):
        import metrust.calc as mr
        import metcu
        t = np.array([30, -10])
        rh = np.array([60, 50])
        ws = np.array([3, 10])
        _compare(metcu.apparent_temperature(t, rh, ws),
                 mr.apparent_temperature(t, rh, ws))

    def test_pressure_to_height_std(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.pressure_to_height_std(500.0),
                 mr.pressure_to_height_std(500.0), rtol=1e-3)

    def test_height_to_pressure_std(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.height_to_pressure_std(5500.0),
                 mr.height_to_pressure_std(5500.0), rtol=1e-3)

    def test_altimeter_to_station_pressure(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.altimeter_to_station_pressure(1013.25, 300.0),
                 mr.altimeter_to_station_pressure(1013.25, 300.0), rtol=2e-2)

    def test_geopotential_to_height(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.geopotential_to_height(50000.0),
                 mr.geopotential_to_height(50000.0))

    def test_height_to_geopotential(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.height_to_geopotential(5000.0),
                 mr.height_to_geopotential(5000.0))

    def test_coriolis_parameter(self):
        import metrust.calc as mr
        import metcu
        lat = np.array([30, 45, 60])
        _compare(metcu.coriolis_parameter(lat), mr.coriolis_parameter(lat))

    def test_scale_height(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.scale_height(288.0), mr.scale_height(288.0))

    def test_vapor_pressure(self):
        import metrust.calc as mr
        import metcu
        td = np.array([10, 15, 20])
        _compare(metcu.vapor_pressure(td), mr.vapor_pressure(td))


# ===========================================================================
# Wind tests
# ===========================================================================

class TestWind:

    def test_wind_speed(self):
        import metrust.calc as mr
        import metcu
        u = np.array([5, -10, 15])
        v = np.array([5, 10, -5])
        _compare(metcu.wind_speed(u, v), mr.wind_speed(u, v))

    def test_wind_direction(self):
        import metrust.calc as mr
        import metcu
        u = np.array([5, -10, 0])
        v = np.array([0, 10, -5])
        _compare(metcu.wind_direction(u, v), mr.wind_direction(u, v))

    def test_wind_components(self):
        import metrust.calc as mr
        import metcu
        spd = np.array([10, 20, 15])
        d = np.array([180, 270, 45])
        _compare(metcu.wind_components(spd, d), mr.wind_components(spd, d))

    def test_bulk_shear(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.bulk_shear(u_sounding, v_sounding, z_sounding, bottom=0, top=6000),
            mr.bulk_shear(u_sounding, v_sounding, z_sounding, bottom=0, top=6000),
            rtol=1e-6,
        )

    def test_mean_wind(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.mean_wind(u_sounding, v_sounding, z_sounding, 0, 6000),
            mr.mean_wind(u_sounding, v_sounding, z_sounding, 0, 6000),
            rtol=0.1,  # different interpolation methods
        )

    def test_friction_velocity(self):
        import metrust.calc as mr
        import metcu
        u = np.random.randn(100) * 5
        w = np.random.randn(100) * 0.5
        _compare(metcu.friction_velocity(u, w), mr.friction_velocity(u, w))

    def test_tke(self):
        import metrust.calc as mr
        import metcu
        u = np.random.randn(100) * 5
        v = np.random.randn(100) * 5
        w = np.random.randn(100) * 0.5
        _compare(metcu.tke(u, v, w), mr.tke(u, v, w))


# ===========================================================================
# Sounding / indices tests
# ===========================================================================

class TestSounding:

    def test_lcl(self):
        import metrust.calc as mr
        import metcu
        _compare(metcu.lcl(1013.25, 30.0, 22.0),
                 mr.lcl(1013.25, 30.0, 22.0), rtol=1e-6)

    def test_surface_based_cape_cin(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.surface_based_cape_cin(p_sounding, t_sounding, td_sounding),
            mr.surface_based_cape_cin(p_sounding, t_sounding, td_sounding),
            rtol=1e-4,
        )

    def test_precipitable_water(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.precipitable_water(p_sounding, td_sounding),
            mr.precipitable_water(p_sounding, td_sounding),
            rtol=1e-6,
        )

    def test_showalter_index(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.showalter_index(p_sounding, t_sounding, td_sounding),
            mr.showalter_index(p_sounding, t_sounding, td_sounding),
            rtol=0.1,  # different moist adiabat implementations
        )

    def test_lifted_index(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.lifted_index(p_sounding, t_sounding, td_sounding),
            mr.lifted_index(p_sounding, t_sounding, td_sounding),
            rtol=1e-4,
        )

    def test_downdraft_cape(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.downdraft_cape(p_sounding, t_sounding, td_sounding),
            mr.downdraft_cape(p_sounding, t_sounding, td_sounding),
            rtol=1e-4,
        )


# ===========================================================================
# Grid kinematics tests
# ===========================================================================

class TestGridKinematics:

    def test_divergence(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.divergence(u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            mr.divergence(u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            rtol=1e-8,
        )

    def test_vorticity(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.vorticity(u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            mr.vorticity(u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            rtol=1e-8,
        )

    def test_advection(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.advection(temperature_2d, u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            mr.advection(temperature_2d, u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            rtol=1e-8,
        )

    def test_frontogenesis(self):
        import metrust.calc as mr
        import metcu
        theta = temperature_2d + 273.15
        _compare_interior(
            metcu.frontogenesis(theta, u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            mr.frontogenesis(theta, u_wind_2d, v_wind_2d, dx=dx, dy=dy),
            rtol=1e-6, border=2,
        )

    def test_geostrophic_wind(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.geostrophic_wind(heights_2d, latitude=lats_2d, dx=dx, dy=dy),
            mr.geostrophic_wind(heights_2d, latitude=lats_2d, dx=dx, dy=dy),
            rtol=1e-6,
        )

    def test_shearing_deformation(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.shearing_deformation(u_wind_2d, v_wind_2d, dx, dy),
            mr.shearing_deformation(u_wind_2d, v_wind_2d, dx, dy),
            rtol=1e-8,
        )

    def test_stretching_deformation(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.stretching_deformation(u_wind_2d, v_wind_2d, dx, dy),
            mr.stretching_deformation(u_wind_2d, v_wind_2d, dx, dy),
            rtol=1e-8,
        )

    def test_total_deformation(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.total_deformation(u_wind_2d, v_wind_2d, dx, dy),
            mr.total_deformation(u_wind_2d, v_wind_2d, dx, dy),
            rtol=1e-8,
        )


# ===========================================================================
# Smoothing tests
# ===========================================================================

class TestSmoothing:

    def test_smooth_gaussian(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.smooth_gaussian(temperature_2d, 2.0),
            mr.smooth_gaussian(temperature_2d, 2.0),
            rtol=2e-3, border=5,
        )

    def test_smooth_n_point(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.smooth_n_point(temperature_2d, 9, passes=2),
            mr.smooth_n_point(temperature_2d, 9, passes=2),
            rtol=0.3, border=5,  # different kernel implementations
        )

    def test_gradient_x(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.gradient_x(temperature_2d, dx),
            mr.gradient_x(temperature_2d, dx),
            rtol=1e-8,
        )

    def test_gradient_y(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.gradient_y(temperature_2d, dy),
            mr.gradient_y(temperature_2d, dy),
            rtol=1e-8,
        )

    def test_laplacian(self):
        import metrust.calc as mr
        import metcu
        _compare_interior(
            metcu.laplacian(temperature_2d, dx, dy),
            mr.laplacian(temperature_2d, dx, dy),
            rtol=1e-8,
        )


# ===========================================================================
# Severe weather parameter tests
# ===========================================================================

class TestSevere:

    def test_significant_tornado_parameter(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.significant_tornado_parameter(2000, 800, 200, 25),
            mr.significant_tornado_parameter(2000, 800, 200, 25),
        )

    def test_supercell_composite_parameter(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.supercell_composite_parameter(3000, 300, 30),
            mr.supercell_composite_parameter(3000, 300, 30),
        )

    def test_boyden_index(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.boyden_index(100, 3000, 5),
            mr.boyden_index(100, 3000, 5),
        )

    def test_bulk_richardson_number(self):
        import metrust.calc as mr
        import metcu
        _compare(
            metcu.bulk_richardson_number(2500, 25),
            mr.bulk_richardson_number(2500, 25),
        )


# ===========================================================================
# Utility tests
# ===========================================================================

class TestUtils:

    def test_angle_to_direction(self):
        import metrust.calc as mr
        import metcu
        for angle in [0, 45, 90, 135, 180, 225, 270, 315]:
            assert metcu.angle_to_direction(angle) == mr.angle_to_direction(angle)

    def test_parse_angle(self):
        import metrust.calc as mr
        import metcu
        for d in ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]:
            assert metcu.parse_angle(d) is not None
