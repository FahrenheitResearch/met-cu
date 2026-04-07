from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

import metpy.calc as mpcalc
import metcu

from metpy.units import units

from tests._metpy_conformance import assert_runtime_close, sounding_profile


@pytest.fixture(scope="module")
def thermo_context(sounding_profile):
    parcel_profile = mpcalc.parcel_profile(
        sounding_profile["pressure"],
        sounding_profile["temperature"][0],
        sounding_profile["dewpoint"][0],
    )
    return {
        "p": sounding_profile["pressure"],
        "h": sounding_profile["height"],
        "t": sounding_profile["temperature"],
        "td": sounding_profile["dewpoint"],
        "rh": np.linspace(0.55, 0.9, 10) * units.dimensionless,
        "parcel_profile": parcel_profile,
    }


def _case_relative_humidity_wet_psychrometric(ctx):
    return (900 * units.hPa, 20 * units.degC, 17 * units.degC), {}


def _case_weighted_continuous_average(ctx):
    return (ctx["p"], ctx["t"]), {"bottom": 900 * units.hPa, "depth": 150 * units.hPa}


def _case_add_height_to_pressure(ctx):
    return (850 * units.hPa, 150 * units.m), {}


def _case_add_pressure_to_height(ctx):
    return (1500 * units.m, 25 * units.hPa), {}


def _case_thickness_hydrostatic_from_relative_humidity(ctx):
    return (ctx["p"][:10], ctx["t"][:10], ctx["rh"]), {}


def _case_ccl(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_density(ctx):
    return (900 * units.hPa, 20 * units.degC, 0.01 * units("kg/kg")), {}


def _case_dry_static_energy(ctx):
    return (1500 * units.m, 290 * units.kelvin), {}


def _case_geopotential_to_height(ctx):
    return (15000 * units("m^2/s^2"),), {}


def _case_get_layer_heights(ctx):
    return (ctx["h"], 2500 * units.m, ctx["t"]), {"bottom": 500 * units.m}


def _case_height_to_geopotential(ctx):
    return (1500 * units.m,), {}


def _case_moist_air_gas_constant(ctx):
    return (0.01 * units("kg/kg"),), {}


def _case_moist_air_specific_heat_pressure(ctx):
    return (0.01 * units("kg/kg"),), {}


def _case_moist_air_poisson_exponent(ctx):
    return (0.01 * units("kg/kg"),), {}


def _case_moist_static_energy(ctx):
    return (1500 * units.m, 290 * units.kelvin, 0.008 * units("kg/kg")), {}


def _case_montgomery_streamfunction(ctx):
    return (1500 * units.m, 290 * units.kelvin), {}


def _case_most_unstable_cape_cin(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_saturation_equivalent_potential_temperature(ctx):
    return (850 * units.hPa, 20 * units.degC), {}


def _case_scale_height(ctx):
    return (20 * units.degC, -50 * units.degC), {}


def _case_specific_humidity_from_dewpoint(ctx):
    return (900 * units.hPa, 15 * units.degC), {}


def _case_surface_based_cape_cin(ctx):
    return (ctx["p"], ctx["t"], ctx["td"]), {}


def _case_lifted_index(ctx):
    return (ctx["p"], ctx["t"], ctx["parcel_profile"]), {}


THERMO_LAYER_CASES = [
    pytest.param("relative_humidity_wet_psychrometric", _case_relative_humidity_wet_psychrometric, 1e-12, id="relative_humidity_wet_psychrometric"),
    pytest.param("weighted_continuous_average", _case_weighted_continuous_average, 1e-12, id="weighted_continuous_average"),
    pytest.param("add_height_to_pressure", _case_add_height_to_pressure, 1e-3, id="add_height_to_pressure"),
    pytest.param("add_pressure_to_height", _case_add_pressure_to_height, 1e-5, id="add_pressure_to_height"),
    pytest.param("thickness_hydrostatic_from_relative_humidity", _case_thickness_hydrostatic_from_relative_humidity, 0.1, id="thickness_hydrostatic_from_relative_humidity"),
    pytest.param("ccl", _case_ccl, 1e-9, id="ccl"),
    pytest.param("density", _case_density, 1e-12, id="density"),
    pytest.param("dry_static_energy", _case_dry_static_energy, 1e-12, id="dry_static_energy"),
    pytest.param("geopotential_to_height", _case_geopotential_to_height, 1e-12, id="geopotential_to_height"),
    pytest.param("get_layer_heights", _case_get_layer_heights, 1e-12, id="get_layer_heights"),
    pytest.param("height_to_geopotential", _case_height_to_geopotential, 1e-12, id="height_to_geopotential"),
    pytest.param("moist_air_gas_constant", _case_moist_air_gas_constant, 1e-12, id="moist_air_gas_constant"),
    pytest.param("moist_air_specific_heat_pressure", _case_moist_air_specific_heat_pressure, 1e-12, id="moist_air_specific_heat_pressure"),
    pytest.param("moist_air_poisson_exponent", _case_moist_air_poisson_exponent, 1e-12, id="moist_air_poisson_exponent"),
    pytest.param("moist_static_energy", _case_moist_static_energy, 1e-12, id="moist_static_energy"),
    pytest.param("montgomery_streamfunction", _case_montgomery_streamfunction, 1e-12, id="montgomery_streamfunction"),
    pytest.param("most_unstable_cape_cin", _case_most_unstable_cape_cin, 10.0, id="most_unstable_cape_cin"),
    pytest.param("saturation_equivalent_potential_temperature", _case_saturation_equivalent_potential_temperature, 1e-12, id="saturation_equivalent_potential_temperature"),
    pytest.param("scale_height", _case_scale_height, 1e-12, id="scale_height"),
    pytest.param("specific_humidity_from_dewpoint", _case_specific_humidity_from_dewpoint, 1e-12, id="specific_humidity_from_dewpoint"),
    pytest.param("surface_based_cape_cin", _case_surface_based_cape_cin, 5.0, id="surface_based_cape_cin"),
    pytest.param("lifted_index", _case_lifted_index, 1e-12, id="lifted_index"),
]


@pytest.mark.parametrize(("name", "builder", "atol"), THERMO_LAYER_CASES)
def test_runtime_parity_thermo_layers(thermo_context, name, builder, atol):
    args, kwargs = builder(thermo_context)
    actual = getattr(metcu, name)(*args, **kwargs)
    expected = getattr(mpcalc, name)(*args, **kwargs)
    assert_runtime_close(actual, expected, atol)


@pytest.fixture(scope="module")
def wind_profile_context(sounding_profile):
    u, v = mpcalc.wind_components(sounding_profile["speed"], sounding_profile["direction"])
    theta = mpcalc.potential_temperature(sounding_profile["pressure"], sounding_profile["temperature"])
    return {
        "pressure": sounding_profile["pressure"],
        "temperature": sounding_profile["temperature"],
        "dewpoint": sounding_profile["dewpoint"],
        "height": sounding_profile["height"],
        "speed": sounding_profile["speed"],
        "direction": sounding_profile["direction"],
        "u": u,
        "v": v,
        "theta": theta,
    }


@pytest.fixture(scope="module")
def turbulence_context():
    rng = np.random.default_rng(0)
    u = (8 + rng.normal(0, 1.5, (3, 256))) * units("m/s")
    v = (2 + rng.normal(0, 1.0, (3, 256))) * units("m/s")
    w = rng.normal(0, 0.35, (3, 256)) * units("m/s")
    up = u - u.mean(axis=1, keepdims=True)
    vp = v - v.mean(axis=1, keepdims=True)
    wp = w - w.mean(axis=1, keepdims=True)
    return {"u": u, "v": v, "w": w, "up": up, "vp": vp, "wp": wp}


def test_wet_bulb_potential_temperature_runtime_parity(wind_profile_context):
    actual = metcu.wet_bulb_potential_temperature(
        wind_profile_context["pressure"][:5],
        wind_profile_context["temperature"][:5],
        wind_profile_context["dewpoint"][:5],
    )
    expected = mpcalc.wet_bulb_potential_temperature(
        wind_profile_context["pressure"][:5],
        wind_profile_context["temperature"][:5],
        wind_profile_context["dewpoint"][:5],
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_mixed_parcel_runtime_parity(wind_profile_context):
    kwargs = {"depth": 150 * units.hPa}
    actual = metcu.mixed_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    expected = mpcalc.mixed_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_most_unstable_parcel_runtime_parity(wind_profile_context):
    kwargs = {"depth": 300 * units.hPa}
    actual = metcu.most_unstable_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    expected = mpcalc.most_unstable_parcel(
        wind_profile_context["pressure"],
        wind_profile_context["temperature"],
        wind_profile_context["dewpoint"],
        **kwargs,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_psychrometric_vapor_pressure_wet_runtime_parity():
    kwargs = {"psychrometer_coefficient": 7e-4 / units.kelvin}
    actual = metcu.psychrometric_vapor_pressure_wet(
        958 * units.hPa,
        25 * units.degC,
        12 * units.degC,
        **kwargs,
    )
    expected = mpcalc.psychrometric_vapor_pressure_wet(
        958 * units.hPa,
        25 * units.degC,
        12 * units.degC,
        **kwargs,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_storm_relative_helicity_runtime_parity(wind_profile_context):
    kwargs = {
        "bottom": 500 * units.meter,
        "storm_u": 5 * units("m/s"),
        "storm_v": -2 * units("m/s"),
    }
    actual = metcu.storm_relative_helicity(
        wind_profile_context["height"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        3 * units.kilometer,
        **kwargs,
    )
    expected = mpcalc.storm_relative_helicity(
        wind_profile_context["height"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        3 * units.kilometer,
        **kwargs,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_corfidi_storm_motion_runtime_parity(wind_profile_context):
    kwargs = {
        "u_llj": wind_profile_context["u"][1],
        "v_llj": wind_profile_context["v"][1],
    }
    actual = metcu.corfidi_storm_motion(
        wind_profile_context["pressure"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        **kwargs,
    )
    expected = mpcalc.corfidi_storm_motion(
        wind_profile_context["pressure"],
        wind_profile_context["u"],
        wind_profile_context["v"],
        **kwargs,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_friction_velocity_runtime_parity(turbulence_context):
    actual = metcu.friction_velocity(
        turbulence_context["u"],
        turbulence_context["w"],
        v=turbulence_context["v"],
        axis=1,
    )
    expected = mpcalc.friction_velocity(
        turbulence_context["u"],
        turbulence_context["w"],
        v=turbulence_context["v"],
        axis=1,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_tke_runtime_parity(turbulence_context):
    actual = metcu.tke(
        turbulence_context["up"],
        turbulence_context["vp"],
        turbulence_context["wp"],
        perturbation=True,
        axis=1,
    )
    expected = mpcalc.tke(
        turbulence_context["up"],
        turbulence_context["vp"],
        turbulence_context["wp"],
        perturbation=True,
        axis=1,
    )
    assert_runtime_close(actual, expected, 1e-10)


def test_gradient_richardson_number_runtime_parity(wind_profile_context):
    height_2d = np.column_stack(
        [wind_profile_context["height"][:20].magnitude, wind_profile_context["height"][:20].magnitude]
    ) * wind_profile_context["height"].units
    theta_2d = np.column_stack(
        [wind_profile_context["theta"][:20].magnitude, wind_profile_context["theta"][:20].magnitude + 1.0]
    ) * wind_profile_context["theta"].units
    u_2d = np.column_stack(
        [wind_profile_context["u"][:20].magnitude, wind_profile_context["u"][:20].magnitude]
    ) * wind_profile_context["u"].units
    v_2d = np.column_stack(
        [wind_profile_context["v"][:20].magnitude, wind_profile_context["v"][:20].magnitude]
    ) * wind_profile_context["v"].units

    actual = metcu.gradient_richardson_number(
        height_2d,
        theta_2d,
        u_2d,
        v_2d,
        vertical_dim=0,
    )
    expected = mpcalc.gradient_richardson_number(
        height_2d,
        theta_2d,
        u_2d,
        v_2d,
        vertical_dim=0,
    )
    assert_runtime_close(actual, expected, 1e-10)


@pytest.fixture(scope="module")
def kinematics_context():
    x = np.linspace(0, 300000, 4) * units.m
    y = np.linspace(0, 200000, 3) * units.m
    xx, yy = np.meshgrid(x.magnitude, y.magnitude)
    latitude = np.full_like(xx, 35.0) * units.degrees
    u = ((xx / 100000.0) + 2.0 * (yy / 100000.0)) * units("m/s")
    v = ((yy / 100000.0) - (xx / 150000.0)) * units("m/s")
    height = (5600.0 + 10.0 * (xx / 100000.0) - 20.0 * (yy / 100000.0)) * units.meter

    pressure_levels = np.array([900.0, 800.0, 700.0]) * units.hPa
    pressure_3d = pressure_levels[:, None, None] * np.ones((3, *u.shape))
    theta = np.stack([
        300.0 + 0.01 * xx + 0.02 * yy,
        305.0 + 0.01 * xx + 0.02 * yy,
        310.0 + 0.01 * xx + 0.02 * yy,
    ]) * units.kelvin
    u3 = np.stack([u.magnitude, u.magnitude + 1.0, u.magnitude + 2.0]) * units("m/s")
    v3 = np.stack([v.magnitude, v.magnitude + 0.5, v.magnitude + 1.0]) * units("m/s")

    cross = xr.Dataset(
        {
            "u": (("isobaric", "index"), np.array([[10, 11, 12, 13], [14, 15, 16, 17]]), {"units": "m/s"}),
            "v": (("isobaric", "index"), np.array([[5, 6, 7, 8], [9, 10, 11, 12]]), {"units": "m/s"}),
        },
        coords={
            "isobaric": ("isobaric", np.array([900.0, 800.0]), {"units": "hPa"}),
            "index": ("index", np.arange(4)),
            "latitude": ("index", np.array([35.0, 35.1, 35.2, 35.3]), {"units": "degrees_north"}),
            "longitude": ("index", np.array([-97.0, -96.9, -96.8, -96.7]), {"units": "degrees_east"}),
        },
    ).metpy.assign_crs(grid_mapping_name="latitude_longitude").metpy.quantify()

    u_geostrophic, v_geostrophic = mpcalc.geostrophic_wind(
        height,
        dx=100000 * units.m,
        dy=100000 * units.m,
        latitude=latitude,
    )

    return {
        "u": u,
        "v": v,
        "height": height,
        "latitude": latitude,
        "pressure_3d": pressure_3d,
        "theta": theta,
        "u3": u3,
        "v3": v3,
        "u_cs": cross["u"],
        "v_cs": cross["v"],
        "u_geostrophic": u_geostrophic,
        "v_geostrophic": v_geostrophic,
        "vel_series": np.array([1.0, 2.0, 3.0, 4.0]) * units("m/s"),
        "scalar_series": np.array([4.0, 5.0, 6.0, 7.0]) * units.kelvin,
    }


def _case_absolute_vorticity(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_ageostrophic_wind(ctx):
    return (ctx["height"], ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_potential_vorticity_baroclinic(ctx):
    return (ctx["theta"], ctx["pressure_3d"], ctx["u3"], ctx["v3"]), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"], "vertical_dim": 0}


def _case_potential_vorticity_barotropic(ctx):
    return (ctx["height"], ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_normal_component(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_tangential_component(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_unit_vectors_from_cross_section(ctx):
    return (ctx["u_cs"],), {}


def _case_vector_derivative(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m}


def _case_absolute_momentum(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_cross_section_components(ctx):
    return (ctx["u_cs"], ctx["v_cs"]), {}


def _case_curvature_vorticity(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m}


def _case_coriolis_parameter(ctx):
    return (ctx["latitude"],), {}


def _case_inertial_advective_wind(ctx):
    return (ctx["u"], ctx["v"], ctx["u_geostrophic"], ctx["v_geostrophic"]), {"dx": 100000 * units.m, "dy": 100000 * units.m, "latitude": ctx["latitude"]}


def _case_kinematic_flux(ctx):
    return (ctx["vel_series"], ctx["scalar_series"]), {}


def _case_shear_vorticity(ctx):
    return (ctx["u"], ctx["v"]), {"dx": 100000 * units.m, "dy": 100000 * units.m}


KINEMATICS_CASES = [
    pytest.param("absolute_vorticity", _case_absolute_vorticity, 1e-9, id="absolute_vorticity"),
    pytest.param("ageostrophic_wind", _case_ageostrophic_wind, 1e-6, id="ageostrophic_wind"),
    pytest.param("potential_vorticity_baroclinic", _case_potential_vorticity_baroclinic, 1e-6, id="potential_vorticity_baroclinic"),
    pytest.param("potential_vorticity_barotropic", _case_potential_vorticity_barotropic, 1e-9, id="potential_vorticity_barotropic"),
    pytest.param("normal_component", _case_normal_component, 1e-9, id="normal_component"),
    pytest.param("tangential_component", _case_tangential_component, 1e-9, id="tangential_component"),
    pytest.param("unit_vectors_from_cross_section", _case_unit_vectors_from_cross_section, 1e-9, id="unit_vectors_from_cross_section"),
    pytest.param("vector_derivative", _case_vector_derivative, 1e-9, id="vector_derivative"),
    pytest.param("absolute_momentum", _case_absolute_momentum, 1e-6, id="absolute_momentum"),
    pytest.param("cross_section_components", _case_cross_section_components, 1e-9, id="cross_section_components"),
    pytest.param("curvature_vorticity", _case_curvature_vorticity, 1e-8, id="curvature_vorticity"),
    pytest.param("coriolis_parameter", _case_coriolis_parameter, 1e-12, id="coriolis_parameter"),
    pytest.param("inertial_advective_wind", _case_inertial_advective_wind, 1e-6, id="inertial_advective_wind"),
    pytest.param("kinematic_flux", _case_kinematic_flux, 1e-12, id="kinematic_flux"),
    pytest.param("shear_vorticity", _case_shear_vorticity, 1e-8, id="shear_vorticity"),
]


@pytest.mark.parametrize(("name", "builder", "atol"), KINEMATICS_CASES)
def test_runtime_parity_kinematics_extra(kinematics_context, name, builder, atol):
    args, kwargs = builder(kinematics_context)
    actual = getattr(metcu, name)(*args, **kwargs)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="invalid value encountered in divide", category=RuntimeWarning)
        expected = getattr(mpcalc, name)(*args, **kwargs)
    assert_runtime_close(actual, expected, atol)


def test_galvez_davison_and_standard_atmosphere_runtime_parity():
    pressure = np.array([1000, 950, 850, 700, 500]) * units.hPa
    temperature = np.array([28, 24, 18, 8, -5]) * units.degC
    mixing_ratio = np.array([15, 13, 9, 4, 1]) * units("g/kg")

    assert_runtime_close(
        metcu.galvez_davison_index(pressure, temperature, mixing_ratio, pressure[0]),
        mpcalc.galvez_davison_index(pressure, temperature, mixing_ratio, pressure[0]),
        1e-9,
    )
    assert_runtime_close(
        metcu.pressure_to_height_std(np.array([1000, 900]) * units.hPa),
        mpcalc.pressure_to_height_std(np.array([1000, 900]) * units.hPa),
        1e-9,
    )
    assert_runtime_close(
        metcu.height_to_pressure_std(np.array([0, 1000]) * units.m),
        mpcalc.height_to_pressure_std(np.array([0, 1000]) * units.m),
        1e-9,
    )
    assert_runtime_close(
        metcu.altimeter_to_station_pressure(29.92 * units.inHg, 350 * units.m),
        mpcalc.altimeter_to_station_pressure(29.92 * units.inHg, 350 * units.m),
        1e-9,
    )
    assert_runtime_close(
        metcu.altimeter_to_sea_level_pressure(29.92 * units.inHg, 350 * units.m, 20 * units.degC),
        mpcalc.altimeter_to_sea_level_pressure(29.92 * units.inHg, 350 * units.m, 20 * units.degC),
        1e-9,
    )
    assert_runtime_close(
        metcu.sigma_to_pressure(np.linspace(0, 1, 5), 1000 * units.hPa, 100 * units.hPa),
        mpcalc.sigma_to_pressure(np.linspace(0, 1, 5), 1000 * units.hPa, 100 * units.hPa),
        1e-9,
    )


def test_smoothing_runtime_parity():
    field = np.arange(25.0).reshape(5, 5)
    window = np.ones((3, 3))

    assert_runtime_close(metcu.smooth_gaussian(field, 2), mpcalc.smooth_gaussian(field, 2), 1e-12)
    assert_runtime_close(metcu.smooth_rectangular(field, 3, passes=2), mpcalc.smooth_rectangular(field, 3, passes=2), 1e-12)
    assert_runtime_close(metcu.smooth_circular(field, 1, passes=2), mpcalc.smooth_circular(field, 1, passes=2), 1e-12)
    assert_runtime_close(metcu.smooth_n_point(field, 5, passes=2), mpcalc.smooth_n_point(field, 5, passes=2), 1e-12)
    assert_runtime_close(metcu.smooth_window(field, window, passes=2), mpcalc.smooth_window(field, window, passes=2), 1e-12)


def test_gradient_and_laplacian_runtime_parity():
    field = np.arange(12.0).reshape(3, 4) * units.kelvin
    coordinates = (
        np.array([0.0, 2.0, 4.0]) * units.m,
        np.array([0.0, 5.0, 10.0, 15.0]) * units.m,
    )

    actual_gradient = metcu.gradient(field, coordinates=coordinates)
    expected_gradient = mpcalc.gradient(field, coordinates=coordinates)
    for actual, expected in zip(actual_gradient, expected_gradient):
        assert_runtime_close(actual, expected, 1e-12)

    assert_runtime_close(
        metcu.laplacian(field, coordinates=coordinates),
        mpcalc.laplacian(field, coordinates=coordinates),
        1e-12,
    )


def test_lat_lon_grid_deltas_runtime_parity():
    longitude = np.linspace(-100, -98, 4) * units.deg
    latitude = np.linspace(35, 36, 3) * units.deg

    actual_dx, actual_dy = metcu.lat_lon_grid_deltas(longitude, latitude)
    expected_dx, expected_dy = mpcalc.lat_lon_grid_deltas(longitude, latitude)
    assert_runtime_close(actual_dx, expected_dx, 1e-6)
    assert_runtime_close(actual_dy, expected_dy, 1e-6)


def test_angle_helpers_runtime_parity():
    assert_runtime_close(
        metcu.angle_to_direction(np.array([0, 45, 225]) * units.deg, level=3, full=True),
        mpcalc.angle_to_direction(np.array([0, 45, 225]) * units.deg, level=3, full=True),
        1e-12,
    )
    assert_runtime_close(
        metcu.parse_angle(["north", "SW"]),
        mpcalc.parse_angle(["north", "SW"]),
        1e-12,
    )


def test_find_bounding_indices_runtime_parity():
    arr = np.array([[1000.0, 900.0], [800.0, 700.0]])
    actual_above, actual_below, actual_good = metcu.find_bounding_indices(arr, [850.0], axis=0)
    expected_above, expected_below, expected_good = mpcalc.find_bounding_indices(arr, [850.0], axis=0)

    for actual, expected in zip(actual_above, expected_above):
        np.testing.assert_array_equal(actual, expected)
    for actual, expected in zip(actual_below, expected_below):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual_good, expected_good)


def test_intersection_helpers_runtime_parity():
    x = np.array([0.0, 1.0, 2.0])
    a = np.array([0.0, 2.0, 0.0])
    b = np.array([1.0, 1.0, 1.0])

    actual_x, actual_y = metcu.find_intersections(x, a, b)
    expected_x, expected_y = mpcalc.find_intersections(x, a, b)
    assert_runtime_close(actual_x, expected_x, 1e-12)
    assert_runtime_close(actual_y, expected_y, 1e-12)
    assert_runtime_close(metcu.nearest_intersection_idx(a, b), mpcalc.nearest_intersection_idx(a, b), 1e-12)


def test_peak_and_resample_helpers_runtime_parity():
    field = np.array(
        [
            [5.0, 1.0, 5.0, 1.0],
            [1.0, 2.0, 1.0, 2.0],
            [4.0, 1.0, 6.0, 1.0],
            [1.0, 3.0, 1.0, 4.0],
        ]
    )

    actual_persistence = metcu.peak_persistence(field)
    expected_persistence = mpcalc.peak_persistence(field)
    assert actual_persistence == expected_persistence

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        actual_peaks = list(metcu.find_peaks(field, iqr_ratio=0))
        expected_peaks = list(mpcalc.find_peaks(field, iqr_ratio=0))
    assert actual_peaks == expected_peaks

    assert metcu.resample_nn_1d(np.array([0, 1, 2, 3]), np.array([0.1, 2.1])) == mpcalc.resample_nn_1d(
        np.array([0, 1, 2, 3]),
        np.array([0.1, 2.1]),
    )


def test_azimuth_range_to_lat_lon_runtime_parity():
    azimuths = np.array([0.0, 90.0]) * units.deg
    ranges = np.array([0.0, 1000.0]) * units.m

    actual_lon, actual_lat = metcu.azimuth_range_to_lat_lon(azimuths, ranges, -97.5, 35.4)
    expected_lon, expected_lat = mpcalc.azimuth_range_to_lat_lon(azimuths, ranges, -97.5, 35.4)
    assert_runtime_close(actual_lon, expected_lon, 1e-9)
    assert_runtime_close(actual_lat, expected_lat, 1e-9)


def test_zoom_xarray_runtime_parity():
    data = xr.DataArray(
        np.arange(9.0).reshape(3, 3),
        dims=("y", "x"),
        coords={"y": [0.0, 1.0, 2.0], "x": [10.0, 20.0, 30.0]},
    )
    assert_runtime_close(metcu.zoom_xarray(data, 2), mpcalc.zoom_xarray(data, 2), 1e-12)


def test_mixed_layer_cape_cin_runtime_parity(sounding_profile):
    actual = metcu.mixed_layer_cape_cin(
        sounding_profile["pressure"],
        sounding_profile["temperature"],
        sounding_profile["dewpoint"],
        depth=50 * units.hPa,
    )
    expected = mpcalc.mixed_layer_cape_cin(
        sounding_profile["pressure"],
        sounding_profile["temperature"],
        sounding_profile["dewpoint"],
        depth=50 * units.hPa,
    )
    assert_runtime_close(actual, expected, 10.0)


def test_get_perturbation_runtime_parity():
    values = np.array([290.0, 292.0, 288.0, 294.0]) * units.kelvin
    assert_runtime_close(metcu.get_perturbation(values), mpcalc.get_perturbation(values), 1e-12)
