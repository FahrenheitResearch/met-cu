import inspect
import numpy as np
import cupy as cp

import metcu
import metcu.calc as mc_calc
from metcu.kernels import thermo as kthermo
from metcu.kernels import wind as kwind
import metrust.calc as mr
from metrust.units import units


def _to_numpy(value):
    if isinstance(value, tuple):
        return tuple(_to_numpy(v) for v in value)
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    if hasattr(value, "magnitude"):
        return np.asarray(value.magnitude)
    return np.asarray(value)


def _assert_close(actual, expected, rtol=1e-6, atol=1e-6):
    a = _to_numpy(actual)
    e = _to_numpy(expected)
    if isinstance(a, tuple):
        assert isinstance(e, tuple)
        assert len(a) == len(e)
        for av, ev in zip(a, e):
            _assert_close(av, ev, rtol=rtol, atol=atol)
        return
    np.testing.assert_allclose(np.asarray(a, dtype=np.float64),
                               np.asarray(e, dtype=np.float64),
                               rtol=rtol, atol=atol)


def _public_functions(mod):
    funcs = {}
    for name in dir(mod):
        if name.startswith("_"):
            continue
        obj = getattr(mod, name)
        if inspect.isfunction(obj) or inspect.isbuiltin(obj):
            funcs[name] = str(inspect.signature(obj))
    return funcs


def _sample_grid_fields():
    nz, ny, nx = 5, 2, 3
    pressure = np.broadcast_to(
        np.array([100000.0, 90000.0, 80000.0, 70000.0, 60000.0])[:, None, None],
        (nz, ny, nx),
    ).copy()
    temperature = np.broadcast_to(
        np.array([28.0, 20.0, 12.0, 4.0, -8.0])[:, None, None],
        (nz, ny, nx),
    ).copy()
    qvapor = np.broadcast_to(
        np.array([0.014, 0.011, 0.008, 0.004, 0.0015])[:, None, None],
        (nz, ny, nx),
    ).copy()
    height = np.broadcast_to(
        np.array([0.0, 900.0, 1900.0, 3200.0, 5000.0])[:, None, None],
        (nz, ny, nx),
    ).copy()
    u = np.broadcast_to(
        np.array([5.0, 10.0, 15.0, 20.0, 25.0])[:, None, None],
        (nz, ny, nx),
    ).copy()
    v = np.broadcast_to(
        np.array([1.0, 3.0, 6.0, 9.0, 12.0])[:, None, None],
        (nz, ny, nx),
    ).copy()
    qrain = np.broadcast_to(
        np.array([0.0, 0.0002, 0.0004, 0.0007, 0.0001])[:, None, None],
        (nz, ny, nx),
    ).copy()
    qsnow = np.broadcast_to(
        np.array([0.0, 0.0, 0.0001, 0.0003, 0.0005])[:, None, None],
        (nz, ny, nx),
    ).copy()
    qgraup = np.broadcast_to(
        np.array([0.0, 0.0, 0.00005, 0.0002, 0.0004])[:, None, None],
        (nz, ny, nx),
    ).copy()
    psfc = np.full((ny, nx), 100000.0)
    t2 = np.full((ny, nx), 301.15)
    q2 = np.full((ny, nx), 0.014)
    return {
        "pressure": pressure,
        "temperature": temperature,
        "qvapor": qvapor,
        "height": height,
        "u": u,
        "v": v,
        "qrain": qrain,
        "qsnow": qsnow,
        "qgraup": qgraup,
        "psfc": psfc,
        "t2": t2,
        "q2": q2,
    }


def test_vapor_pressure_accepts_convertible_mixing_ratio_units():
    pressure = np.array([90000.0, 100000.0]) * units.Pa
    mixing_ratio = np.array([10.0, 15.0]) * units("g/kg")
    _assert_close(
        metcu.vapor_pressure(pressure, mixing_ratio),
        mr.vapor_pressure(pressure, mixing_ratio),
    )


def test_public_function_signatures_match_metrust():
    metcu_funcs = _public_functions(mc_calc)
    metrust_funcs = _public_functions(mr)
    missing = sorted(set(metrust_funcs) - set(metcu_funcs))
    mismatches = sorted(
        (name, metcu_funcs[name], metrust_funcs[name])
        for name in set(metcu_funcs) & set(metrust_funcs)
        if metcu_funcs[name] != metrust_funcs[name]
    )
    assert missing == []
    assert mismatches == []


def test_aliases_match_metrust():
    _assert_close(
        metcu.significant_tornado(np.array([2000.0]), np.array([1000.0]), np.array([200.0]), np.array([25.0])),
        mr.significant_tornado(2000.0 * units("J/kg"), 1000.0 * units.m, 200.0 * units("m**2/s**2"), 25.0 * units("m/s")),
        atol=1e-9,
    )
    _assert_close(
        metcu.supercell_composite(np.array([3000.0]), np.array([300.0]), np.array([30.0])),
        mr.supercell_composite(3000.0 * units("J/kg"), 300.0 * units("m**2/s**2"), 30.0 * units("m/s")),
        atol=1e-9,
    )
    _assert_close(
        metcu.total_totals_index(15.0, 12.0, -15.0),
        mr.total_totals_index(15.0, 12.0, -15.0),
        atol=1e-9,
    )


def test_parcel_profile_with_lcl_matches_metrust_shape_and_values():
    pressure = np.array([1000.0, 900.0, 800.0, 700.0])
    actual_p, actual_t = metcu.parcel_profile_with_lcl(pressure, 25.0, 20.0)
    expected_p, expected_t = mr.parcel_profile_with_lcl(
        pressure * units.hPa,
        25.0 * units.degC,
        20.0 * units.degC,
    )
    _assert_close(actual_p, expected_p, atol=1e-6)
    _assert_close(actual_t, expected_t, atol=1e-9)


def test_get_layer_interpolates_pressure_bounds():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0])
    temperature = np.array([25.0, 22.0, 18.0, 8.0])
    dewpoint = np.array([20.0, 18.0, 14.0, -2.0])
    actual = metcu.get_layer(pressure, temperature, dewpoint, bottom=950.0, depth=150.0)
    expected = mr.get_layer(
        pressure * units.hPa,
        temperature * units.degC,
        dewpoint * units.degC,
        bottom=950.0 * units.hPa,
        depth=150.0 * units.hPa,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_get_layer_heights_matches_metrust():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0])
    heights = np.array([100.0, 800.0, 1500.0, 3000.0])
    actual = metcu.get_layer_heights(pressure, heights, 950.0, 800.0)
    expected = mr.get_layer_heights(
        pressure * units.hPa,
        heights * units.m,
        950.0 * units.hPa,
        800.0 * units.hPa,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_mixed_layer_wrapper_matches_metrust():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0])
    temperature = np.array([25.0, 22.0, 18.0, 8.0])
    dewpoint = np.array([20.0, 18.0, 14.0, -2.0])
    actual = metcu.mixed_layer(pressure, temperature, dewpoint, depth=100.0)
    expected = mr.mixed_layer(
        pressure * units.hPa,
        temperature * units.degC,
        dewpoint * units.degC,
        depth=100.0 * units.hPa,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_mixed_layer_kernel_matches_metrust_value_average():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0])
    temperature = np.array([25.0, 22.0, 18.0, 8.0])
    dewpoint = np.array([20.0, 18.0, 14.0, -2.0])
    t_ml, td_ml = kthermo.mixed_layer(pressure, temperature, dewpoint, depth=100.0)
    expected_t = mr.mixed_layer(pressure * units.hPa, temperature * units.degC, depth=100.0 * units.hPa)
    expected_td = mr.mixed_layer(pressure * units.hPa, dewpoint * units.degC, depth=100.0 * units.hPa)
    _assert_close(t_ml, expected_t, atol=1e-6)
    _assert_close(td_ml, expected_td, atol=1e-6)


def test_get_mixed_layer_parcel_matches_metrust():
    pressure = np.array([1000.0, 925.0, 850.0, 700.0])
    temperature = np.array([25.0, 22.0, 18.0, 8.0])
    dewpoint = np.array([20.0, 18.0, 14.0, -2.0])
    actual = metcu.get_mixed_layer_parcel(pressure, temperature, dewpoint, depth=100.0)
    expected = mr.get_mixed_layer_parcel(
        pressure * units.hPa,
        temperature * units.degC,
        dewpoint * units.degC,
        100.0 * units.hPa,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_surface_based_cape_cin_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    actual = metcu.surface_based_cape_cin(pressure, temperature, dewpoint)
    expected = mr.surface_based_cape_cin(
        pressure * units.hPa,
        temperature * units.degC,
        dewpoint * units.degC,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_mixed_layer_cape_cin_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    actual = metcu.mixed_layer_cape_cin(pressure, temperature, dewpoint, depth=100.0)
    expected = mr.mixed_layer_cape_cin(
        pressure * units.hPa,
        temperature * units.degC,
        dewpoint * units.degC,
        depth=100.0 * units.hPa,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_bunkers_storm_motion_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    u = np.linspace(5.0, 40.0, 40)
    v = np.linspace(0.0, 20.0, 40)
    heights = np.linspace(0.0, 16000.0, 40)
    actual = metcu.bunkers_storm_motion(pressure, u, v, heights)
    expected = mr.bunkers_storm_motion(
        pressure * units.hPa,
        u * units("m/s"),
        v * units("m/s"),
        heights * units.m,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_mean_wind_matches_metrust():
    u = np.linspace(5.0, 40.0, 40)
    v = np.linspace(0.0, 20.0, 40)
    heights = np.linspace(0.0, 16000.0, 40)
    actual = metcu.mean_wind(u, v, heights, 0.0, 6000.0)
    expected = mr.mean_wind(
        u * units("m/s"),
        v * units("m/s"),
        heights * units.m,
        0.0 * units.m,
        6000.0 * units.m,
    )
    _assert_close(actual, expected, atol=1e-6)


def test_corfidi_storm_motion_matches_metrust():
    u = np.linspace(5.0, 40.0, 40)
    v = np.linspace(0.0, 20.0, 40)
    heights = np.linspace(0.0, 16000.0, 40)
    actual = metcu.corfidi_storm_motion(u, v, heights, 15.0, 8.0)
    expected = mr.corfidi_storm_motion(
        u * units("m/s"),
        v * units("m/s"),
        heights * units.m,
        15.0 * units("m/s"),
        8.0 * units("m/s"),
    )
    _assert_close(actual, expected, atol=1e-6)


def test_storm_relative_helicity_matches_metrust():
    u = np.linspace(5.0, 40.0, 40)
    v = np.linspace(0.0, 20.0, 40)
    heights = np.linspace(0.0, 16000.0, 40)
    _assert_close(
        metcu.storm_relative_helicity(heights, u, v, depth=3000.0, storm_u=5.0, storm_v=5.0),
        mr.storm_relative_helicity(
            heights * units.m,
            u * units("m/s"),
            v * units("m/s"),
            depth=3000.0 * units.m,
            storm_u=5.0 * units("m/s"),
            storm_v=5.0 * units("m/s"),
        ),
        atol=1e-9,
    )


def test_kernel_storm_relative_helicity_matches_metrust():
    u = np.linspace(5.0, 40.0, 40)
    v = np.linspace(0.0, 20.0, 40)
    heights = np.linspace(0.0, 16000.0, 40)
    _assert_close(
        kwind.storm_relative_helicity(u, v, heights, 3000.0, 5.0, 5.0),
        mr.storm_relative_helicity(
            heights * units.m,
            u * units("m/s"),
            v * units("m/s"),
            depth=3000.0 * units.m,
            storm_u=5.0 * units("m/s"),
            storm_v=5.0 * units("m/s"),
        ),
        atol=1e-9,
    )


def test_absolute_momentum_matches_metrust():
    u = np.array([5.0, 10.0])
    lats = np.array([35.0, 36.0])
    y_dist = np.array([0.0, 1000.0])
    _assert_close(
        metcu.absolute_momentum(u, lats, y_dist),
        mr.absolute_momentum(u * units("m/s"), lats * units.degree, y_dist * units.m),
        atol=1e-6,
    )


def test_cross_section_components_matches_metrust():
    u = np.array([5.0, 10.0])
    v = np.array([1.0, 2.0])
    _assert_close(
        metcu.cross_section_components(u, v, 35.0, -100.0, 36.0, -99.0),
        mr.cross_section_components(
            u * units("m/s"),
            v * units("m/s"),
            35.0 * units.degree,
            -100.0 * units.degree,
            36.0 * units.degree,
            -99.0 * units.degree,
        ),
        atol=1e-6,
    )


def test_lfc_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    _assert_close(
        metcu.lfc(pressure, temperature, dewpoint),
        mr.lfc(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-6,
    )


def test_el_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    _assert_close(
        metcu.el(pressure, temperature, dewpoint),
        mr.el(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-9,
    )


def test_kernel_lfc_and_el_match_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    _assert_close(
        kthermo.lfc(pressure, temperature, dewpoint),
        mr.lfc(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-6,
    )
    _assert_close(
        kthermo.el(pressure, temperature, dewpoint),
        mr.el(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-9,
    )


def test_cape_cin_wrapper_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    _assert_close(
        metcu.cape_cin(pressure, temperature, dewpoint),
        mr.cape_cin(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-6,
    )


def test_standard_atmosphere_and_pressure_conversions_match_metrust():
    pressure = np.array([1000.0, 900.0, 800.0])
    heights = np.array([0.0, 1000.0, 2000.0])
    altimeter = pressure + 13.0
    temperature = np.array([25.0, 15.0, 5.0])
    sigma = np.array([0.1, 0.5, 0.9])

    expected_h = np.array([mr.pressure_to_height_std(float(p) * units.hPa).magnitude for p in pressure])
    expected_p = np.array([mr.height_to_pressure_std(float(z) * units.m).magnitude for z in heights])
    expected_station = np.array([
        mr.altimeter_to_station_pressure(float(a) * units.hPa, float(z) * units.m).magnitude
        for a, z in zip(altimeter, heights)
    ])
    expected_alt = np.array([
        mr.station_to_altimeter_pressure(float(p) * units.hPa, float(z) * units.m).magnitude
        for p, z in zip(pressure, heights)
    ])
    expected_slp = np.array([
        mr.altimeter_to_sea_level_pressure(float(a) * units.hPa, float(z) * units.m, float(t) * units.degC).magnitude
        for a, z, t in zip(altimeter, heights, temperature)
    ])
    expected_sigma = np.array([
        mr.sigma_to_pressure(float(s), float(p) * units.hPa, 50.0 * units.hPa).magnitude
        for s, p in zip(sigma, pressure)
    ])

    _assert_close(metcu.pressure_to_height_std(pressure), expected_h, atol=1e-9)
    _assert_close(metcu.height_to_pressure_std(heights), expected_p, atol=1e-9)
    _assert_close(metcu.altimeter_to_station_pressure(altimeter, heights), expected_station, atol=1e-9)
    _assert_close(metcu.station_to_altimeter_pressure(pressure, heights), expected_alt, atol=1e-9)
    _assert_close(metcu.altimeter_to_sea_level_pressure(altimeter, heights, temperature), expected_slp, atol=1e-9)
    _assert_close(metcu.sigma_to_pressure(sigma, pressure, np.full(3, 50.0)), expected_sigma, atol=1e-9)


def test_showalter_and_convective_inhibition_depth_match_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    dewpoint = np.linspace(20.0, -65.0, 40)
    _assert_close(
        metcu.showalter_index(pressure, temperature, dewpoint),
        mr.showalter_index(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-6,
    )
    _assert_close(
        metcu.convective_inhibition_depth(pressure, temperature, dewpoint),
        mr.convective_inhibition_depth(pressure * units.hPa, temperature * units.degC, dewpoint * units.degC),
        atol=1e-6,
    )


def test_fire_and_psychrometric_indices_match_metrust():
    _assert_close(
        metcu.compute_dcp(np.array([800.0]), np.array([2000.0]), np.array([25.0]), np.array([12.0])),
        np.array([mr.compute_dcp(
            np.array([[800.0]]) * units("J/kg"),
            np.array([[2000.0]]) * units("J/kg"),
            np.array([[25.0]]) * units("m/s"),
            np.array([[12.0]]) * units("g/kg"),
        ).magnitude.item()]),
        atol=1e-9,
    )
    _assert_close(
        metcu.fosberg_fire_weather_index(np.array([80.0]), np.array([30.0]), np.array([15.0])),
        mr.fosberg_fire_weather_index(80.0 * units.degF, 30.0 * units.percent, 15.0 * units("mph")),
        atol=1e-9,
    )
    _assert_close(
        metcu.hot_dry_windy(np.array([35.0]), np.array([15.0]), np.array([10.0])),
        mr.hot_dry_windy(35.0 * units.degC, 15.0 * units.percent, 10.0 * units("m/s")),
        atol=1e-9,
    )
    _assert_close(
        metcu.galvez_davison_index(18.0, 15.0, 5.0, -15.0, 16.0, 12.0, -3.0, 28.0),
        mr.galvez_davison_index(
            18.0 * units.degC,
            15.0 * units.degC,
            5.0 * units.degC,
            -15.0 * units.degC,
            16.0 * units.degC,
            12.0 * units.degC,
            -3.0 * units.degC,
            28.0 * units.degC,
        ),
        atol=1e-9,
    )
    _assert_close(
        metcu.relative_humidity_wet_psychrometric(np.array([25.0]), np.array([20.0]), np.array([1000.0])),
        mr.relative_humidity_wet_psychrometric(
            np.array([25.0]) * units.degC,
            np.array([20.0]) * units.degC,
            np.array([1000.0]) * units.hPa,
        ),
        atol=1e-9,
    )


def test_dendritic_growth_zone_matches_metrust():
    pressure = np.linspace(1000.0, 100.0, 40)
    temperature = np.linspace(25.0, -60.0, 40)
    _assert_close(
        metcu.dendritic_growth_zone(temperature, pressure),
        mr.dendritic_growth_zone(temperature * units.degC, pressure * units.hPa),
        atol=1e-6,
    )


def test_grid_deformation_and_critical_angle_match_metrust():
    y, x = np.mgrid[0:5, 0:6]
    u = 2.0 * x + 0.5 * y
    v = -1.0 * x + 3.0 * y
    dx = 3000.0
    dy = 3000.0
    _assert_close(
        metcu.shearing_deformation(u, v, dx, dy),
        mr.shearing_deformation(u * units("m/s"), v * units("m/s"), dx=dx * units.m, dy=dy * units.m),
        atol=1e-9,
    )
    _assert_close(
        metcu.stretching_deformation(u, v, dx, dy),
        mr.stretching_deformation(u * units("m/s"), v * units("m/s"), dx=dx * units.m, dy=dy * units.m),
        atol=1e-9,
    )
    _assert_close(
        metcu.total_deformation(u, v, dx, dy),
        mr.total_deformation(u * units("m/s"), v * units("m/s"), dx=dx * units.m, dy=dy * units.m),
        atol=1e-9,
    )
    _assert_close(
        metcu.compute_grid_critical_angle(
            np.full((3, 4), 5.0),
            np.full((3, 4), 5.0),
            np.full((3, 4), 10.0),
            np.full((3, 4), 15.0),
        ),
        mr.compute_grid_critical_angle(
            np.full((3, 4), 5.0) * units("m/s"),
            np.full((3, 4), 5.0) * units("m/s"),
            np.full((3, 4), 10.0) * units("m/s"),
            np.full((3, 4), 15.0) * units("m/s"),
        ),
        atol=1e-9,
    )


def test_grid_wrapper_parity_matches_metrust():
    data = _sample_grid_fields()
    _assert_close(
        metcu.compute_cape_cin(
            data["pressure"], data["temperature"], data["qvapor"],
            data["height"], data["psfc"], data["t2"], data["q2"],
        ),
        mr.compute_cape_cin(
            data["pressure"], data["temperature"], data["qvapor"],
            data["height"], data["psfc"], data["t2"], data["q2"],
        ),
    )
    _assert_close(
        metcu.compute_srh(data["u"], data["v"], data["height"], top_m=1000.0),
        mr.compute_srh(data["u"], data["v"], data["height"], top_m=1000.0),
    )
    _assert_close(
        metcu.compute_shear(data["u"], data["v"], data["height"], bottom_m=0.0, top_m=6000.0),
        mr.compute_shear(data["u"], data["v"], data["height"], bottom_m=0.0, top_m=6000.0),
        atol=1e-6,
    )
    _assert_close(
        metcu.compute_pw(data["qvapor"], data["pressure"]),
        mr.compute_pw(data["qvapor"], data["pressure"]),
        atol=1e-6,
    )
    _assert_close(
        metcu.composite_reflectivity_from_hydrometeors(
            data["pressure"],
            data["temperature"],
            data["qrain"],
            data["qsnow"],
            data["qgraup"],
        ),
        mr.composite_reflectivity_from_hydrometeors(
            data["pressure"],
            data["temperature"],
            data["qrain"],
            data["qsnow"],
            data["qgraup"],
        ),
        atol=1e-3,
    )


def test_compute_cape_cin_matches_metrust_on_stable_profile():
    nz, ny, nx = 30, 2, 2
    pressure = np.broadcast_to(
        np.linspace(100000.0, 10000.0, nz)[:, None, None],
        (nz, ny, nx),
    ).copy()
    height = np.broadcast_to(
        np.linspace(0.0, 15000.0, nz)[:, None, None],
        (nz, ny, nx),
    ).copy()
    temperature = np.broadcast_to(
        np.linspace(303.0, 260.0, nz)[:, None, None],
        (nz, ny, nx),
    ).copy()
    qvapor = np.broadcast_to(
        np.linspace(0.012, 0.0005, nz)[:, None, None],
        (nz, ny, nx),
    ).copy()
    psfc = np.full((ny, nx), 100000.0)
    t2 = np.full((ny, nx), 303.0)
    q2 = np.full((ny, nx), 0.012)
    _assert_close(
        metcu.compute_cape_cin(pressure, temperature, qvapor, height, psfc, t2, q2),
        mr.compute_cape_cin(pressure, temperature, qvapor, height, psfc, t2, q2),
    )


def test_compute_cape_cin_matches_metrust_on_deep_profile():
    nz = 260
    pressure = np.broadcast_to(
        np.linspace(100000.0, 10000.0, nz)[:, None, None],
        (nz, 1, 1),
    ).copy()
    height = np.broadcast_to(
        np.linspace(0.0, 17000.0, nz)[:, None, None],
        (nz, 1, 1),
    ).copy()
    temperature = np.broadcast_to(
        np.linspace(300.0, 210.0, nz)[:, None, None],
        (nz, 1, 1),
    ).copy()
    qvapor = np.broadcast_to(
        np.linspace(0.014, 0.0001, nz)[:, None, None],
        (nz, 1, 1),
    ).copy()
    psfc = np.array([[100000.0]])
    t2 = np.array([[300.0]])
    q2 = np.array([[0.014]])
    _assert_close(
        metcu.compute_cape_cin(pressure, temperature, qvapor, height, psfc, t2, q2),
        mr.compute_cape_cin(pressure, temperature, qvapor, height, psfc, t2, q2),
    )


def test_compute_cape_cin_respects_parcel_type_and_top_m():
    nz, ny, nx = 12, 1, 1
    pressure = np.broadcast_to(
        np.linspace(100000.0, 20000.0, nz)[:, None, None],
        (nz, ny, nx),
    ).copy()
    height = np.broadcast_to(
        np.array([0, 150, 350, 700, 1100, 1700, 2400, 3200, 4300, 5800, 7600, 9800])[:, None, None],
        (nz, ny, nx),
    ).copy()
    temperature = np.broadcast_to(
        np.array([29, 27, 24, 20, 15, 10, 5, 0, -8, -18, -30, -44])[:, None, None],
        (nz, ny, nx),
    ).copy()
    qvapor = np.broadcast_to(
        np.array([0.016, 0.015, 0.013, 0.011, 0.009, 0.0065, 0.0045, 0.003, 0.0018, 0.001, 0.0004, 0.0001])[:, None, None],
        (nz, ny, nx),
    ).copy()
    psfc = np.array([[100000.0]])
    t2 = np.array([[302.15]])
    q2 = np.array([[0.016]])

    for kwargs in (
        {"parcel_type": "surface"},
        {"parcel_type": "mixed-layer"},
        {"parcel_type": "most-unstable"},
        {"parcel_type": "surface", "top_m": 3000.0},
    ):
        _assert_close(
            metcu.compute_cape_cin(pressure, temperature, qvapor, height, psfc, t2, q2, **kwargs),
            mr.compute_cape_cin(pressure, temperature, qvapor, height, psfc, t2, q2, **kwargs),
        )


def test_compute_srh_matches_metrust_when_lowest_level_is_above_ground():
    u = np.array([[[5.0]], [[10.0]], [[15.0]], [[20.0]]])
    v = np.array([[[0.0]], [[3.0]], [[7.0]], [[12.0]]])
    height = np.array([[[50.0]], [[300.0]], [[800.0]], [[1400.0]]])

    for top_m in (500.0, 1000.0):
        _assert_close(
            metcu.compute_srh(u, v, height, top_m=top_m),
            mr.compute_srh(u, v, height, top_m=top_m),
        )
