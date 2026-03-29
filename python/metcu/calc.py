"""metcu.calc -- GPU-accelerated meteorological calculations.

Every public function mirrors the metrust.calc API exactly: same names,
same parameter signatures, same return semantics.  Inputs are automatically
moved to the GPU (cupy arrays), the CUDA kernel is executed, and results
are returned as cupy arrays on the device.

Callers can use ``.get()`` on any result to obtain a numpy array on the CPU.

Unit conventions (same as metrust)
----------------------------------
- Pressure:  hPa (millibars)
- Temperature:  Celsius  (potential temperature in Kelvin)
- Mixing ratio:  g/kg
- Relative humidity:  percent [0-100]
- Wind speed:  m/s
- Height:  meters
- Grid spacings (dx, dy):  meters
- Angles:  degrees
"""

import cupy as cp
import numpy as np

from metcu.utils import to_gpu, to_cpu, strip_units

try:
    from metrust.units import units
except Exception:
    units = None

try:
    import xarray as xr
except Exception:
    xr = None

# --- thermo kernels ---
from metcu.kernels.thermo import (
    potential_temperature as _k_potential_temperature,
    equivalent_potential_temperature as _k_equivalent_potential_temperature,
    saturation_vapor_pressure as _k_saturation_vapor_pressure,
    saturation_mixing_ratio as _k_saturation_mixing_ratio,
    wet_bulb_temperature as _k_wet_bulb_temperature,
    dewpoint_from_relative_humidity as _k_dewpoint_from_relative_humidity,
    relative_humidity_from_dewpoint as _k_relative_humidity_from_dewpoint,
    virtual_temperature as _k_virtual_temperature,
    virtual_temperature_from_dewpoint as _k_virtual_temperature_from_dewpoint,
    mixing_ratio as _k_mixing_ratio,
    density as _k_density,
    dewpoint as _k_dewpoint,
    dewpoint_from_specific_humidity as _k_dewpoint_from_specific_humidity,
    dry_lapse as _k_dry_lapse,
    dry_static_energy as _k_dry_static_energy,
    exner_function as _k_exner_function,
    moist_lapse as _k_moist_lapse,
    moist_static_energy as _k_moist_static_energy,
    parcel_profile as _k_parcel_profile,
    temperature_from_potential_temperature as _k_temperature_from_potential_temperature,
    vertical_velocity as _k_vertical_velocity,
    vertical_velocity_pressure as _k_vertical_velocity_pressure,
    virtual_potential_temperature as _k_virtual_potential_temperature,
    wet_bulb_potential_temperature as _k_wet_bulb_potential_temperature,
    saturation_equivalent_potential_temperature as _k_saturation_equivalent_potential_temperature,
    vapor_pressure as _k_vapor_pressure,
    vapor_pressure_from_mixing_ratio as _k_vapor_pressure_from_mixing_ratio,
    specific_humidity_from_mixing_ratio as _k_specific_humidity_from_mixing_ratio,
    mixing_ratio_from_relative_humidity as _k_mixing_ratio_from_relative_humidity,
    mixing_ratio_from_specific_humidity as _k_mixing_ratio_from_specific_humidity,
    relative_humidity_from_mixing_ratio as _k_relative_humidity_from_mixing_ratio,
    relative_humidity_from_specific_humidity as _k_relative_humidity_from_specific_humidity,
    specific_humidity_from_dewpoint as _k_specific_humidity_from_dewpoint,
    frost_point as _k_frost_point,
    heat_index as _k_heat_index,
    windchill as _k_windchill,
    apparent_temperature as _k_apparent_temperature,
    moist_air_gas_constant as _k_moist_air_gas_constant,
    moist_air_specific_heat_pressure as _k_moist_air_specific_heat_pressure,
    moist_air_poisson_exponent as _k_moist_air_poisson_exponent,
    water_latent_heat_vaporization as _k_water_latent_heat_vaporization,
    water_latent_heat_melting as _k_water_latent_heat_melting,
    water_latent_heat_sublimation as _k_water_latent_heat_sublimation,
    psychrometric_vapor_pressure as _k_psychrometric_vapor_pressure,
    add_height_to_pressure as _k_add_height_to_pressure,
    add_pressure_to_height as _k_add_pressure_to_height,
    thickness_hydrostatic as _k_thickness_hydrostatic,
    scale_height as _k_scale_height,
    geopotential_to_height as _k_geopotential_to_height,
    height_to_geopotential as _k_height_to_geopotential,
    pressure_to_height_std as _k_pressure_to_height_std,
    height_to_pressure_std as _k_height_to_pressure_std,
    altimeter_to_station_pressure as _k_altimeter_to_station_pressure,
    station_to_altimeter_pressure as _k_station_to_altimeter_pressure,
    altimeter_to_sea_level_pressure as _k_altimeter_to_sea_level_pressure,
    sigma_to_pressure as _k_sigma_to_pressure,
    montgomery_streamfunction as _k_montgomery_streamfunction,
    cape_cin as _k_cape_cin,
    lcl as _k_lcl,
    lfc as _k_lfc,
    el as _k_el,
    lifted_index as _k_lifted_index,
    ccl as _k_ccl,
    precipitable_water as _k_precipitable_water,
    mixed_layer as _k_mixed_layer,
    downdraft_cape as _k_downdraft_cape,
    brunt_vaisala_frequency as _k_brunt_vaisala_frequency,
    brunt_vaisala_frequency_squared as _k_brunt_vaisala_frequency_squared,
    brunt_vaisala_period as _k_brunt_vaisala_period,
    static_stability as _k_static_stability,
    grid_precipitable_water as _k_grid_precipitable_water,
    grid_cape_cin as _k_grid_cape_cin,
)

# --- wind kernels ---
from metcu.kernels.wind import (
    wind_speed as _k_wind_speed,
    wind_direction as _k_wind_direction,
    wind_components as _k_wind_components,
    coriolis_parameter as _k_coriolis_parameter,
    normal_component as _k_normal_component,
    tangential_component as _k_tangential_component,
    friction_velocity as _k_friction_velocity,
    tke as _k_tke,
    bulk_shear as _k_bulk_shear,
    mean_wind as _k_mean_wind,
    storm_relative_helicity as _k_storm_relative_helicity,
    bunkers_storm_motion as _k_bunkers_storm_motion,
    corfidi_storm_motion as _k_corfidi_storm_motion,
    critical_angle as _k_critical_angle,
    get_layer as _k_get_layer,
    gradient_richardson_number as _k_gradient_richardson_number,
    significant_tornado_parameter as _k_significant_tornado_parameter,
    supercell_composite_parameter as _k_supercell_composite_parameter,
    compute_ship as _k_compute_ship,
    compute_ehi as _k_compute_ehi,
    compute_dcp as _k_compute_dcp,
    compute_lapse_rate as _k_compute_lapse_rate,
    bulk_richardson_number as _k_bulk_richardson_number,
    k_index as _k_k_index,
    total_totals as _k_total_totals,
    cross_totals as _k_cross_totals,
    vertical_totals as _k_vertical_totals,
    sweat_index as _k_sweat_index,
    showalter_index as _k_showalter_index,
    boyden_index as _k_boyden_index,
    galvez_davison_index as _k_galvez_davison_index,
    fosberg_fire_weather_index as _k_fosberg_fire_weather_index,
    haines_index as _k_haines_index,
    hot_dry_windy as _k_hot_dry_windy,
    significant_tornado as _k_significant_tornado,
    freezing_rain_composite as _k_freezing_rain_composite,
    warm_nose_check as _k_warm_nose_check,
    dendritic_growth_zone as _k_dendritic_growth_zone,
    convective_inhibition_depth as _k_convective_inhibition_depth,
    grid_storm_relative_helicity as _k_grid_srh,
)

# --- grid kernels ---
from metcu.kernels.grid import (
    divergence as _k_divergence,
    vorticity as _k_vorticity,
    absolute_vorticity as _k_absolute_vorticity,
    advection as _k_advection,
    frontogenesis as _k_frontogenesis,
    geostrophic_wind as _k_geostrophic_wind,
    ageostrophic_wind as _k_ageostrophic_wind,
    potential_vorticity_baroclinic as _k_potential_vorticity_baroclinic,
    potential_vorticity_barotropic as _k_potential_vorticity_barotropic,
    shearing_deformation as _k_shearing_deformation,
    stretching_deformation as _k_stretching_deformation,
    total_deformation as _k_total_deformation,
    curvature_vorticity as _k_curvature_vorticity,
    shear_vorticity as _k_shear_vorticity,
    q_vector as _k_q_vector,
    inertial_advective_wind as _k_inertial_advective_wind,
    lat_lon_grid_deltas as _k_lat_lon_grid_deltas,
    smooth_gaussian as _k_smooth_gaussian,
    smooth_rectangular as _k_smooth_rectangular,
    smooth_circular as _k_smooth_circular,
    smooth_n_point as _k_smooth_n_point,
    smooth_window as _k_smooth_window,
    first_derivative_x as _k_first_derivative_x,
    first_derivative_y as _k_first_derivative_y,
    second_derivative_x as _k_second_derivative_x,
    second_derivative_y as _k_second_derivative_y,
    laplacian as _k_laplacian,
    gradient as _k_gradient,
    composite_reflectivity as _k_composite_reflectivity,
    get_layer_heights as _k_get_layer_heights,
    mean_pressure_weighted as _k_mean_pressure_weighted,
    isentropic_interpolation as _k_isentropic_interpolation,
    composite_reflectivity_from_hydrometeors as _k_composite_reflectivity_from_hydrometeors,
)


# ---------------------------------------------------------------------------
# Inline stubs for kernel functions not yet in the kernel modules
# ---------------------------------------------------------------------------

def _STUB_thickness_from_rh(p, t, rh):
    """Hypsometric thickness from P, T, RH -- inline fallback."""
    import cupy as cp
    w = _k_mixing_ratio_from_relative_humidity(p, t, rh)
    t_k = t + 273.15  # Celsius -> Kelvin
    tv = t_k * (1.0 + w / 0.6219569100577033) / (1.0 + w)
    if float(p[0]) < float(p[-1]):
        p = p[::-1]
        tv = tv[::-1]
    return -(287.05 / 9.80665) * cp.trapz(tv, cp.log(p))


def _as_magnitude_in_units(arr, unit=None):
    """Convert a pint quantity to a target unit and return raw magnitudes."""
    if hasattr(arr, 'to') and unit is not None:
        try:
            return arr.to(unit).magnitude
        except Exception:
            pass
    if hasattr(arr, 'magnitude'):
        return arr.magnitude
    return arr


def _log_interp_pressure_value(target_p, pressure, values):
    """Interpolate a profile value at a pressure using log-pressure spacing."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if p.size != v.size:
        raise ValueError("pressure and values must have the same length")
    if p.size == 0:
        return np.nan
    if p[0] < p[-1]:
        p = p[::-1]
        v = v[::-1]
    if target_p >= p[0]:
        return float(v[0])
    if target_p <= p[-1]:
        return float(v[-1])
    return float(np.interp(np.log(target_p), np.log(p[::-1]), v[::-1]))


def _sharppy_mixratio_gkg(pressure_hpa, dewpoint_c):
    """SHARPpy-style mixing ratio from pressure and dewpoint in g/kg."""
    x = 0.02 * (dewpoint_c - 12.5 + (7500.0 / pressure_hpa))
    enhancement = 1.0 + (0.0000045 * pressure_hpa) + (0.0014 * x * x)
    pol = dewpoint_c * (1.1112018e-17 + (dewpoint_c * -3.0994571e-20))
    pol = dewpoint_c * (2.1874425e-13 + (dewpoint_c * (-1.789232e-15 + pol)))
    pol = dewpoint_c * (4.3884180e-09 + (dewpoint_c * (-2.988388e-11 + pol)))
    pol = dewpoint_c * (7.8736169e-05 + (dewpoint_c * (-6.111796e-07 + pol)))
    pol = 0.99999683 + (dewpoint_c * (-9.082695e-03 + pol))
    vapor_pressure_hpa = enhancement * (6.1078 / (pol ** 8))
    return 621.97 * (vapor_pressure_hpa / (pressure_hpa - vapor_pressure_hpa))


def _sharppy_temp_at_mixrat(mixing_ratio_gkg, pressure_hpa):
    """SHARPpy temp_at_mixrat approximation returning Celsius."""
    c1 = 0.0498646455
    c2 = 2.4082965
    c3 = 7.07475
    c4 = 38.9114
    c5 = 0.0915
    c6 = 1.2035
    x = np.log10(mixing_ratio_gkg * pressure_hpa / (622.0 + mixing_ratio_gkg))
    return (10.0 ** (c1 * x + c2) - c3 + c4 * (10.0 ** (c5 * x) - c6) ** 2) - 273.15


def _extract_layer_1d(pressure, values, p_bottom, p_top, interpolate=True):
    """Extract a 1D pressure layer with MetRust-compatible boundary interpolation."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if p.size != v.size:
        raise ValueError("pressure and values must have the same length")
    if p.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    flipped = p[0] < p[-1]
    if flipped:
        p = p[::-1]
        v = v[::-1]

    p_out = []
    v_out = []

    for i in range(p.size):
        pi = p[i]
        if pi <= p_bottom and pi >= p_top:
            if (
                not p_out and
                i > 0 and
                p[i - 1] > p_bottom and
                interpolate and
                not np.isclose(pi, p_bottom)
            ):
                p_out.append(float(p_bottom))
                v_out.append(_log_interp_pressure_value(p_bottom, p[i - 1:i + 1], v[i - 1:i + 1]))
            if not p_out or not np.isclose(p_out[-1], pi):
                p_out.append(float(pi))
                v_out.append(float(v[i]))
        elif pi < p_top and p_out:
            if (
                i > 0 and
                p[i - 1] >= p_top and
                interpolate and
                not np.isclose(p[i - 1], p_top)
            ):
                p_out.append(float(p_top))
                v_out.append(_log_interp_pressure_value(p_top, p[i - 1:i + 1], v[i - 1:i + 1]))
            break

    if not p_out:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    p_arr = np.asarray(p_out, dtype=np.float64)
    v_arr = np.asarray(v_out, dtype=np.float64)
    if flipped:
        p_arr = p_arr[::-1]
        v_arr = v_arr[::-1]
    return p_arr, v_arr


def _svp_hpa_ambaum(t_c):
    """MetRust/MetPy saturation vapor pressure over liquid water in hPa."""
    t_k = float(t_c) + 273.15
    t0 = 273.16
    lv0 = 2500840.0
    cp_l = 4219.4
    cp_v = 1860.078011865639
    rv = 461.52311572606084
    latent = lv0 - (cp_l - cp_v) * (t_k - t0)
    heat_pow = (cp_l - cp_v) / rv
    exp_term = (lv0 / t0 - latent / t_k) / rv
    return 611.2 * (t0 / t_k) ** heat_pow * np.exp(exp_term) / 100.0


def _lcl_temperature_c(temp_c, dewpoint_c):
    """Bolton-style LCL temperature in Celsius."""
    spread = float(temp_c) - float(dewpoint_c)
    delta = spread * (
        1.2185
        + 0.001278 * float(temp_c)
        + spread * (-0.00219 + 1.173e-5 * spread - 0.0000052 * float(temp_c))
    )
    return float(temp_c) - delta


def _drylift_cpu(pressure_hpa, temp_c, dewpoint_c):
    """Return (p_lcl, t_lcl) using the same drylift relation as the kernels."""
    t_lcl = _lcl_temperature_c(temp_c, dewpoint_c)
    p_lcl = 1000.0 * (
        (t_lcl + 273.15)
        / ((float(temp_c) + 273.15) * (1000.0 / float(pressure_hpa)) ** 0.2857142857142857)
    ) ** (1.0 / 0.2857142857142857)
    return p_lcl, t_lcl


def _vappres_sharppy_hpa(temp_c):
    """SHARPpy/Wexler saturation vapor pressure in hPa."""
    pol = temp_c * (1.1112018e-17 + (temp_c * -3.0994571e-20))
    pol = temp_c * (2.1874425e-13 + (temp_c * (-1.789232e-15 + pol)))
    pol = temp_c * (4.3884180e-09 + (temp_c * (-2.988388e-11 + pol)))
    pol = temp_c * (7.8736169e-05 + (temp_c * (-6.111796e-07 + pol)))
    pol = 0.99999683 + (temp_c * (-9.082695e-03 + pol))
    return 6.1078 / (pol ** 8)


def _wobf_c(temp_c):
    """Wobus function used by SHARPpy-style moist lifting."""
    tc = float(temp_c) - 20.0
    if tc <= 0.0:
        npol = 1.0 + tc * (
            -8.841660499999999e-3
            + tc * (
                1.4714143e-4
                + tc * (-9.671989000000001e-7 + tc * (-3.2607217e-8 + tc * (-3.8598073e-10)))
            )
        )
        n2 = npol * npol
        return 15.13 / (n2 * n2)
    ppol = tc * (
        4.9618922e-07
        + tc * (-6.1059365e-09 + tc * (3.9401551e-11 + tc * (-1.2588129e-13 + tc * 1.6688280e-16)))
    )
    ppol = 1.0 + tc * (3.6182989e-03 + tc * (-1.3603273e-05 + ppol))
    p2 = ppol * ppol
    return (29.93 / (p2 * p2)) + (0.96 * tc) - 14.8


def _thetam_from_lcl(p_lcl, t_lcl):
    """Moist theta used by satlift, in Celsius-space."""
    theta_c = (float(t_lcl) + 273.15) * ((1000.0 / float(p_lcl)) ** 0.2857142857142857) - 273.15
    return theta_c - _wobf_c(theta_c) + _wobf_c(float(t_lcl))


def _satlift_c(pressure_hpa, thetam_c):
    """Lift a saturated parcel to pressure_hpa using the SHARPpy satlift iteration."""
    p = float(pressure_hpa)
    if p >= 1000.0:
        return float(thetam_c)
    pwrp = (p / 1000.0) ** 0.2857142857142857
    t1 = (float(thetam_c) + 273.15) * pwrp - 273.15
    e1 = _wobf_c(t1) - _wobf_c(float(thetam_c))
    rate = 1.0
    for _ in range(7):
        if abs(e1) < 0.001:
            break
        t2 = t1 - (e1 * rate)
        e2 = (t2 + 273.15) / pwrp - 273.15
        e2 += _wobf_c(t2) - _wobf_c(e2) - float(thetam_c)
        denom = e2 - e1
        if abs(denom) < 1e-12:
            break
        rate = (t2 - t1) / denom
        t1 = t2
        e1 = e2
    return t1 - e1 * rate


def _moist_lapse_rate_hpa(p_hpa, t_c):
    """Moist adiabatic lapse rate in K/hPa matching wx-math."""
    t_k = float(t_c) + 273.15
    es = _svp_hpa_ambaum(t_c)
    rs = 0.6219569100577033 * es / (float(p_hpa) - es)
    numerator = (287.04749097718457 * t_k + 2500840.0 * rs) / float(p_hpa)
    denominator = 1004.6662184201462 + (
        (2500840.0 * 2500840.0 * rs * 0.6219569100577033)
        / (287.04749097718457 * t_k * t_k)
    )
    return numerator / denominator


def _moist_lapse_profile_numpy(pressure, t_start_c):
    """RK4 moist-adiabatic parcel profile matching wx-math."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    if p.size == 0:
        return np.empty(0, dtype=np.float64)
    result = np.empty_like(p)
    result[0] = float(t_start_c)
    t_cur = float(t_start_c)
    for i in range(1, p.size):
        dp = p[i] - p[i - 1]
        if abs(dp) < 1e-10:
            result[i] = t_cur
            continue
        n_steps = max(int(abs(dp) / 5.0), 4)
        h = dp / n_steps
        p_cur = p[i - 1]
        for _ in range(n_steps):
            k1 = h * _moist_lapse_rate_hpa(p_cur, t_cur)
            k2 = h * _moist_lapse_rate_hpa(p_cur + h / 2.0, t_cur + k1 / 2.0)
            k3 = h * _moist_lapse_rate_hpa(p_cur + h / 2.0, t_cur + k2 / 2.0)
            k4 = h * _moist_lapse_rate_hpa(p_cur + h, t_cur + k3)
            t_cur += (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0
            p_cur += h
        result[i] = t_cur
    return result


def _lift_parcel_profile_virtual_temperature_numpy(pressure, temperature, dewpoint):
    """MetRust-compatible lifted parcel virtual-temperature profile."""
    p_prof = np.asarray(pressure, dtype=np.float64).ravel()
    t_prof = np.asarray(temperature, dtype=np.float64).ravel()
    td_prof = np.asarray(dewpoint, dtype=np.float64).ravel()
    if p_prof.size == 0:
        return np.nan, np.nan, np.empty(0, dtype=np.float64)

    p_sfc = float(p_prof[0])
    t_sfc = float(t_prof[0])
    td_sfc = float(td_prof[0])
    p_lcl, t_lcl = _drylift_cpu(p_sfc, t_sfc, td_sfc)
    thetam = _thetam_from_lcl(p_lcl, t_lcl)
    theta_dry_k = (t_sfc + 273.15) * ((1000.0 / p_sfc) ** 0.2857142857142857)
    r_parcel_gkg = _sharppy_mixratio_gkg(p_sfc, td_sfc)

    parcel_tv = np.empty_like(p_prof)
    for i, p_val in enumerate(p_prof):
        if p_val > p_lcl:
            t_parc_k = theta_dry_k * ((p_val / 1000.0) ** 0.2857142857142857)
            t_parc = t_parc_k - 273.15
            parcel_tv[i] = (t_parc + 273.15) * (1.0 + 0.61 * (r_parcel_gkg / 1000.0)) - 273.15
        else:
            t_parc = _satlift_c(p_val, thetam)
            parcel_tv[i] = _virtual_temp_from_dewpoint_c(t_parc, p_val, t_parc)

    return p_lcl, t_lcl, parcel_tv


def _parcel_profile_with_lcl_numpy(pressure, t_surface_c, td_surface_c):
    """MetRust-compatible parcel profile with the LCL inserted."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    if p.size == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)

    t_surface_c = float(t_surface_c)
    td_surface_c = float(td_surface_c)
    p_lcl, t_lcl = _drylift_cpu(float(p[0]), t_surface_c, td_surface_c)
    thetam = _thetam_from_lcl(p_lcl, t_lcl)

    p_aug = []
    lcl_inserted = False
    for p_val in p:
        if not lcl_inserted and p_val <= p_lcl:
            if abs(p_val - p_lcl) > 0.01:
                p_aug.append(p_lcl)
            lcl_inserted = True
        p_aug.append(float(p_val))
    if not lcl_inserted:
        p_aug.append(p_lcl)

    t_surface_k = t_surface_c + 273.15
    p_surface = float(p[0])
    t_aug = []
    for p_val in p_aug:
        if p_val > p_lcl:
            t_k = t_surface_k * ((p_val / p_surface) ** 0.2857142857142857)
            t_aug.append(t_k - 273.15)
        else:
            t_aug.append(_satlift_c(p_val, thetam))

    return np.asarray(p_aug, dtype=np.float64), np.asarray(t_aug, dtype=np.float64)


def _virtual_temp_from_dewpoint_c(temp_c, pressure_hpa, dewpoint_c):
    """Virtual temperature in Celsius using the same SHARPpy-style mix ratio."""
    w = _sharppy_mixratio_gkg(float(pressure_hpa), min(float(dewpoint_c), float(temp_c))) / 1000.0
    t_k = float(temp_c) + 273.15
    return t_k * (1.0 + 0.61 * w) - 273.15


def _height_at_pressure_value(target_p, pressure, heights):
    """Linearly interpolate height at target pressure from a descending profile."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    h = np.asarray(heights, dtype=np.float64).ravel()
    if p.size != h.size:
        raise ValueError("pressure and heights must have the same length")
    if p.size == 0:
        return np.nan
    if p[0] < p[-1]:
        p = p[::-1]
        h = h[::-1]
    for i in range(p.size - 1):
        if p[i] >= target_p >= p[i + 1]:
            p0 = p[i]
            p1 = p[i + 1]
            if p1 == p0:
                return float(h[i])
            frac = (target_p - p0) / (p1 - p0)
            return float(h[i] + frac * (h[i + 1] - h[i]))
    if target_p > p[0]:
        return float(h[0])
    if target_p < p[-1]:
        return float(h[-1])
    return np.nan


def _pressure_at_height_value(target_z, heights, pressure):
    """Linearly interpolate pressure at a target height."""
    h = np.asarray(heights, dtype=np.float64).ravel()
    p = np.asarray(pressure, dtype=np.float64).ravel()
    if h.size != p.size:
        raise ValueError("heights and pressure must have the same length")
    if h.size == 0:
        return np.nan
    if h[0] > h[-1]:
        h = h[::-1]
        p = p[::-1]
    if target_z <= h[0]:
        return float(p[0])
    if target_z >= h[-1]:
        return float(p[-1])
    idx = np.searchsorted(h, target_z, side="left")
    h0 = h[idx - 1]
    h1 = h[idx]
    if h1 == h0:
        return float(p[idx - 1])
    frac = (target_z - h0) / (h1 - h0)
    return float(p[idx - 1] + frac * (p[idx] - p[idx - 1]))


def _standard_agl_heights_from_pressure(pressure_hpa):
    """Standard-atmosphere AGL heights for a pressure profile."""
    p = np.asarray(pressure_hpa, dtype=np.float64)
    z = (288.0 / 0.0065) * (1.0 - (p / 1013.25) ** (1.0 / (9.80665 * 0.0289644 / (8.31447 * 0.0065))))
    return z - float(z[0])


def _parcel_profile_cape_cin_numpy(pressure, temperature, dewpoint, parcel_profile):
    """MetRust-compatible CAPE/CIN integration for an explicit parcel profile."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    t = np.asarray(temperature, dtype=np.float64).ravel()
    td = np.asarray(dewpoint, dtype=np.float64).ravel()
    t_parcel = np.asarray(parcel_profile, dtype=np.float64).ravel()

    z = np.zeros(len(p), dtype=np.float64)
    for i in range(1, len(p)):
        if p[i] <= 0.0 or p[i - 1] <= 0.0:
            z[i] = z[i - 1]
            continue
        tv_mean = (
            _virtual_temp_from_dewpoint_c(t[i - 1], p[i - 1], td[i - 1])
            + _virtual_temp_from_dewpoint_c(t[i], p[i], td[i])
        ) / 2.0 + 273.15
        z[i] = z[i - 1] + (287.04749 * tv_mean / 9.80665) * np.log(p[i - 1] / p[i])

    buoyancy = np.zeros(len(p), dtype=np.float64)
    for i in range(len(p)):
        if p[i] <= 0.0:
            continue
        tv_e = _virtual_temp_from_dewpoint_c(t[i], p[i], td[i]) + 273.15
        tv_p = _virtual_temp_from_dewpoint_c(t_parcel[i], p[i], t_parcel[i]) + 273.15
        if tv_e > 0.0:
            buoyancy[i] = (tv_p - tv_e) / tv_e

    lfc_idx = None
    for i in range(1, len(p)):
        if buoyancy[i] > 0.0 and buoyancy[i - 1] <= 0.0:
            lfc_idx = i
            break
    if lfc_idx is None and any(b > 0.0 for b in buoyancy[1:]):
        lfc_idx = 0

    el_idx = len(p) - 1
    if lfc_idx is not None:
        for i in range(lfc_idx + 1, len(p)):
            if buoyancy[i] <= 0.0 and buoyancy[i - 1] > 0.0:
                el_idx = i

    cape_val = 0.0
    cin_val = 0.0
    for i in range(1, len(p)):
        if p[i] <= 0.0:
            continue
        tv_e_lo = _virtual_temp_from_dewpoint_c(t[i - 1], p[i - 1], td[i - 1]) + 273.15
        tv_e_hi = _virtual_temp_from_dewpoint_c(t[i], p[i], td[i]) + 273.15
        tv_p_lo = _virtual_temp_from_dewpoint_c(t_parcel[i - 1], p[i - 1], t_parcel[i - 1]) + 273.15
        tv_p_hi = _virtual_temp_from_dewpoint_c(t_parcel[i], p[i], t_parcel[i]) + 273.15
        dz = z[i] - z[i - 1]
        if abs(dz) < 1e-6 or tv_e_lo <= 0.0 or tv_e_hi <= 0.0:
            continue
        buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo
        buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi
        val = 9.80665 * (buoy_lo + buoy_hi) / 2.0 * dz
        if lfc_idx is not None and i <= el_idx:
            if val > 0.0 and i >= lfc_idx:
                cape_val += val
            elif val < 0.0 and i <= lfc_idx:
                cin_val += val

    if lfc_idx is None or cape_val <= 0.0:
        return 0.0, 0.0
    return cape_val, cin_val


def _cape_cin_core_compatible(
    pressure,
    temperature,
    dewpoint,
    height_agl,
    psfc,
    t2m,
    td2m,
    parcel_type="sb",
    ml_depth=100.0,
    mu_depth=300.0,
    top_m=None,
):
    """MetRust-compatible generic cape_cin wrapper semantics."""
    p_prof = np.asarray(pressure, dtype=np.float64).ravel()
    t_prof = np.asarray(temperature, dtype=np.float64).ravel()
    td_prof = np.asarray(dewpoint, dtype=np.float64).ravel()
    h_prof = np.asarray(height_agl, dtype=np.float64).ravel()

    td2m = min(float(td2m), float(t2m))

    p_aug = np.concatenate(([float(psfc)], p_prof))
    t_aug = np.concatenate(([float(t2m)], t_prof))
    td_aug = np.concatenate(([td2m], np.minimum(td_prof, t_prof)))
    h_aug = np.concatenate(([0.0], h_prof))

    parcel_type = str(parcel_type or "sb").lower()
    if parcel_type == "ml":
        p_start, t_start, td_start = get_mixed_layer_parcel(p_aug, t_aug, td_aug, float(ml_depth))
        p_start = float(cp.asnumpy(cp.asarray(p_start)))
        t_start = float(cp.asnumpy(cp.asarray(t_start)))
        td_start = float(cp.asnumpy(cp.asarray(td_start)))
    elif parcel_type == "mu":
        p_start, t_start, td_start, _ = get_most_unstable_parcel(p_aug, t_aug, td_aug, float(mu_depth))
        p_start = float(cp.asnumpy(cp.asarray(p_start)))
        t_start = float(cp.asnumpy(cp.asarray(t_start)))
        td_start = float(cp.asnumpy(cp.asarray(td_start)))
    else:
        p_start = float(psfc)
        t_start = float(t2m)
        td_start = td2m

    p_lcl, t_lcl = _drylift_cpu(p_start, t_start, td_start)
    h_lcl = _height_at_pressure_value(p_lcl, p_aug, h_aug)
    thetam = _thetam_from_lcl(p_lcl, t_lcl)

    lfc_p = p_lcl
    el_p = p_lcl
    found_positive_layer = False
    in_pos_layer = False

    start_idx = 0
    for i in range(len(p_aug)):
        if p_aug[i] <= p_lcl:
            start_idx = i
            break

    for i in range(start_idx, len(p_aug)):
        p_curr = p_aug[i]
        tv_env = _virtual_temp_from_dewpoint_c(t_aug[i], p_curr, td_aug[i])
        t_parc = _satlift_c(p_curr, thetam)
        tv_parc = _virtual_temp_from_dewpoint_c(t_parc, p_curr, t_parc)
        buoyancy = tv_parc - tv_env

        if buoyancy > 0.0:
            if not in_pos_layer:
                in_pos_layer = True
                if i > 0:
                    p_prev = p_aug[i - 1]
                    tv_env_prev = _virtual_temp_from_dewpoint_c(t_aug[i - 1], p_prev, td_aug[i - 1])
                    t_parc_prev = _satlift_c(p_prev, thetam)
                    tv_parc_prev = _virtual_temp_from_dewpoint_c(t_parc_prev, p_prev, t_parc_prev)
                    buoy_prev = tv_parc_prev - tv_env_prev
                    if buoyancy != buoy_prev:
                        frac = (0.0 - buoy_prev) / (buoyancy - buoy_prev)
                        lfc_p = p_prev + frac * (p_curr - p_prev)
                    else:
                        lfc_p = p_curr
                else:
                    lfc_p = p_curr
                el_p = p_aug[-1]
                found_positive_layer = True
        elif in_pos_layer:
            in_pos_layer = False
            p_prev = p_aug[i - 1]
            tv_env_prev = _virtual_temp_from_dewpoint_c(t_aug[i - 1], p_prev, td_aug[i - 1])
            t_parc_prev = _satlift_c(p_prev, thetam)
            tv_parc_prev = _virtual_temp_from_dewpoint_c(t_parc_prev, p_prev, t_parc_prev)
            buoy_prev = tv_parc_prev - tv_env_prev
            if buoyancy != buoy_prev:
                frac = (0.0 - buoy_prev) / (buoyancy - buoy_prev)
                el_p = p_prev + frac * (p_curr - p_prev)
            else:
                el_p = p_curr

    if in_pos_layer:
        el_p = p_aug[-1]

    if not found_positive_layer:
        return 0.0, 0.0, h_lcl, np.nan

    if np.isnan(lfc_p) or lfc_p > p_lcl:
        lfc_p = p_lcl
    h_lfc = _height_at_pressure_value(lfc_p, p_aug, h_aug)

    theta_dry_k = (t_start + 273.15) * ((1000.0 / p_start) ** 0.2857142857142857)
    w_kgkg = _sharppy_mixratio_gkg(p_start, td_start) / 1000.0
    p_moist = [p_lcl]
    for pi in p_aug:
        if pi < p_lcl and pi > 0.0:
            p_moist.append(float(pi))
    p_moist = np.asarray(p_moist, dtype=np.float64)
    moist_temps = _moist_lapse_profile_numpy(p_moist, t_lcl) if len(p_moist) > 1 else np.asarray([t_lcl], dtype=np.float64)

    n = len(p_aug)
    tv_parc_arr = np.full(n, np.nan, dtype=np.float64)
    tv_env_arr = np.zeros(n, dtype=np.float64)

    for i in range(n):
        if p_aug[i] <= 0.0:
            continue
        tv_env_arr[i] = _virtual_temp_from_dewpoint_c(t_aug[i], p_aug[i], td_aug[i])
        if p_aug[i] >= p_lcl:
            t_parc_k = theta_dry_k * ((p_aug[i] / 1000.0) ** 0.2857142857142857)
            t_parc = t_parc_k - 273.15
            tv_parc_arr[i] = (t_parc + 273.15) * (1.0 + w_kgkg / 0.6219569100577033) / (1.0 + w_kgkg) - 273.15
        else:
            t_parc = _log_interp_pressure_value(p_aug[i], p_moist, moist_temps)
            tv_parc_arr[i] = _virtual_temp_from_dewpoint_c(t_parc, p_aug[i], t_parc)

    z_calc = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if p_aug[i] <= 0.0 or p_aug[i - 1] <= 0.0:
            z_calc[i] = z_calc[i - 1]
            continue
        tv_mean = (tv_env_arr[i - 1] + tv_env_arr[i]) / 2.0 + 273.15
        z_calc[i] = z_calc[i - 1] + (287.058 * tv_mean / 9.80665) * np.log(p_aug[i - 1] / p_aug[i])

    z_use = h_aug if np.any(h_aug > 0.0) else z_calc

    p_top_limit = el_p
    if top_m is not None:
        p_top_m = _pressure_at_height_value(float(top_m), z_use, p_aug)
        if p_top_m >= p_top_limit:
            p_top_limit = max(p_top_m, p_aug[-1])

    last_lfc_idx = None
    for i in range(1, n):
        if not np.isfinite(tv_parc_arr[i]) or not np.isfinite(tv_parc_arr[i - 1]):
            continue
        buoy = tv_parc_arr[i] - tv_env_arr[i]
        buoy_prev = tv_parc_arr[i - 1] - tv_env_arr[i - 1]
        if buoy > 0.0 and buoy_prev <= 0.0:
            last_lfc_idx = i

    total_cape = 0.0
    total_cin = 0.0
    p_top_actual = p_top_limit if p_top_limit > 0.0 else p_aug[-1]

    for i in range(1, n):
        if p_aug[i] <= 0.0 or not np.isfinite(tv_parc_arr[i]) or not np.isfinite(tv_parc_arr[i - 1]):
            continue
        if p_aug[i] < p_top_actual:
            continue
        tv_e_lo = tv_env_arr[i - 1] + 273.15
        tv_e_hi = tv_env_arr[i] + 273.15
        tv_p_lo = tv_parc_arr[i - 1] + 273.15
        tv_p_hi = tv_parc_arr[i] + 273.15
        dz = z_use[i] - z_use[i - 1]
        if abs(dz) < 1e-6 or tv_e_lo <= 0.0 or tv_e_hi <= 0.0:
            continue
        buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo
        buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi
        val = 9.80665 * (buoy_lo + buoy_hi) / 2.0 * dz
        if last_lfc_idx is not None:
            if val > 0.0 and i >= last_lfc_idx:
                total_cape += val
            elif val < 0.0 and i <= last_lfc_idx:
                total_cin += val
        else:
            if val > 0.0:
                total_cape += val
            else:
                total_cin += val

    return total_cape, total_cin, h_lcl, h_lfc


def _cape_cin_from_parcel_state(pressure, temperature, dewpoint, p_start, t_start, td_start):
    """MetRust-compatible CAPE/CIN integration for a specified starting parcel."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    t = np.asarray(temperature, dtype=np.float64).ravel()
    td = np.asarray(dewpoint, dtype=np.float64).ravel()
    if p.size == 0:
        return 0.0, 0.0, np.nan, np.nan, np.nan

    if p[0] < p[-1]:
        p = p[::-1]
        t = t[::-1]
        td = td[::-1]

    t_start = float(t_start)
    td_start = min(float(td_start), t_start)
    p_start = float(p_start)

    p_lcl, t_lcl = _drylift_cpu(p_start, t_start, td_start)
    theta_dry_k = (t_start + 273.15) * ((1000.0 / p_start) ** 0.2857142857142857)
    w_kgkg = _sharppy_mixratio_gkg(p_start, td_start) / 1000.0

    p_moist = [p_lcl]
    for pi in p:
        if pi < p_lcl and pi > 0.0:
            p_moist.append(float(pi))
    p_moist = np.asarray(p_moist, dtype=np.float64)
    moist_profile = _moist_lapse_profile_numpy(p_moist, t_lcl)

    tv_parcel = np.empty_like(p)
    tv_env = np.empty_like(p)
    z = np.zeros_like(p)

    for i, pi in enumerate(p):
        td_i = min(td[i], t[i])
        tv_env[i] = _virtual_temp_from_dewpoint_c(t[i], pi, td_i)
        if pi <= 0.0:
            tv_parcel[i] = np.nan
            continue
        if pi >= p_lcl:
            t_parc_k = theta_dry_k * ((pi / 1000.0) ** 0.2857142857142857)
            t_parc = t_parc_k - 273.15
            tv_parcel[i] = (t_parc + 273.15) * (1.0 + w_kgkg / 0.6219569100577033) / (1.0 + w_kgkg) - 273.15
        else:
            t_parc = _log_interp_pressure_value(pi, p_moist, moist_profile)
            tv_parcel[i] = _virtual_temp_from_dewpoint_c(t_parc, pi, t_parc)

    for i in range(1, p.size):
        if p[i] <= 0.0 or p[i - 1] <= 0.0:
            z[i] = z[i - 1]
            continue
        tv_mean = (tv_env[i - 1] + tv_env[i]) / 2.0 + 273.15
        z[i] = z[i - 1] + (287.04749097718457 * tv_mean / 9.80665) * np.log(p[i - 1] / p[i])

    last_lfc_idx = -1
    for i in range(1, p.size):
        if not np.isfinite(tv_parcel[i]) or not np.isfinite(tv_parcel[i - 1]):
            continue
        buoy = tv_parcel[i] - tv_env[i]
        buoy_prev = tv_parcel[i - 1] - tv_env[i - 1]
        if buoy > 0.0 and buoy_prev <= 0.0:
            last_lfc_idx = i

    el_idx = -1
    found_positive = False
    for i in range(1, p.size):
        if not np.isfinite(tv_parcel[i]) or not np.isfinite(tv_parcel[i - 1]):
            continue
        buoy = tv_parcel[i] - tv_env[i]
        buoy_prev = tv_parcel[i - 1] - tv_env[i - 1]
        if buoy > 0.0:
            found_positive = True
        if found_positive and buoy_prev > 0.0 and buoy <= 0.0:
            el_idx = i

    if last_lfc_idx < 0:
        return 0.0, 0.0, p_lcl, p_lcl, p_lcl

    buoy_prev = tv_parcel[last_lfc_idx - 1] - tv_env[last_lfc_idx - 1]
    buoy = tv_parcel[last_lfc_idx] - tv_env[last_lfc_idx]
    lfc_frac = -buoy_prev / (buoy - buoy_prev)
    lfc_p = p[last_lfc_idx - 1] + lfc_frac * (p[last_lfc_idx] - p[last_lfc_idx - 1])

    if el_idx >= 0:
        buoy_prev = tv_parcel[el_idx - 1] - tv_env[el_idx - 1]
        buoy = tv_parcel[el_idx] - tv_env[el_idx]
        el_frac = -buoy_prev / (buoy - buoy_prev)
        el_p = p[el_idx - 1] + el_frac * (p[el_idx] - p[el_idx - 1])
    else:
        el_p = p[-1]

    cape = 0.0
    cin = 0.0
    for i in range(1, p.size):
        if p[i] > p_start or p[i - 1] > p_start:
            continue
        if not np.isfinite(tv_parcel[i]) or not np.isfinite(tv_parcel[i - 1]):
            continue
        tv_e_lo = tv_env[i - 1] + 273.15
        tv_e_hi = tv_env[i] + 273.15
        tv_p_lo = tv_parcel[i - 1] + 273.15
        tv_p_hi = tv_parcel[i] + 273.15
        dz = z[i] - z[i - 1]
        if abs(dz) < 1e-6 or tv_e_lo <= 0.0 or tv_e_hi <= 0.0:
            continue
        buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo
        buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi
        val = 9.80665 * (buoy_lo + buoy_hi) / 2.0 * dz
        if val > 0.0:
            cape += val
        elif val < 0.0:
            cin += val

    return cape, cin, p_lcl, lfc_p, el_p


def _interp_height_value(target_z, heights, values):
    """Linear interpolation at a height target."""
    h = np.asarray(heights, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    if h.size != v.size:
        raise ValueError("heights and values must have the same length")
    if target_z <= h[0]:
        return float(v[0])
    if target_z >= h[-1]:
        return float(v[-1])
    idx = np.searchsorted(h, target_z, side="left")
    h0 = h[idx - 1]
    h1 = h[idx]
    frac = (target_z - h0) / (h1 - h0)
    return float(v[idx - 1] + frac * (v[idx] - v[idx - 1]))


def _pressure_weighted_height_average(pressure, values, heights, z_bottom, z_top):
    """MetRust-compatible continuous average in a height layer."""
    p = np.asarray(pressure, dtype=np.float64).ravel()
    v = np.asarray(values, dtype=np.float64).ravel()
    h = np.asarray(heights, dtype=np.float64).ravel()
    layer_vals = [_interp_height_value(z_bottom, h, v)]
    layer_p = [_interp_height_value(z_bottom, h, p)]
    for i in range(h.size):
        if h[i] <= z_bottom:
            continue
        if h[i] >= z_top:
            break
        layer_vals.append(float(v[i]))
        layer_p.append(float(p[i]))
    layer_vals.append(_interp_height_value(z_top, h, v))
    layer_p.append(_interp_height_value(z_top, h, p))
    layer_vals = np.asarray(layer_vals, dtype=np.float64)
    layer_p = np.asarray(layer_p, dtype=np.float64)
    if layer_vals.size < 2:
        return float(layer_vals[0])
    dp = np.diff(layer_p)
    num = np.sum((layer_vals[1:] + layer_vals[:-1]) / 2.0 * dp)
    den = np.sum(dp)
    return float(num / den) if abs(den) > 1e-10 else float(layer_vals[0])


def _height_weighted_layer_average(values, heights, z_bottom, z_top):
    """MetRust-compatible trapezoidal height average over an interpolated layer."""
    v = np.asarray(values, dtype=np.float64).ravel()
    h = np.asarray(heights, dtype=np.float64).ravel()
    layer_h = [float(z_bottom)]
    layer_v = [_interp_height_value(z_bottom, h, v)]
    for i in range(h.size):
        if h[i] <= z_bottom:
            continue
        if h[i] >= z_top:
            break
        layer_h.append(float(h[i]))
        layer_v.append(float(v[i]))
    layer_h.append(float(z_top))
    layer_v.append(_interp_height_value(z_top, h, v))
    layer_h = np.asarray(layer_h, dtype=np.float64)
    layer_v = np.asarray(layer_v, dtype=np.float64)
    if layer_h.size < 2:
        return float(layer_v[0])
    dz = np.diff(layer_h)
    num = np.sum((layer_v[1:] + layer_v[:-1]) / 2.0 * dz)
    den = np.sum(dz)
    return float(num / den) if abs(den) > 1e-10 else float(layer_v[0])


def _storm_relative_helicity_numpy(u_prof, v_prof, height_prof, depth_m, storm_u, storm_v, bottom_m=None):
    """MetRust-compatible storm-relative helicity integration."""
    u = np.asarray(u_prof, dtype=np.float64).ravel()
    v = np.asarray(v_prof, dtype=np.float64).ravel()
    h = np.asarray(height_prof, dtype=np.float64).ravel()
    if u.size != v.size or u.size != h.size:
        raise ValueError("u, v, and height must have the same length")
    if u.size < 2:
        return 0.0, 0.0, 0.0

    h_start = float(h[0]) if bottom_m is None else float(bottom_m)
    h_end = h_start + float(depth_m)
    heights = [h_start]
    us = [_interp_height_value(h_start, h, u)]
    vs = [_interp_height_value(h_start, h, v)]

    for i in range(u.size):
        if h[i] > h_start and h[i] < h_end:
            heights.append(float(h[i]))
            us.append(float(u[i]))
            vs.append(float(v[i]))

    heights.append(h_end)
    us.append(_interp_height_value(h_end, h, u))
    vs.append(_interp_height_value(h_end, h, v))

    pos_srh = 0.0
    neg_srh = 0.0
    for i in range(len(heights) - 1):
        sru_i = us[i] - float(storm_u)
        srv_i = vs[i] - float(storm_v)
        sru_ip1 = us[i + 1] - float(storm_u)
        srv_ip1 = vs[i + 1] - float(storm_v)
        contrib = (sru_ip1 * srv_i) - (sru_i * srv_ip1)
        if contrib > 0.0:
            pos_srh += contrib
        else:
            neg_srh += contrib

    return pos_srh, neg_srh, pos_srh + neg_srh


def _STUB_parcel_profile_with_lcl(p, t, td):
    """Parcel profile with LCL inserted -- inline fallback."""
    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    if p_np.size == 0:
        empty = cp.empty(0, dtype=cp.float64)
        return empty, empty

    p_aug, t_aug = _parcel_profile_with_lcl_numpy(p_np, float(t), float(td))
    return cp.asarray(p_aug), cp.asarray(t_aug)


def _STUB_get_mixed_layer_parcel(p, t, td, d):
    """Mixed layer parcel -- inline fallback."""
    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    t_np = cp.asnumpy(t).astype(np.float64, copy=False)
    td_np = cp.asnumpy(td).astype(np.float64, copy=False)

    sfc_p = float(p_np[0])
    top_p = sfc_p - float(d)

    theta_sfc = (t_np[0] + 273.15) * (1000.0 / sfc_p) ** 0.2857142857142857
    td_sfc = float(td_np[0])

    t_top = _log_interp_pressure_value(top_p, p_np, t_np)
    td_top = _log_interp_pressure_value(top_p, p_np, td_np)
    theta_top = (t_top + 273.15) * (1000.0 / top_p) ** 0.2857142857142857

    sum_theta = theta_sfc + theta_top
    sum_p = sfc_p + top_p
    sum_td = td_sfc + td_top
    count = 2.0

    for i in range(1, p_np.size):
        pi = p_np[i]
        if pi <= top_p:
            break
        theta_i = (t_np[i] + 273.15) * (1000.0 / pi) ** 0.2857142857142857
        sum_theta += 2.0 * theta_i
        sum_p += 2.0 * pi
        sum_td += 2.0 * td_np[i]
        count += 2.0

    avg_theta = sum_theta / count
    avg_p = sum_p / count
    avg_td = sum_td / count

    avg_t = avg_theta * (sfc_p / 1000.0) ** 0.2857142857142857 - 273.15
    avg_w = _sharppy_mixratio_gkg(avg_p, avg_td)
    parcel_td = _sharppy_temp_at_mixrat(avg_w, sfc_p)

    return (
        cp.asarray(sfc_p, dtype=cp.float64),
        cp.asarray(avg_t, dtype=cp.float64),
        cp.asarray(parcel_td, dtype=cp.float64),
    )


def _STUB_get_most_unstable_parcel(p, t, td, d):
    """Most unstable parcel -- inline fallback."""
    import cupy as cp
    theta_e = _k_equivalent_potential_temperature(p, t, td)
    p_bot = float(p[0])
    mask = p >= (p_bot - d)
    idx = int(cp.argmax(theta_e * mask))
    return (p[idx], t[idx], td[idx], idx)


def _STUB_vector_derivative(u_arr, v_arr, dx_val, dy_val):
    """All four partial derivatives of a 2-D vector field."""
    dudx = _gpu_first_derivative_uniform_2d(u_arr, dx_val, axis=1)
    dudy = _gpu_first_derivative_uniform_2d(u_arr, dy_val, axis=0)
    dvdx = _gpu_first_derivative_uniform_2d(v_arr, dx_val, axis=1)
    dvdy = _gpu_first_derivative_uniform_2d(v_arr, dy_val, axis=0)
    return (dudx, dudy, dvdx, dvdy)


def _STUB_kinematic_flux(v_arr, s_arr):
    """Kinematic flux -- element-wise product."""
    return v_arr * s_arr


def _STUB_absolute_momentum(u_arr, lat_arr, yd):
    """Absolute momentum -- u + f*y."""
    import cupy as cp
    f = 2.0 * 7.2921159e-5 * cp.sin(cp.deg2rad(lat_arr))
    return u_arr - f * yd


def _STUB_cross_section_components(u_arr, v_arr, slat, slon, elat, elon):
    """Decompose (u,v) into parallel/perpendicular components."""
    import math
    to_rad = math.pi / 180.0
    dlat = (elat - slat) * to_rad
    dlon = (elon - slon) * to_rad
    mean_lat = ((slat + elat) / 2.0) * to_rad
    de = dlon * math.cos(mean_lat)
    dn = dlat
    mag = math.sqrt(de**2 + dn**2)
    if mag < 1e-12:
        return u_arr * 0.0, v_arr * 0.0
    tx, ty = de / mag, dn / mag
    parallel = u_arr * tx + v_arr * ty
    perpendicular = -u_arr * ty + v_arr * tx
    return parallel, perpendicular


def _STUB_advection_3d(s, u_arr, v_arr, w_arr, dx_val, dy_val, dz_val):
    """3-D advection -- inline fallback."""
    import cupy as cp
    dsdz, dsdy, dsdx = cp.gradient(s, dz_val, dy_val, dx_val)
    return -(u_arr * dsdx + v_arr * dsdy + w_arr * dsdz)


def _STUB_geospatial_gradient(d, lat, lon):
    """Gradient on lat/lon grid -- inline fallback."""
    import cupy as cp
    R = 6371229.0
    lat_rad = cp.deg2rad(lat)
    lon_rad = cp.deg2rad(lon)
    dy = R * cp.gradient(lat_rad, axis=0)
    dx = R * cp.cos(lat_rad) * cp.gradient(lon_rad, axis=1)
    ddy = cp.gradient(d, axis=0) / dy
    ddx = cp.gradient(d, axis=1) / dx
    return (ddx, ddy)


def _STUB_geospatial_laplacian(d, lat, lon):
    """Laplacian on lat/lon grid -- inline fallback."""
    import cupy as cp
    R = 6371229.0
    lat_rad = cp.deg2rad(lat)
    lon_rad = cp.deg2rad(lon)
    dy = R * cp.gradient(lat_rad, axis=0)
    dx = R * cp.cos(lat_rad) * cp.gradient(lon_rad, axis=1)
    d2dx2 = cp.gradient(cp.gradient(d, axis=1) / dx, axis=1) / dx
    d2dy2 = cp.gradient(cp.gradient(d, axis=0) / dy, axis=0) / dy
    return d2dx2 + d2dy2


def _STUB_first_derivative(d_arr, ds, axis):
    """First derivative along a chosen axis."""
    if d_arr.ndim != 2:
        raise NotImplementedError("first_derivative currently expects 2-D input")
    # MetRust axis convention: axis=0 -> x/columns, axis=1 -> y/rows.
    if axis == 0:
        return _gpu_first_derivative_uniform_2d(d_arr, ds, axis=1)
    return _gpu_first_derivative_uniform_2d(d_arr, ds, axis=0)


def _STUB_second_derivative(d_arr, ds, axis):
    """Second derivative along a chosen axis."""
    if d_arr.ndim != 2:
        raise NotImplementedError("second_derivative currently expects 2-D input")
    if axis == 0:
        return _gpu_second_derivative_uniform_2d(d_arr, ds, axis=1)
    return _gpu_second_derivative_uniform_2d(d_arr, ds, axis=0)




def _STUB_compute_grid_scp(mc, s, sh, ci):
    """Enhanced SCP with CIN term (grid-scale formula with shear/40)."""
    import cupy as cp
    scp = (mc / 1000.0) * (s / 50.0) * (sh / 40.0)
    scp = cp.maximum(scp, 0.0)
    cin_term = cp.where(ci > -40.0, 1.0, -40.0 / ci)
    return scp * cin_term


def _STUB_compute_grid_critical_angle(us, vs, ush, vsh):
    """Critical angle between two vectors on a 2D grid."""
    import cupy as cp
    inflow_u = -us
    inflow_v = -vs
    dot = inflow_u * ush + inflow_v * vsh
    mag1 = cp.sqrt(inflow_u**2 + inflow_v**2)
    mag2 = cp.sqrt(ush**2 + vsh**2)
    cos_angle = cp.clip(dot / (mag1 * mag2 + 1e-12), -1.0, 1.0)
    angle = cp.rad2deg(cp.arccos(cos_angle))
    return cp.where((mag1 < 0.01) | (mag2 < 0.01), cp.nan, angle)




# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_gpu(arr):
    """Convert numpy/list/scalar to cupy array on GPU, stripping pint units."""
    if isinstance(arr, cp.ndarray):
        return arr
    if hasattr(arr, 'magnitude'):
        arr = arr.magnitude
    return cp.asarray(arr, dtype=cp.float64)


def _to_cpu(arr):
    """Convert cupy array back to numpy for compatibility."""
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def _to_metrust_input(value):
    """Convert GPU-backed inputs to CPU arrays before calling metrust."""
    if isinstance(value, tuple):
        return tuple(_to_metrust_input(v) for v in value)
    if isinstance(value, list):
        return [_to_metrust_input(v) for v in value]
    if isinstance(value, cp.ndarray):
        return cp.asnumpy(value)
    return value


def _from_metrust_result(value):
    """Convert metrust outputs to met-cu style GPU arrays."""
    if isinstance(value, tuple):
        return tuple(_from_metrust_result(v) for v in value)
    if isinstance(value, list):
        return [_from_metrust_result(v) for v in value]
    if hasattr(value, "magnitude"):
        value = value.magnitude
    if isinstance(value, cp.ndarray):
        return value
    if isinstance(value, np.ndarray):
        return cp.asarray(value, dtype=cp.float64)
    if np.isscalar(value):
        return float(value)
    return value


def _metrust_gpu_fallback(function_name, *args, **kwargs):
    """Call the metrust reference implementation and move results back to GPU."""
    try:
        import metrust.calc as _mr
    except ImportError as exc:
        raise NotImplementedError(f"{function_name} requires metrust") from exc
    result = getattr(_mr, function_name)(
        *[_to_metrust_input(arg) for arg in args],
        **{key: _to_metrust_input(val) for key, val in kwargs.items()},
    )
    return _from_metrust_result(result)


def _scalar(arr):
    """Strip pint units and convert to Python float."""
    if hasattr(arr, 'magnitude'):
        return float(arr.magnitude)
    return float(arr)


def _1d(arr):
    """Strip pint, convert to 1-D cupy float64."""
    if hasattr(arr, 'magnitude'):
        arr = arr.magnitude
    return cp.ascontiguousarray(cp.asarray(arr, dtype=cp.float64).ravel())


def _2d(arr):
    """Strip pint, convert to contiguous 2-D cupy float64."""
    if hasattr(arr, 'magnitude'):
        arr = arr.magnitude
    a = cp.asarray(arr, dtype=cp.float64)
    return cp.ascontiguousarray(a)


def _mean_spacing(val):
    """Extract a scalar grid spacing from a scalar or array."""
    if hasattr(val, 'magnitude'):
        arr = cp.asarray(val.magnitude, dtype=cp.float64)
    else:
        arr = cp.asarray(val, dtype=cp.float64)
    return float(arr.mean()) if arr.ndim > 0 and arr.size > 1 else float(arr)


def _gpu_first_derivative_uniform_2d(arr, spacing, axis):
    """Second-order uniform-grid first derivative with one-sided edges."""
    a = cp.asarray(arr, dtype=cp.float64)
    ds = float(spacing)
    out = cp.empty_like(a)
    if axis == 1:
        if a.shape[1] < 2:
            out.fill(0.0)
            return out
        if a.shape[1] == 2:
            out[:] = (a[:, 1:2] - a[:, 0:1]) / ds
            return out
        out[:, 1:-1] = (a[:, 2:] - a[:, :-2]) / (2.0 * ds)
        out[:, 0] = (-3.0 * a[:, 0] + 4.0 * a[:, 1] - a[:, 2]) / (2.0 * ds)
        out[:, -1] = (3.0 * a[:, -1] - 4.0 * a[:, -2] + a[:, -3]) / (2.0 * ds)
        return out
    if a.shape[0] < 2:
        out.fill(0.0)
        return out
    if a.shape[0] == 2:
        out[:] = (a[1:2, :] - a[0:1, :]) / ds
        return out
    out[1:-1, :] = (a[2:, :] - a[:-2, :]) / (2.0 * ds)
    out[0, :] = (-3.0 * a[0, :] + 4.0 * a[1, :] - a[2, :]) / (2.0 * ds)
    out[-1, :] = (3.0 * a[-1, :] - 4.0 * a[-2, :] + a[-3, :]) / (2.0 * ds)
    return out


def _gpu_second_derivative_uniform_2d(arr, spacing, axis):
    """Second-order uniform-grid second derivative with one-sided edges."""
    a = cp.asarray(arr, dtype=cp.float64)
    ds2 = float(spacing) ** 2
    out = cp.empty_like(a)
    if axis == 1:
        if a.shape[1] < 3:
            out.fill(0.0)
            return out
        out[:, 1:-1] = (a[:, 2:] - 2.0 * a[:, 1:-1] + a[:, :-2]) / ds2
        out[:, 0] = (a[:, 2] - 2.0 * a[:, 1] + a[:, 0]) / ds2
        out[:, -1] = (a[:, -1] - 2.0 * a[:, -2] + a[:, -3]) / ds2
        return out
    if a.shape[0] < 3:
        out.fill(0.0)
        return out
    out[1:-1, :] = (a[2:, :] - 2.0 * a[1:-1, :] + a[:-2, :]) / ds2
    out[0, :] = (a[2, :] - 2.0 * a[1, :] + a[0, :]) / ds2
    out[-1, :] = (a[-1, :] - 2.0 * a[-2, :] + a[-3, :]) / ds2
    return out


# ===========================================================================
# Thermodynamic functions
# ===========================================================================

def potential_temperature(pressure, temperature):
    """Potential temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (Kelvin)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    return _k_potential_temperature(p, t)


def equivalent_potential_temperature(pressure, temperature, dewpoint):
    """Equivalent potential temperature (theta-e).

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (Kelvin)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    td = _to_gpu(dewpoint)
    return _k_equivalent_potential_temperature(p, t, td)


def saturation_vapor_pressure(temperature, phase="liquid"):
    """Saturation vapor pressure.

    Parameters
    ----------
    temperature : array-like (Celsius)
    phase : str
        "liquid" (default), "ice", or "auto".

    Returns
    -------
    cupy.ndarray (Pa)
    """
    t = _to_gpu(temperature)
    return _k_saturation_vapor_pressure(t) * 100.0  # hPa -> Pa


def saturation_mixing_ratio(pressure, temperature, phase="liquid"):
    """Saturation mixing ratio.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    phase : str

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    return _k_saturation_mixing_ratio(p, t)


def wet_bulb_temperature(pressure, temperature, dewpoint):
    """Wet-bulb temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    td = _to_gpu(dewpoint)
    return _k_wet_bulb_temperature(p, t, td)


def dewpoint_from_relative_humidity(temperature, relative_humidity):
    """Dewpoint from temperature and relative humidity.

    Parameters
    ----------
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent 0-100)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    rh = _to_gpu(relative_humidity)
    return _k_dewpoint_from_relative_humidity(t, rh)


def relative_humidity_from_dewpoint(temperature, dewpoint, phase="liquid"):
    """Relative humidity from temperature and dewpoint.

    Parameters
    ----------
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)
    phase : str

    Returns
    -------
    cupy.ndarray (dimensionless, 0-1)
    """
    t = _to_gpu(temperature)
    td = _to_gpu(dewpoint)
    return _k_relative_humidity_from_dewpoint(t, td) / 100.0  # percent -> fractional


def virtual_temperature(temperature, pressure_or_mixing_ratio, dewpoint=None,
                        molecular_weight_ratio=0.6219569100577033):
    """Virtual temperature.

    Can be called as:
    - ``virtual_temperature(T, mixing_ratio)`` (MetPy-compatible)
    - ``virtual_temperature(T, pressure, dewpoint)`` (Rust-native)

    Parameters
    ----------
    temperature : array-like (Celsius)
    pressure_or_mixing_ratio : array-like
    dewpoint : array-like (Celsius), optional

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    pmr = _to_gpu(pressure_or_mixing_ratio)
    if dewpoint is None:
        # MetPy path: T * (1 + w/eps) / (1 + w)
        eps = molecular_weight_ratio
        t_k = t + 273.15
        tv_k = t_k * (1.0 + pmr / eps) / (1.0 + pmr)
        return tv_k - 273.15
    td = _to_gpu(dewpoint)
    return _k_virtual_temperature_from_dewpoint(pmr, t, td)


def virtual_temperature_from_dewpoint(pressure, temperature, dewpoint,
                                      molecular_weight_ratio=None, phase=None):
    """Virtual temperature from pressure, temperature, and dewpoint.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    td = _to_gpu(dewpoint)
    p = _to_gpu(pressure)
    return _k_virtual_temperature_from_dewpoint(p, t, td)


def mixing_ratio(partial_press_or_pressure, total_press_or_temperature,
                 molecular_weight_ratio=0.6219569100577033):
    """Mixing ratio.

    Can be called as:
    - ``mixing_ratio(pressure, temperature)`` -- from pressure & temperature
    - ``mixing_ratio(partial_pressure, total_pressure)`` -- from vapor/total pressure

    Parameters
    ----------
    partial_press_or_pressure : array-like (hPa or Pa)
    total_press_or_temperature : array-like

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    a = _to_gpu(partial_press_or_pressure)
    b = _to_gpu(total_press_or_temperature)
    # If the second arg looks like temperature (values roughly -100..60),
    # compute saturation mixing ratio from (pressure, temperature)
    # Otherwise compute from (vapor_pressure, total_pressure)
    b_mean = float(b.mean()) if b.size > 0 else 0.0
    if abs(b_mean) < 200.0:
        # Likely (pressure, temperature) form. Match metrust's SHARPpy/Wexler
        # saturation mixing ratio path rather than the Ambaum sat-mixing kernel.
        x = 0.02 * (b - 12.5 + (7500.0 / a))
        enhancement = 1.0 + (0.0000045 * a) + (0.0014 * x * x)
        pol = b * (1.1112018e-17 + (b * -3.0994571e-20))
        pol = b * (2.1874425e-13 + (b * (-1.789232e-15 + pol)))
        pol = b * (4.3884180e-09 + (b * (-2.988388e-11 + pol)))
        pol = b * (7.8736169e-05 + (b * (-6.111796e-07 + pol)))
        pol = 0.99999683 + (b * (-9.082695e-03 + pol))
        vapor_pressure_hpa = enhancement * (6.1078 / (pol ** 8))
        return 621.97 * (vapor_pressure_hpa / (a - vapor_pressure_hpa)) / 1000.0
    return _k_mixing_ratio(a, b)


def density(pressure, temperature, mixing_ratio):
    """Air density from pressure, temperature, and mixing ratio.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    mixing_ratio : array-like (g/kg)

    Returns
    -------
    cupy.ndarray (kg/m^3)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    w = _to_gpu(mixing_ratio) / 1000.0  # g/kg -> kg/kg (kernel expects kg/kg)
    return _k_density(p, t, w)


def dewpoint(vapor_pressure_val):
    """Dewpoint from vapor pressure.

    Parameters
    ----------
    vapor_pressure_val : array-like (hPa)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    e = _to_gpu(vapor_pressure_val)
    return _k_dewpoint(e)


def dewpoint_from_specific_humidity(pressure, specific_humidity):
    """Dewpoint from pressure and specific humidity.

    Parameters
    ----------
    pressure : array-like (hPa)
    specific_humidity : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    p = _to_gpu(pressure)
    q = _to_gpu(specific_humidity)
    return _k_dewpoint_from_specific_humidity(p, q)


def dry_lapse(pressure, t_surface):
    """Dry adiabatic lapse rate temperature profile.

    Parameters
    ----------
    pressure : array-like (hPa)
    t_surface : float (Celsius)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    p = _1d(pressure)
    t = _scalar(t_surface)
    p_ref = float(p[0])
    return _k_dry_lapse(p, p_ref, t)


def dry_static_energy(height, temperature):
    """Dry static energy.

    Parameters
    ----------
    height : array-like (m)
    temperature : array-like (K)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    h = _to_gpu(height)
    t = _to_gpu(temperature)
    return _k_dry_static_energy(h, t)


def exner_function(pressure):
    """Exner function.

    Parameters
    ----------
    pressure : array-like (hPa)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    p = _to_gpu(pressure)
    return _k_exner_function(p)


def moist_lapse(pressure, t_start, reference_pressure=None):
    """Moist adiabatic lapse rate temperature profile.

    Parameters
    ----------
    pressure : array-like (hPa)
    t_start : float (Celsius)
    reference_pressure : float (hPa), optional

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    p = _1d(pressure)
    t = _scalar(t_start)
    return _k_moist_lapse(p, t)


def moist_static_energy(height, temperature, specific_humidity):
    """Moist static energy.

    Parameters
    ----------
    height : array-like (m)
    temperature : array-like (K)
    specific_humidity : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    h = _to_gpu(height)
    t = _to_gpu(temperature)
    q = _to_gpu(specific_humidity)
    return _k_moist_static_energy(h, t, q)


def parcel_profile(pressure, temperature, dewpoint):
    """Parcel temperature profile.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : float (Celsius)
    dewpoint : float (Celsius)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    p = _1d(pressure)
    t = _scalar(temperature)
    td = _scalar(dewpoint)
    return _k_parcel_profile(p, t, td)


def temperature_from_potential_temperature(pressure, theta):
    """Temperature from potential temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    theta : array-like (K)

    Returns
    -------
    cupy.ndarray (K)
    """
    p = _to_gpu(pressure)
    th = _to_gpu(theta)
    return _k_temperature_from_potential_temperature(p, th)


def vertical_velocity(omega, pressure, temperature):
    """Convert pressure vertical velocity (omega) to w (m/s).

    Parameters
    ----------
    omega : array-like (Pa/s)
    pressure : array-like (hPa)
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (m/s)
    """
    o = _to_gpu(omega)
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    return _k_vertical_velocity(o, p, t)


def vertical_velocity_pressure(w, pressure, temperature):
    """Convert w (m/s) to pressure vertical velocity (omega, Pa/s).

    Parameters
    ----------
    w : array-like (m/s)
    pressure : array-like (hPa)
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (Pa/s)
    """
    ww = _to_gpu(w)
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    return _k_vertical_velocity_pressure(ww, p, t)


def virtual_potential_temperature(pressure, temperature, mixing_ratio_val):
    """Virtual potential temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    mixing_ratio_val : array-like (g/kg)

    Returns
    -------
    cupy.ndarray (K)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    w = _to_gpu(mixing_ratio_val) / 1000.0  # g/kg -> kg/kg (kernel expects kg/kg)
    return _k_virtual_potential_temperature(p, t, w)


def wet_bulb_potential_temperature(pressure, temperature, dewpoint):
    """Wet-bulb potential temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (K)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    td = _to_gpu(dewpoint)
    return _k_wet_bulb_potential_temperature(p, t, td)


def saturation_equivalent_potential_temperature(pressure, temperature):
    """Saturation equivalent potential temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (K)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    return _k_saturation_equivalent_potential_temperature(p, t)


def vapor_pressure(pressure_or_dewpoint, mixing_ratio=None,
                   molecular_weight_ratio=0.6219569100577033):
    """Vapor pressure from dewpoint or from pressure and mixing ratio.

    Parameters
    ----------
    pressure_or_dewpoint : array-like
    mixing_ratio : array-like, optional

    Returns
    -------
    cupy.ndarray (Pa)
    """
    if mixing_ratio is not None:
        p = _to_gpu(_as_magnitude_in_units(pressure_or_dewpoint, "Pa"))
        w = _to_gpu(_as_magnitude_in_units(mixing_ratio, "kg/kg"))
        return p * w / (molecular_weight_ratio + w)
    td = _to_gpu(_as_magnitude_in_units(pressure_or_dewpoint, "degC"))
    return _k_vapor_pressure(td) * 100.0  # hPa -> Pa


def specific_humidity_from_mixing_ratio(mixing_ratio):
    """Specific humidity from mixing ratio.

    Parameters
    ----------
    mixing_ratio : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    w = _to_gpu(mixing_ratio)
    return _k_specific_humidity_from_mixing_ratio(w)


def mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity):
    """Mixing ratio from pressure, temperature, and relative humidity.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent 0-100)

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    rh = _to_gpu(relative_humidity)
    return _k_mixing_ratio_from_relative_humidity(p, t, rh)


def mixing_ratio_from_specific_humidity(specific_humidity):
    """Mixing ratio from specific humidity.

    Parameters
    ----------
    specific_humidity : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    q = _to_gpu(specific_humidity)
    return _k_mixing_ratio_from_specific_humidity(q)


def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio_val):
    """Relative humidity from mixing ratio.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    mixing_ratio_val : array-like (g/kg)

    Returns
    -------
    cupy.ndarray (dimensionless, 0-1)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    w = _to_gpu(mixing_ratio_val) / 1000.0  # g/kg -> kg/kg (kernel expects kg/kg)
    return _k_relative_humidity_from_mixing_ratio(p, t, w) / 100.0  # percent -> fractional


def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
    """Relative humidity from specific humidity.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    specific_humidity : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (dimensionless, 0-1)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    q = _to_gpu(specific_humidity)
    return _k_relative_humidity_from_specific_humidity(p, t, q) / 100.0  # percent -> fractional


def specific_humidity_from_dewpoint(pressure, dewpoint_val):
    """Specific humidity from pressure and dewpoint.

    Parameters
    ----------
    pressure : array-like (hPa)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    p = _to_gpu(pressure)
    td = _to_gpu(dewpoint_val)
    return _k_specific_humidity_from_dewpoint(p, td)


def frost_point(temperature, relative_humidity):
    """Frost point temperature.

    Parameters
    ----------
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent 0-100)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    rh = _to_gpu(relative_humidity)
    return _k_frost_point(t, rh)


def moist_air_gas_constant(mixing_ratio_kgkg):
    """Gas constant for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (J/(kg*K))
    """
    w = _to_gpu(mixing_ratio_kgkg)
    return _k_moist_air_gas_constant(w)


def moist_air_specific_heat_pressure(mixing_ratio_kgkg):
    """Specific heat at constant pressure for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (J/(kg*K))
    """
    w = _to_gpu(mixing_ratio_kgkg)
    return _k_moist_air_specific_heat_pressure(w)


def moist_air_poisson_exponent(mixing_ratio_kgkg):
    """Poisson exponent (kappa) for moist air.

    Parameters
    ----------
    mixing_ratio_kgkg : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    w = _to_gpu(mixing_ratio_kgkg)
    return _k_moist_air_poisson_exponent(w)


def water_latent_heat_vaporization(temperature):
    """Latent heat of vaporization (temperature-dependent).

    Parameters
    ----------
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    t = _to_gpu(temperature)
    return _k_water_latent_heat_vaporization(t)


def water_latent_heat_melting(temperature):
    """Latent heat of melting (temperature-dependent).

    Parameters
    ----------
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    t = _to_gpu(temperature)
    return _k_water_latent_heat_melting(t)


def water_latent_heat_sublimation(temperature):
    """Latent heat of sublimation (temperature-dependent).

    Parameters
    ----------
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    t = _to_gpu(temperature)
    return _k_water_latent_heat_sublimation(t)


def relative_humidity_wet_psychrometric(temperature, wet_bulb, pressure):
    """Relative humidity from dry-bulb, wet-bulb, and pressure.

    Parameters
    ----------
    temperature : array-like (Celsius)
    wet_bulb : array-like (Celsius)
    pressure : array-like (hPa)

    Returns
    -------
    cupy.ndarray (percent)
    """
    t = _to_gpu(temperature)
    tw = _to_gpu(wet_bulb)
    p = _to_gpu(pressure)
    es_tw = _k_saturation_vapor_pressure(tw)
    es_t = _k_saturation_vapor_pressure(t)
    e = es_tw - (0.000799 * p * (t - tw))
    rh = cp.where(es_t > 0.0, 100.0 * e / es_t, 0.0)
    return cp.clip(rh, 0.0, 100.0)


def psychrometric_vapor_pressure(temperature, wet_bulb, pressure):
    """Psychrometric vapor pressure from dry-bulb, wet-bulb, and pressure.

    Parameters
    ----------
    temperature : array-like (Celsius)
    wet_bulb : array-like (Celsius)
    pressure : array-like (hPa)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    t = _to_gpu(temperature)
    tw = _to_gpu(wet_bulb)
    p = _to_gpu(pressure)
    return _k_psychrometric_vapor_pressure(t, tw, p)


def psychrometric_vapor_pressure_wet(temperature, wet_bulb, pressure):
    """Alias for psychrometric_vapor_pressure."""
    return psychrometric_vapor_pressure(temperature, wet_bulb, pressure)


def add_height_to_pressure(pressure, delta_height):
    """New pressure after ascending/descending by a height increment.

    Parameters
    ----------
    pressure : array-like (hPa)
    delta_height : array-like (m)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    p = _to_gpu(pressure)
    dh = _to_gpu(delta_height)
    return _k_add_height_to_pressure(p, dh)


def add_pressure_to_height(height, delta_pressure):
    """New height after a pressure increment.

    Parameters
    ----------
    height : array-like (m)
    delta_pressure : array-like (hPa)

    Returns
    -------
    cupy.ndarray (m)
    """
    h = _to_gpu(height)
    dp = _to_gpu(delta_pressure)
    return _k_add_pressure_to_height(h, dp)


def thickness_hydrostatic(pressure_or_bottom, temperature_or_top, t_mean=None,
                          mixing_ratio=None,
                          molecular_weight_ratio=0.6219569100577033,
                          bottom=None, depth=None):
    """Hypsometric thickness.

    Supports both scalar form ``thickness_hydrostatic(p_bottom, p_top, t_mean)``
    and profile form ``thickness_hydrostatic(pressure, temperature, ...)``.

    Returns
    -------
    cupy.ndarray or float (m)
    """
    if t_mean is not None:
        p_bot = _to_gpu(pressure_or_bottom)
        p_top_val = _to_gpu(temperature_or_top)
        tm = _to_gpu(t_mean)
        return _k_thickness_hydrostatic(p_bot, p_top_val, tm)
    # Profile form
    p = _1d(pressure_or_bottom)
    t = _1d(temperature_or_top)
    if mixing_ratio is not None:
        w = _1d(mixing_ratio)
        eps = molecular_weight_ratio
        t = t * (1.0 + w / eps) / (1.0 + w)
    if float(p[0]) < float(p[-1]):
        p = p[::-1]
        t = t[::-1]
    return -(287.05 / 9.80665) * cp.trapz(t, cp.log(p))


def thickness_hydrostatic_from_relative_humidity(pressure, temperature,
                                                 relative_humidity):
    """Hypsometric thickness from pressure, temperature, and relative humidity.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent 0-100)

    Returns
    -------
    cupy.ndarray (m)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    rh = _1d(relative_humidity)
    return _STUB_thickness_from_rh(p, t, rh)


def scale_height(temperature):
    """Atmospheric scale height.

    Parameters
    ----------
    temperature : array-like (K)

    Returns
    -------
    cupy.ndarray (m)
    """
    t = _to_gpu(temperature)
    return _k_scale_height(t)


def geopotential_to_height(geopotential):
    """Convert geopotential to height.

    Parameters
    ----------
    geopotential : array-like (m^2/s^2)

    Returns
    -------
    cupy.ndarray (m)
    """
    gp = _to_gpu(geopotential)
    return _k_geopotential_to_height(gp)


def height_to_geopotential(height):
    """Convert height to geopotential.

    Parameters
    ----------
    height : array-like (m)

    Returns
    -------
    cupy.ndarray (m^2/s^2)
    """
    h = _to_gpu(height)
    return _k_height_to_geopotential(h)


def weighted_continuous_average(values, weights):
    """Trapezoidal weighted average over a coordinate.

    Parameters
    ----------
    values : array-like
    weights : array-like

    Returns
    -------
    float
    """
    v = _1d(values)
    w = _1d(weights)
    # Trapezoidal rule: sum((v[i]+v[i+1])/2 * (w[i+1]-w[i])) / (w[-1]-w[0])
    dw = w[1:] - w[:-1]
    avg_v = (v[:-1] + v[1:]) / 2.0
    return float(cp.sum(avg_v * dw) / cp.sum(dw))


def get_perturbation(values):
    """Anomaly (perturbation) from the mean.

    Parameters
    ----------
    values : array-like

    Returns
    -------
    cupy.ndarray
    """
    v = _to_gpu(values)
    return v - cp.mean(v)


# ===========================================================================
# Standard atmosphere and comfort indices
# ===========================================================================

def pressure_to_height_std(pressure):
    """Convert pressure to height using the US Standard Atmosphere 1976.

    Parameters
    ----------
    pressure : array-like (hPa)

    Returns
    -------
    cupy.ndarray (m)
    """
    p = _to_gpu(pressure)
    baro_exp = 9.80665 * 0.0289644 / (8.31447 * 0.0065)
    return (288.0 / 0.0065) * (1.0 - (p / 1013.25) ** (1.0 / baro_exp))


def height_to_pressure_std(height):
    """Convert height to pressure using the US Standard Atmosphere 1976.

    Parameters
    ----------
    height : array-like (m)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    h = _to_gpu(height)
    baro_exp = 9.80665 * 0.0289644 / (8.31447 * 0.0065)
    return 1013.25 * (1.0 - 0.0065 * h / 288.0) ** baro_exp


def altimeter_to_station_pressure(altimeter, elevation):
    """Convert altimeter setting to station pressure.

    Parameters
    ----------
    altimeter : array-like (hPa)
    elevation : array-like (m)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    a = _to_gpu(altimeter)
    e = _to_gpu(elevation)
    baro_exp = 9.80665 * 0.0289644 / (8.31447 * 0.0065)
    n = 1.0 / baro_exp
    return (a ** n - 1013.25 ** n * 0.0065 * e / 288.0) ** (1.0 / n) + 0.3


def station_to_altimeter_pressure(station_pressure, elevation):
    """Convert station pressure to altimeter setting.

    Parameters
    ----------
    station_pressure : array-like (hPa)
    elevation : array-like (m)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    s = _to_gpu(station_pressure)
    e = _to_gpu(elevation)
    baro_exp = 9.80665 * 0.0289644 / (8.31447 * 0.0065)
    n = 1.0 / baro_exp
    return ((s - 0.3) ** n + 1013.25 ** n * 0.0065 * e / 288.0) ** (1.0 / n)


def altimeter_to_sea_level_pressure(altimeter, elevation, temperature):
    """Convert altimeter setting to sea-level pressure.

    Parameters
    ----------
    altimeter : array-like (hPa)
    elevation : array-like (m)
    temperature : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    a = _to_gpu(altimeter)
    e = _to_gpu(elevation)
    t = _to_gpu(temperature)
    p_stn = altimeter_to_station_pressure(a, e)
    t_mean_k = t + 273.15 + 0.5 * 0.0065 * e
    return p_stn * cp.exp(9.80665 * e / (287.058 * t_mean_k))


def sigma_to_pressure(sigma, psfc, ptop):
    """Convert a sigma coordinate to pressure.

    Parameters
    ----------
    sigma : float
    psfc : float (hPa)
    ptop : float (hPa)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    s = _to_gpu(sigma)
    ps = _to_gpu(psfc)
    pt = _to_gpu(ptop)
    return s * (ps - pt) + pt


def heat_index(temperature, relative_humidity):
    """Heat index (NWS Rothfusz regression).

    Parameters
    ----------
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    rh = _to_gpu(relative_humidity)
    return _k_heat_index(t, rh)


def windchill(temperature, wind_speed_val):
    """Wind chill index (NWS formula).

    Parameters
    ----------
    temperature : array-like (Celsius)
    wind_speed_val : array-like (m/s)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    ws = _to_gpu(wind_speed_val)
    return _k_windchill(t, ws)


def apparent_temperature(temperature, relative_humidity, wind_speed_val):
    """Apparent temperature combining heat index and wind chill.

    Parameters
    ----------
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent)
    wind_speed_val : array-like (m/s)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    t = _to_gpu(temperature)
    rh = _to_gpu(relative_humidity)
    ws = _to_gpu(wind_speed_val)
    return _k_apparent_temperature(t, rh, ws)


# ===========================================================================
# Sounding / profile functions
# ===========================================================================

def lfc(pressure, temperature, dewpoint, parcel_temperature_profile=None,
        dewpoint_start=None, which="top"):
    """Level of Free Convection.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        LFC pressure (hPa) and parcel temperature at the LFC (Celsius).
    """
    p = cp.asnumpy(_1d(pressure)).astype(np.float64, copy=False)
    t = cp.asnumpy(_1d(temperature)).astype(np.float64, copy=False)
    td = cp.asnumpy(_1d(dewpoint)).astype(np.float64, copy=False)
    if p.size == 0:
        nan = cp.asarray([np.nan], dtype=cp.float64)
        return nan, nan

    p_lcl, _, parcel_tv = _lift_parcel_profile_virtual_temperature_numpy(p, t, td)

    for i in range(1, len(p)):
        if p[i] > p_lcl:
            continue
        tv_env_prev = _virtual_temp_from_dewpoint_c(t[i - 1], p[i - 1], td[i - 1])
        tv_env = _virtual_temp_from_dewpoint_c(t[i], p[i], td[i])
        buoy_prev = parcel_tv[i - 1] - tv_env_prev
        buoy = parcel_tv[i] - tv_env
        if buoy_prev <= 0.0 and buoy > 0.0:
            frac = (0.0 - buoy_prev) / (buoy - buoy_prev)
            p_lfc = p[i - 1] + frac * (p[i] - p[i - 1])
            t_lfc = t[i - 1] + frac * (t[i] - t[i - 1])
            return cp.asarray([p_lfc], dtype=cp.float64), cp.asarray([t_lfc], dtype=cp.float64)
        if buoy > 0.0 and p[i] <= p_lcl and p[i - 1] > p_lcl:
            return cp.asarray([p[i]], dtype=cp.float64), cp.asarray([t[i]], dtype=cp.float64)

    nan = cp.asarray([np.nan], dtype=cp.float64)
    return nan, nan


def el(pressure, temperature, dewpoint, parcel_temperature_profile=None,
       which="top"):
    """Equilibrium Level.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        EL pressure (hPa) and parcel temperature at the EL (Celsius).
    """
    p = cp.asnumpy(_1d(pressure)).astype(np.float64, copy=False)
    t = cp.asnumpy(_1d(temperature)).astype(np.float64, copy=False)
    td = cp.asnumpy(_1d(dewpoint)).astype(np.float64, copy=False)
    if p.size == 0:
        nan = cp.asarray([np.nan], dtype=cp.float64)
        return nan, nan

    p_lcl, _, parcel_tv = _lift_parcel_profile_virtual_temperature_numpy(p, t, td)
    found_positive = False
    crossings = []

    for i in range(1, len(p)):
        if p[i] > p_lcl:
            continue
        tv_env_prev = _virtual_temp_from_dewpoint_c(t[i - 1], p[i - 1], td[i - 1])
        tv_env = _virtual_temp_from_dewpoint_c(t[i], p[i], td[i])
        buoy_prev = parcel_tv[i - 1] - tv_env_prev
        buoy = parcel_tv[i] - tv_env

        if buoy > 0.0:
            found_positive = True

        if found_positive and buoy_prev > 0.0 and buoy <= 0.0:
            frac = (0.0 - buoy_prev) / (buoy - buoy_prev)
            p_el = p[i - 1] + frac * (p[i] - p[i - 1])
            t_el = t[i - 1] + frac * (t[i] - t[i - 1])
            crossings.append((p_el, t_el))

    if not crossings:
        nan = cp.asarray([np.nan], dtype=cp.float64)
        return nan, nan

    which_key = str(which).lower()
    p_el, t_el = crossings[0] if which_key == "bottom" else crossings[-1]
    return cp.asarray([p_el], dtype=cp.float64), cp.asarray([t_el], dtype=cp.float64)


def lcl(pressure, temperature, dewpoint):
    """Lifting Condensation Level.

    Parameters
    ----------
    pressure : float (hPa)
    temperature : float (Celsius)
    dewpoint : float (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        LCL pressure (hPa) and temperature (Celsius).
    """
    p = _scalar(pressure)
    t = _scalar(temperature)
    td = _scalar(dewpoint)
    return _k_lcl(p, t, td)


def cape_cin(pressure, temperature, dewpoint, parcel_profile_or_height=None,
             *args, parcel_profile=None, which_lfc="bottom", which_el="top",
             parcel_type="sb", ml_depth=100.0, mu_depth=300.0, top_m=None,
             **kwargs):
    """CAPE and CIN for a sounding.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)
    parcel_profile_or_height : array-like, optional
    parcel_type : str
    ml_depth : float
    mu_depth : float
    top_m : float, optional

    Returns
    -------
    tuple of cupy.ndarray
        CAPE (J/kg), CIN (J/kg), LCL_p, LFC_p, EL_p.
    """
    psfc = kwargs.pop("psfc", None)
    t2m = kwargs.pop("t2m", None)
    td2m = kwargs.pop("td2m", None)
    if kwargs:
        raise TypeError(f"Unexpected keyword arguments: {sorted(kwargs)}")

    if parcel_profile is not None:
        parcel_profile_or_height = parcel_profile
    if len(args) >= 3:
        psfc, t2m, td2m = args[:3]

    p = cp.asnumpy(_1d(pressure)).astype(np.float64, copy=False)
    t = cp.asnumpy(_1d(temperature)).astype(np.float64, copy=False)
    td = cp.asnumpy(_1d(dewpoint)).astype(np.float64, copy=False)

    fourth = parcel_profile_or_height
    metpy_profile_form = False
    if fourth is not None:
        fourth_arr = np.asarray(strip_units(fourth), dtype=np.float64)
        if parcel_profile is not None or (fourth_arr.size == p.size and np.nanmax(np.abs(fourth_arr)) < 150.0):
            metpy_profile_form = True

    if metpy_profile_form:
        t_parcel = np.asarray(strip_units(fourth), dtype=np.float64).ravel()
        cape_val, cin_val = _parcel_profile_cape_cin_numpy(p, t, td, t_parcel)
        return cp.asarray([cape_val], dtype=cp.float64), cp.asarray([cin_val], dtype=cp.float64)

    if fourth is not None:
        h = np.asarray(strip_units(fourth), dtype=np.float64).ravel()
    else:
        h = _standard_agl_heights_from_pressure(p)
        if psfc is None:
            psfc = pressure[0] if hasattr(pressure, "__getitem__") else pressure
        if t2m is None:
            t2m = temperature[0] if hasattr(temperature, "__getitem__") else temperature
        if td2m is None:
            td2m = dewpoint[0] if hasattr(dewpoint, "__getitem__") else dewpoint

    ps = _scalar(psfc if psfc is not None else p[0])
    t2 = _scalar(t2m if t2m is not None else t[0])
    td2 = _scalar(td2m if td2m is not None else td[0])
    cape_val, cin_val, h_lcl, h_lfc = _cape_cin_core_compatible(
        p, t, td, h, ps, t2, td2,
        parcel_type=parcel_type,
        ml_depth=float(ml_depth),
        mu_depth=float(mu_depth),
        top_m=top_m,
    )
    return (
        cp.asarray([cape_val], dtype=cp.float64),
        cp.asarray([cin_val], dtype=cp.float64),
        cp.asarray([h_lcl], dtype=cp.float64),
        cp.asarray([h_lfc], dtype=cp.float64),
    )


def surface_based_cape_cin(pressure, temperature, dewpoint):
    """Surface-based CAPE and CIN.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        CAPE (J/kg) and CIN (J/kg).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    t_np = cp.asnumpy(t).astype(np.float64, copy=False)
    td_np = cp.asnumpy(td).astype(np.float64, copy=False)
    cape, cin, _, _, _ = _cape_cin_from_parcel_state(
        p_np,
        t_np,
        td_np,
        float(p_np[0]),
        float(t_np[0]),
        float(td_np[0]),
    )
    return cp.asarray([cape], dtype=cp.float64), cp.asarray([cin], dtype=cp.float64)


def mixed_layer_cape_cin(pressure, temperature, dewpoint, depth=100.0):
    """Mixed-layer CAPE and CIN.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        CAPE (J/kg) and CIN (J/kg).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    d = _scalar(depth)
    parcel_p, parcel_t, parcel_td = get_mixed_layer_parcel(p, t, td, d)
    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    t_np = cp.asnumpy(t).astype(np.float64, copy=False)
    td_np = cp.asnumpy(td).astype(np.float64, copy=False)
    cape, cin, _, _, _ = _cape_cin_from_parcel_state(
        p_np,
        t_np,
        td_np,
        float(cp.asnumpy(cp.asarray(parcel_p))),
        float(cp.asnumpy(cp.asarray(parcel_t))),
        float(cp.asnumpy(cp.asarray(parcel_td))),
    )
    return cp.asarray([cape], dtype=cp.float64), cp.asarray([cin], dtype=cp.float64)


def most_unstable_cape_cin(pressure, temperature, dewpoint, depth=300, **kwargs):
    """Most-unstable CAPE and CIN.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        CAPE (J/kg) and CIN (J/kg).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    d = _scalar(depth)
    mu_p, mu_t, mu_td, mu_idx = _STUB_get_most_unstable_parcel(p, t, td, d)
    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    t_np = cp.asnumpy(t).astype(np.float64, copy=False)
    td_np = cp.asnumpy(td).astype(np.float64, copy=False)
    cape, cin, _, _, _ = _cape_cin_from_parcel_state(
        p_np,
        t_np,
        td_np,
        float(cp.asnumpy(cp.asarray(mu_p))),
        float(cp.asnumpy(cp.asarray(mu_t))),
        float(cp.asnumpy(cp.asarray(mu_td))),
    )
    return cp.asarray([cape], dtype=cp.float64), cp.asarray([cin], dtype=cp.float64)


def downdraft_cape(pressure, temperature, dewpoint):
    """Downdraft CAPE (DCAPE).

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    return _k_downdraft_cape(p, t, td)


def showalter_index(pressure, temperature, dewpoint):
    """Showalter Index.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    p = cp.asnumpy(_1d(pressure)).astype(np.float64, copy=False)
    t = cp.asnumpy(_1d(temperature)).astype(np.float64, copy=False)
    td = cp.asnumpy(_1d(dewpoint)).astype(np.float64, copy=False)
    t850 = _log_interp_pressure_value(850.0, p, t)
    td850 = _log_interp_pressure_value(850.0, p, td)
    t500_env = _log_interp_pressure_value(500.0, p, t)
    p_lcl, t_lcl = _drylift_cpu(850.0, t850, td850)
    thetam = _thetam_from_lcl(p_lcl, t_lcl)
    t500_parcel = _satlift_c(500.0, thetam)
    return t500_env - t500_parcel


def k_index(*args, vertical_dim=0):
    """K-Index.

    Called as ``k_index(t850, td850, t700, td700, t500)``.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) == 5:
        t850, td850, t700, td700, t500 = [_to_gpu(a) for a in args]
        return _k_k_index(t850, t700, t500, td850, td700)
    elif len(args) == 3:
        p, t, td = [_1d(a) for a in args]
        return _k_k_index(p, t, td)
    raise TypeError("k_index expects (t850, td850, t700, td700, t500) or (pressure, temperature, dewpoint)")


def total_totals(*args, vertical_dim=0):
    """Total Totals Index.

    Called as ``total_totals(t850, td850, t500)``.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) == 3:
        t850, td850, t500 = [_to_gpu(a) for a in args]
        return _k_total_totals(t850, t500, td850)
    raise TypeError("total_totals expects (t850, td850, t500)")


total_totals_index = total_totals


def cross_totals(*args, vertical_dim=0):
    """Cross Totals: Td850 - T500.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) in (2, 3):
        return _k_cross_totals(*[_to_gpu(a) for a in args])
    raise TypeError("cross_totals expects (pressure, temperature, dewpoint) or (td850, t500)")


def vertical_totals(*args, vertical_dim=0):
    """Vertical Totals: T850 - T500.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) == 2:
        return _k_vertical_totals(*[_to_gpu(a) for a in args])
    raise TypeError("vertical_totals expects (pressure, temperature) or (t850, t500)")


def sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500):
    """SWEAT Index.

    Parameters
    ----------
    t850, td850, t500 : float (Celsius)
    dd850, dd500 : float (degrees)
    ff850, ff500 : float (knots)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    from metcu.kernels.wind import sweat_index_direct as _k_sweat_index_direct
    return _k_sweat_index_direct(
        _to_gpu(t850), _to_gpu(td850), _to_gpu(t500),
        _to_gpu(dd850), _to_gpu(dd500), _to_gpu(ff850), _to_gpu(ff500),
    )


def lifted_index(pressure, temperature, dewpoint):
    """Lifted Index.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    return _k_lifted_index(p, t, td)


def ccl(pressure, temperature, dewpoint):
    """Convective Condensation Level.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) or None
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    return _k_ccl(p, t, td)


def precipitable_water(pressure, dewpoint):
    """Precipitable water.

    Parameters
    ----------
    pressure : array-like (hPa)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (mm)
    """
    p = _1d(pressure)
    td = _1d(dewpoint)
    return _k_precipitable_water(p, td)


def brunt_vaisala_frequency(height, potential_temp):
    """Brunt-Vaisala frequency at each level.

    Parameters
    ----------
    height : array-like (m)
    potential_temp : array-like (K)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    z = _1d(height)
    theta = _1d(potential_temp)
    return _k_brunt_vaisala_frequency(z, theta)


def brunt_vaisala_period(height, potential_temp):
    """Brunt-Vaisala period at each level.

    Parameters
    ----------
    height : array-like (m)
    potential_temp : array-like (K)

    Returns
    -------
    cupy.ndarray (s)
    """
    z = _1d(height)
    theta = _1d(potential_temp)
    bvf = _k_brunt_vaisala_frequency(z, theta)
    return _k_brunt_vaisala_period(bvf)


def brunt_vaisala_frequency_squared(height, potential_temp):
    """Brunt-Vaisala frequency squared (N^2) at each level.

    Parameters
    ----------
    height : array-like (m)
    potential_temp : array-like (K)

    Returns
    -------
    cupy.ndarray (1/s^2)
    """
    z = _1d(height)
    theta = _1d(potential_temp)
    return _k_brunt_vaisala_frequency_squared(z, theta)


def static_stability(pressure, temperature):
    """Static stability.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (K)

    Returns
    -------
    cupy.ndarray (K/Pa)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    return _k_static_stability(p, t)


def parcel_profile_with_lcl(pressure, t_surface, td_surface):
    """Parcel temperature profile with the LCL level inserted.

    Parameters
    ----------
    pressure : array-like (hPa)
    t_surface : float (Celsius)
    td_surface : float (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        Pressure levels (with LCL) and parcel temperatures.
    """
    p = _1d(pressure)
    t = _scalar(t_surface)
    td = _scalar(td_surface)
    return _STUB_parcel_profile_with_lcl(p, t, td)


def get_layer(pressure, *args, p_bottom=None, p_top=None,
              bottom=None, depth=None, interpolate=True):
    """Extract one or more fields from a sounding layer.

    Parameters
    ----------
    pressure : array-like (hPa)
    *args : array-like
        One or more value profiles.
    p_bottom, p_top : float, optional
    bottom, depth : float, optional

    Returns
    -------
    tuple of cupy.ndarray
    """
    p = _1d(pressure)
    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    value_arrays = [_1d(a) for a in args]
    pb = _scalar(p_bottom) if p_bottom is not None else (_scalar(bottom) if bottom is not None else float(p_np[0]))
    if p_top is not None:
        pt = _scalar(p_top)
    elif depth is not None:
        pt = pb - _scalar(depth)
    else:
        raise TypeError("get_layer requires either p_top or depth")

    results = []
    p_layer = None
    for v_arr in value_arrays:
        v_np = cp.asnumpy(v_arr).astype(np.float64, copy=False)
        p_l, v_l = _extract_layer_1d(p_np, v_np, pb, pt, interpolate=interpolate)
        if p_layer is None:
            p_layer = cp.asarray(p_l, dtype=cp.float64)
        results.append(cp.asarray(v_l, dtype=cp.float64))
    if p_layer is None:
        p_layer = p
    if len(results) == 1:
        return p_layer, results[0]
    return (p_layer,) + tuple(results)


def get_layer_heights(pressure, heights, p_bottom, p_top):
    """Extract layer heights between two pressures.

    Parameters
    ----------
    pressure : array-like (hPa)
    heights : array-like (m)
    p_bottom : float (hPa)
    p_top : float (hPa)

    Returns
    -------
    tuple of (float, float, float) -- (p_layer, h_bottom, h_top)
    """
    return get_layer(pressure, heights, p_bottom=p_bottom, p_top=p_top, interpolate=True)


def mixed_layer(pressure, *args, height=None, bottom=None, depth=100.0,
                interpolate=True):
    """Mixed-layer mean of one or more profiles.

    Parameters
    ----------
    pressure : array-like (hPa)
    *args : array-like (one or two profiles)
    depth : float (hPa)

    Returns
    -------
    float or tuple of floats
    """
    p = _1d(pressure)
    d = _scalar(depth)
    if len(args) == 1:
        # Single profile: pass it as both T and Td, return only first result
        v = _1d(args[0])
        t_ml, td_ml = _k_mixed_layer(p, v, v, d)
        return t_ml
    elif len(args) >= 2:
        t = _1d(args[0])
        td = _1d(args[1])
        return _k_mixed_layer(p, t, td, d)
    raise TypeError("mixed_layer requires at least one profile argument")


def mean_pressure_weighted(pressure, values):
    """Pressure-weighted mean of a quantity.

    Parameters
    ----------
    pressure : array-like (hPa)
    values : array-like
    Returns
    -------
    float
    """
    p = _1d(pressure)
    v = _1d(values)
    pb = float(cp.asnumpy(p[0]))
    pt = float(cp.asnumpy(p[-1]))
    # 1D case: pressure-weighted average using trapezoidal rule
    p_np = cp.asnumpy(p)
    v_np = cp.asnumpy(v)
    mask = (p_np <= pb) & (p_np >= pt)
    if mask.sum() < 2:
        return float(v_np[mask].mean()) if mask.sum() > 0 else 0.0
    p_sub = p_np[mask]
    v_sub = v_np[mask]
    dp = np.abs(np.diff(p_sub))
    avg_v = (v_sub[:-1] + v_sub[1:]) / 2.0
    return float(np.sum(avg_v * dp) / np.sum(dp))


def get_mixed_layer_parcel(pressure, temperature, dewpoint, depth=100.0):
    """Get mixed-layer parcel properties.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray)
        Parcel pressure, temperature, dewpoint.
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    d = _scalar(depth)
    return _STUB_get_mixed_layer_parcel(p, t, td, d)


def get_most_unstable_parcel(pressure, temperature, dewpoint,
                             height=None, bottom=None, depth=300.0):
    """Get most-unstable parcel properties.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray, int)
        Parcel pressure, temperature, dewpoint, source index.
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint)
    d = _scalar(depth)
    return _STUB_get_most_unstable_parcel(p, t, td, d)


def mixed_parcel(pressure, temperature, dewpoint, parcel_start_pressure=None,
                 height=None, bottom=None, depth=100, interpolate=True):
    """Mixed-layer parcel (MetPy-compatible alias).

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray)
    """
    d = _scalar(depth)
    return get_mixed_layer_parcel(pressure, temperature, dewpoint, d)


def most_unstable_parcel(pressure, temperature, dewpoint, height=None,
                         bottom=None, depth=300):
    """Alias for get_most_unstable_parcel."""
    return get_most_unstable_parcel(pressure, temperature, dewpoint,
                                    height=height, bottom=bottom, depth=depth)


def isentropic_interpolation(theta_levels, pressure_3d, temperature_3d,
                              fields, nx=None, ny=None, nz=None):
    """Interpolate fields to isentropic surfaces.

    Parameters
    ----------
    theta_levels : array-like (K) -- target theta values
    pressure_3d : 3-D array (hPa)
    temperature_3d : 3-D array (K) -- this is treated as potential temperature
    fields : list of 3-D arrays to interpolate
    nx, ny, nz : int, optional

    Returns
    -------
    list of cupy.ndarray, each shaped (n_theta, ny, nx)
    """
    theta_targets = _1d(theta_levels)
    p_arr = _to_gpu(pressure_3d)
    theta_3d = _to_gpu(temperature_3d)  # assumed to be potential temperature
    field_list = [_to_gpu(f) for f in fields]
    # Kernel interpolates one field at a time
    results = []
    for f in field_list:
        result = _k_isentropic_interpolation(theta_3d, p_arr, f, theta_targets)
        results.append(result)
    return results


def montgomery_streamfunction(height_or_theta, temperature_or_pressure=None,
                              temperature=None, height=None):
    """Montgomery streamfunction on isentropic surfaces.

    Parameters
    ----------
    height : array-like (m)
    temperature : array-like (K)

    Returns
    -------
    cupy.ndarray (J/kg or kJ/kg)
    """
    if temperature_or_pressure is not None and temperature is None and height is None:
        h = _to_gpu(height_or_theta)
        t = _to_gpu(temperature_or_pressure)
        cp_d = 1004.6662184201462
        g = 9.80665
        return (cp_d * t + g * h) / 1000.0
    elif temperature is not None and height is not None:
        th = _to_gpu(height_or_theta)
        p = _to_gpu(temperature_or_pressure)
        t = _to_gpu(temperature)
        h = _to_gpu(height)
        return _k_montgomery_streamfunction(th, p, t, h)
    raise TypeError(
        "montgomery_streamfunction expects (height, temperature) or "
        "(theta, pressure, temperature, height)"
    )


def find_intersections(x, y1, y2):
    """Find intersections of two curves.

    Parameters
    ----------
    x : array-like
    y1, y2 : array-like

    Returns
    -------
    list of (x, y) tuples
    """
    # CPU-only utility: transfer to CPU and compute
    x_arr = cp.asnumpy(_1d(x))
    y1_arr = cp.asnumpy(_1d(y1))
    y2_arr = cp.asnumpy(_1d(y2))
    try:
        from metrust.calc import find_intersections as _find
        return _find(x_arr, y1_arr, y2_arr)
    except ImportError:
        # Simple fallback: find sign changes in y1-y2
        diff = y1_arr - y2_arr
        result = []
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] < 0:
                frac = diff[i] / (diff[i] - diff[i + 1])
                xi = x_arr[i] + frac * (x_arr[i + 1] - x_arr[i])
                yi = y1_arr[i] + frac * (y1_arr[i + 1] - y1_arr[i])
                result.append((xi, yi))
        return result


def convective_inhibition_depth(pressure, temperature, dewpoint):
    """Convective inhibition depth.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    p = cp.asnumpy(_1d(pressure)).astype(np.float64, copy=False)
    t = cp.asnumpy(_1d(temperature)).astype(np.float64, copy=False)
    td = cp.asnumpy(_1d(dewpoint)).astype(np.float64, copy=False)
    if len(p) == 0:
        return 0.0
    p_sfc = p[0]
    t_sfc = t[0]
    td_sfc = td[0]
    p_lcl, t_lcl = _drylift_cpu(p_sfc, t_sfc, td_sfc)
    thetam = _thetam_from_lcl(p_lcl, t_lcl)
    for k in range(len(p)):
        if p[k] > p_lcl:
            continue
        tv_env = _virtual_temp_from_dewpoint_c(t[k], p[k], td[k])
        t_parcel = _satlift_c(p[k], thetam)
        tv_parcel = _virtual_temp_from_dewpoint_c(t_parcel, p[k], t_parcel)
        if tv_parcel > tv_env:
            return float(p_sfc - p[k])
    return float(p_sfc - p[-1])


# ===========================================================================
# Wind / kinematics (1-D profile functions)
# ===========================================================================

def wind_speed(u, v):
    """Wind speed from (u, v) components.

    Parameters
    ----------
    u, v : array-like (m/s)

    Returns
    -------
    cupy.ndarray (m/s)
    """
    u_arr = _to_gpu(u)
    v_arr = _to_gpu(v)
    return _k_wind_speed(u_arr, v_arr)


def wind_direction(u, v):
    """Meteorological wind direction from (u, v).

    Parameters
    ----------
    u, v : array-like (m/s)

    Returns
    -------
    cupy.ndarray (degrees)
    """
    u_arr = _to_gpu(u)
    v_arr = _to_gpu(v)
    return _k_wind_direction(u_arr, v_arr)


def wind_components(speed, direction):
    """Convert (speed, direction) to (u, v) components.

    Parameters
    ----------
    speed : array-like (m/s)
    direction : array-like (degrees)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    spd = _to_gpu(speed)
    dirn = _to_gpu(direction)
    return _k_wind_components(spd, dirn)


def bulk_shear(pressure_or_u, u_or_v, v_or_height=None, height=None,
               bottom=None, depth=None, top=None):
    """Bulk wind shear over a height layer.

    Parameters
    ----------
    pressure_or_u : array-like
    u_or_v : array-like
    v_or_height : array-like, optional
    height : array-like (m), optional
    bottom : float (m), optional
    depth : float (m), optional
    top : float (m), optional

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    if height is not None:
        u_arr = _1d(u_or_v)
        v_arr = _1d(v_or_height)
        h_arr = _1d(height)
    else:
        u_arr = _1d(pressure_or_u)
        v_arr = _1d(u_or_v)
        h_arr = _1d(v_or_height)
    bot = _scalar(bottom) if bottom is not None else float(h_arr[0])
    if top is not None:
        top_val = _scalar(top)
    elif depth is not None:
        top_val = bot + _scalar(depth)
    else:
        top_val = float(h_arr[-1])
    return _k_bulk_shear(u_arr, v_arr, h_arr, bot, top_val)


def mean_wind(u, v, height, bottom, top):
    """Mean wind over a height layer.

    Parameters
    ----------
    u, v : array-like (m/s)
    height : array-like (m)
    bottom, top : float (m)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    u_arr = _1d(u)
    v_arr = _1d(v)
    h_arr = _1d(height)
    bot = _scalar(bottom)
    top_val = _scalar(top)
    u_np = cp.asnumpy(u_arr).astype(np.float64, copy=False)
    v_np = cp.asnumpy(v_arr).astype(np.float64, copy=False)
    h_np = cp.asnumpy(h_arr).astype(np.float64, copy=False)
    return (
        _height_weighted_layer_average(u_np, h_np, bot, top_val),
        _height_weighted_layer_average(v_np, h_np, bot, top_val),
    )


def storm_relative_helicity(*args, bottom=None, depth=None,
                            storm_u=None, storm_v=None):
    """Storm-relative helicity.

    Parameters
    ----------
    height : array-like (m)
    u, v : array-like (m/s)
    depth : float (m)
    storm_u, storm_v : float (m/s)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray)
        Positive, negative, and total SRH (m^2/s^2).
    """
    if len(args) == 6:
        u, v, height_a, depth_a, storm_u, storm_v = args
    elif len(args) == 4:
        height_a, u, v, depth_a = args
        if depth is not None:
            depth_a = depth
    elif len(args) == 3:
        height_a, u, v = args
        depth_a = depth
    else:
        raise TypeError("storm_relative_helicity expects 3-6 positional args")
    u_arr = _1d(u)
    v_arr = _1d(v)
    h_arr = _1d(height_a)
    d = _scalar(depth_a)
    if storm_u is None or storm_v is None:
        (rm_u, rm_v), _, _ = bunkers_storm_motion(u_arr, v_arr, h_arr)
        if storm_u is None:
            storm_u = rm_u
        if storm_v is None:
            storm_v = rm_v
    su = _scalar(storm_u)
    sv = _scalar(storm_v)
    pos, neg, total = _storm_relative_helicity_numpy(
        cp.asnumpy(u_arr).astype(np.float64, copy=False),
        cp.asnumpy(v_arr).astype(np.float64, copy=False),
        cp.asnumpy(h_arr).astype(np.float64, copy=False),
        d,
        su,
        sv,
        bottom_m=_scalar(bottom) if bottom is not None else None,
    )
    return (
        cp.asarray([pos], dtype=cp.float64),
        cp.asarray([neg], dtype=cp.float64),
        cp.asarray([total], dtype=cp.float64),
    )


def bunkers_storm_motion(pressure_or_u, u_or_v, v_or_height, height=None):
    """Bunkers storm motion (right-mover, left-mover, mean wind).

    Parameters
    ----------
    pressure_or_u : array-like
    u_or_v : array-like
    v_or_height : array-like
    height : array-like (m), optional

    Returns
    -------
    tuple of 3 tuples, each (cupy.ndarray, cupy.ndarray)
    """
    if height is not None:
        p = _1d(pressure_or_u)
        u = _1d(u_or_v)
        v = _1d(v_or_height)
        h = _1d(height)
    else:
        u = _1d(pressure_or_u)
        v = _1d(u_or_v)
        h = _1d(v_or_height)
        p = height_to_pressure_std(h)

    p_np = cp.asnumpy(p).astype(np.float64, copy=False)
    u_np = cp.asnumpy(u).astype(np.float64, copy=False)
    v_np = cp.asnumpy(v).astype(np.float64, copy=False)
    h_np = cp.asnumpy(h).astype(np.float64, copy=False)

    z_sfc = float(h_np[0])
    mean_u = _pressure_weighted_height_average(p_np, u_np, h_np, z_sfc, z_sfc + 6000.0)
    mean_v = _pressure_weighted_height_average(p_np, v_np, h_np, z_sfc, z_sfc + 6000.0)
    u_500 = _pressure_weighted_height_average(p_np, u_np, h_np, z_sfc, z_sfc + 500.0)
    v_500 = _pressure_weighted_height_average(p_np, v_np, h_np, z_sfc, z_sfc + 500.0)
    u_5500 = _pressure_weighted_height_average(p_np, u_np, h_np, z_sfc + 5500.0, z_sfc + 6000.0)
    v_5500 = _pressure_weighted_height_average(p_np, v_np, h_np, z_sfc + 5500.0, z_sfc + 6000.0)
    shear_u = u_5500 - u_500
    shear_v = v_5500 - v_500
    shear_mag = float(np.hypot(shear_u, shear_v))
    if shear_mag < 1e-10:
        return (mean_u, mean_v), (mean_u, mean_v), (mean_u, mean_v)
    perp_u = shear_v / shear_mag
    perp_v = -shear_u / shear_mag
    deviation = 7.5
    return (
        (mean_u + deviation * perp_u, mean_v + deviation * perp_v),
        (mean_u - deviation * perp_u, mean_v - deviation * perp_v),
        (mean_u, mean_v),
    )


def corfidi_storm_motion(pressure_or_u, u_or_v, v_or_height, *args,
                         u_llj=None, v_llj=None):
    """Corfidi upwind and downwind vectors for MCS motion.

    Returns
    -------
    tuple of 2 tuples, each (cupy.ndarray, cupy.ndarray)
    """
    if len(args) != 2:
        raise TypeError("corfidi_storm_motion expects 5 positional args")
    u_850, v_850 = args
    u_arr = _1d(pressure_or_u)
    v_arr = _1d(u_or_v)
    h_arr = _1d(v_or_height)
    u8 = _scalar(u_850)
    v8 = _scalar(v_850)
    mw_u, mw_v = mean_wind(u_arr, v_arr, h_arr, 0.0, 6000.0)
    prop_u = mw_u - u8
    prop_v = mw_v - v8
    return (prop_u, prop_v), (mw_u + prop_u, mw_v + prop_v)


def friction_velocity(u, w):
    """Friction velocity from time series of u and w components.

    Parameters
    ----------
    u, w : array-like (m/s)

    Returns
    -------
    cupy.ndarray (m/s)
    """
    u_arr = _1d(u)
    w_arr = _1d(w)
    return _k_friction_velocity(u_arr, w_arr)


def tke(u, v, w):
    """Turbulent kinetic energy from wind component time series.

    Parameters
    ----------
    u, v, w : array-like (m/s)

    Returns
    -------
    cupy.ndarray (m^2/s^2)
    """
    u_arr = _1d(u)
    v_arr = _1d(v)
    w_arr = _1d(w)
    return _k_tke(u_arr, v_arr, w_arr)


def gradient_richardson_number(height, potential_temperature, u, v):
    """Gradient Richardson number at each level.

    Parameters
    ----------
    height : array-like (m)
    potential_temperature : array-like (K)
    u, v : array-like (m/s)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    z = _1d(height)
    theta = _1d(potential_temperature)
    u_arr = _1d(u)
    v_arr = _1d(v)
    return _k_gradient_richardson_number(z, theta, u_arr, v_arr)


def coriolis_parameter(latitude):
    """Coriolis parameter.

    Parameters
    ----------
    latitude : array-like (degrees)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    lat = _to_gpu(latitude)
    return _k_coriolis_parameter(lat)


# ===========================================================================
# 2-D grid kinematics
# ===========================================================================

def divergence(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
               parallel_scale=None, meridional_scale=None,
               latitude=None, longitude=None, crs=None):
    """Horizontal divergence on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dudx = _gpu_first_derivative_uniform_2d(u_arr, dx_val, axis=1)
    dvdy = _gpu_first_derivative_uniform_2d(v_arr, dy_val, axis=0)
    return dudx + dvdy


def vorticity(u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
              parallel_scale=None, meridional_scale=None,
              latitude=None, longitude=None, crs=None):
    """Relative vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dvdx = _gpu_first_derivative_uniform_2d(v_arr, dx_val, axis=1)
    dudy = _gpu_first_derivative_uniform_2d(u_arr, dy_val, axis=0)
    return dvdx - dudy


def absolute_vorticity(u, v, lats=None, dx=None, dy=None, latitude=None,
                       longitude=None, x_dim=-1, y_dim=-2, crs=None):
    """Absolute vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    lats : 2-D array (degrees)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    lat_source = lats if lats is not None else latitude
    lats_arr = _2d(lat_source)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    f_arr = _k_coriolis_parameter(lats_arr)
    return vorticity(u_arr, v_arr, dx=dx_val, dy=dy_val) + f_arr


def advection(scalar, *args, dx=None, dy=None, dz=None, x_dim=-1, y_dim=-2,
              vertical_dim=-3, parallel_scale=None, meridional_scale=None,
              latitude=None, longitude=None, crs=None):
    """Advection of a scalar field by a 2-D wind.

    Parameters
    ----------
    scalar : 2-D array-like
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray
    """
    if len(args) >= 2:
        u, v = args[0], args[1]
        if len(args) >= 4 and dx is None:
            dx = args[2]
            dy = args[3]
    else:
        raise TypeError("advection requires at least (scalar, u, v)")
    s_arr = _2d(scalar)
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dsdx = _gpu_first_derivative_uniform_2d(s_arr, dx_val, axis=1)
    dsdy = _gpu_first_derivative_uniform_2d(s_arr, dy_val, axis=0)
    return -(u_arr * dsdx + v_arr * dsdy)


def frontogenesis(theta, u, v, dx=None, dy=None, x_dim=-1, y_dim=-2,
                  parallel_scale=None, meridional_scale=None,
                  latitude=None, longitude=None, crs=None):
    """2-D Petterssen frontogenesis function.

    Parameters
    ----------
    theta : 2-D array-like (K)
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (K/m/s)
    """
    t_arr = _2d(theta)
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dtdx = _gpu_first_derivative_uniform_2d(t_arr, dx_val, axis=1)
    dtdy = _gpu_first_derivative_uniform_2d(t_arr, dy_val, axis=0)
    dudx = _gpu_first_derivative_uniform_2d(u_arr, dx_val, axis=1)
    dudy = _gpu_first_derivative_uniform_2d(u_arr, dy_val, axis=0)
    dvdx = _gpu_first_derivative_uniform_2d(v_arr, dx_val, axis=1)
    dvdy = _gpu_first_derivative_uniform_2d(v_arr, dy_val, axis=0)
    mag = cp.sqrt(dtdx * dtdx + dtdy * dtdy)
    mag = cp.where(mag < 1e-30, cp.nan, mag)
    return -(
        dtdx * dtdx * dudx
        + dtdy * dtdy * dvdy
        + dtdx * dtdy * (dvdx + dudy)
    ) / mag


def geostrophic_wind(heights, dx=None, dy=None, latitude=None, x_dim=-1,
                     y_dim=-2, parallel_scale=None, meridional_scale=None,
                     longitude=None, crs=None):
    """Geostrophic wind from geopotential height.

    Parameters
    ----------
    heights : 2-D array-like (m) -- geopotential height
    latitude : 2-D array-like (degrees)
    dx, dy : float (m)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    h_arr = _2d(heights)
    lats_arr = _2d(latitude)
    f_arr = _k_coriolis_parameter(lats_arr)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dZdx = _gpu_first_derivative_uniform_2d(h_arr, dx_val, axis=1)
    dZdy = _gpu_first_derivative_uniform_2d(h_arr, dy_val, axis=0)
    safe_f = cp.where(cp.abs(f_arr) < 1e-20, cp.nan, f_arr)
    ug = -(9.80665 / safe_f) * dZdy
    vg = (9.80665 / safe_f) * dZdx
    ug = cp.nan_to_num(ug)
    vg = cp.nan_to_num(vg)
    return ug, vg


def ageostrophic_wind(u, v, heights, lats, dx, dy):
    """Ageostrophic wind: total wind minus geostrophic wind.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    heights : 2-D array-like (m)
    lats : 2-D array-like (degrees)
    dx, dy : float (m)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    h_arr = _2d(heights)
    lats_arr = _2d(lats)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    ug, vg = geostrophic_wind(h_arr, dx=dx_val, dy=dy_val, latitude=lats_arr)
    return u_arr - ug, v_arr - vg


def potential_vorticity_baroclinic(potential_temp, pressure, *args, dx=None,
                                   dy=None, latitude=None, x_dim=-1, y_dim=-2,
                                   vertical_dim=-3, longitude=None, crs=None):
    """Baroclinic (Ertel) potential vorticity.

    Returns
    -------
    cupy.ndarray (K*m^2/(kg*s))
    """
    gpu_args = [_2d(a) for a in args]
    pt = _2d(potential_temp)
    p = _to_gpu(pressure)
    dx_val = _mean_spacing(dx) if dx is not None else None
    dy_val = _mean_spacing(dy) if dy is not None else None
    return _k_potential_vorticity_baroclinic(pt, p, gpu_args, dx_val, dy_val, latitude)


def potential_vorticity_barotropic(heights, u, v, lats, dx, dy):
    """Barotropic potential vorticity.

    Parameters
    ----------
    heights : 2-D array-like (m)
    u, v : 2-D array-like (m/s)
    lats : 2-D array-like (degrees)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/(m*s))
    """
    h_arr = _2d(heights)
    u_arr = _2d(u)
    v_arr = _2d(v)
    lats_arr = _2d(lats)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    f_arr = _k_coriolis_parameter(lats_arr)
    zeta = vorticity(u_arr, v_arr, dx=dx_val, dy=dy_val)
    depth = cp.where(cp.abs(h_arr) > 1e-10, h_arr, cp.nan)
    out = (f_arr + zeta) / depth
    return cp.nan_to_num(out)


def normal_component(u, v, start, end):
    """Normal (perpendicular) component of wind relative to a cross-section.

    Parameters
    ----------
    u, v : array-like (m/s)
    start, end : tuple of (lat, lon)

    Returns
    -------
    cupy.ndarray (m/s)
    """
    u_arr = _1d(u)
    v_arr = _1d(v)
    return _k_normal_component(u_arr, v_arr, start, end)


def tangential_component(u, v, start, end):
    """Tangential (parallel) component of wind relative to a cross-section.

    Parameters
    ----------
    u, v : array-like (m/s)
    start, end : tuple of (lat, lon)

    Returns
    -------
    cupy.ndarray (m/s)
    """
    u_arr = _1d(u)
    v_arr = _1d(v)
    return _k_tangential_component(u_arr, v_arr, start, end)


def unit_vectors_from_cross_section(start, end):
    """Tangent and normal unit vectors for a cross-section line.

    Parameters
    ----------
    start, end : tuple of (lat, lon)

    Returns
    -------
    tuple of ((east, north), (east, north))
    """
    dlat = end[0] - start[0]
    dlon = end[1] - start[1]
    mag = np.sqrt(dlat**2 + dlon**2)
    if mag < 1e-12:
        return (0.0, 0.0), (0.0, 0.0)
    tang = (dlon / mag, dlat / mag)
    norm = (-dlat / mag, dlon / mag)
    return tang, norm


def vector_derivative(u, v, dx, dy):
    """All four partial derivatives of a 2-D vector field.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    tuple of four cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    return _STUB_vector_derivative(u_arr, v_arr, dx_val, dy_val)


def absolute_momentum(u, lats, y_distances):
    """Absolute momentum.

    Parameters
    ----------
    u : array-like (m/s)
    lats : array-like (degrees)
    y_distances : array-like (m)

    Returns
    -------
    cupy.ndarray (m/s)
    """
    u_arr = _1d(u)
    lat_arr = _1d(lats)
    yd = _1d(y_distances)
    return _STUB_absolute_momentum(u_arr, lat_arr, yd)


def cross_section_components(u, v, start_lat, start_lon, end_lat, end_lon):
    """Decompose wind into parallel and perpendicular cross-section components.

    Parameters
    ----------
    u, v : array-like (m/s)
    start_lat, start_lon, end_lat, end_lon : float (degrees)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    u_arr = _1d(u)
    v_arr = _1d(v)
    slat = _scalar(start_lat)
    slon = _scalar(start_lon)
    elat = _scalar(end_lat)
    elon = _scalar(end_lon)
    return _STUB_cross_section_components(u_arr, v_arr, slat, slon, elat, elon)


def curvature_vorticity(u, v, dx, dy):
    """Curvature vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    uc = u_arr
    vc = v_arr
    spd2 = uc * uc + vc * vc
    dudx = _gpu_first_derivative_uniform_2d(u_arr, dx_val, axis=1)
    dudy = _gpu_first_derivative_uniform_2d(u_arr, dy_val, axis=0)
    dvdx = _gpu_first_derivative_uniform_2d(v_arr, dx_val, axis=1)
    dvdy = _gpu_first_derivative_uniform_2d(v_arr, dy_val, axis=0)
    out = (uc * uc * dvdx - vc * vc * dudy + uc * vc * (dvdy - dudx)) / spd2
    out = cp.where(spd2 < 1e-20, 0.0, out)
    return out


def inertial_advective_wind(u, v, u_geo, v_geo, dx, dy):
    """Inertial-advective wind.

    Parameters
    ----------
    u, v : 2-D array-like (m/s) -- actual wind (not used in kernel)
    u_geo, v_geo : 2-D array-like (m/s) -- geostrophic wind
    dx, dy : float (m)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    ug = _2d(u_geo)
    vg = _2d(v_geo)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dugdx = _gpu_first_derivative_uniform_2d(ug, dx_val, axis=1)
    dugdy = _gpu_first_derivative_uniform_2d(ug, dy_val, axis=0)
    dvgdx = _gpu_first_derivative_uniform_2d(vg, dx_val, axis=1)
    dvgdy = _gpu_first_derivative_uniform_2d(vg, dy_val, axis=0)
    return u_arr * dugdx + v_arr * dugdy, u_arr * dvgdx + v_arr * dvgdy


def kinematic_flux(v_component, scalar):
    """Kinematic flux (element-wise product).

    Parameters
    ----------
    v_component : array-like (m/s)
    scalar : array-like

    Returns
    -------
    cupy.ndarray
    """
    v_arr = _1d(v_component)
    s_arr = _1d(scalar)
    return _STUB_kinematic_flux(v_arr, s_arr)


def q_vector(u, v, temperature, pressure, dx=None, dy=None, **kwargs):
    """Q-vector on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    temperature : 2-D array-like (K)
    pressure : float (hPa)
    dx, dy : float (m)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
    """
    t_arr = _2d(temperature)
    u_arr = _2d(u)
    v_arr = _2d(v)
    p_val = _scalar(pressure)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dTdx = _gpu_first_derivative_uniform_2d(t_arr, dx_val, axis=1)
    dTdy = _gpu_first_derivative_uniform_2d(t_arr, dy_val, axis=0)
    dudx = _gpu_first_derivative_uniform_2d(u_arr, dx_val, axis=1)
    dudy = _gpu_first_derivative_uniform_2d(u_arr, dy_val, axis=0)
    dvdx = _gpu_first_derivative_uniform_2d(v_arr, dx_val, axis=1)
    dvdy = _gpu_first_derivative_uniform_2d(v_arr, dy_val, axis=0)
    coeff = -287.04749097718457 / (p_val * 100.0)
    q1 = coeff * (dudx * dTdx + dvdx * dTdy)
    q2 = coeff * (dudy * dTdx + dvdy * dTdy)
    return q1, q2


def shear_vorticity(u, v, dx, dy):
    """Shear vorticity on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    return vorticity(u_arr, v_arr, dx=dx_val, dy=dy_val) - curvature_vorticity(
        u_arr, v_arr, dx_val, dy_val
    )


def shearing_deformation(u, v, dx, dy):
    """Shearing deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dvdx = _gpu_first_derivative_uniform_2d(v_arr, dx_val, axis=1)
    dudy = _gpu_first_derivative_uniform_2d(u_arr, dy_val, axis=0)
    return dvdx + dudy


def stretching_deformation(u, v, dx, dy):
    """Stretching deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dudx = _gpu_first_derivative_uniform_2d(u_arr, dx_val, axis=1)
    dvdy = _gpu_first_derivative_uniform_2d(v_arr, dy_val, axis=0)
    return dudx - dvdy


def total_deformation(u, v, dx, dy):
    """Total deformation on a 2-D grid.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray (1/s)
    """
    stretch = stretching_deformation(u, v, dx, dy)
    shear = shearing_deformation(u, v, dx, dy)
    return cp.sqrt(stretch * stretch + shear * shear)


def geospatial_gradient(data, lats, lons):
    """Gradient of a scalar field on a lat/lon grid.

    Parameters
    ----------
    data : 2-D array-like
    lats, lons : 2-D array-like (degrees)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
    """
    d = _2d(data)
    lat = _2d(lats)
    lon = _2d(lons)
    return _STUB_geospatial_gradient(d, lat, lon)


def geospatial_laplacian(data, lats, lons):
    """Laplacian of a scalar field on a lat/lon grid.

    Parameters
    ----------
    data : 2-D array-like
    lats, lons : 2-D array-like (degrees)

    Returns
    -------
    cupy.ndarray
    """
    d = _2d(data)
    lat = _2d(lats)
    lon = _2d(lons)
    return _STUB_geospatial_laplacian(d, lat, lon)


def advection_3d(scalar, u, v, w, dx, dy, dz):
    """Advection of a scalar field by a 3-D wind.

    Parameters
    ----------
    scalar : 3-D array-like
    u, v, w : 3-D array-like (m/s)
    dx, dy, dz : float (m)

    Returns
    -------
    cupy.ndarray
    """
    s = _to_gpu(scalar)
    u_arr = _to_gpu(u)
    v_arr = _to_gpu(v)
    w_arr = _to_gpu(w)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    dz_val = _scalar(dz)
    return _STUB_advection_3d(s, u_arr, v_arr, w_arr, dx_val, dy_val, dz_val)


def lat_lon_grid_deltas(longitude, latitude, x_dim=-1, y_dim=-2, geod=None):
    """Physical grid spacings (dx, dy) in meters from lat/lon grids.

    Parameters
    ----------
    longitude : 2-D array-like (degrees)
    latitude : 2-D array-like (degrees)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m)
    """
    lon = _2d(longitude)
    lat = _2d(latitude)
    return _k_lat_lon_grid_deltas(lon, lat)


# ===========================================================================
# Severe weather composite parameters
# ===========================================================================

def significant_tornado_parameter(sbcape, lcl_height, srh_0_1km,
                                  bulk_shear_0_6km):
    """Significant Tornado Parameter (fixed-layer STP).

    Parameters
    ----------
    sbcape : array-like (J/kg)
    lcl_height : array-like (m)
    srh_0_1km : array-like (m^2/s^2)
    bulk_shear_0_6km : array-like (m/s)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    cape = _to_gpu(sbcape)
    lcl_h = _to_gpu(lcl_height)
    srh = _to_gpu(srh_0_1km)
    shear = _to_gpu(bulk_shear_0_6km)
    return _k_significant_tornado_parameter(cape, lcl_h, srh, shear)


def supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff):
    """Supercell Composite Parameter (SCP).

    Parameters
    ----------
    mucape : array-like (J/kg)
    srh_eff : array-like (m^2/s^2)
    bulk_shear_eff : array-like (m/s)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    cape = _to_gpu(mucape)
    srh = _to_gpu(srh_eff)
    shear = _to_gpu(bulk_shear_eff)
    return _k_supercell_composite_parameter(cape, srh, shear)


significant_tornado = significant_tornado_parameter
supercell_composite = supercell_composite_parameter


def critical_angle(*args):
    """Critical angle between storm-relative inflow and 0-500m shear.

    Can be called as:
    - ``critical_angle(u_storm, v_storm, u_shear, v_shear)`` -- 4 args
    - ``critical_angle(u_storm, v_storm, u_sfc, v_sfc, u_500, v_500)`` -- 6 args

    Returns
    -------
    cupy.ndarray (degrees)
    """
    if len(args) == 4:
        # 4-arg form: (storm_u, storm_v, u_shear, v_shear)
        # Convert to 6-arg form by setting sfc=(0,0) and 500=(shear_u, shear_v)
        storm_u, storm_v, u_shear, v_shear = [_to_gpu(a) for a in args]
        u_sfc = _to_gpu(0.0) * cp.ones_like(storm_u)
        v_sfc = _to_gpu(0.0) * cp.ones_like(storm_v)
        return _k_critical_angle(storm_u, storm_v, u_sfc, v_sfc, u_shear, v_shear)
    gpu_args = [_to_gpu(a) for a in args]
    return _k_critical_angle(*gpu_args)


def boyden_index(z1000, z700, t700):
    """Boyden Index.

    Parameters
    ----------
    z1000 : array-like (m)
    z700 : array-like (m)
    t700 : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    return _k_boyden_index(_to_gpu(z1000), _to_gpu(z700), _to_gpu(t700))


def bulk_richardson_number(cape, shear_0_6km):
    """Bulk Richardson Number.

    Parameters
    ----------
    cape : array-like (J/kg)
    shear_0_6km : array-like (m/s)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    return _k_bulk_richardson_number(_to_gpu(cape), _to_gpu(shear_0_6km))


def dendritic_growth_zone(temperature, pressure):
    """Dendritic growth zone bounds.

    Parameters
    ----------
    temperature : array-like (Celsius)
    pressure : array-like (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
    """
    t = cp.asnumpy(_1d(temperature)).astype(np.float64, copy=False)
    p = cp.asnumpy(_1d(pressure)).astype(np.float64, copy=False)
    n = min(len(t), len(p))
    if n < 2:
        return float("nan"), float("nan")

    p_top = np.nan
    p_bottom = np.nan
    for k in range(n):
        temp = t[k]
        if temp <= -12.0 and temp >= -18.0:
            if np.isnan(p_bottom):
                p_bottom = p[k]
            p_top = p[k]

    if not np.isnan(p_bottom):
        for k in range(n - 1):
            if ((t[k] > -12.0 and t[k + 1] <= -12.0) or (t[k] <= -12.0 and t[k + 1] > -12.0)):
                frac = (-12.0 - t[k]) / (t[k + 1] - t[k])
                p_bottom = p[k] + frac * (p[k + 1] - p[k])
                break
        for k in range(n - 1):
            if ((t[k] > -18.0 and t[k + 1] <= -18.0) or (t[k] <= -18.0 and t[k + 1] > -18.0)):
                frac = (-18.0 - t[k]) / (t[k + 1] - t[k])
                p_top = p[k] + frac * (p[k + 1] - p[k])
                break

    return float(p_top), float(p_bottom)


def fosberg_fire_weather_index(temperature, relative_humidity, wind_speed_val):
    """Fosberg Fire Weather Index.

    Parameters
    ----------
    temperature : array-like (Fahrenheit)
    relative_humidity : array-like (percent)
    wind_speed_val : array-like (mph)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    t = _to_gpu(temperature)
    rh = cp.clip(_to_gpu(relative_humidity), 0.0, 100.0)
    ws = _to_gpu(wind_speed_val)
    emc = cp.where(
        rh <= 10.0,
        0.03229 + 0.281073 * rh - 0.000578 * rh * t,
        cp.where(
            rh <= 50.0,
            2.22749 + 0.160107 * rh - 0.01478 * t,
            21.0606 + 0.005565 * rh * rh - 0.00035 * rh * t - 0.483199 * rh,
        ),
    )
    m = cp.maximum(emc / 30.0, 0.0)
    eta = 1.0 - 2.0 * m + 1.5 * m * m - 0.5 * m * m * m
    fw = eta * cp.sqrt(1.0 + ws * ws)
    return cp.clip(fw * (10.0 / 3.0), 0.0, 100.0)


def freezing_rain_composite(temperature, pressure, precip_type):
    """Freezing rain composite index.

    Parameters
    ----------
    temperature : array-like (Celsius)
    pressure : array-like (hPa)
    precip_type : int

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    t = _1d(temperature)
    p = _1d(pressure)
    return _k_freezing_rain_composite(t, p, int(precip_type))


def haines_index(t_950, t_850, td_850):
    """Haines Index (fire weather).

    Parameters
    ----------
    t_950, t_850, td_850 : float (Celsius)

    Returns
    -------
    int
    """
    return _k_haines_index(_to_gpu(t_950), _to_gpu(t_850), _to_gpu(td_850))


def hot_dry_windy(temperature, relative_humidity, wind_speed_val, vpd=0.0):
    """Hot-Dry-Windy Index.

    Parameters
    ----------
    temperature : array-like (Celsius)
    relative_humidity : array-like (percent)
    wind_speed_val : array-like (m/s)
    vpd : float

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    t = _to_gpu(temperature)
    rh = _to_gpu(relative_humidity)
    ws = _to_gpu(wind_speed_val)
    vpd_val = float(vpd)
    if vpd_val > 0.0:
        vpd_arr = cp.full_like(t, vpd_val)
    else:
        es = _vappres_sharppy_hpa(t)
        ea = es * (rh / 100.0)
        vpd_arr = cp.maximum(es - ea, 0.0)
    return vpd_arr * ws


def warm_nose_check(temperature, pressure):
    """Check for a warm nose (melting layer above freezing aloft).

    Parameters
    ----------
    temperature : array-like (Celsius)
    pressure : array-like (hPa)

    Returns
    -------
    bool
    """
    t = _1d(temperature)
    p = _1d(pressure)
    return _k_warm_nose_check(t, p)


def galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst):
    """Galvez-Davison Index (tropical thunderstorm potential).

    Parameters
    ----------
    t950, t850, t700, t500, td950, td850, td700, sst : float (Celsius)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    t950 = _to_gpu(t950)
    t850 = _to_gpu(t850)
    t700 = _to_gpu(t700)
    t500 = _to_gpu(t500)
    td950 = _to_gpu(td950)
    td850 = _to_gpu(td850)
    td700 = _to_gpu(td700)
    sst = _to_gpu(sst)
    thetae_950 = equivalent_potential_temperature(cp.full_like(t950, 950.0), t950, td950)
    thetae_850 = equivalent_potential_temperature(cp.full_like(t850, 850.0), t850, td850)
    thetae_700 = equivalent_potential_temperature(cp.full_like(t700, 700.0), t700, td700)
    thetae_low = (thetae_950 + thetae_850) / 2.0
    cbi = thetae_low - thetae_700
    mwi = ((t500 + 273.15) - 243.15) * 1.5
    ii = cp.maximum((sst + 273.15) - 298.15, 0.0) * 5.0
    return cbi + ii - mwi


# ===========================================================================
# Grid composite operations (parallelized over grid points)
# ===========================================================================

def compute_cape_cin(pressure_3d, temperature_c_3d, qvapor_3d,
                     height_agl_3d, psfc, t2, q2,
                     parcel_type="surface", top_m=None):
    """CAPE/CIN for every grid point.

    3-D inputs: shape (nz, ny, nx).
    pressure_3d in Pa, temperature_c_3d in Celsius,
    qvapor_3d in kg/kg mixing ratio, height_agl_3d in m AGL,
    psfc in Pa, t2 in K (or C), q2 in kg/kg mixing ratio.
    Returns (cape, cin, lcl_height, lfc_height) each shaped (ny, nx).
    """
    parcel = str(parcel_type or "surface").strip().lower()
    if parcel == "ml":
        parcel_code = 1
    elif parcel == "mu":
        parcel_code = 2
    else:
        parcel_code = 0

    p3 = _to_gpu(pressure_3d)
    t3 = _to_gpu(temperature_c_3d)
    q3 = _to_gpu(qvapor_3d)
    h3 = _to_gpu(height_agl_3d)
    ps = _to_gpu(psfc)
    t2_arr = _to_gpu(t2)
    q2_arr = _to_gpu(q2)
    top = -1.0 if top_m is None else float(top_m)

    if p3.ndim == 3:
        if float(cp.asnumpy(p3[0, 0, 0])) < float(cp.asnumpy(p3[-1, 0, 0])):
            p3 = cp.flip(p3, axis=0)
            t3 = cp.flip(t3, axis=0)
            q3 = cp.flip(q3, axis=0)
            h3 = cp.flip(h3, axis=0)
        nz, ny, nx = p3.shape
        ncols = ny * nx
        p_2d = cp.ascontiguousarray(p3.reshape(nz, ncols).T)
        t_2d = cp.ascontiguousarray(t3.reshape(nz, ncols).T)
        q_2d = cp.ascontiguousarray(q3.reshape(nz, ncols).T)
        h_2d = cp.ascontiguousarray(h3.reshape(nz, ncols).T)
        ps_1d = cp.ascontiguousarray(ps.reshape(ncols))
        t2_1d = cp.ascontiguousarray(t2_arr.reshape(ncols))
        q2_1d = cp.ascontiguousarray(q2_arr.reshape(ncols))
        cape, cin, lcl, lfc = _k_grid_cape_cin(
            p_2d, t_2d, q_2d, h_2d,
            ps_1d, t2_1d, q2_1d,
            parcel_type_code=parcel_code,
            top_m=top,
        )
        return (
            cape.reshape(ny, nx),
            cin.reshape(ny, nx),
            lcl.reshape(ny, nx),
            lfc.reshape(ny, nx),
        )

    if p3.shape[1] > 1 and float(cp.asnumpy(p3[0, 0])) < float(cp.asnumpy(p3[0, -1])):
        p3 = cp.flip(p3, axis=1)
        t3 = cp.flip(t3, axis=1)
        q3 = cp.flip(q3, axis=1)
        h3 = cp.flip(h3, axis=1)
    return _k_grid_cape_cin(
        p3, t3, q3, h3,
        cp.ascontiguousarray(ps.ravel()),
        cp.ascontiguousarray(t2_arr.ravel()),
        cp.ascontiguousarray(q2_arr.ravel()),
        parcel_type_code=parcel_code,
        top_m=top,
    )


def compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0):
    """Storm-relative helicity for every grid point.

    3-D inputs: shape (nz, ny, nx).
    Returns SRH shaped (ny, nx) in m^2/s^2.
    """
    u3 = _to_gpu(u_3d)
    v3 = _to_gpu(v_3d)
    h3 = _to_gpu(height_agl_3d)
    top = float(top_m)
    if u3.ndim == 3:
        nz, ny, nx = u3.shape
        ncols = ny * nx
        u_2d = cp.ascontiguousarray(u3.reshape(nz, ncols).T)
        v_2d = cp.ascontiguousarray(v3.reshape(nz, ncols).T)
        h_2d = cp.ascontiguousarray(h3.reshape(nz, ncols).T)
        _, _, total = _k_grid_srh(u_2d, v_2d, h_2d, top)
        return total.reshape(ny, nx)
    _, _, total = _k_grid_srh(u3, v3, h3, top)
    return total


def compute_shear(u_3d, v_3d, height_agl_3d, bottom_m=0.0, top_m=6000.0):
    """Bulk wind shear for every grid point.

    3-D inputs: shape (nz, ny, nx).
    Returns shear magnitude shaped (ny, nx) in m/s.
    """
    u3 = _to_gpu(u_3d)
    v3 = _to_gpu(v_3d)
    h3 = _to_gpu(height_agl_3d)
    bottom = float(bottom_m)
    top = float(top_m)
    if u3.ndim == 3:
        nz, ny, nx = u3.shape
        ncols = ny * nx
        # Reshape (nz, ny, nx) -> (ncols, nz) for column kernels
        u_2d = cp.ascontiguousarray(u3.reshape(nz, ncols).T)
        v_2d = cp.ascontiguousarray(v3.reshape(nz, ncols).T)
        h_2d = cp.ascontiguousarray(h3.reshape(nz, ncols).T)
        su, sv = _k_bulk_shear(u_2d, v_2d, h_2d, bottom, top)
        return cp.sqrt(su ** 2 + sv ** 2).reshape(ny, nx)
    # 2-D input (ncols, nlevels)
    su, sv = _k_bulk_shear(u3, v3, h3, bottom, top)
    return cp.sqrt(su ** 2 + sv ** 2)


def compute_lapse_rate(temperature_c_3d, qvapor_3d, height_agl_3d,
                       bottom_km=0.0, top_km=3.0):
    """Environmental lapse rate for every grid point (C/km).

    3-D inputs: shape (nz, ny, nx).
    Returns lapse rate shaped (ny, nx) in C/km.
    """
    t3 = _to_gpu(temperature_c_3d)
    h3 = _to_gpu(height_agl_3d)
    bottom_m = _scalar(bottom_km) * 1000.0
    top_m = _scalar(top_km) * 1000.0
    if t3.ndim == 3:
        nz, ny, nx = t3.shape
        # Reshape (nz, ny, nx) -> (ny*nx, nz) for the column kernel
        t_2d = t3.reshape(nz, ny * nx).T.copy()  # (ny*nx, nz)
        h_2d = h3.reshape(nz, ny * nx).T.copy()  # (ny*nx, nz)
        result = _k_compute_lapse_rate(t_2d, h_2d, bottom_m, top_m)
        return result.reshape(ny, nx)
    return _k_compute_lapse_rate(t3, h3, bottom_m, top_m)


def compute_pw(qvapor_3d, pressure_3d):
    """Precipitable water for every grid point (mm).

    3-D inputs: shape (nz, ny, nx).
    Returns PW shaped (ny, nx) in mm.
    """
    q3 = _to_gpu(qvapor_3d)
    p3 = _to_gpu(pressure_3d)
    if q3.ndim == 3:
        nz, ny, nx = q3.shape
        ncols = ny * nx
        # Reshape (nz, ny, nx) -> (ncols, nz) for column kernel
        q_2d = cp.ascontiguousarray(q3.reshape(nz, ncols).T)
        p_2d = cp.ascontiguousarray(p3.reshape(nz, ncols).T)
        pw = _k_grid_precipitable_water(p_2d, q_2d)
        return pw.reshape(ny, nx)
    return _k_grid_precipitable_water(p3, q3)


def compute_stp(cape, lcl_height, srh_1km, shear_6km):
    """Significant Tornado Parameter on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    Returns STP shaped (ny, nx), dimensionless.
    """
    c = _2d(cape)
    l = _2d(lcl_height)
    s = _2d(srh_1km)
    sh = _2d(shear_6km)
    return _k_significant_tornado_parameter(c, l, s, sh)


def compute_scp(mucape, srh_3km, shear_6km):
    """Supercell Composite Parameter on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    Returns SCP shaped (ny, nx), dimensionless.
    Uses the grid-scale SCP formula with shear/40 (vs scalar shear/30).
    """
    c = _2d(mucape)
    s = _2d(srh_3km)
    sh = _2d(shear_6km)
    scp = (c / 1000.0) * (s / 50.0) * (sh / 40.0)
    return cp.maximum(scp, 0.0)


def compute_ehi(cape, srh):
    """Energy-Helicity Index on pre-computed fields.

    Inputs can be scalar, 1-D, or 2-D.
    Returns EHI with same shape, dimensionless.
    """
    c = _to_gpu(cape)
    s = _to_gpu(srh)
    return _k_compute_ehi(c, s)


def compute_ship(cape, shear06, t500, lr_700_500, mixing_ratio_gkg):
    """Significant Hail Parameter (SHIP).

    Inputs can be scalar, 1-D, or 2-D.
    Returns SHIP, dimensionless.
    """
    c = _to_gpu(cape)
    sh = _to_gpu(shear06)
    t5 = _to_gpu(t500)
    lr = _to_gpu(lr_700_500)
    mr = _to_gpu(mixing_ratio_gkg)
    return _k_compute_ship(c, sh, t5, lr, mr)


def compute_dcp(dcape, mu_cape, shear06, mu_mixing_ratio):
    """Derecho Composite Parameter (DCP).

    Inputs can be scalar, 1-D, or 2-D.
    Returns DCP, dimensionless.
    """
    d = _to_gpu(dcape)
    mc = _to_gpu(mu_cape)
    sh = _to_gpu(shear06)
    mr = _to_gpu(mu_mixing_ratio)
    return (
        cp.maximum(d / 980.0, 0.0)
        * cp.maximum(mc / 2000.0, 0.0)
        * cp.maximum(sh / 20.0, 0.0)
        * cp.maximum(mr / 11.0, 0.0)
    )


def compute_grid_scp(mu_cape, srh, shear_06, mu_cin):
    """Enhanced Supercell Composite with CIN term on 2-D fields.

    All inputs: shape (ny, nx).
    Returns SCP shaped (ny, nx), dimensionless.
    """
    mc = _2d(mu_cape)
    s = _2d(srh)
    sh = _2d(shear_06)
    ci = _2d(mu_cin)
    return _STUB_compute_grid_scp(mc, s, sh, ci)


def compute_grid_critical_angle(u_storm, v_storm, u_shear, v_shear):
    """Critical angle on 2-D fields.

    All inputs: shape (ny, nx).
    Returns angle in degrees (0-180).
    """
    us = _2d(u_storm)
    vs = _2d(v_storm)
    ush = _2d(u_shear)
    vsh = _2d(v_shear)
    return _STUB_compute_grid_critical_angle(us, vs, ush, vsh)


def composite_reflectivity(refl_3d):
    """Composite reflectivity (column max) from a 3-D reflectivity field.

    Input: shape (nz, ny, nx) in dBZ.
    Returns composite reflectivity shaped (ny, nx).
    """
    r3 = _to_gpu(refl_3d)
    return _k_composite_reflectivity(r3)


def composite_reflectivity_from_hydrometeors(pressure_3d, temperature_c_3d,
                                             qrain_3d, qsnow_3d, qgraup_3d):
    """Composite reflectivity from hydrometeor mixing ratios.

    All 3-D inputs: shape (nz, ny, nx).
    pressure in Pa, temperature in C, mixing ratios in kg/kg.
    Returns composite reflectivity shaped (ny, nx) in dBZ.
    """
    return _k_composite_reflectivity_from_hydrometeors(
        _to_gpu(pressure_3d),
        _to_gpu(temperature_c_3d),
        _to_gpu(qrain_3d),
        _to_gpu(qsnow_3d),
        _to_gpu(qgraup_3d),
    )


# ===========================================================================
# Smoothing / spatial derivatives
# ===========================================================================

def smooth_gaussian(data, sigma):
    """2-D Gaussian smoothing.

    Parameters
    ----------
    data : 2-D array-like
    sigma : float (grid-point units)

    Returns
    -------
    cupy.ndarray
    """
    arr = _2d(data)
    return _k_smooth_gaussian(arr, float(sigma))


def smooth_rectangular(data, size, passes=1):
    """Rectangular (box) smoothing.

    Parameters
    ----------
    data : 2-D array-like
    size : int
    passes : int

    Returns
    -------
    cupy.ndarray
    """
    arr = _2d(data)
    return _k_smooth_rectangular(arr, int(size), int(passes))


def smooth_circular(data, radius, passes=1):
    """Circular (disk) smoothing.

    Parameters
    ----------
    data : 2-D array-like
    radius : float
    passes : int

    Returns
    -------
    cupy.ndarray
    """
    arr = _2d(data)
    result = _k_smooth_circular(arr, int(radius))
    for _ in range(int(passes) - 1):
        result = _k_smooth_circular(result, int(radius))
    return result


def smooth_n_point(data, n, passes=1):
    """N-point smoother (5 or 9).

    Parameters
    ----------
    data : 2-D array-like
    n : int
    passes : int

    Returns
    -------
    cupy.ndarray
    """
    arr = _2d(data)
    return _k_smooth_n_point(arr, int(n), int(passes))


def smooth_window(data, window, passes=1, normalize_weights=True):
    """Generic 2-D convolution with a user-supplied kernel.

    Parameters
    ----------
    data : 2-D array-like
    window : 2-D array-like (kernel)
    passes : int
    normalize_weights : bool

    Returns
    -------
    cupy.ndarray
    """
    d_arr = _2d(data)
    w_arr = _2d(window)
    return _k_smooth_window(d_arr, w_arr, int(passes), bool(normalize_weights))


def gradient(f, **kwargs):
    """Calculate the gradient of a scalar field.

    Parameters
    ----------
    f : array-like
    **kwargs : deltas, axes

    Returns
    -------
    list of cupy.ndarray
    """
    data = _to_gpu(f)
    deltas = kwargs.get("deltas", None)
    if data.ndim == 2 and deltas is not None and len(deltas) >= 2:
        dy_val = _scalar(deltas[0])
        dx_val = _scalar(deltas[1])
        gx = _k_first_derivative_x(data, dx_val)
        gy = _k_first_derivative_y(data, dy_val)
        return [gy, gx]
    # General fallback using cupy.gradient
    if deltas is not None:
        spacing = [float(strip_units(d)) for d in deltas]
        result = cp.gradient(data, *spacing)
    else:
        result = cp.gradient(data)
    if isinstance(result, cp.ndarray):
        result = [result]
    return list(result)


def gradient_x(data, dx):
    """Partial derivative df/dx.

    Parameters
    ----------
    data : 2-D array-like
    dx : float (m)

    Returns
    -------
    cupy.ndarray
    """
    d_arr = _2d(data)
    dx_val = _mean_spacing(dx)
    return _gpu_first_derivative_uniform_2d(d_arr, dx_val, axis=1)


def gradient_y(data, dy):
    """Partial derivative df/dy.

    Parameters
    ----------
    data : 2-D array-like
    dy : float (m)

    Returns
    -------
    cupy.ndarray
    """
    d_arr = _2d(data)
    dy_val = _mean_spacing(dy)
    return _gpu_first_derivative_uniform_2d(d_arr, dy_val, axis=0)


def laplacian(data, dx, dy):
    """Laplacian (d2f/dx2 + d2f/dy2).

    Parameters
    ----------
    data : 2-D array-like
    dx, dy : float (m)

    Returns
    -------
    cupy.ndarray
    """
    d_arr = _2d(data)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    return (
        _gpu_second_derivative_uniform_2d(d_arr, dx_val, axis=1)
        + _gpu_second_derivative_uniform_2d(d_arr, dy_val, axis=0)
    )


def first_derivative(data, axis_spacing=None, axis=0, x=None, delta=None):
    """First derivative along a chosen axis.

    Parameters
    ----------
    data : array-like
    axis_spacing : float (m)
    axis : int

    Returns
    -------
    cupy.ndarray
    """
    d_arr = _to_gpu(data)
    if delta is not None:
        axis_spacing = delta
    elif x is not None and axis_spacing is None:
        axis_spacing = x
    if axis_spacing is None:
        raise TypeError("first_derivative requires axis spacing")
    ds = _mean_spacing(axis_spacing)
    return _STUB_first_derivative(d_arr, ds, int(axis))


def second_derivative(data, axis_spacing=None, axis=0, x=None, delta=None):
    """Second derivative along a chosen axis.

    Parameters
    ----------
    data : array-like
    axis_spacing : float (m)
    axis : int

    Returns
    -------
    cupy.ndarray
    """
    d_arr = _to_gpu(data)
    if delta is not None:
        axis_spacing = delta
    elif x is not None and axis_spacing is None:
        axis_spacing = x
    if axis_spacing is None:
        raise TypeError("second_derivative requires axis spacing")
    ds = _mean_spacing(axis_spacing)
    return _STUB_second_derivative(d_arr, ds, int(axis))


# ===========================================================================
# CPU-only utility functions (no GPU benefit)
# ===========================================================================

def angle_to_direction(degrees, level=16, full=False):
    """Convert a meteorological angle to a cardinal direction string.

    Parameters
    ----------
    degrees : float
    level : int (8, 16, or 32)
    full : bool

    Returns
    -------
    str
    """
    try:
        from metrust.calc import angle_to_direction as _fn
        return _fn(degrees, level, full)
    except ImportError:
        d = _scalar(degrees) % 360
        dirs_16 = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                    "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
        idx = int((d + 11.25) / 22.5) % 16
        return dirs_16[idx]


def parse_angle(direction):
    """Parse a cardinal direction string to degrees.

    Parameters
    ----------
    direction : str

    Returns
    -------
    float or None
    """
    try:
        from metrust.calc import parse_angle as _fn
        return _fn(direction)
    except ImportError:
        dirs = {
            "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5,
            "E": 90, "ESE": 112.5, "SE": 135, "SSE": 157.5,
            "S": 180, "SSW": 202.5, "SW": 225, "WSW": 247.5,
            "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
        }
        return dirs.get(direction.strip().upper())


def find_bounding_indices(values, target):
    """Find two indices that bracket a target value.

    Parameters
    ----------
    values : array-like
    target : float

    Returns
    -------
    tuple of (int, int) or None
    """
    try:
        from metrust.calc import find_bounding_indices as _fn
        return _fn(values, target)
    except ImportError:
        v = np.asarray(strip_units(values), dtype=np.float64).ravel()
        t = float(strip_units(target))
        for i in range(len(v) - 1):
            if (v[i] <= t <= v[i + 1]) or (v[i] >= t >= v[i + 1]):
                return (i, i + 1)
        return None


def nearest_intersection_idx(x, y1, y2):
    """Find the index nearest to where two series cross.

    Parameters
    ----------
    x, y1, y2 : array-like

    Returns
    -------
    int or None
    """
    try:
        from metrust.calc import nearest_intersection_idx as _fn
        return _fn(x, y1, y2)
    except ImportError:
        y1a = np.asarray(strip_units(y1), dtype=np.float64)
        y2a = np.asarray(strip_units(y2), dtype=np.float64)
        diff = y1a - y2a
        for i in range(len(diff) - 1):
            if diff[i] * diff[i + 1] <= 0:
                return i
        return None


def resample_nn_1d(x, xp, fp):
    """Nearest-neighbour 1-D resampling.

    Parameters
    ----------
    x, xp, fp : array-like

    Returns
    -------
    cupy.ndarray
    """
    try:
        from metrust.calc import resample_nn_1d as _fn
        result = _fn(
            np.asarray(strip_units(x)),
            np.asarray(strip_units(xp)),
            np.asarray(strip_units(fp)),
        )
        return cp.asarray(result)
    except ImportError:
        x_arr = np.asarray(strip_units(x), dtype=np.float64)
        xp_arr = np.asarray(strip_units(xp), dtype=np.float64)
        fp_arr = np.asarray(strip_units(fp), dtype=np.float64)
        indices = np.searchsorted(xp_arr, x_arr)
        indices = np.clip(indices, 0, len(xp_arr) - 1)
        return cp.asarray(fp_arr[indices])


def find_peaks(data, maxima=True, iqr_ratio=0.0):
    """Find peaks (or troughs) in a 1-D array.

    Parameters
    ----------
    data : array-like
    maxima : bool
    iqr_ratio : float

    Returns
    -------
    cupy.ndarray of int
    """
    try:
        from metrust.calc import find_peaks as _fn
        result = _fn(data, maxima, iqr_ratio)
        return cp.asarray(result)
    except ImportError:
        d = np.asarray(strip_units(data), dtype=np.float64)
        peaks = []
        for i in range(1, len(d) - 1):
            if maxima and d[i] > d[i - 1] and d[i] > d[i + 1]:
                peaks.append(i)
            elif not maxima and d[i] < d[i - 1] and d[i] < d[i + 1]:
                peaks.append(i)
        return cp.asarray(peaks, dtype=cp.int64)


def peak_persistence(data, maxima=True):
    """Topological persistence-based peak detection.

    Parameters
    ----------
    data : array-like
    maxima : bool

    Returns
    -------
    list of (int, float)
    """
    try:
        from metrust.calc import peak_persistence as _fn
        return _fn(data, maxima)
    except ImportError:
        # Simplified fallback
        d = np.asarray(strip_units(data), dtype=np.float64)
        peaks = []
        for i in range(1, len(d) - 1):
            if maxima and d[i] > d[i - 1] and d[i] > d[i + 1]:
                pers = d[i] - min(d[i - 1], d[i + 1])
                peaks.append((i, pers))
            elif not maxima and d[i] < d[i - 1] and d[i] < d[i + 1]:
                pers = max(d[i - 1], d[i + 1]) - d[i]
                peaks.append((i, pers))
        return sorted(peaks, key=lambda x: -x[1])


def reduce_point_density(lats, lons, radius):
    """Reduce point density by removing points too close together.

    Parameters
    ----------
    lats, lons : array-like (degrees)
    radius : float (degrees)

    Returns
    -------
    list of bool
    """
    try:
        from metrust.calc import reduce_point_density as _fn
        return _fn(lats, lons, radius)
    except ImportError:
        lat_arr = np.asarray(strip_units(lats), dtype=np.float64)
        lon_arr = np.asarray(strip_units(lons), dtype=np.float64)
        r = float(strip_units(radius))
        n = len(lat_arr)
        keep = [True] * n
        for i in range(n):
            if not keep[i]:
                continue
            for j in range(i + 1, n):
                if not keep[j]:
                    continue
                dist = np.sqrt((lat_arr[i] - lat_arr[j])**2 +
                               (lon_arr[i] - lon_arr[j])**2)
                if dist < r:
                    keep[j] = False
        return keep


def azimuth_range_to_lat_lon(azimuths, ranges, center_lat, center_lon):
    """Convert radar azimuth/range to latitude/longitude.

    Parameters
    ----------
    azimuths : array-like (degrees)
    ranges : array-like (m)
    center_lat, center_lon : float (degrees)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
    """
    try:
        from metrust.calc import azimuth_range_to_lat_lon as _fn
        lats, lons = _fn(azimuths, ranges, center_lat, center_lon)
        return cp.asarray(lats), cp.asarray(lons)
    except ImportError:
        az = np.asarray(strip_units(azimuths), dtype=np.float64)
        rng = np.asarray(strip_units(ranges), dtype=np.float64)
        R = 6371229.0
        clat = float(strip_units(center_lat))
        clon = float(strip_units(center_lon))
        az_rad = np.deg2rad(az)
        delta = rng / R
        lat_rad = np.arcsin(
            np.sin(np.deg2rad(clat)) * np.cos(delta) +
            np.cos(np.deg2rad(clat)) * np.sin(delta) * np.cos(az_rad)
        )
        lon_rad = np.deg2rad(clon) + np.arctan2(
            np.sin(az_rad) * np.sin(delta) * np.cos(np.deg2rad(clat)),
            np.cos(delta) - np.sin(np.deg2rad(clat)) * np.sin(lat_rad)
        )
        return cp.asarray(np.rad2deg(lat_rad)), cp.asarray(np.rad2deg(lon_rad))


# ===========================================================================
# xarray Dataset wrappers (CPU-side, use GPU for computation)
# ===========================================================================

def parcel_profile_with_lcl_as_dataset(pressure, temperature, dewpoint):
    """Calculate parcel profile and return as xarray Dataset with LCL inserted.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint : array-like (Celsius)

    Returns
    -------
    xarray.Dataset
    """
    try:
        from metrust.calc import parcel_profile_with_lcl_as_dataset as _fn
        return _fn(pressure, temperature, dewpoint)
    except ImportError:
        import xarray as xr
        p_out, t_parcel = parcel_profile_with_lcl(pressure, temperature, dewpoint)
        p_cpu = to_cpu(p_out)
        t_cpu = to_cpu(t_parcel)
        return xr.Dataset(
            {"parcel_temperature": xr.Variable("isobaric", t_cpu)},
            coords={"isobaric": xr.Variable("isobaric", p_cpu)},
        )


def isentropic_interpolation_as_dataset(levels, temperature, *args,
                                         max_iters=50, eps=1e-6,
                                         bottom_up_search=True, pressure=None):
    """Interpolate to isentropic surfaces and return as xarray Dataset.

    Parameters
    ----------
    levels : array-like (K)
    temperature : xarray.DataArray
    *args : xarray.DataArray

    Returns
    -------
    xarray.Dataset
    """
    try:
        from metrust.calc import isentropic_interpolation_as_dataset as _fn
        return _fn(levels, temperature, *args, max_iters=max_iters, eps=eps,
                   bottom_up_search=bottom_up_search, pressure=pressure)
    except ImportError:
        raise NotImplementedError(
            "isentropic_interpolation_as_dataset requires metrust"
        )


def zoom_xarray(input_field, zoom, output=None, order=3, mode="constant",
                cval=0.0, prefilter=True):
    """Zoom/interpolate an xarray DataArray using scipy.ndimage.zoom.

    Parameters
    ----------
    input_field : xarray.DataArray
    zoom : float or sequence of float

    Returns
    -------
    xarray.DataArray
    """
    try:
        from metrust.calc import zoom_xarray as _fn
        return _fn(input_field, zoom, output=output, order=order, mode=mode,
                   cval=cval, prefilter=prefilter)
    except ImportError:
        import xarray as xr
        from scipy.ndimage import zoom as _zoom
        data = np.asarray(input_field.values, dtype=np.float64)
        zoomed = _zoom(data, zoom, output=output, order=order, mode=mode,
                       cval=cval, prefilter=prefilter)
        return xr.DataArray(zoomed, dims=input_field.dims)


# ===========================================================================
# Interpolation functions (CPU-side, matching metrust interface)
# ===========================================================================

def inverse_distance_to_grid(xp, yp, variable, grid_x, grid_y, r,
                              gamma=None, kappa=None, min_neighbors=3,
                              kind='cressman'):
    """Interpolate using inverse-distance weighting to a grid."""
    try:
        from metrust.calc import inverse_distance_to_grid as _fn
        return _fn(xp, yp, variable, grid_x, grid_y, r,
                   gamma=gamma, kappa=kappa, min_neighbors=min_neighbors,
                   kind=kind)
    except ImportError:
        raise NotImplementedError("inverse_distance_to_grid requires metrust or scipy")


def inverse_distance_to_points(points, values, xi, r,
                                gamma=None, kappa=None, min_neighbors=3,
                                kind='cressman'):
    """Interpolate using inverse-distance weighting to arbitrary points."""
    try:
        from metrust.calc import inverse_distance_to_points as _fn
        return _fn(points, values, xi, r, gamma=gamma, kappa=kappa,
                   min_neighbors=min_neighbors, kind=kind)
    except ImportError:
        raise NotImplementedError("inverse_distance_to_points requires metrust or scipy")


def remove_nan_observations(x, y, z):
    """Remove all observations where any of x, y, or z is NaN."""
    x = cp.asarray(strip_units(x), dtype=cp.float64)
    y = cp.asarray(strip_units(y), dtype=cp.float64)
    z = cp.asarray(strip_units(z), dtype=cp.float64)
    mask = ~(cp.isnan(x) | cp.isnan(y) | cp.isnan(z))
    return x[mask], y[mask], z[mask]


def remove_observations_below_value(x, y, z, val=0):
    """Remove observations where z < val."""
    x = cp.asarray(strip_units(x), dtype=cp.float64)
    y = cp.asarray(strip_units(y), dtype=cp.float64)
    z = cp.asarray(strip_units(z), dtype=cp.float64)
    mask = z >= val
    return x[mask], y[mask], z[mask]


def remove_repeat_coordinates(x, y, z):
    """Remove duplicate (x, y) coordinate pairs, keeping the first."""
    # CuPy doesn't have unique with axis, fall back to numpy
    x_np = np.asarray(strip_units(x), dtype=np.float64)
    y_np = np.asarray(strip_units(y), dtype=np.float64)
    z_np = np.asarray(strip_units(z), dtype=np.float64)
    coords = np.column_stack([x_np, y_np])
    _, idx = np.unique(coords, axis=0, return_index=True)
    idx = np.sort(idx)
    return cp.asarray(x_np[idx]), cp.asarray(y_np[idx]), cp.asarray(z_np[idx])


def interpolate_nans_1d(x, y, kind='linear'):
    """Interpolate NaN values in a 1D array."""
    try:
        from metrust.calc import interpolate_nans_1d as _fn
        return _fn(x, y, kind=kind)
    except ImportError:
        from scipy.interpolate import interp1d
        xa = np.asarray(strip_units(x), dtype=np.float64)
        ya = np.asarray(strip_units(y), dtype=np.float64)
        mask = ~np.isnan(ya)
        if mask.all() or not mask.any():
            return ya.copy()
        f = interp1d(xa[mask], ya[mask], kind=kind, bounds_error=False,
                     fill_value=np.nan)
        out = ya.copy()
        out[~mask] = f(xa[~mask])
        return out


def interpolate_to_grid(x, y, z, interp_type='linear', hres=50000,
                         minimum_neighbors=3, search_radius=None,
                         gamma=None, kappa=None, rbf_func='linear',
                         rbf_smooth=0):
    """Interpolate observations to a grid."""
    try:
        from metrust.calc import interpolate_to_grid as _fn
        return _fn(x, y, z, interp_type=interp_type, hres=hres,
                   minimum_neighbors=minimum_neighbors,
                   search_radius=search_radius, gamma=gamma, kappa=kappa,
                   rbf_func=rbf_func, rbf_smooth=rbf_smooth)
    except ImportError:
        raise NotImplementedError("interpolate_to_grid requires metrust or scipy")


def interpolate_to_points(points, values, xi, interp_type='linear',
                           minimum_neighbors=3, search_radius=None,
                           gamma=None, kappa=None, rbf_func='linear',
                           rbf_smooth=0):
    """Interpolate observations to arbitrary points."""
    try:
        from metrust.calc import interpolate_to_points as _fn
        return _fn(points, values, xi, interp_type=interp_type,
                   minimum_neighbors=minimum_neighbors,
                   search_radius=search_radius, gamma=gamma, kappa=kappa,
                   rbf_func=rbf_func, rbf_smooth=rbf_smooth)
    except ImportError:
        raise NotImplementedError("interpolate_to_points requires metrust or scipy")


def natural_neighbor_to_grid(xp, yp, variable, grid_x, grid_y):
    """Interpolate using natural-neighbor-like method to a grid."""
    try:
        from metrust.calc import natural_neighbor_to_grid as _fn
        return _fn(xp, yp, variable, grid_x, grid_y)
    except ImportError:
        from scipy.interpolate import griddata
        pts = np.column_stack([
            np.asarray(strip_units(xp)),
            np.asarray(strip_units(yp)),
        ])
        result = griddata(pts, np.asarray(strip_units(variable)),
                          (np.asarray(strip_units(grid_x)),
                           np.asarray(strip_units(grid_y))),
                          method='cubic')
        return result


def natural_neighbor_to_points(points, values, xi):
    """Interpolate using natural-neighbor-like method to arbitrary points."""
    try:
        from metrust.calc import natural_neighbor_to_points as _fn
        return _fn(points, values, xi)
    except ImportError:
        from scipy.interpolate import griddata
        return griddata(np.asarray(strip_units(points)),
                        np.asarray(strip_units(values)),
                        np.asarray(strip_units(xi)), method='cubic')


def interpolate_to_isosurface(level_var, interp_var, level,
                               bottom_up_search=True):
    """Interpolate a variable to an isosurface of another variable."""
    try:
        from metrust.calc import interpolate_to_isosurface as _fn
        return _fn(level_var, interp_var, level,
                   bottom_up_search=bottom_up_search)
    except ImportError:
        lv = np.asarray(strip_units(level_var), dtype=np.float64)
        iv = np.asarray(strip_units(interp_var), dtype=np.float64)
        lvl = float(level)
        nz = lv.shape[0]
        shape_2d = lv.shape[1:]
        result = np.full(shape_2d, np.nan)
        rng = range(nz - 1) if bottom_up_search else range(nz - 2, -1, -1)
        for k in rng:
            k2 = k + 1 if bottom_up_search else k - 1
            if k2 < 0 or k2 >= nz:
                continue
            below, above = lv[k], lv[k2]
            mask = (np.minimum(below, above) <= lvl) & (np.maximum(below, above) >= lvl) & np.isnan(result)
            if not mask.any():
                continue
            denom = above[mask] - below[mask]
            denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
            frac = (lvl - below[mask]) / denom
            result[mask] = iv[k][mask] + frac * (iv[k2][mask] - iv[k][mask])
        return result


def interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan,
                   return_list_always=False):
    """Interpolate 1D data along an axis."""
    try:
        from metrust.calc import interpolate_1d as _fn
        return _fn(x, xp, *args, axis=axis, fill_value=fill_value,
                   return_list_always=return_list_always)
    except ImportError:
        x_np = np.asarray(strip_units(x), dtype=np.float64)
        xp_np = np.asarray(strip_units(xp), dtype=np.float64)
        results = []
        for a in args:
            a_np = np.asarray(strip_units(a), dtype=np.float64)
            results.append(np.interp(x_np, xp_np, a_np, left=fill_value,
                                     right=fill_value))
        if len(results) == 1 and not return_list_always:
            return results[0]
        return results


def log_interpolate_1d(x, xp, *args, axis=0, fill_value=np.nan):
    """Interpolate in log-space along an axis."""
    log_x = np.log(np.asarray(strip_units(x), dtype=np.float64))
    log_xp = np.log(np.asarray(strip_units(xp), dtype=np.float64))
    return interpolate_1d(log_x, log_xp, *args, axis=axis,
                          fill_value=fill_value, return_list_always=True)


def cross_section(data, start, end, steps=100, interp_type='linear'):
    """Extract a cross-section from gridded data."""
    try:
        from metrust.calc import cross_section as _fn
        return _fn(data, start, end, steps=steps, interp_type=interp_type)
    except ImportError:
        raise NotImplementedError("cross_section requires metrust or metpy")


def interpolate_to_slice(data, points, interp_type='linear'):
    """Interpolate data to a slice along a set of points."""
    try:
        from metrust.calc import interpolate_to_slice as _fn
        return _fn(data, points, interp_type=interp_type)
    except ImportError:
        raise NotImplementedError("interpolate_to_slice requires metrust or metpy")


def geodesic(crs, start, end, steps):
    """Calculate points along a geodesic between two points."""
    try:
        from metrust.calc import geodesic as _fn
        return _fn(crs, start, end, steps)
    except ImportError:
        try:
            from pyproj import Geod
            g = Geod(ellps='WGS84')
            lon_start, lat_start = start
            lon_end, lat_end = end
            pts = g.npts(lon_start, lat_start, lon_end, lat_end, steps)
            lons = [lon_start] + [p[0] for p in pts] + [lon_end]
            lats = [lat_start] + [p[1] for p in pts] + [lat_end]
            return np.array(list(zip(lons, lats)))
        except ImportError:
            raise NotImplementedError("geodesic requires metrust, metpy, or pyproj")


# ===========================================================================
# Exception class
# ===========================================================================

class InvalidSoundingError(Exception):
    """Raised when sounding data is invalid or insufficient."""
    pass


# ===========================================================================
# __all__ -- explicit public API (mirrors metrust.calc exactly)
# ===========================================================================

__all__ = [
    # thermo
    "potential_temperature",
    "equivalent_potential_temperature",
    "saturation_vapor_pressure",
    "saturation_mixing_ratio",
    "wet_bulb_temperature",
    "lfc",
    "el",
    "lcl",
    "dewpoint_from_relative_humidity",
    "relative_humidity_from_dewpoint",
    "virtual_temperature",
    "virtual_temperature_from_dewpoint",
    "cape_cin",
    "mixing_ratio",
    "showalter_index",
    "k_index",
    "total_totals",
    "total_totals_index",
    "downdraft_cape",
    "cross_totals",
    "vertical_totals",
    "sweat_index",
    "brunt_vaisala_frequency",
    "brunt_vaisala_period",
    "brunt_vaisala_frequency_squared",
    "precipitable_water",
    "parcel_profile_with_lcl",
    "moist_air_gas_constant",
    "moist_air_specific_heat_pressure",
    "moist_air_poisson_exponent",
    "water_latent_heat_vaporization",
    "water_latent_heat_melting",
    "water_latent_heat_sublimation",
    "relative_humidity_wet_psychrometric",
    "weighted_continuous_average",
    "get_perturbation",
    "add_height_to_pressure",
    "add_pressure_to_height",
    "thickness_hydrostatic",
    "specific_humidity_from_mixing_ratio",
    "thickness_hydrostatic_from_relative_humidity",
    "vapor_pressure",
    "ccl",
    "lifted_index",
    "density",
    "dewpoint",
    "dewpoint_from_specific_humidity",
    "dry_lapse",
    "dry_static_energy",
    "exner_function",
    "find_intersections",
    "geopotential_to_height",
    "get_layer",
    "get_layer_heights",
    "height_to_geopotential",
    "isentropic_interpolation",
    "mean_pressure_weighted",
    "mixed_layer",
    "mixed_layer_cape_cin",
    "mixing_ratio_from_relative_humidity",
    "mixing_ratio_from_specific_humidity",
    "moist_lapse",
    "moist_static_energy",
    "montgomery_streamfunction",
    "most_unstable_cape_cin",
    "parcel_profile",
    "reduce_point_density",
    "relative_humidity_from_mixing_ratio",
    "relative_humidity_from_specific_humidity",
    "saturation_equivalent_potential_temperature",
    "scale_height",
    "specific_humidity_from_dewpoint",
    "static_stability",
    "surface_based_cape_cin",
    "temperature_from_potential_temperature",
    "vertical_velocity",
    "vertical_velocity_pressure",
    "virtual_potential_temperature",
    "wet_bulb_potential_temperature",
    "get_mixed_layer_parcel",
    "get_most_unstable_parcel",
    "psychrometric_vapor_pressure",
    "frost_point",
    "mixed_parcel",
    "most_unstable_parcel",
    "psychrometric_vapor_pressure_wet",
    # wind
    "wind_speed",
    "wind_direction",
    "wind_components",
    "bulk_shear",
    "mean_wind",
    "storm_relative_helicity",
    "bunkers_storm_motion",
    "corfidi_storm_motion",
    "friction_velocity",
    "tke",
    "gradient_richardson_number",
    # kinematics
    "divergence",
    "vorticity",
    "absolute_vorticity",
    "advection",
    "frontogenesis",
    "geostrophic_wind",
    "ageostrophic_wind",
    "potential_vorticity_baroclinic",
    "potential_vorticity_barotropic",
    "normal_component",
    "tangential_component",
    "unit_vectors_from_cross_section",
    "vector_derivative",
    "absolute_momentum",
    "coriolis_parameter",
    "cross_section_components",
    "curvature_vorticity",
    "inertial_advective_wind",
    "kinematic_flux",
    "q_vector",
    "shear_vorticity",
    "shearing_deformation",
    "stretching_deformation",
    "total_deformation",
    "geospatial_gradient",
    "geospatial_laplacian",
    # severe
    "significant_tornado_parameter",
    "significant_tornado",
    "supercell_composite_parameter",
    "supercell_composite",
    "critical_angle",
    "boyden_index",
    "bulk_richardson_number",
    "convective_inhibition_depth",
    "dendritic_growth_zone",
    "fosberg_fire_weather_index",
    "freezing_rain_composite",
    "haines_index",
    "hot_dry_windy",
    "warm_nose_check",
    "galvez_davison_index",
    # grid composites
    "compute_cape_cin",
    "compute_srh",
    "compute_shear",
    "compute_lapse_rate",
    "compute_pw",
    "compute_stp",
    "compute_scp",
    "compute_ehi",
    "compute_ship",
    "compute_dcp",
    "compute_grid_scp",
    "compute_grid_critical_angle",
    "composite_reflectivity",
    "composite_reflectivity_from_hydrometeors",
    # atmo
    "pressure_to_height_std",
    "height_to_pressure_std",
    "altimeter_to_station_pressure",
    "station_to_altimeter_pressure",
    "altimeter_to_sea_level_pressure",
    "sigma_to_pressure",
    "heat_index",
    "windchill",
    "apparent_temperature",
    # smooth
    "smooth_gaussian",
    "smooth_rectangular",
    "smooth_circular",
    "smooth_n_point",
    "smooth_window",
    "gradient",
    "gradient_x",
    "gradient_y",
    "laplacian",
    "first_derivative",
    "second_derivative",
    "lat_lon_grid_deltas",
    # utils
    "angle_to_direction",
    "parse_angle",
    "find_bounding_indices",
    "nearest_intersection_idx",
    "resample_nn_1d",
    "find_peaks",
    "peak_persistence",
    "azimuth_range_to_lat_lon",
    "advection_3d",
    # exceptions
    "InvalidSoundingError",
    # xarray dataset wrappers
    "parcel_profile_with_lcl_as_dataset",
    "isentropic_interpolation_as_dataset",
    "zoom_xarray",
    # interpolation
    "inverse_distance_to_grid",
    "inverse_distance_to_points",
    "remove_nan_observations",
    "remove_observations_below_value",
    "remove_repeat_coordinates",
    "interpolate_nans_1d",
    "interpolate_to_grid",
    "interpolate_to_points",
    "natural_neighbor_to_grid",
    "natural_neighbor_to_points",
    "interpolate_to_isosurface",
    "interpolate_1d",
    "log_interpolate_1d",
    "cross_section",
    "interpolate_to_slice",
    "geodesic",
    "units",
    "xr",
]
