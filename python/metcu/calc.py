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
from metcu.kernels.thermo import (
    potential_temperature_kernel,
    equivalent_potential_temperature_kernel,
    saturation_vapor_pressure_kernel,
    saturation_mixing_ratio_kernel,
    wet_bulb_temperature_kernel,
    dewpoint_from_rh_kernel,
    rh_from_dewpoint_kernel,
    virtual_temperature_kernel,
    virtual_temperature_from_dewpoint_kernel,
    mixing_ratio_kernel,
    density_kernel,
    dewpoint_kernel,
    dewpoint_from_specific_humidity_kernel,
    dry_lapse_kernel,
    dry_static_energy_kernel,
    exner_function_kernel,
    moist_lapse_kernel,
    moist_static_energy_kernel,
    parcel_profile_kernel,
    temperature_from_potential_temperature_kernel,
    vertical_velocity_kernel,
    vertical_velocity_pressure_kernel,
    virtual_potential_temperature_kernel,
    wet_bulb_potential_temperature_kernel,
    saturation_equivalent_potential_temperature_kernel,
    vapor_pressure_kernel,
    specific_humidity_from_mixing_ratio_kernel,
    mixing_ratio_from_relative_humidity_kernel,
    mixing_ratio_from_specific_humidity_kernel,
    relative_humidity_from_mixing_ratio_kernel,
    relative_humidity_from_specific_humidity_kernel,
    specific_humidity_from_dewpoint_kernel,
    frost_point_kernel,
    heat_index_kernel,
    windchill_kernel,
    apparent_temperature_kernel,
    moist_air_gas_constant_kernel,
    moist_air_specific_heat_pressure_kernel,
    moist_air_poisson_exponent_kernel,
    water_latent_heat_vaporization_kernel,
    water_latent_heat_melting_kernel,
    water_latent_heat_sublimation_kernel,
    relative_humidity_wet_psychrometric_kernel,
    psychrometric_vapor_pressure_kernel,
    add_height_to_pressure_kernel,
    add_pressure_to_height_kernel,
    thickness_hydrostatic_kernel,
    scale_height_kernel,
    geopotential_to_height_kernel,
    height_to_geopotential_kernel,
    pressure_to_height_std_kernel,
    height_to_pressure_std_kernel,
    altimeter_to_station_pressure_kernel,
    station_to_altimeter_pressure_kernel,
    altimeter_to_sea_level_pressure_kernel,
    sigma_to_pressure_kernel,
    coriolis_parameter_kernel,
)
from metcu.kernels.wind import (
    wind_speed_kernel,
    wind_direction_kernel,
    wind_components_kernel,
    bulk_shear_kernel,
    mean_wind_kernel,
    storm_relative_helicity_kernel,
    bunkers_storm_motion_kernel,
    corfidi_storm_motion_kernel,
    friction_velocity_kernel,
    tke_kernel,
    gradient_richardson_number_kernel,
)
from metcu.kernels.kinematics import (
    divergence_kernel,
    vorticity_kernel,
    absolute_vorticity_kernel,
    advection_kernel,
    frontogenesis_kernel,
    geostrophic_wind_kernel,
    ageostrophic_wind_kernel,
    potential_vorticity_baroclinic_kernel,
    potential_vorticity_barotropic_kernel,
    vector_derivative_kernel,
    shearing_deformation_kernel,
    stretching_deformation_kernel,
    total_deformation_kernel,
    curvature_vorticity_kernel,
    shear_vorticity_kernel,
    q_vector_kernel,
    inertial_advective_wind_kernel,
    advection_3d_kernel,
    geospatial_gradient_kernel,
    geospatial_laplacian_kernel,
    kinematic_flux_kernel,
    absolute_momentum_kernel,
    normal_component_kernel,
    tangential_component_kernel,
    cross_section_components_kernel,
    lat_lon_grid_deltas_kernel,
)
from metcu.kernels.smooth import (
    smooth_gaussian_kernel,
    smooth_rectangular_kernel,
    smooth_circular_kernel,
    smooth_n_point_kernel,
    smooth_window_kernel,
    gradient_x_kernel,
    gradient_y_kernel,
    laplacian_kernel,
    first_derivative_kernel,
    second_derivative_kernel,
)
from metcu.kernels.sounding import (
    lfc_kernel,
    el_kernel,
    lcl_kernel,
    cape_cin_kernel,
    surface_based_cape_cin_kernel,
    mixed_layer_cape_cin_kernel,
    most_unstable_cape_cin_kernel,
    downdraft_cape_kernel,
    showalter_index_kernel,
    k_index_kernel,
    total_totals_kernel,
    cross_totals_kernel,
    vertical_totals_kernel,
    sweat_index_kernel,
    lifted_index_kernel,
    ccl_kernel,
    precipitable_water_kernel,
    brunt_vaisala_frequency_kernel,
    brunt_vaisala_period_kernel,
    brunt_vaisala_frequency_squared_kernel,
    static_stability_kernel,
    parcel_profile_with_lcl_kernel,
    get_layer_kernel,
    get_layer_heights_kernel,
    mixed_layer_kernel,
    mean_pressure_weighted_kernel,
    get_mixed_layer_parcel_kernel,
    get_most_unstable_parcel_kernel,
    isentropic_interpolation_kernel,
    thickness_hydrostatic_from_rh_kernel,
    convective_inhibition_depth_kernel,
    montgomery_streamfunction_kernel,
)
from metcu.kernels.severe import (
    significant_tornado_parameter_kernel,
    supercell_composite_parameter_kernel,
    critical_angle_kernel,
    boyden_index_kernel,
    bulk_richardson_number_kernel,
    dendritic_growth_zone_kernel,
    fosberg_fire_weather_index_kernel,
    freezing_rain_composite_kernel,
    haines_index_kernel,
    hot_dry_windy_kernel,
    warm_nose_check_kernel,
    galvez_davison_index_kernel,
)
from metcu.kernels.grid import (
    compute_cape_cin_kernel,
    compute_srh_kernel,
    compute_shear_kernel,
    compute_lapse_rate_kernel,
    compute_pw_kernel,
    compute_stp_kernel,
    compute_scp_kernel,
    compute_ehi_kernel,
    compute_ship_kernel,
    compute_dcp_kernel,
    compute_grid_scp_kernel,
    compute_grid_critical_angle_kernel,
    composite_reflectivity_kernel,
    composite_reflectivity_from_hydrometeors_kernel,
)


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
    return potential_temperature_kernel(p, t)


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
    return equivalent_potential_temperature_kernel(p, t, td)


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
    return saturation_vapor_pressure_kernel(t, phase)


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
    return saturation_mixing_ratio_kernel(p, t, phase)


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
    return wet_bulb_temperature_kernel(p, t, td)


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
    return dewpoint_from_rh_kernel(t, rh)


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
    return rh_from_dewpoint_kernel(t, td, phase)


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
    return virtual_temperature_kernel(t, pmr, td)


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
    return virtual_temperature_from_dewpoint_kernel(t, td, p)


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
    return mixing_ratio_kernel(a, b, molecular_weight_ratio)


def density(pressure, temperature, mixing_ratio_val):
    """Air density from pressure, temperature, and mixing ratio.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    mixing_ratio_val : array-like (g/kg)

    Returns
    -------
    cupy.ndarray (kg/m^3)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    w = _to_gpu(mixing_ratio_val)
    return density_kernel(p, t, w)


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
    return dewpoint_kernel(e)


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
    return dewpoint_from_specific_humidity_kernel(p, q)


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
    return dry_lapse_kernel(p, t)


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
    return dry_static_energy_kernel(h, t)


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
    return exner_function_kernel(p)


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
    ref_p = _scalar(reference_pressure) if reference_pressure is not None else None
    return moist_lapse_kernel(p, t, ref_p)


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
    return moist_static_energy_kernel(h, t, q)


def parcel_profile(pressure, temperature, dewpoint_val):
    """Parcel temperature profile.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : float (Celsius)
    dewpoint_val : float (Celsius)

    Returns
    -------
    cupy.ndarray (Celsius)
    """
    p = _1d(pressure)
    t = _scalar(temperature)
    td = _scalar(dewpoint_val)
    return parcel_profile_kernel(p, t, td)


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
    return temperature_from_potential_temperature_kernel(p, th)


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
    return vertical_velocity_kernel(o, p, t)


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
    return vertical_velocity_pressure_kernel(ww, p, t)


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
    w = _to_gpu(mixing_ratio_val)
    return virtual_potential_temperature_kernel(p, t, w)


def wet_bulb_potential_temperature(pressure, temperature, dewpoint_val):
    """Wet-bulb potential temperature.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (K)
    """
    p = _to_gpu(pressure)
    t = _to_gpu(temperature)
    td = _to_gpu(dewpoint_val)
    return wet_bulb_potential_temperature_kernel(p, t, td)


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
    return saturation_equivalent_potential_temperature_kernel(p, t)


def vapor_pressure(pressure_or_dewpoint, mixing_ratio_val=None,
                   molecular_weight_ratio=0.6219569100577033):
    """Vapor pressure from dewpoint or from pressure and mixing ratio.

    Parameters
    ----------
    pressure_or_dewpoint : array-like
    mixing_ratio_val : array-like, optional

    Returns
    -------
    cupy.ndarray (Pa)
    """
    if mixing_ratio_val is not None:
        p = _to_gpu(pressure_or_dewpoint)
        w = _to_gpu(mixing_ratio_val)
        return p * w / (molecular_weight_ratio + w)
    td = _to_gpu(pressure_or_dewpoint)
    return vapor_pressure_kernel(td)


def specific_humidity_from_mixing_ratio(mixing_ratio_val):
    """Specific humidity from mixing ratio.

    Parameters
    ----------
    mixing_ratio_val : array-like (kg/kg)

    Returns
    -------
    cupy.ndarray (kg/kg)
    """
    w = _to_gpu(mixing_ratio_val)
    return specific_humidity_from_mixing_ratio_kernel(w)


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
    return mixing_ratio_from_relative_humidity_kernel(p, t, rh)


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
    return mixing_ratio_from_specific_humidity_kernel(q)


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
    w = _to_gpu(mixing_ratio_val)
    return relative_humidity_from_mixing_ratio_kernel(p, t, w)


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
    return relative_humidity_from_specific_humidity_kernel(p, t, q)


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
    return specific_humidity_from_dewpoint_kernel(p, td)


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
    return frost_point_kernel(t, rh)


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
    return moist_air_gas_constant_kernel(w)


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
    return moist_air_specific_heat_pressure_kernel(w)


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
    return moist_air_poisson_exponent_kernel(w)


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
    return water_latent_heat_vaporization_kernel(t)


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
    return water_latent_heat_melting_kernel(t)


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
    return water_latent_heat_sublimation_kernel(t)


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
    return relative_humidity_wet_psychrometric_kernel(t, tw, p)


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
    return psychrometric_vapor_pressure_kernel(t, tw, p)


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
    return add_height_to_pressure_kernel(p, dh)


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
    return add_pressure_to_height_kernel(h, dp)


def thickness_hydrostatic(pressure_or_bottom, temperature_or_top, t_mean=None,
                          mixing_ratio_val=None,
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
        return thickness_hydrostatic_kernel(p_bot, p_top_val, tm)
    # Profile form
    p = _1d(pressure_or_bottom)
    t = _1d(temperature_or_top)
    if mixing_ratio_val is not None:
        w = _1d(mixing_ratio_val)
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
    return thickness_hydrostatic_from_rh_kernel(p, t, rh)


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
    return scale_height_kernel(t)


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
    return geopotential_to_height_kernel(gp)


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
    return height_to_geopotential_kernel(h)


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
    return pressure_to_height_std_kernel(p)


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
    return height_to_pressure_std_kernel(h)


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
    return altimeter_to_station_pressure_kernel(a, e)


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
    return station_to_altimeter_pressure_kernel(s, e)


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
    return altimeter_to_sea_level_pressure_kernel(a, e, t)


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
    return sigma_to_pressure_kernel(s, ps, pt)


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
    return heat_index_kernel(t, rh)


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
    return windchill_kernel(t, ws)


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
    return apparent_temperature_kernel(t, rh, ws)


# ===========================================================================
# Sounding / profile functions
# ===========================================================================

def lfc(pressure, temperature, dewpoint_val, parcel_temperature_profile=None,
        dewpoint_start=None, which="top"):
    """Level of Free Convection.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        LFC pressure (hPa) and parcel temperature at the LFC (Celsius).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return lfc_kernel(p, t, td)


def el(pressure, temperature, dewpoint_val, parcel_temperature_profile=None,
       which="top"):
    """Equilibrium Level.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        EL pressure (hPa) and parcel temperature at the EL (Celsius).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return el_kernel(p, t, td)


def lcl(pressure, temperature, dewpoint_val):
    """Lifting Condensation Level.

    Parameters
    ----------
    pressure : float (hPa)
    temperature : float (Celsius)
    dewpoint_val : float (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        LCL pressure (hPa) and temperature (Celsius).
    """
    p = _scalar(pressure)
    t = _scalar(temperature)
    td = _scalar(dewpoint_val)
    return lcl_kernel(p, t, td)


def cape_cin(pressure, temperature, dewpoint_val, parcel_profile_or_height=None,
             *args, parcel_profile=None, which_lfc="bottom", which_el="top",
             parcel_type="sb", ml_depth=100.0, mu_depth=300.0, top_m=None,
             **kwargs):
    """CAPE and CIN for a sounding.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)
    parcel_profile_or_height : array-like, optional
    parcel_type : str
    ml_depth : float
    mu_depth : float
    top_m : float, optional

    Returns
    -------
    tuple of cupy.ndarray
        CAPE (J/kg), CIN (J/kg), optionally LCL height (m), LFC height (m).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    h = _1d(parcel_profile_or_height) if parcel_profile_or_height is not None else None
    psfc = _scalar(kwargs.get("psfc", pressure[0] if hasattr(pressure, '__getitem__') else pressure))
    t2m = _scalar(kwargs.get("t2m", temperature[0] if hasattr(temperature, '__getitem__') else temperature))
    td2m = _scalar(kwargs.get("td2m", dewpoint_val[0] if hasattr(dewpoint_val, '__getitem__') else dewpoint_val))
    return cape_cin_kernel(p, t, td, h, psfc, t2m, td2m, parcel_type,
                           float(ml_depth), float(mu_depth), top_m)


def surface_based_cape_cin(pressure, temperature, dewpoint_val):
    """Surface-based CAPE and CIN.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        CAPE (J/kg) and CIN (J/kg).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return surface_based_cape_cin_kernel(p, t, td)


def mixed_layer_cape_cin(pressure, temperature, dewpoint_val, depth=100.0):
    """Mixed-layer CAPE and CIN.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        CAPE (J/kg) and CIN (J/kg).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    d = _scalar(depth)
    return mixed_layer_cape_cin_kernel(p, t, td, d)


def most_unstable_cape_cin(pressure, temperature, dewpoint_val, depth=300, **kwargs):
    """Most-unstable CAPE and CIN.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray)
        CAPE (J/kg) and CIN (J/kg).
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    d = _scalar(depth)
    return most_unstable_cape_cin_kernel(p, t, td, d)


def downdraft_cape(pressure, temperature, dewpoint_val):
    """Downdraft CAPE (DCAPE).

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (J/kg)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return downdraft_cape_kernel(p, t, td)


def showalter_index(pressure, temperature, dewpoint_val):
    """Showalter Index.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return showalter_index_kernel(p, t, td)


def k_index(*args, vertical_dim=0):
    """K-Index.

    Parameters
    ----------
    Either (pressure, temperature, dewpoint) arrays or 5 scalar level values.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) == 5:
        return k_index_kernel(*[_to_gpu(a) for a in args])
    elif len(args) == 3:
        p, t, td = [_1d(a) for a in args]
        return k_index_kernel(p, t, td)
    raise TypeError("k_index expects (pressure, temperature, dewpoint) or 5 scalar level values")


def total_totals(*args, vertical_dim=0):
    """Total Totals Index.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) == 3:
        return total_totals_kernel(*[_to_gpu(a) for a in args])
    raise TypeError("total_totals expects (pressure, temperature, dewpoint) or 3 scalar level values")


def cross_totals(*args, vertical_dim=0):
    """Cross Totals: Td850 - T500.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) in (2, 3):
        return cross_totals_kernel(*[_to_gpu(a) for a in args])
    raise TypeError("cross_totals expects (pressure, temperature, dewpoint) or (td850, t500)")


def vertical_totals(*args, vertical_dim=0):
    """Vertical Totals: T850 - T500.

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    if len(args) == 2:
        return vertical_totals_kernel(*[_to_gpu(a) for a in args])
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
    return sweat_index_kernel(
        _to_gpu(t850), _to_gpu(td850), _to_gpu(t500),
        _to_gpu(dd850), _to_gpu(dd500), _to_gpu(ff850), _to_gpu(ff500),
    )


def lifted_index(pressure, temperature, dewpoint_val):
    """Lifted Index.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (delta_degC)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return lifted_index_kernel(p, t, td)


def ccl(pressure, temperature, dewpoint_val):
    """Convective Condensation Level.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) or None
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return ccl_kernel(p, t, td)


def precipitable_water(pressure, dewpoint_val):
    """Precipitable water.

    Parameters
    ----------
    pressure : array-like (hPa)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (mm)
    """
    p = _1d(pressure)
    td = _1d(dewpoint_val)
    return precipitable_water_kernel(p, td)


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
    return brunt_vaisala_frequency_kernel(z, theta)


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
    return brunt_vaisala_period_kernel(z, theta)


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
    return brunt_vaisala_frequency_squared_kernel(z, theta)


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
    return static_stability_kernel(p, t)


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
    return parcel_profile_with_lcl_kernel(p, t, td)


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
    value_arrays = [_1d(a) for a in args]
    pb = _scalar(p_bottom) if p_bottom is not None else (_scalar(bottom) if bottom is not None else float(p[0]))
    if p_top is not None:
        pt = _scalar(p_top)
    elif depth is not None:
        pt = pb - _scalar(depth)
    else:
        raise TypeError("get_layer requires either p_top or depth")
    results = get_layer_kernel(p, value_arrays, pb, pt)
    if len(results) == 2:
        return results[0], results[1]
    return results


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
    tuple of (cupy.ndarray, cupy.ndarray)
    """
    p = _1d(pressure)
    h = _1d(heights)
    pb = _scalar(p_bottom)
    pt = _scalar(p_top)
    return get_layer_heights_kernel(p, h, pb, pt)


def mixed_layer(pressure, *args, height=None, bottom=None, depth=100.0,
                interpolate=True):
    """Mixed-layer mean of one or more profiles.

    Parameters
    ----------
    pressure : array-like (hPa)
    *args : array-like
    depth : float (hPa)

    Returns
    -------
    float or tuple of floats
    """
    p = _1d(pressure)
    value_arrays = [_1d(a) for a in args]
    d = _scalar(depth)
    results = mixed_layer_kernel(p, value_arrays, d)
    if len(results) == 1:
        return results[0]
    return tuple(results)


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
    return mean_pressure_weighted_kernel(p, v)


def get_mixed_layer_parcel(pressure, temperature, dewpoint_val, depth=100.0):
    """Get mixed-layer parcel properties.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray)
        Parcel pressure, temperature, dewpoint.
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    d = _scalar(depth)
    return get_mixed_layer_parcel_kernel(p, t, td, d)


def get_most_unstable_parcel(pressure, temperature, dewpoint_val,
                             height=None, bottom=None, depth=300.0):
    """Get most-unstable parcel properties.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)
    depth : float (hPa)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray, int)
        Parcel pressure, temperature, dewpoint, source index.
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    d = _scalar(depth)
    return get_most_unstable_parcel_kernel(p, t, td, d)


def mixed_parcel(pressure, temperature, dewpoint_val, parcel_start_pressure=None,
                 height=None, bottom=None, depth=100, interpolate=True):
    """Mixed-layer parcel (MetPy-compatible alias).

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray, cupy.ndarray)
    """
    d = _scalar(depth)
    return get_mixed_layer_parcel(pressure, temperature, dewpoint_val, d)


def most_unstable_parcel(pressure, temperature, dewpoint_val, height=None,
                         bottom=None, depth=300):
    """Alias for get_most_unstable_parcel."""
    return get_most_unstable_parcel(pressure, temperature, dewpoint_val,
                                    height=height, bottom=bottom, depth=depth)


def isentropic_interpolation(theta_levels, pressure_3d, temperature_3d,
                              fields, nx=None, ny=None, nz=None):
    """Interpolate fields to isentropic surfaces.

    Parameters
    ----------
    theta_levels : array-like (K)
    pressure_3d : 3-D array (hPa)
    temperature_3d : 3-D array (K)
    fields : list of 3-D arrays
    nx, ny, nz : int, optional

    Returns
    -------
    list of cupy.ndarray
    """
    theta = _1d(theta_levels)
    p_arr = _to_gpu(pressure_3d)
    t_arr = _to_gpu(temperature_3d)
    if p_arr.ndim == 3 and nz is None:
        nz, ny, nx = p_arr.shape
    field_list = [_to_gpu(f) for f in fields]
    return isentropic_interpolation_kernel(theta, p_arr, t_arr, field_list,
                                            nx, ny, nz)


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
        return montgomery_streamfunction_kernel(th, p, t, h)
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


def convective_inhibition_depth(pressure, temperature, dewpoint_val):
    """Convective inhibition depth.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    cupy.ndarray (hPa)
    """
    p = _1d(pressure)
    t = _1d(temperature)
    td = _1d(dewpoint_val)
    return convective_inhibition_depth_kernel(p, t, td)


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
    return wind_speed_kernel(u_arr, v_arr)


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
    return wind_direction_kernel(u_arr, v_arr)


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
    return wind_components_kernel(spd, dirn)


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
    return bulk_shear_kernel(u_arr, v_arr, h_arr, bot, top_val)


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
    return mean_wind_kernel(u_arr, v_arr, h_arr, bot, top_val)


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
    su = _scalar(storm_u) if storm_u is not None else None
    sv = _scalar(storm_v) if storm_v is not None else None
    return storm_relative_helicity_kernel(u_arr, v_arr, h_arr, d, su, sv)


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
        p = None
    return bunkers_storm_motion_kernel(p, u, v, h)


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
    return corfidi_storm_motion_kernel(u_arr, v_arr, h_arr, u8, v8)


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
    return friction_velocity_kernel(u_arr, w_arr)


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
    return tke_kernel(u_arr, v_arr, w_arr)


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
    return gradient_richardson_number_kernel(z, theta, u_arr, v_arr)


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
    return coriolis_parameter_kernel(lat)


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
    return divergence_kernel(u_arr, v_arr, dx_val, dy_val)


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
    return vorticity_kernel(u_arr, v_arr, dx_val, dy_val)


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
    return absolute_vorticity_kernel(u_arr, v_arr, lats_arr, dx_val, dy_val)


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
    return advection_kernel(s_arr, u_arr, v_arr, dx_val, dy_val)


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
    return frontogenesis_kernel(t_arr, u_arr, v_arr, dx_val, dy_val)


def geostrophic_wind(heights, dx=None, dy=None, latitude=None, x_dim=-1,
                     y_dim=-2, parallel_scale=None, meridional_scale=None,
                     longitude=None, crs=None):
    """Geostrophic wind from geopotential height.

    Parameters
    ----------
    heights : 2-D array-like (m)
    latitude : 2-D array-like (degrees)
    dx, dy : float (m)

    Returns
    -------
    tuple of (cupy.ndarray, cupy.ndarray) (m/s)
    """
    h_arr = _2d(heights)
    lats_arr = _2d(latitude)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    return geostrophic_wind_kernel(h_arr, lats_arr, dx_val, dy_val)


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
    return ageostrophic_wind_kernel(u_arr, v_arr, h_arr, lats_arr, dx_val, dy_val)


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
    return potential_vorticity_baroclinic_kernel(pt, p, gpu_args, dx_val, dy_val, latitude)


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
    return potential_vorticity_barotropic_kernel(h_arr, u_arr, v_arr, lats_arr,
                                                 dx_val, dy_val)


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
    return normal_component_kernel(u_arr, v_arr, start, end)


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
    return tangential_component_kernel(u_arr, v_arr, start, end)


def unit_vectors_from_cross_section(start, end):
    """Tangent and normal unit vectors for a cross-section line.

    Parameters
    ----------
    start, end : tuple of (lat, lon)

    Returns
    -------
    tuple of ((east, north), (east, north))
    """
    # CPU-only utility
    try:
        from metrust.calc import unit_vectors_from_cross_section as _uv
        return _uv(start, end)
    except ImportError:
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
    return vector_derivative_kernel(u_arr, v_arr, dx_val, dy_val)


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
    return absolute_momentum_kernel(u_arr, lat_arr, yd)


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
    return cross_section_components_kernel(u_arr, v_arr, slat, slon, elat, elon)


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
    return curvature_vorticity_kernel(u_arr, v_arr, dx_val, dy_val)


def inertial_advective_wind(u, v, u_geo, v_geo, dx, dy):
    """Inertial-advective wind.

    Parameters
    ----------
    u, v : 2-D array-like (m/s)
    u_geo, v_geo : 2-D array-like (m/s)
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
    return inertial_advective_wind_kernel(u_arr, v_arr, ug, vg, dx_val, dy_val)


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
    return kinematic_flux_kernel(v_arr, s_arr)


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
    return q_vector_kernel(t_arr, u_arr, v_arr, p_val, dx_val, dy_val)


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
    return shear_vorticity_kernel(u_arr, v_arr, dx_val, dy_val)


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
    return shearing_deformation_kernel(u_arr, v_arr, dx_val, dy_val)


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
    return stretching_deformation_kernel(u_arr, v_arr, dx_val, dy_val)


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
    u_arr = _2d(u)
    v_arr = _2d(v)
    dx_val = _mean_spacing(dx)
    dy_val = _mean_spacing(dy)
    return total_deformation_kernel(u_arr, v_arr, dx_val, dy_val)


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
    return geospatial_gradient_kernel(d, lat, lon)


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
    return geospatial_laplacian_kernel(d, lat, lon)


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
    return advection_3d_kernel(s, u_arr, v_arr, w_arr, dx_val, dy_val, dz_val)


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
    return lat_lon_grid_deltas_kernel(lon, lat)


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
    return significant_tornado_parameter_kernel(cape, lcl_h, srh, shear)


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
    return supercell_composite_parameter_kernel(cape, srh, shear)


def critical_angle(*args):
    """Critical angle between storm-relative inflow and 0-500m shear.

    Returns
    -------
    cupy.ndarray (degrees)
    """
    gpu_args = [_to_gpu(a) for a in args]
    return critical_angle_kernel(*gpu_args)


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
    return boyden_index_kernel(_to_gpu(z1000), _to_gpu(z700), _to_gpu(t700))


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
    return bulk_richardson_number_kernel(_to_gpu(cape), _to_gpu(shear_0_6km))


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
    t = _1d(temperature)
    p = _1d(pressure)
    return dendritic_growth_zone_kernel(t, p)


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
    rh = _to_gpu(relative_humidity)
    ws = _to_gpu(wind_speed_val)
    return fosberg_fire_weather_index_kernel(t, rh, ws)


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
    return freezing_rain_composite_kernel(t, p, int(precip_type))


def haines_index(t_950, t_850, td_850):
    """Haines Index (fire weather).

    Parameters
    ----------
    t_950, t_850, td_850 : float (Celsius)

    Returns
    -------
    int
    """
    return haines_index_kernel(_to_gpu(t_950), _to_gpu(t_850), _to_gpu(td_850))


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
    return hot_dry_windy_kernel(t, rh, ws, float(vpd))


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
    return warm_nose_check_kernel(t, p)


def galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst):
    """Galvez-Davison Index (tropical thunderstorm potential).

    Parameters
    ----------
    t950, t850, t700, t500, td950, td850, td700, sst : float (Celsius)

    Returns
    -------
    cupy.ndarray (dimensionless)
    """
    return galvez_davison_index_kernel(
        _to_gpu(t950), _to_gpu(t850), _to_gpu(t700), _to_gpu(t500),
        _to_gpu(td950), _to_gpu(td850), _to_gpu(td700), _to_gpu(sst),
    )


# ===========================================================================
# Grid composite operations (parallelized over grid points)
# ===========================================================================

def compute_cape_cin(pressure_3d, temperature_c_3d, qvapor_3d,
                     height_agl_3d, psfc, t2, q2,
                     parcel_type="surface", top_m=None):
    """CAPE/CIN for every grid point.

    3-D inputs: shape (nz, ny, nx).
    Returns (cape, cin, lcl_height, lfc_height) each shaped (ny, nx).
    """
    p3 = _to_gpu(pressure_3d)
    t3 = _to_gpu(temperature_c_3d)
    q3 = _to_gpu(qvapor_3d)
    h3 = _to_gpu(height_agl_3d)
    ps = _to_gpu(psfc)
    t2v = _to_gpu(t2)
    q2v = _to_gpu(q2)
    return compute_cape_cin_kernel(p3, t3, q3, h3, ps, t2v, q2v,
                                   parcel_type, top_m)


def compute_srh(u_3d, v_3d, height_agl_3d, top_m=1000.0):
    """Storm-relative helicity for every grid point.

    3-D inputs: shape (nz, ny, nx).
    Returns SRH shaped (ny, nx) in m^2/s^2.
    """
    u3 = _to_gpu(u_3d)
    v3 = _to_gpu(v_3d)
    h3 = _to_gpu(height_agl_3d)
    return compute_srh_kernel(u3, v3, h3, _scalar(top_m))


def compute_shear(u_3d, v_3d, height_agl_3d, bottom_m=0.0, top_m=6000.0):
    """Bulk wind shear for every grid point.

    3-D inputs: shape (nz, ny, nx).
    Returns shear magnitude shaped (ny, nx) in m/s.
    """
    u3 = _to_gpu(u_3d)
    v3 = _to_gpu(v_3d)
    h3 = _to_gpu(height_agl_3d)
    return compute_shear_kernel(u3, v3, h3, _scalar(bottom_m), _scalar(top_m))


def compute_lapse_rate(temperature_c_3d, qvapor_3d, height_agl_3d,
                       bottom_km=0.0, top_km=3.0):
    """Environmental lapse rate for every grid point (C/km).

    3-D inputs: shape (nz, ny, nx).
    Returns lapse rate shaped (ny, nx) in C/km.
    """
    t3 = _to_gpu(temperature_c_3d)
    q3 = _to_gpu(qvapor_3d)
    h3 = _to_gpu(height_agl_3d)
    return compute_lapse_rate_kernel(t3, q3, h3, _scalar(bottom_km),
                                     _scalar(top_km))


def compute_pw(qvapor_3d, pressure_3d):
    """Precipitable water for every grid point (mm).

    3-D inputs: shape (nz, ny, nx).
    Returns PW shaped (ny, nx) in mm.
    """
    q3 = _to_gpu(qvapor_3d)
    p3 = _to_gpu(pressure_3d)
    return compute_pw_kernel(q3, p3)


def compute_stp(cape, lcl_height, srh_1km, shear_6km):
    """Significant Tornado Parameter on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    Returns STP shaped (ny, nx), dimensionless.
    """
    c = _2d(cape)
    l = _2d(lcl_height)
    s = _2d(srh_1km)
    sh = _2d(shear_6km)
    return compute_stp_kernel(c, l, s, sh)


def compute_scp(mucape, srh_3km, shear_6km):
    """Supercell Composite Parameter on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    Returns SCP shaped (ny, nx), dimensionless.
    """
    c = _2d(mucape)
    s = _2d(srh_3km)
    sh = _2d(shear_6km)
    return compute_scp_kernel(c, s, sh)


def compute_ehi(cape, srh):
    """Energy-Helicity Index on pre-computed 2-D fields.

    All inputs: shape (ny, nx).
    Returns EHI shaped (ny, nx), dimensionless.
    """
    c = _2d(cape)
    s = _2d(srh)
    return compute_ehi_kernel(c, s)


def compute_ship(cape, shear06, t500, lr_700_500, mixing_ratio_gkg):
    """Significant Hail Parameter (SHIP) on 2-D fields.

    All inputs: shape (ny, nx).
    Returns SHIP shaped (ny, nx), dimensionless.
    """
    c = _2d(cape)
    sh = _2d(shear06)
    t5 = _2d(t500)
    lr = _2d(lr_700_500)
    mr = _2d(mixing_ratio_gkg)
    return compute_ship_kernel(c, sh, t5, lr, mr)


def compute_dcp(dcape, mu_cape, shear06, mu_mixing_ratio):
    """Derecho Composite Parameter (DCP) on 2-D fields.

    All inputs: shape (ny, nx).
    Returns DCP shaped (ny, nx), dimensionless.
    """
    d = _2d(dcape)
    mc = _2d(mu_cape)
    sh = _2d(shear06)
    mr = _2d(mu_mixing_ratio)
    return compute_dcp_kernel(d, mc, sh, mr)


def compute_grid_scp(mu_cape, srh, shear_06, mu_cin):
    """Enhanced Supercell Composite with CIN term on 2-D fields.

    All inputs: shape (ny, nx).
    Returns SCP shaped (ny, nx), dimensionless.
    """
    mc = _2d(mu_cape)
    s = _2d(srh)
    sh = _2d(shear_06)
    ci = _2d(mu_cin)
    return compute_grid_scp_kernel(mc, s, sh, ci)


def compute_grid_critical_angle(u_storm, v_storm, u_shear, v_shear):
    """Critical angle on 2-D fields.

    All inputs: shape (ny, nx).
    Returns angle in degrees (0-180).
    """
    us = _2d(u_storm)
    vs = _2d(v_storm)
    ush = _2d(u_shear)
    vsh = _2d(v_shear)
    return compute_grid_critical_angle_kernel(us, vs, ush, vsh)


def composite_reflectivity(refl_3d):
    """Composite reflectivity (column max) from a 3-D reflectivity field.

    Input: shape (nz, ny, nx) in dBZ.
    Returns composite reflectivity shaped (ny, nx).
    """
    r3 = _to_gpu(refl_3d)
    return composite_reflectivity_kernel(r3)


def composite_reflectivity_from_hydrometeors(pressure_3d, temperature_c_3d,
                                             qrain_3d, qsnow_3d, qgraup_3d):
    """Composite reflectivity from hydrometeor mixing ratios.

    All 3-D inputs: shape (nz, ny, nx).
    Returns composite reflectivity shaped (ny, nx) in dBZ.
    """
    p3 = _to_gpu(pressure_3d)
    t3 = _to_gpu(temperature_c_3d)
    qr = _to_gpu(qrain_3d)
    qs = _to_gpu(qsnow_3d)
    qg = _to_gpu(qgraup_3d)
    return composite_reflectivity_from_hydrometeors_kernel(p3, t3, qr, qs, qg)


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
    return smooth_gaussian_kernel(arr, float(sigma))


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
    return smooth_rectangular_kernel(arr, int(size), int(passes))


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
    return smooth_circular_kernel(arr, float(radius), int(passes))


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
    return smooth_n_point_kernel(arr, int(n), int(passes))


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
    return smooth_window_kernel(d_arr, w_arr, int(passes), bool(normalize_weights))


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
        gx = gradient_x_kernel(data, dx_val)
        gy = gradient_y_kernel(data, dy_val)
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
    return gradient_x_kernel(d_arr, dx_val)


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
    return gradient_y_kernel(d_arr, dy_val)


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
    return laplacian_kernel(d_arr, dx_val, dy_val)


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
    return first_derivative_kernel(d_arr, ds, int(axis))


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
    return second_derivative_kernel(d_arr, ds, int(axis))


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

def parcel_profile_with_lcl_as_dataset(pressure, temperature, dewpoint_val):
    """Calculate parcel profile and return as xarray Dataset with LCL inserted.

    Parameters
    ----------
    pressure : array-like (hPa)
    temperature : array-like (Celsius)
    dewpoint_val : array-like (Celsius)

    Returns
    -------
    xarray.Dataset
    """
    try:
        from metrust.calc import parcel_profile_with_lcl_as_dataset as _fn
        return _fn(pressure, temperature, dewpoint_val)
    except ImportError:
        import xarray as xr
        p_out, t_parcel = parcel_profile_with_lcl(pressure, temperature, dewpoint_val)
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
    "supercell_composite_parameter",
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
]
