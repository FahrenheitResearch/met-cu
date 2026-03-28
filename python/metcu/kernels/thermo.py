"""
CUDA thermodynamic kernels for met-cu.

Every function has a custom CUDA kernel -- no CPU fallbacks.
Uses CuPy ElementwiseKernel for per-element operations and
RawKernel for complex column/iterative operations.

Conventions (matching metrust):
  - Temperatures: Celsius (unless noted as Kelvin)
  - Pressures: hPa (millibars)
  - Mixing ratio: kg/kg (NOT g/kg) at the Python API boundary
  - Relative humidity: percent (0-100) at the Python API boundary
  - Specific humidity: kg/kg
"""

import cupy as cp
import numpy as np

# ---------------------------------------------------------------------------
# Physical constants (MetPy-exact, matching metrust wx-math)
# ---------------------------------------------------------------------------
RD = 287.04749097718457       # J/(kg*K) -- dry air gas constant
RV = 461.52311572606084       # J/(kg*K) -- water vapor gas constant
CP = 1004.6662184201462       # J/(kg*K) -- Cp_d
G = 9.80665                   # m/s^2
ROCP = 0.2857142857142857     # Rd/Cp = 2/7
ZEROCNK = 273.15              # 0 C in K
EPS = 0.6219569100577033      # Rd/Rv (epsilon)
LV = 2500840.0                # J/kg -- latent heat of vaporization at 0 C
LAPSE_STD = 0.0065            # K/m -- standard atmosphere lapse rate
P0_STD = 1013.25              # hPa -- standard sea level pressure
T0_STD = 288.15               # K -- standard sea level temperature

# Ambaum (2020) constants for SVP
_T0 = 273.16
_SAT_PRESSURE_0C = 611.2      # Pa
_CP_L = 4219.4
_CP_V = 1860.078011865639
_CP_I = 2090.0
_RV_METPY = 461.52311572606084
_LV_0 = 2500840.0
_LS_0 = 2834540.0

# Shared CUDA constant block injected into every kernel
_CUDA_CONSTANTS = r'''
__device__ const double RD   = 287.04749097718457;
__device__ const double RV   = 461.52311572606084;
__device__ const double CP_D = 1004.6662184201462;
__device__ const double G0   = 9.80665;
__device__ const double ROCP = 0.2857142857142857;
__device__ const double ZEROCNK = 273.15;
__device__ const double EPS  = 0.6219569100577033;
__device__ const double LV0  = 2500840.0;
__device__ const double LS0  = 2834540.0;
__device__ const double LAPSE_STD = 0.0065;
__device__ const double P0_STD   = 1013.25;
__device__ const double T0_STD   = 288.15;

// Ambaum (2020) SVP constants
__device__ const double SAT_PRESSURE_0C = 611.2;  // Pa
__device__ const double T0_TRIP = 273.16;
__device__ const double CP_L = 4219.4;
__device__ const double CP_V = 1860.078011865639;
__device__ const double CP_I = 2090.0;
__device__ const double RV_METPY = 461.52311572606084;

// Inline SVP over liquid water (Pa) -- Ambaum (2020)
__device__ double svp_liquid_pa(double t_k) {
    double latent = LV0 - (CP_L - CP_V) * (t_k - T0_TRIP);
    double heat_pow = (CP_L - CP_V) / RV_METPY;
    double exp_term = (LV0 / T0_TRIP - latent / t_k) / RV_METPY;
    return SAT_PRESSURE_0C * pow(T0_TRIP / t_k, heat_pow) * exp(exp_term);
}

// SVP in hPa from Celsius
__device__ double svp_hpa(double t_c) {
    return svp_liquid_pa(t_c + ZEROCNK) / 100.0;
}

// Saturation mixing ratio (kg/kg) from p (hPa) and T (C)
__device__ double sat_mixing_ratio(double p_hpa, double t_c) {
    double es = svp_hpa(t_c);
    double ws = EPS * es / (p_hpa - es);
    return ws > 0.0 ? ws : 0.0;
}

// SHARPpy mixing ratio (g/kg) with Wexler enhancement
__device__ double vappres_sharppy(double t) {
    double pol = t * (1.1112018e-17 + (t * -3.0994571e-20));
    pol = t * (2.1874425e-13 + (t * (-1.789232e-15 + pol)));
    pol = t * (4.3884180e-09 + (t * (-2.988388e-11 + pol)));
    pol = t * (7.8736169e-05 + (t * (-6.111796e-07 + pol)));
    pol = 0.99999683 + (t * (-9.082695e-03 + pol));
    double p8 = pol*pol; p8 *= p8; p8 *= p8;
    return 6.1078 / p8;
}

__device__ double mixratio_gkg(double p, double t) {
    double x = 0.02 * (t - 12.5 + (7500.0 / p));
    double wfw = 1.0 + (0.0000045 * p) + (0.0014 * x * x);
    double fwesw = wfw * vappres_sharppy(t);
    return 621.97 * (fwesw / (p - fwesw));
}

// Virtual temperature (Celsius)
__device__ double virtual_temp(double t, double p, double td) {
    double w = mixratio_gkg(p, td) / 1000.0;
    double tk = t + ZEROCNK;
    return tk * (1.0 + 0.61 * w) - ZEROCNK;
}

// Wobus function for moist adiabat
__device__ double wobf(double t) {
    double tc = t - 20.0;
    if (tc <= 0.0) {
        double npol = 1.0
            + tc * (-8.841660499999999e-3
                + tc * (1.4714143e-4
                    + tc * (-9.671989000000001e-7
                        + tc * (-3.2607217e-8 + tc * (-3.8598073e-10)))));
        double n2 = npol * npol;
        return 15.13 / (n2 * n2);
    } else {
        double ppol = tc
            * (4.9618922e-07
                + tc * (-6.1059365e-09
                    + tc * (3.9401551e-11
                        + tc * (-1.2588129e-13 + tc * (1.6688280e-16)))));
        ppol = 1.0 + tc * (3.6182989e-03 + tc * (-1.3603273e-05 + ppol));
        double p2 = ppol * ppol;
        return (29.93 / (p2 * p2)) + (0.96 * tc) - 14.8;
    }
}

// Saturated lift
__device__ double satlift(double p, double thetam) {
    if (p >= 1000.0) return thetam;
    double pwrp = pow(p / 1000.0, ROCP);
    double t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK;
    double e1 = wobf(t1) - wobf(thetam);
    double rate = 1.0;
    for (int iter = 0; iter < 7; iter++) {
        if (fabs(e1) < 0.001) break;
        double t2 = t1 - (e1 * rate);
        double e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK;
        e2 += wobf(t2) - wobf(e2) - thetam;
        rate = (t2 - t1) / (e2 - e1);
        t1 = t2;
        e1 = e2;
    }
    return t1 - e1 * rate;
}

// LCL temperature from T, Td (Celsius)
__device__ double lcltemp(double t, double td) {
    double s = t - td;
    double dlt = s * (1.2185 + 0.001278 * t + s * (-0.00219 + 1.173e-5 * s - 0.0000052 * t));
    return t - dlt;
}

// Dry lift to LCL: returns (p_lcl, t_lcl)
__device__ void drylift(double p, double t, double td, double *p_lcl, double *t_lcl) {
    *t_lcl = lcltemp(t, td);
    *p_lcl = 1000.0 * pow((*t_lcl + ZEROCNK) / ((t + ZEROCNK) * pow(1000.0 / p, ROCP)), 1.0 / ROCP);
}

// Dewpoint from vapor pressure (hPa) -- inverse Bolton
__device__ double dewpoint_from_vp(double e_hpa) {
    if (e_hpa <= 0.0) return -ZEROCNK;
    double ln_ratio = log(e_hpa / 6.112);
    return 243.5 * ln_ratio / (17.67 - ln_ratio);
}

// Moist lapse rate dT/dp (K/hPa)
__device__ double moist_lapse_rate(double p_hpa, double t_c) {
    double t_k = t_c + ZEROCNK;
    double es = svp_hpa(t_c);
    double rs = EPS * es / (p_hpa - es);
    if (rs < 0.0) rs = 0.0;
    double num = (RD * t_k + LV0 * rs) / p_hpa;
    double den = CP_D + (LV0 * LV0 * rs * EPS) / (RD * t_k * t_k);
    return num / den;
}

// Single RK4 step for moist adiabat
__device__ double moist_rk4_step(double p, double t, double dp) {
    double k1 = dp * moist_lapse_rate(p, t);
    double k2 = dp * moist_lapse_rate(p + dp/2.0, t + k1/2.0);
    double k3 = dp * moist_lapse_rate(p + dp/2.0, t + k2/2.0);
    double k4 = dp * moist_lapse_rate(p + dp, t + k3);
    return t + (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
}
'''

# ============================================================================
# 1. Potential temperature
# ============================================================================
potential_temperature_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 temperature',
    'float64 theta',
    '''
    double t_k = temperature + 273.15;
    double p_ratio = 1000.0 / pressure;
    theta = t_k * pow(p_ratio, 0.2857142857142857);
    ''',
    'potential_temperature_kernel'
)

# ============================================================================
# 2. Temperature from potential temperature
# ============================================================================
temperature_from_potential_temperature_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 theta',
    'float64 temperature',
    '''
    temperature = theta * pow(pressure / 1000.0, 0.2857142857142857);
    ''',
    'temperature_from_potential_temperature_kernel'
)

# ============================================================================
# 3. Virtual temperature (from T and mixing ratio kg/kg)
# ============================================================================
virtual_temperature_kernel = cp.ElementwiseKernel(
    'float64 temperature, float64 mixing_ratio',
    'float64 tv',
    '''
    double t_k = temperature + 273.15;
    double w = mixing_ratio;
    double eps = 0.6219569100577033;
    tv = t_k * (1.0 + w / eps) / (1.0 + w) - 273.15;
    ''',
    'virtual_temperature_kernel'
)

# ============================================================================
# 4. Virtual temperature from dewpoint (T, Td, P)
# ============================================================================
_vt_dewpoint_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void virtual_temperature_from_dewpoint_kernel(
    const double* temperature, const double* dewpoint, const double* pressure,
    double* tv_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double t = temperature[i];
    double p = pressure[i];
    double td = dewpoint[i];
    tv_out[i] = virtual_temp(t, p, td);
}
'''
_vt_dewpoint_raw = cp.RawKernel(_vt_dewpoint_code, 'virtual_temperature_from_dewpoint_kernel')

# ============================================================================
# 5. Virtual potential temperature (p hPa, T C, w kg/kg)
# ============================================================================
virtual_potential_temperature_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 temperature, float64 mixing_ratio',
    'float64 theta_v',
    '''
    double t_k = temperature + 273.15;
    double theta = t_k * pow(1000.0 / pressure, 0.2857142857142857);
    double w = mixing_ratio;
    theta_v = theta * (1.0 + 0.61 * w);
    ''',
    'virtual_potential_temperature_kernel'
)

# ============================================================================
# 6. Equivalent potential temperature (Bolton 1980)
# ============================================================================
_thetae_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void equivalent_potential_temperature_kernel(
    const double* pressure, const double* temperature, const double* dewpoint,
    double* thetae_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];
    double t_k = t_c + ZEROCNK;
    double td_k = td_c + ZEROCNK;
    // Bolton LCL temperature (Bolton 1980 eq 15)
    double t_lcl = 56.0 + 1.0 / (1.0/(td_k - 56.0) + log(t_k/td_k)/800.0);
    // Vapor pressure and mixing ratio at dewpoint (kg/kg)
    double e = svp_hpa(td_c);
    double r = EPS * e / (p - e);
    // Bolton (1980) eq 39
    double theta_dl = t_k * pow(1000.0/(p - e), ROCP) * pow(t_k/t_lcl, 0.28*r);
    thetae_out[i] = theta_dl * exp((3036.0/t_lcl - 1.78) * r * (1.0 + 0.448*r));
}
'''
_thetae_raw = cp.RawKernel(_thetae_code, 'equivalent_potential_temperature_kernel')

# ============================================================================
# 7. Saturation equivalent potential temperature
# ============================================================================
_sat_thetae_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void saturation_equivalent_potential_temperature_kernel(
    const double* pressure, const double* temperature,
    double* thetae_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    // Same as theta_e but with Td = T (100% RH)
    double t_k = t_c + ZEROCNK;
    double t_lcl = 56.0 + 1.0 / (1.0/(t_k - 56.0) + log(t_k/t_k)/800.0);
    // When Td=T, log(T/Td)=0, so t_lcl diverges to infinity -- use actual T
    t_lcl = t_k;  // at saturation, LCL = surface, so T_LCL = T
    double e = svp_hpa(t_c);
    double r = EPS * e / (p - e);
    double theta_dl = t_k * pow(1000.0/(p - e), ROCP) * pow(t_k/t_lcl, 0.28*r);
    thetae_out[i] = theta_dl * exp((3036.0/t_lcl - 1.78) * r * (1.0 + 0.448*r));
}
'''
_sat_thetae_raw = cp.RawKernel(_sat_thetae_code, 'saturation_equivalent_potential_temperature_kernel')

# ============================================================================
# 8. Wet bulb potential temperature (RawKernel -- iterative satlift)
# ============================================================================
_wbpt_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void wet_bulb_potential_temperature_kernel(
    const double* pressure, const double* temperature, const double* dewpoint,
    double* wbpt_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];
    // Lift to LCL
    double p_lcl, t_lcl;
    drylift(p, t_c, td_c, &p_lcl, &t_lcl);
    // Compute thetam for moist descent
    double theta_k = (t_lcl + ZEROCNK) * pow(1000.0 / p_lcl, ROCP);
    double theta_c = theta_k - ZEROCNK;
    double thetam = theta_c - wobf(theta_c) + wobf(t_lcl);
    // Descend to 1000 hPa
    double tw_1000 = satlift(1000.0, thetam);
    wbpt_out[i] = tw_1000 + ZEROCNK;  // return in K
}
'''
_wbpt_raw = cp.RawKernel(_wbpt_code, 'wet_bulb_potential_temperature_kernel')

# ============================================================================
# 9. Wet bulb temperature (RawKernel -- iterative)
# ============================================================================
_wbt_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void wet_bulb_temperature_raw_kernel(
    const double* pressure, const double* temperature, const double* dewpoint,
    double* wbt_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];
    // Lift to LCL
    double p_lcl, t_lcl;
    drylift(p, t_c, td_c, &p_lcl, &t_lcl);
    // Compute thetam for moist descent
    double theta_k = (t_lcl + ZEROCNK) * pow(1000.0 / p_lcl, ROCP);
    double theta_c = theta_k - ZEROCNK;
    double thetam = theta_c - wobf(theta_c) + wobf(t_lcl);
    // Descend to original pressure
    wbt_out[i] = satlift(p, thetam);
}
'''
_wbt_raw = cp.RawKernel(_wbt_code, 'wet_bulb_temperature_raw_kernel')

# ============================================================================
# 10. Saturation vapor pressure (Ambaum 2020)
# ============================================================================
_svp_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void saturation_vapor_pressure_kernel(
    const double* temperature, double* es_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    es_out[i] = svp_hpa(temperature[i]);
}
'''
_svp_raw = cp.RawKernel(_svp_code, 'saturation_vapor_pressure_kernel')

# ============================================================================
# 11. Vapor pressure (e from mixing ratio: e = w*p/(epsilon+w))
# ============================================================================
vapor_pressure_from_mixing_ratio_kernel = cp.ElementwiseKernel(
    'float64 mixing_ratio, float64 pressure',
    'float64 e',
    '''
    double eps = 0.6219569100577033;
    e = mixing_ratio * pressure / (eps + mixing_ratio);
    ''',
    'vapor_pressure_from_mixing_ratio_kernel'
)

# Vapor pressure from dewpoint (= SVP at Td)
_vp_from_td_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void vapor_pressure_from_dewpoint_kernel(
    const double* dewpoint, double* e_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    e_out[i] = svp_hpa(dewpoint[i]);
}
'''
_vp_from_td_raw = cp.RawKernel(_vp_from_td_code, 'vapor_pressure_from_dewpoint_kernel')

# ============================================================================
# 12. Dewpoint from vapor pressure (inverse Bolton)
# ============================================================================
dewpoint_kernel = cp.ElementwiseKernel(
    'float64 vapor_pressure',
    'float64 td',
    '''
    double ln_ratio = log(vapor_pressure / 6.112);
    td = 243.5 * ln_ratio / (17.67 - ln_ratio);
    ''',
    'dewpoint_kernel'
)

# ============================================================================
# 13. Dewpoint from relative humidity
# ============================================================================
_td_rh_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void dewpoint_from_relative_humidity_kernel(
    const double* temperature, const double* rh, double* td_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double es = svp_hpa(temperature[i]);
    double e = (rh[i] / 100.0) * es;
    double ln_ratio = log(e / 6.112);
    td_out[i] = 243.5 * ln_ratio / (17.67 - ln_ratio);
}
'''
_td_rh_raw = cp.RawKernel(_td_rh_code, 'dewpoint_from_relative_humidity_kernel')

# ============================================================================
# 14. Dewpoint from specific humidity
# ============================================================================
_td_q_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void dewpoint_from_specific_humidity_kernel(
    const double* pressure, const double* specific_humidity, double* td_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double q = specific_humidity[i];
    double p = pressure[i];
    double w = q / (1.0 - q);
    double e = w * p / (EPS + w);
    td_out[i] = dewpoint_from_vp(e);
}
'''
_td_q_raw = cp.RawKernel(_td_q_code, 'dewpoint_from_specific_humidity_kernel')

# ============================================================================
# 15. Mixing ratio (from vapor pressure and total pressure)
# ============================================================================
mixing_ratio_kernel = cp.ElementwiseKernel(
    'float64 vapor_pres, float64 total_pres',
    'float64 w',
    '''
    double eps = 0.6219569100577033;
    w = eps * vapor_pres / (total_pres - vapor_pres);
    ''',
    'mixing_ratio_kernel'
)

# ============================================================================
# 16. Saturation mixing ratio
# ============================================================================
_sat_mr_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void saturation_mixing_ratio_kernel(
    const double* pressure, const double* temperature, double* ws_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    ws_out[i] = sat_mixing_ratio(pressure[i], temperature[i]);
}
'''
_sat_mr_raw = cp.RawKernel(_sat_mr_code, 'saturation_mixing_ratio_kernel')

# ============================================================================
# 17. Mixing ratio from relative humidity
# ============================================================================
_mr_rh_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void mixing_ratio_from_relative_humidity_kernel(
    const double* pressure, const double* temperature, const double* rh,
    double* w_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double ws = sat_mixing_ratio(pressure[i], temperature[i]);
    w_out[i] = ws * rh[i] / 100.0;
}
'''
_mr_rh_raw = cp.RawKernel(_mr_rh_code, 'mixing_ratio_from_relative_humidity_kernel')

# ============================================================================
# 18. Mixing ratio from specific humidity
# ============================================================================
mixing_ratio_from_specific_humidity_kernel = cp.ElementwiseKernel(
    'float64 specific_humidity',
    'float64 w',
    '''
    w = specific_humidity / (1.0 - specific_humidity);
    ''',
    'mixing_ratio_from_specific_humidity_kernel'
)

# ============================================================================
# 19. Specific humidity from dewpoint
# ============================================================================
_q_td_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void specific_humidity_from_dewpoint_kernel(
    const double* pressure, const double* dewpoint, double* q_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double e = svp_hpa(dewpoint[i]);
    double w = EPS * e / (pressure[i] - e);
    q_out[i] = w / (1.0 + w);
}
'''
_q_td_raw = cp.RawKernel(_q_td_code, 'specific_humidity_from_dewpoint_kernel')

# ============================================================================
# 20. Specific humidity from mixing ratio
# ============================================================================
specific_humidity_from_mixing_ratio_kernel = cp.ElementwiseKernel(
    'float64 mixing_ratio',
    'float64 q',
    '''
    q = mixing_ratio / (1.0 + mixing_ratio);
    ''',
    'specific_humidity_from_mixing_ratio_kernel'
)

# ============================================================================
# 21. Relative humidity from dewpoint
# ============================================================================
_rh_td_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void relative_humidity_from_dewpoint_kernel(
    const double* temperature, const double* dewpoint, double* rh_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double es_t = svp_hpa(temperature[i]);
    double es_td = svp_hpa(dewpoint[i]);
    rh_out[i] = (es_td / es_t) * 100.0;
}
'''
_rh_td_raw = cp.RawKernel(_rh_td_code, 'relative_humidity_from_dewpoint_kernel')

# ============================================================================
# 22. Relative humidity from mixing ratio
# ============================================================================
_rh_mr_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void relative_humidity_from_mixing_ratio_kernel(
    const double* pressure, const double* temperature, const double* mixing_ratio,
    double* rh_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double ws = sat_mixing_ratio(pressure[i], temperature[i]);
    rh_out[i] = (ws > 0.0) ? (mixing_ratio[i] / ws) * 100.0 : 0.0;
}
'''
_rh_mr_raw = cp.RawKernel(_rh_mr_code, 'relative_humidity_from_mixing_ratio_kernel')

# ============================================================================
# 23. Relative humidity from specific humidity
# ============================================================================
_rh_q_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void relative_humidity_from_specific_humidity_kernel(
    const double* pressure, const double* temperature, const double* specific_humidity,
    double* rh_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double q = specific_humidity[i];
    double w = q / (1.0 - q);
    double ws = sat_mixing_ratio(pressure[i], temperature[i]);
    rh_out[i] = (ws > 0.0) ? (w / ws) * 100.0 : 0.0;
}
'''
_rh_q_raw = cp.RawKernel(_rh_q_code, 'relative_humidity_from_specific_humidity_kernel')

# ============================================================================
# 24. Density
# ============================================================================
density_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 temperature, float64 mixing_ratio',
    'float64 rho',
    '''
    double p_pa = pressure * 100.0;
    double t_k = temperature + 273.15;
    double w = mixing_ratio;
    double tv_k = t_k * (1.0 + 0.61 * w);
    rho = p_pa / (287.04749097718457 * tv_k);
    ''',
    'density_kernel'
)

# ============================================================================
# 25. Dry static energy
# ============================================================================
dry_static_energy_kernel = cp.ElementwiseKernel(
    'float64 height, float64 temperature',
    'float64 dse',
    '''
    dse = 1004.6662184201462 * temperature + 9.80665 * height;
    ''',
    'dry_static_energy_kernel'
)

# ============================================================================
# 26. Moist static energy
# ============================================================================
moist_static_energy_kernel = cp.ElementwiseKernel(
    'float64 height, float64 temperature, float64 specific_humidity',
    'float64 mse',
    '''
    mse = 1004.6662184201462 * temperature + 9.80665 * height + 2500840.0 * specific_humidity;
    ''',
    'moist_static_energy_kernel'
)

# ============================================================================
# 27. Exner function
# ============================================================================
exner_function_kernel = cp.ElementwiseKernel(
    'float64 pressure',
    'float64 exner',
    '''
    exner = pow(pressure / 1000.0, 0.2857142857142857);
    ''',
    'exner_function_kernel'
)

# ============================================================================
# 28. Dry lapse (T at new pressure assuming dry adiabatic from surface)
# ============================================================================
dry_lapse_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 reference_pressure, float64 t_surface',
    'float64 t_out',
    '''
    double t_k = t_surface + 273.15;
    t_out = t_k * pow(pressure / reference_pressure, 0.2857142857142857) - 273.15;
    ''',
    'dry_lapse_kernel'
)

# ============================================================================
# 29. Height to pressure (standard atmosphere)
# ============================================================================
height_to_pressure_std_kernel = cp.ElementwiseKernel(
    'float64 height',
    'float64 pressure',
    '''
    double P0 = 1013.25;
    double T0 = 288.15;
    double L  = 0.0065;
    double Rd = 287.04749097718457;
    double g  = 9.80665;
    pressure = P0 * pow(1.0 - L * height / T0, g / (Rd * L));
    ''',
    'height_to_pressure_std_kernel'
)

# ============================================================================
# 30. Pressure to height (standard atmosphere)
# ============================================================================
pressure_to_height_std_kernel = cp.ElementwiseKernel(
    'float64 pressure',
    'float64 height',
    '''
    double P0 = 1013.25;
    double T0 = 288.15;
    double L  = 0.0065;
    double Rd = 287.04749097718457;
    double g  = 9.80665;
    height = (T0 / L) * (1.0 - pow(pressure / P0, (Rd * L) / g));
    ''',
    'pressure_to_height_std_kernel'
)

# ============================================================================
# 31. Add height to pressure (hypsometric via standard atmosphere)
# ============================================================================
add_height_to_pressure_kernel = cp.ElementwiseKernel(
    'float64 pressure, float64 delta_height',
    'float64 p_new',
    '''
    double P0 = 1013.25;
    double T0 = 288.15;
    double L  = 0.0065;
    double Rd = 287.04749097718457;
    double g  = 9.80665;
    // Convert p to height, add delta, convert back
    double h = (T0 / L) * (1.0 - pow(pressure / P0, (Rd * L) / g));
    p_new = P0 * pow(1.0 - L * (h + delta_height) / T0, g / (Rd * L));
    ''',
    'add_height_to_pressure_kernel'
)

# ============================================================================
# 32. Add pressure to height (hypsometric via standard atmosphere)
# ============================================================================
add_pressure_to_height_kernel = cp.ElementwiseKernel(
    'float64 height, float64 delta_pressure',
    'float64 h_new',
    '''
    double P0 = 1013.25;
    double T0 = 288.15;
    double L  = 0.0065;
    double Rd = 287.04749097718457;
    double g  = 9.80665;
    double p = P0 * pow(1.0 - L * height / T0, g / (Rd * L));
    h_new = (T0 / L) * (1.0 - pow((p + delta_pressure) / P0, (Rd * L) / g));
    ''',
    'add_pressure_to_height_kernel'
)

# ============================================================================
# 33. Altimeter to sea level pressure
# ============================================================================
altimeter_to_sea_level_pressure_kernel = cp.ElementwiseKernel(
    'float64 altimeter, float64 elevation, float64 temperature',
    'float64 slp',
    '''
    double ROCP_v = 0.2857142857142857;
    double T0_v = 288.15;
    double L = 0.0065;
    double g = 9.80665;
    double Rd = 287.058;
    // Step 1: altimeter to station pressure (Smithsonian)
    double n = 1.0 / ROCP_v;
    double ratio = 1.0 - (L * elevation) / (T0_v + L * elevation);
    double p_stn = altimeter * pow(ratio, 1.0 / ROCP_v) + 0.3;
    // Step 2: hypsometric SLP
    double t_sfc_k = temperature + 273.15;
    double t_mean_k = t_sfc_k + 0.5 * L * elevation;
    slp = p_stn * exp(g * elevation / (Rd * t_mean_k));
    ''',
    'altimeter_to_sea_level_pressure_kernel'
)

# ============================================================================
# 34. Altimeter to station pressure
# ============================================================================
altimeter_to_station_pressure_kernel = cp.ElementwiseKernel(
    'float64 altimeter, float64 elevation',
    'float64 p_stn',
    '''
    double k = 0.2857142857142857;
    double T0_v = 288.15;
    double L = 0.0065;
    double ratio = 1.0 - (L * elevation) / (T0_v + L * elevation);
    p_stn = altimeter * pow(ratio, 1.0 / k);
    ''',
    'altimeter_to_station_pressure_kernel'
)

# ============================================================================
# 35. Station to altimeter pressure
# ============================================================================
station_to_altimeter_pressure_kernel = cp.ElementwiseKernel(
    'float64 station_pressure, float64 elevation',
    'float64 altimeter',
    '''
    double BARO_EXP = 0.190284;
    double P0 = 1013.25;
    double LAPSE = 0.0065;
    double T0_v = 288.15;
    double n = 1.0 / BARO_EXP;
    double term = pow(station_pressure - 0.3, n) + pow(P0, n) * LAPSE * elevation / T0_v;
    altimeter = pow(term, 1.0 / n);
    ''',
    'station_to_altimeter_pressure_kernel'
)

# ============================================================================
# 36. Sigma to pressure
# ============================================================================
sigma_to_pressure_kernel = cp.ElementwiseKernel(
    'float64 sigma, float64 psfc, float64 ptop',
    'float64 pressure',
    '''
    pressure = sigma * (psfc - ptop) + ptop;
    ''',
    'sigma_to_pressure_kernel'
)

# ============================================================================
# 37. Geopotential to height
# ============================================================================
geopotential_to_height_kernel = cp.ElementwiseKernel(
    'float64 geopotential',
    'float64 height',
    '''
    height = geopotential / 9.80665;
    ''',
    'geopotential_to_height_kernel'
)

# ============================================================================
# 38. Height to geopotential
# ============================================================================
height_to_geopotential_kernel = cp.ElementwiseKernel(
    'float64 height',
    'float64 geopotential',
    '''
    geopotential = 9.80665 * height;
    ''',
    'height_to_geopotential_kernel'
)

# ============================================================================
# 39. Scale height
# ============================================================================
scale_height_kernel = cp.ElementwiseKernel(
    'float64 temperature',
    'float64 H',
    '''
    H = 287.04749097718457 * temperature / 9.80665;
    ''',
    'scale_height_kernel'
)

# ============================================================================
# 40. Thickness hydrostatic
# ============================================================================
thickness_hydrostatic_kernel = cp.ElementwiseKernel(
    'float64 p_bottom, float64 p_top, float64 t_mean_k',
    'float64 dz',
    '''
    dz = (287.04749097718457 * t_mean_k / 9.80665) * log(p_bottom / p_top);
    ''',
    'thickness_hydrostatic_kernel'
)

# ============================================================================
# 41. Brunt-Vaisala frequency squared (per-element finite difference)
# ============================================================================
_bvf2_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void brunt_vaisala_frequency_squared_kernel(
    const double* height, const double* theta, double* n2_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double dtheta, dz;
    if (i == 0) {
        dtheta = theta[1] - theta[0];
        dz = height[1] - height[0];
    } else if (i == n - 1) {
        dtheta = theta[n-1] - theta[n-2];
        dz = height[n-1] - height[n-2];
    } else {
        dtheta = theta[i+1] - theta[i-1];
        dz = height[i+1] - height[i-1];
    }
    if (fabs(dz) < 1e-10 || fabs(theta[i]) < 1e-10)
        n2_out[i] = 0.0;
    else
        n2_out[i] = (G0 / theta[i]) * (dtheta / dz);
}
'''
_bvf2_raw = cp.RawKernel(_bvf2_code, 'brunt_vaisala_frequency_squared_kernel')

# ============================================================================
# 42. Brunt-Vaisala frequency
# ============================================================================
_bvf_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void brunt_vaisala_frequency_kernel(
    const double* height, const double* theta, double* n_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double dtheta, dz;
    if (i == 0) {
        dtheta = theta[1] - theta[0];
        dz = height[1] - height[0];
    } else if (i == n - 1) {
        dtheta = theta[n-1] - theta[n-2];
        dz = height[n-1] - height[n-2];
    } else {
        dtheta = theta[i+1] - theta[i-1];
        dz = height[i+1] - height[i-1];
    }
    if (fabs(dz) < 1e-10 || fabs(theta[i]) < 1e-10)
        n_out[i] = 0.0;
    else {
        double n2 = (G0 / theta[i]) * (dtheta / dz);
        n_out[i] = (n2 > 0.0) ? sqrt(n2) : 0.0;
    }
}
'''
_bvf_raw = cp.RawKernel(_bvf_code, 'brunt_vaisala_frequency_kernel')

# ============================================================================
# 43. Brunt-Vaisala period
# ============================================================================
brunt_vaisala_period_kernel = cp.ElementwiseKernel(
    'float64 bvf',
    'float64 period',
    '''
    period = (bvf > 0.0) ? (2.0 * 3.14159265358979323846 / bvf) : 1.0e30;
    ''',
    'brunt_vaisala_period_kernel'
)

# ============================================================================
# 44. Static stability
# ============================================================================
_static_stab_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void static_stability_kernel(
    const double* pressure, const double* temperature_k,
    double* sigma_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    // Compute theta at this level
    double theta_i = temperature_k[i] * pow(1000.0 / pressure[i], ROCP);
    double dtheta, dp;
    if (i == 0) {
        double theta_1 = temperature_k[1] * pow(1000.0 / pressure[1], ROCP);
        dtheta = theta_1 - theta_i;
        dp = pressure[1] - pressure[0];
    } else if (i == n - 1) {
        double theta_prev = temperature_k[n-2] * pow(1000.0 / pressure[n-2], ROCP);
        dtheta = theta_i - theta_prev;
        dp = pressure[n-1] - pressure[n-2];
    } else {
        double theta_prev = temperature_k[i-1] * pow(1000.0 / pressure[i-1], ROCP);
        double theta_next = temperature_k[i+1] * pow(1000.0 / pressure[i+1], ROCP);
        dtheta = theta_next - theta_prev;
        dp = pressure[i+1] - pressure[i-1];
    }
    if (fabs(dp) < 1e-10 || fabs(theta_i) < 1e-10)
        sigma_out[i] = 0.0;
    else
        sigma_out[i] = -(temperature_k[i] / theta_i) * (dtheta / (dp * 100.0));
}
'''
_static_stab_raw = cp.RawKernel(_static_stab_code, 'static_stability_kernel')

# ============================================================================
# 45. Vertical velocity (omega to w)
# ============================================================================
vertical_velocity_kernel = cp.ElementwiseKernel(
    'float64 omega, float64 pressure, float64 temperature',
    'float64 w',
    '''
    double t_k = temperature + 273.15;
    double p_pa = pressure * 100.0;
    double rho = p_pa / (287.04749097718457 * t_k);
    w = -omega / (rho * 9.80665);
    ''',
    'vertical_velocity_kernel'
)

# ============================================================================
# 46. Vertical velocity pressure (w to omega)
# ============================================================================
vertical_velocity_pressure_kernel = cp.ElementwiseKernel(
    'float64 w, float64 pressure, float64 temperature',
    'float64 omega',
    '''
    double t_k = temperature + 273.15;
    double p_pa = pressure * 100.0;
    double rho = p_pa / (287.04749097718457 * t_k);
    omega = -rho * 9.80665 * w;
    ''',
    'vertical_velocity_pressure_kernel'
)

# ============================================================================
# 47. Montgomery streamfunction
# ============================================================================
montgomery_streamfunction_kernel = cp.ElementwiseKernel(
    'float64 height, float64 temperature',
    'float64 psi',
    '''
    psi = 1004.6662184201462 * temperature + 9.80665 * height;
    ''',
    'montgomery_streamfunction_kernel'
)

# ============================================================================
# 48. Heat index (NWS Rothfusz regression) -- input Celsius, output Celsius
# ============================================================================
heat_index_kernel = cp.ElementwiseKernel(
    'float64 temperature, float64 relative_humidity',
    'float64 hi',
    '''
    double t_f = temperature * 9.0 / 5.0 + 32.0;
    double rh = relative_humidity;
    double hi_f;
    if (t_f < 80.0) {
        hi_f = 0.5 * (t_f + 61.0 + (t_f - 68.0) * 1.2 + rh * 0.094);
    } else {
        hi_f = -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh
            - 0.22475541 * t_f * rh
            - 0.00683783 * t_f * t_f
            - 0.05481717 * rh * rh
            + 0.00122874 * t_f * t_f * rh
            + 0.00085282 * t_f * rh * rh
            - 0.00000199 * t_f * t_f * rh * rh;
        if (rh < 13.0 && t_f >= 80.0 && t_f <= 112.0) {
            hi_f -= ((13.0 - rh) / 4.0) * sqrt((17.0 - fabs(t_f - 95.0)) / 17.0);
        } else if (rh > 85.0 && t_f >= 80.0 && t_f <= 87.0) {
            hi_f += ((rh - 85.0) / 10.0) * ((87.0 - t_f) / 5.0);
        }
    }
    hi = (hi_f - 32.0) * 5.0 / 9.0;
    ''',
    'heat_index_kernel'
)

# ============================================================================
# 49. Wind chill (NWS formula) -- input C and m/s, output C
# ============================================================================
windchill_kernel = cp.ElementwiseKernel(
    'float64 temperature, float64 wind_speed',
    'float64 wc',
    '''
    double wind_kmh = wind_speed * 3.6;
    double spf = pow(wind_kmh, 0.16);
    wc = (0.6215 + 0.3965 * spf) * temperature - 11.37 * spf + 13.12;
    ''',
    'windchill_kernel'
)

# ============================================================================
# 50. Apparent temperature
# ============================================================================
_apparent_temp_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void apparent_temperature_kernel(
    const double* temperature, const double* rh, const double* wind_speed,
    double* at_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double t_c = temperature[i];
    double rh_pct = rh[i];
    double ws = wind_speed[i];
    double t_f = t_c * 9.0 / 5.0 + 32.0;
    double wind_mph = ws * 2.23694;

    if (t_f >= 80.0) {
        // Heat index
        double hi_f = -42.379
            + 2.04901523 * t_f
            + 10.14333127 * rh_pct
            - 0.22475541 * t_f * rh_pct
            - 0.00683783 * t_f * t_f
            - 0.05481717 * rh_pct * rh_pct
            + 0.00122874 * t_f * t_f * rh_pct
            + 0.00085282 * t_f * rh_pct * rh_pct
            - 0.00000199 * t_f * t_f * rh_pct * rh_pct;
        if (rh_pct < 13.0 && t_f >= 80.0 && t_f <= 112.0) {
            hi_f -= ((13.0 - rh_pct) / 4.0) * sqrt((17.0 - fabs(t_f - 95.0)) / 17.0);
        } else if (rh_pct > 85.0 && t_f >= 80.0 && t_f <= 87.0) {
            hi_f += ((rh_pct - 85.0) / 10.0) * ((87.0 - t_f) / 5.0);
        }
        at_out[i] = (hi_f - 32.0) * 5.0 / 9.0;
    } else if (t_f <= 50.0 && wind_mph > 3.0) {
        // Wind chill
        double wind_kmh = ws * 3.6;
        double spf = pow(wind_kmh, 0.16);
        at_out[i] = (0.6215 + 0.3965 * spf) * t_c - 11.37 * spf + 13.12;
    } else {
        at_out[i] = t_c;
    }
}
'''
_apparent_temp_raw = cp.RawKernel(_apparent_temp_code, 'apparent_temperature_kernel')

# ============================================================================
# 51. Frost point
# ============================================================================
_frost_point_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void frost_point_kernel(
    const double* temperature, const double* rh, double* fp_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double es_water = svp_hpa(temperature[i]);
    double e = (rh[i] / 100.0) * es_water;
    // Invert Magnus over ice: ei = 6.112 * exp(22.46*T/(T+272.62))
    double ln_ratio = log(e / 6.112);
    fp_out[i] = 272.62 * ln_ratio / (22.46 - ln_ratio);
}
'''
_frost_point_raw = cp.RawKernel(_frost_point_code, 'frost_point_kernel')

# ============================================================================
# 52. Psychrometric vapor pressure
# ============================================================================
_psychro_vp_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void psychrometric_vapor_pressure_kernel(
    const double* temperature, const double* wet_bulb, const double* pressure,
    double* e_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double es_tw = svp_hpa(wet_bulb[i]);
    double A = 6.6e-4;
    e_out[i] = es_tw - A * pressure[i] * (temperature[i] - wet_bulb[i]);
}
'''
_psychro_vp_raw = cp.RawKernel(_psychro_vp_code, 'psychrometric_vapor_pressure_kernel')

# ============================================================================
# 53. Water latent heat of vaporization (T-dependent)
# ============================================================================
water_latent_heat_vaporization_kernel = cp.ElementwiseKernel(
    'float64 temperature',
    'float64 lv',
    '''
    lv = 2.501e6 - 2370.0 * temperature;
    ''',
    'water_latent_heat_vaporization_kernel'
)

# ============================================================================
# 54. Water latent heat of sublimation
# ============================================================================
water_latent_heat_sublimation_kernel = cp.ElementwiseKernel(
    'float64 temperature',
    'float64 ls',
    '''
    double lv = 2.501e6 - 2370.0 * temperature;
    double lf = 3.34e5 + 2106.0 * temperature;
    ls = lv + lf;
    ''',
    'water_latent_heat_sublimation_kernel'
)

# ============================================================================
# 55. Water latent heat of melting
# ============================================================================
water_latent_heat_melting_kernel = cp.ElementwiseKernel(
    'float64 temperature',
    'float64 lf',
    '''
    lf = 3.34e5 + 2106.0 * temperature;
    ''',
    'water_latent_heat_melting_kernel'
)

# ============================================================================
# 56. Moist air gas constant
# ============================================================================
moist_air_gas_constant_kernel = cp.ElementwiseKernel(
    'float64 mixing_ratio',
    'float64 r_moist',
    '''
    double Rd = 287.058;
    double eps = 0.622;
    r_moist = Rd * (1.0 + mixing_ratio / eps) / (1.0 + mixing_ratio);
    ''',
    'moist_air_gas_constant_kernel'
)

# ============================================================================
# 57. Moist air specific heat at constant pressure
# ============================================================================
moist_air_specific_heat_pressure_kernel = cp.ElementwiseKernel(
    'float64 mixing_ratio',
    'float64 cp_moist',
    '''
    double Cp_d = 1005.7;
    double Cp_v = 1875.0;
    cp_moist = Cp_d * (1.0 + (Cp_v / Cp_d) * mixing_ratio) / (1.0 + mixing_ratio);
    ''',
    'moist_air_specific_heat_pressure_kernel'
)

# ============================================================================
# 58. Moist air Poisson exponent
# ============================================================================
moist_air_poisson_exponent_kernel = cp.ElementwiseKernel(
    'float64 mixing_ratio',
    'float64 kappa',
    '''
    double Rd = 287.058;
    double eps = 0.622;
    double Cp_d = 1005.7;
    double Cp_v = 1875.0;
    double r_m = Rd * (1.0 + mixing_ratio / eps) / (1.0 + mixing_ratio);
    double cp_m = Cp_d * (1.0 + (Cp_v / Cp_d) * mixing_ratio) / (1.0 + mixing_ratio);
    kappa = r_m / cp_m;
    ''',
    'moist_air_poisson_exponent_kernel'
)

# ============================================================================
# 59. Moist lapse (RawKernel -- RK4 integration per column)
# ============================================================================
_moist_lapse_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void moist_lapse_kernel(
    const double* pressure,   // (nlevels,)
    const double* t_start,    // (ncols,) starting temperature in C
    double* t_out,            // (ncols, nlevels) output temperatures in C
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double t = t_start[col];
    t_out[col * nlevels] = t;

    for (int k = 1; k < nlevels; k++) {
        double dp = pressure[k] - pressure[k-1];
        if (fabs(dp) < 1e-10) {
            t_out[col * nlevels + k] = t;
            continue;
        }
        // RK4 with sub-steps
        int n_steps = (int)(fabs(dp) / 5.0);
        if (n_steps < 4) n_steps = 4;
        double h = dp / (double)n_steps;
        double pc = pressure[k-1];
        for (int s = 0; s < n_steps; s++) {
            double k1 = h * moist_lapse_rate(pc, t);
            double k2 = h * moist_lapse_rate(pc + h/2.0, t + k1/2.0);
            double k3 = h * moist_lapse_rate(pc + h/2.0, t + k2/2.0);
            double k4 = h * moist_lapse_rate(pc + h, t + k3);
            t += (k1 + 2.0*k2 + 2.0*k3 + k4) / 6.0;
            pc += h;
        }
        t_out[col * nlevels + k] = t;
    }
}
'''
_moist_lapse_raw = cp.RawKernel(_moist_lapse_code, 'moist_lapse_kernel')

# ============================================================================
# 60. Parcel profile (RawKernel -- dry below LCL, moist above)
# ============================================================================
_parcel_profile_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void parcel_profile_kernel(
    const double* pressure,     // (nlevels,) hPa, surface first
    const double* t_surface,    // (ncols,) surface T in C
    const double* td_surface,   // (ncols,) surface Td in C
    double* t_out,              // (ncols, nlevels) parcel T in C
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double t_sfc = t_surface[col];
    double td_sfc = td_surface[col];
    double p_sfc = pressure[0];
    double t_k_sfc = t_sfc + ZEROCNK;

    // Find LCL
    double p_lcl, t_lcl;
    drylift(p_sfc, t_sfc, td_sfc, &p_lcl, &t_lcl);

    // Compute thetam for moist ascent
    double theta_dry_k = t_k_sfc * pow(1000.0 / p_sfc, ROCP);

    // Moist adiabat: build from LCL upward using RK4
    double t_moist = t_lcl;

    // We do two passes: first dry, then moist
    // Track the moist adiabat temperature as we go up
    double prev_p = p_lcl;
    double prev_t_moist = t_lcl;

    for (int k = 0; k < nlevels; k++) {
        double p = pressure[k];
        if (p > p_lcl) {
            // Below LCL: dry adiabat
            t_out[col * nlevels + k] = theta_dry_k * pow(p / 1000.0, ROCP) - ZEROCNK;
        } else {
            // Above LCL: moist adiabat via RK4
            double dp = p - prev_p;
            if (fabs(dp) < 1e-10) {
                t_out[col * nlevels + k] = prev_t_moist;
                continue;
            }
            int n_steps = (int)(fabs(dp) / 5.0);
            if (n_steps < 4) n_steps = 4;
            double h = dp / (double)n_steps;
            double pc = prev_p;
            double tc = prev_t_moist;
            for (int s = 0; s < n_steps; s++) {
                double rk1 = h * moist_lapse_rate(pc, tc);
                double rk2 = h * moist_lapse_rate(pc + h/2.0, tc + rk1/2.0);
                double rk3 = h * moist_lapse_rate(pc + h/2.0, tc + rk2/2.0);
                double rk4 = h * moist_lapse_rate(pc + h, tc + rk3);
                tc += (rk1 + 2.0*rk2 + 2.0*rk3 + rk4) / 6.0;
                pc += h;
            }
            t_out[col * nlevels + k] = tc;
            prev_p = p;
            prev_t_moist = tc;
        }
    }
}
'''
_parcel_profile_raw = cp.RawKernel(_parcel_profile_code, 'parcel_profile_kernel')

# ============================================================================
# 61. CAPE/CIN (RawKernel -- full column integration with LFC-bounded CIN)
# ============================================================================
_cape_cin_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void cape_cin_kernel(
    const double* pressure,    // (nlevels,) in hPa
    const double* temperature, // (ncols, nlevels) in Celsius
    const double* dewpoint,    // (ncols, nlevels) in Celsius
    double* cape_out,          // (ncols,)
    double* cin_out,           // (ncols,)
    double* lcl_out,           // (ncols,) pressure in hPa
    double* lfc_out,           // (ncols,) pressure in hPa
    double* el_out,            // (ncols,) pressure in hPa
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double sfc_t  = temperature[col * nlevels];
    double sfc_td = dewpoint[col * nlevels];
    double sfc_p  = pressure[0];

    // Ensure Td <= T
    if (sfc_td > sfc_t) sfc_td = sfc_t;

    // 1. Compute LCL
    double p_lcl, t_lcl;
    drylift(sfc_p, sfc_t, sfc_td, &p_lcl, &t_lcl);
    lcl_out[col] = p_lcl;

    // 2. Build parcel profile: dry below LCL, moist above
    double theta_dry_k = (sfc_t + ZEROCNK) * pow(1000.0 / sfc_p, ROCP);
    double r_parcel_gkg = mixratio_gkg(sfc_p, sfc_td);
    double w_kgkg = r_parcel_gkg / 1000.0;

    // Compute parcel Tv and env Tv at each level
    // Also compute heights from hypsometric equation
    // Use stack-allocated arrays via local memory (max 200 levels)
    // For larger profiles we'd need shared/global, but 200 is typical
    double tv_parc[200];
    double tv_env[200];
    double z[200];

    int nlev = nlevels;
    if (nlev > 200) nlev = 200;

    // Environment Tv and initial height
    for (int k = 0; k < nlev; k++) {
        double t_e = temperature[col * nlevels + k];
        double td_e = dewpoint[col * nlevels + k];
        if (td_e > t_e) td_e = t_e;
        tv_env[k] = virtual_temp(t_e, pressure[k], td_e);
    }

    z[0] = 0.0;
    for (int k = 1; k < nlev; k++) {
        if (pressure[k] <= 0.0 || pressure[k-1] <= 0.0) {
            z[k] = z[k-1];
            continue;
        }
        double tv_mean = (tv_env[k-1] + tv_env[k]) / 2.0 + ZEROCNK;
        z[k] = z[k-1] + (RD * tv_mean / G0) * log(pressure[k-1] / pressure[k]);
    }

    // Parcel Tv: dry below LCL, moist above via RK4
    double moist_t = t_lcl;
    double moist_p = p_lcl;

    for (int k = 0; k < nlev; k++) {
        double p = pressure[k];
        if (p <= 0.0) { tv_parc[k] = -9999.0; continue; }
        if (p >= p_lcl) {
            // Below LCL: dry adiabat with moisture correction
            double t_parc_k = theta_dry_k * pow(p / 1000.0, ROCP);
            double t_parc = t_parc_k - ZEROCNK;
            tv_parc[k] = (t_parc + ZEROCNK) * (1.0 + w_kgkg / EPS) / (1.0 + w_kgkg) - ZEROCNK;
        } else {
            // Above LCL: moist adiabat via RK4 sub-stepping at 10 hPa
            double dp = p - moist_p;
            if (fabs(dp) > 1e-10) {
                int n_steps = (int)(fabs(dp) / 10.0);
                if (n_steps < 4) n_steps = 4;
                double h = dp / (double)n_steps;
                double pc = moist_p;
                double tc = moist_t;
                for (int s = 0; s < n_steps; s++) {
                    double rk1 = h * moist_lapse_rate(pc, tc);
                    double rk2 = h * moist_lapse_rate(pc + h/2.0, tc + rk1/2.0);
                    double rk3 = h * moist_lapse_rate(pc + h/2.0, tc + rk2/2.0);
                    double rk4 = h * moist_lapse_rate(pc + h, tc + rk3);
                    tc += (rk1 + 2.0*rk2 + 2.0*rk3 + rk4) / 6.0;
                    pc += h;
                }
                moist_t = tc;
                moist_p = p;
            }
            // Saturated parcel: Td = T
            tv_parc[k] = virtual_temp(moist_t, p, moist_t);
        }
    }

    // 3. Find the LAST negative->positive buoyancy crossing (the true LFC)
    int last_lfc_idx = -1;
    for (int k = 1; k < nlev; k++) {
        if (tv_parc[k] < -9000.0 || tv_parc[k-1] < -9000.0) continue;
        double buoy     = (tv_parc[k]   + ZEROCNK) - (tv_env[k]   + ZEROCNK);
        double buoy_prev = (tv_parc[k-1] + ZEROCNK) - (tv_env[k-1] + ZEROCNK);
        if (buoy > 0.0 && buoy_prev <= 0.0) {
            last_lfc_idx = k;
        }
    }

    // Find EL: last positive->negative after any positive layer
    int el_idx = -1;
    bool found_pos = false;
    for (int k = 1; k < nlev; k++) {
        if (tv_parc[k] < -9000.0 || tv_parc[k-1] < -9000.0) continue;
        double buoy     = (tv_parc[k]   + ZEROCNK) - (tv_env[k]   + ZEROCNK);
        double buoy_prev = (tv_parc[k-1] + ZEROCNK) - (tv_env[k-1] + ZEROCNK);
        if (buoy > 0.0) found_pos = true;
        if (found_pos && buoy_prev > 0.0 && buoy <= 0.0) {
            el_idx = k;
        }
    }

    if (last_lfc_idx < 0) {
        // No instability found
        cape_out[col] = 0.0;
        cin_out[col] = 0.0;
        lfc_out[col] = p_lcl;
        el_out[col] = p_lcl;
        return;
    }

    // Interpolate LFC pressure
    {
        double buoy_prev = (tv_parc[last_lfc_idx-1] + ZEROCNK) - (tv_env[last_lfc_idx-1] + ZEROCNK);
        double buoy      = (tv_parc[last_lfc_idx]   + ZEROCNK) - (tv_env[last_lfc_idx]   + ZEROCNK);
        double frac = -buoy_prev / (buoy - buoy_prev);
        lfc_out[col] = pressure[last_lfc_idx-1] + frac * (pressure[last_lfc_idx] - pressure[last_lfc_idx-1]);
    }

    if (el_idx >= 0) {
        double buoy_prev = (tv_parc[el_idx-1] + ZEROCNK) - (tv_env[el_idx-1] + ZEROCNK);
        double buoy      = (tv_parc[el_idx]   + ZEROCNK) - (tv_env[el_idx]   + ZEROCNK);
        double frac = -buoy_prev / (buoy - buoy_prev);
        el_out[col] = pressure[el_idx-1] + frac * (pressure[el_idx] - pressure[el_idx-1]);
    } else {
        el_out[col] = pressure[nlev - 1];
    }

    // 4. Integrate CAPE and CIN (trapezoidal, LFC-bounded CIN)
    double cape = 0.0;
    double cin = 0.0;

    for (int k = 1; k < nlev; k++) {
        if (pressure[k] <= 0.0 || tv_parc[k] < -9000.0 || tv_parc[k-1] < -9000.0) continue;

        double tv_e_lo = tv_env[k-1] + ZEROCNK;
        double tv_e_hi = tv_env[k] + ZEROCNK;
        double tv_p_lo = tv_parc[k-1] + ZEROCNK;
        double tv_p_hi = tv_parc[k] + ZEROCNK;
        double dz = z[k] - z[k-1];
        if (fabs(dz) < 1e-6 || tv_e_lo <= 0.0 || tv_e_hi <= 0.0) continue;

        double buoy_lo = (tv_p_lo - tv_e_lo) / tv_e_lo;
        double buoy_hi = (tv_p_hi - tv_e_hi) / tv_e_hi;
        double val = G0 * (buoy_lo + buoy_hi) / 2.0 * dz;

        if (val > 0.0 && k >= last_lfc_idx) {
            cape += val;
        } else if (val < 0.0 && k <= last_lfc_idx) {
            cin += val;
        }
    }

    cape_out[col] = cape;
    cin_out[col] = cin;
}
'''
_cape_cin_raw = cp.RawKernel(_cape_cin_code, 'cape_cin_kernel')

# ============================================================================
# 62. LCL (RawKernel -- per column)
# ============================================================================
_lcl_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void lcl_kernel(
    const double* pressure, const double* temperature, const double* dewpoint,
    double* p_lcl_out, double* t_lcl_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p_lcl, t_lcl;
    drylift(pressure[i], temperature[i], dewpoint[i], &p_lcl, &t_lcl);
    p_lcl_out[i] = p_lcl;
    t_lcl_out[i] = t_lcl;
}
'''
_lcl_raw = cp.RawKernel(_lcl_code, 'lcl_kernel')

# ============================================================================
# 63. LFC (RawKernel -- per column)
# ============================================================================
_lfc_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void lfc_kernel(
    const double* pressure,    // (nlevels,)
    const double* temperature, // (ncols, nlevels)
    const double* dewpoint,    // (ncols, nlevels)
    double* lfc_p_out,         // (ncols,)
    double* lfc_t_out,         // (ncols,)
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    int nlev = nlevels < 200 ? nlevels : 200;

    double t_sfc = temperature[col * nlevels];
    double td_sfc = dewpoint[col * nlevels];
    double p_sfc = pressure[0];

    double p_lcl, t_lcl;
    drylift(p_sfc, t_sfc, td_sfc, &p_lcl, &t_lcl);

    double theta_dry_k = (t_sfc + ZEROCNK) * pow(1000.0 / p_sfc, ROCP);
    double r_gkg = mixratio_gkg(p_sfc, td_sfc);
    double theta_k = (t_lcl + ZEROCNK) * pow(1000.0 / p_lcl, ROCP);
    double theta_c = theta_k - ZEROCNK;
    double thetam = theta_c - wobf(theta_c) + wobf(t_lcl);

    double prev_buoy = 0.0;
    bool first = true;

    for (int k = 0; k < nlev; k++) {
        if (pressure[k] > p_lcl) continue;
        double t_e = temperature[col * nlevels + k];
        double td_e = dewpoint[col * nlevels + k];
        double tv_env_val = virtual_temp(t_e, pressure[k], td_e);
        double t_parc = satlift(pressure[k], thetam);
        double tv_parc_val = virtual_temp(t_parc, pressure[k], t_parc);
        double buoy = tv_parc_val - tv_env_val;

        if (!first && buoy > 0.0 && prev_buoy <= 0.0) {
            // Found crossing
            double frac = -prev_buoy / (buoy - prev_buoy);
            lfc_p_out[col] = pressure[k-1] + frac * (pressure[k] - pressure[k-1]);
            lfc_t_out[col] = t_e;
            return;
        }
        prev_buoy = buoy;
        first = false;
    }
    lfc_p_out[col] = -9999.0;
    lfc_t_out[col] = -9999.0;
}
'''
_lfc_raw = cp.RawKernel(_lfc_code, 'lfc_kernel')

# ============================================================================
# 64. EL (RawKernel -- per column)
# ============================================================================
_el_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void el_kernel(
    const double* pressure,    // (nlevels,)
    const double* temperature, // (ncols, nlevels)
    const double* dewpoint,    // (ncols, nlevels)
    double* el_p_out,          // (ncols,)
    double* el_t_out,          // (ncols,)
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    int nlev = nlevels < 200 ? nlevels : 200;

    double t_sfc = temperature[col * nlevels];
    double td_sfc = dewpoint[col * nlevels];
    double p_sfc = pressure[0];

    double p_lcl, t_lcl;
    drylift(p_sfc, t_sfc, td_sfc, &p_lcl, &t_lcl);

    double theta_k = (t_lcl + ZEROCNK) * pow(1000.0 / p_lcl, ROCP);
    double theta_c = theta_k - ZEROCNK;
    double thetam = theta_c - wobf(theta_c) + wobf(t_lcl);

    bool found_pos = false;
    double prev_buoy = 0.0;
    bool first = true;
    double last_el_p = -9999.0;
    double last_el_t = -9999.0;

    for (int k = 0; k < nlev; k++) {
        if (pressure[k] > p_lcl) continue;
        double t_e = temperature[col * nlevels + k];
        double td_e = dewpoint[col * nlevels + k];
        double tv_env_val = virtual_temp(t_e, pressure[k], td_e);
        double t_parc = satlift(pressure[k], thetam);
        double tv_parc_val = virtual_temp(t_parc, pressure[k], t_parc);
        double buoy = tv_parc_val - tv_env_val;

        if (buoy > 0.0) found_pos = true;
        if (!first && found_pos && prev_buoy > 0.0 && buoy <= 0.0) {
            double frac = -prev_buoy / (buoy - prev_buoy);
            last_el_p = pressure[k-1] + frac * (pressure[k] - pressure[k-1]);
            last_el_t = t_e;
        }
        prev_buoy = buoy;
        first = false;
    }
    el_p_out[col] = last_el_p;
    el_t_out[col] = last_el_t;
}
'''
_el_raw = cp.RawKernel(_el_code, 'el_kernel')

# ============================================================================
# 65. Lifted index (RawKernel)
# ============================================================================
_li_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void lifted_index_kernel(
    const double* pressure,    // (nlevels,)
    const double* temperature, // (ncols, nlevels)
    const double* dewpoint,    // (ncols, nlevels)
    double* li_out,            // (ncols,)
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double t_sfc = temperature[col * nlevels];
    double td_sfc = dewpoint[col * nlevels];
    double p_sfc = pressure[0];

    double p_lcl, t_lcl;
    drylift(p_sfc, t_sfc, td_sfc, &p_lcl, &t_lcl);

    double t_parcel_500;
    if (500.0 >= p_lcl) {
        double theta_k = (t_sfc + ZEROCNK) * pow(1000.0 / p_sfc, ROCP);
        t_parcel_500 = theta_k * pow(500.0 / 1000.0, ROCP) - ZEROCNK;
    } else {
        double theta_k = (t_lcl + ZEROCNK) * pow(1000.0 / p_lcl, ROCP);
        double theta_c = theta_k - ZEROCNK;
        double thetam = theta_c - wobf(theta_c) + wobf(t_lcl);
        t_parcel_500 = satlift(500.0, thetam);
    }

    // Interpolate env temperature at 500 hPa (log-p)
    double t_env_500 = temperature[col * nlevels + nlevels - 1];
    for (int k = 0; k < nlevels - 1; k++) {
        if (pressure[k] >= 500.0 && pressure[k+1] <= 500.0) {
            double log_p0 = log(pressure[k]);
            double log_p1 = log(pressure[k+1]);
            double log_pt = log(500.0);
            double frac = (log_pt - log_p0) / (log_p1 - log_p0);
            t_env_500 = temperature[col*nlevels+k] + frac * (temperature[col*nlevels+k+1] - temperature[col*nlevels+k]);
            break;
        }
    }
    li_out[col] = t_env_500 - t_parcel_500;
}
'''
_li_raw = cp.RawKernel(_li_code, 'lifted_index_kernel')

# ============================================================================
# 66. Precipitable water (RawKernel -- integrate moisture per column)
# ============================================================================
_pw_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void precipitable_water_kernel(
    const double* pressure,   // (nlevels,) hPa
    const double* dewpoint,   // (ncols, nlevels) C
    double* pw_out,           // (ncols,) mm
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double pw = 0.0;
    for (int k = 0; k < nlevels - 1; k++) {
        double w0 = mixratio_gkg(pressure[k],   dewpoint[col*nlevels+k])   / 1000.0;
        double w1 = mixratio_gkg(pressure[k+1], dewpoint[col*nlevels+k+1]) / 1000.0;
        double dp = (pressure[k] - pressure[k+1]) * 100.0;  // hPa -> Pa
        pw += (w0 + w1) / 2.0 * dp;
    }
    pw_out[col] = pw / G0;
}
'''
_pw_raw = cp.RawKernel(_pw_code, 'precipitable_water_kernel')

# ============================================================================
# 67. Mixed layer (RawKernel -- average T, Td in lowest N hPa)
# ============================================================================
_mixed_layer_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void mixed_layer_kernel(
    const double* pressure,    // (nlevels,) hPa
    const double* temperature, // (ncols, nlevels) C
    const double* dewpoint,    // (ncols, nlevels) C
    double* t_ml_out,          // (ncols,) mean T in C
    double* td_ml_out,         // (ncols,) mean Td in C
    double depth,              // hPa
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double sfc_p = pressure[0];
    double top_p = sfc_p - depth;

    // Pressure-weighted average using trapezoidal rule
    // Convert T to theta for proper averaging, then convert back
    double sum_theta = 0.0;
    double sum_td = 0.0;
    double total_dp = 0.0;

    for (int k = 0; k < nlevels - 1; k++) {
        if (pressure[k] < top_p) break;
        double p_top_layer = pressure[k+1];
        if (p_top_layer < top_p) p_top_layer = top_p;
        double dp = pressure[k] - p_top_layer;
        if (dp <= 0.0) continue;

        double th0 = (temperature[col*nlevels+k]   + ZEROCNK) * pow(1000.0/pressure[k],   ROCP);
        double th1 = (temperature[col*nlevels+k+1] + ZEROCNK) * pow(1000.0/pressure[k+1], ROCP);
        double td0 = dewpoint[col*nlevels+k];
        double td1 = dewpoint[col*nlevels+k+1];

        sum_theta += (th0 + th1) / 2.0 * dp;
        sum_td    += (td0 + td1) / 2.0 * dp;
        total_dp  += dp;
    }

    if (total_dp > 0.0) {
        double avg_theta = sum_theta / total_dp;
        // Convert avg theta back to T at surface pressure
        t_ml_out[col] = avg_theta * pow(sfc_p / 1000.0, ROCP) - ZEROCNK;
        td_ml_out[col] = sum_td / total_dp;
    } else {
        t_ml_out[col] = temperature[col*nlevels];
        td_ml_out[col] = dewpoint[col*nlevels];
    }
}
'''
_mixed_layer_raw = cp.RawKernel(_mixed_layer_code, 'mixed_layer_kernel')

# ============================================================================
# 68. Downdraft CAPE (RawKernel)
# ============================================================================
_dcape_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void downdraft_cape_kernel(
    const double* pressure,    // (nlevels,) hPa
    const double* temperature, // (ncols, nlevels) C
    const double* dewpoint,    // (ncols, nlevels) C
    double* dcape_out,         // (ncols,)
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    int nlev = nlevels < 200 ? nlevels : 200;

    double sfc_p = pressure[0];
    double limit_p = sfc_p - 400.0;

    // Find level of minimum theta-e
    double min_te = 1e30;
    int min_idx = 0;
    for (int k = 0; k < nlev; k++) {
        if (pressure[k] < limit_p) break;
        double t_c = temperature[col*nlevels+k];
        double td_c = dewpoint[col*nlevels+k];
        // Compute theta-e using SHARPpy thetae
        double p_lcl_loc, t_lcl_loc;
        drylift(pressure[k], t_c, td_c, &p_lcl_loc, &t_lcl_loc);
        double theta = (t_lcl_loc + ZEROCNK) * pow(1000.0/p_lcl_loc, ROCP);
        double r = mixratio_gkg(pressure[k], td_c) / 1000.0;
        double lc = 2500.0 - 2.37 * t_lcl_loc;
        double te = theta * exp((lc * 1000.0 * r) / (CP_D * (t_lcl_loc + ZEROCNK))) - ZEROCNK;
        if (te < min_te) { min_te = te; min_idx = k; }
    }

    if (min_idx == 0) { dcape_out[col] = 0.0; return; }

    // Descend moist adiabatically from min theta-e level to surface
    double dcape = 0.0;
    double t_parc = temperature[col*nlevels+min_idx];
    double prev_p = pressure[min_idx];

    for (int k = min_idx - 1; k >= 0; k--) {
        double p = pressure[k];
        double dp_desc = p - prev_p;  // positive going down
        if (fabs(dp_desc) < 1e-10) continue;

        // RK4 moist descent
        int n_steps = (int)(fabs(dp_desc) / 5.0);
        if (n_steps < 4) n_steps = 4;
        double h = dp_desc / (double)n_steps;
        double pc = prev_p;
        double tc = t_parc;
        for (int s = 0; s < n_steps; s++) {
            double rk1 = h * moist_lapse_rate(pc, tc);
            double rk2 = h * moist_lapse_rate(pc + h/2.0, tc + rk1/2.0);
            double rk3 = h * moist_lapse_rate(pc + h/2.0, tc + rk2/2.0);
            double rk4 = h * moist_lapse_rate(pc + h, tc + rk3);
            tc += (rk1 + 2.0*rk2 + 2.0*rk3 + rk4) / 6.0;
            pc += h;
        }
        t_parc = tc;
        prev_p = p;

        double tv_parc_val = virtual_temp(t_parc, p, t_parc);
        double tv_env_val = virtual_temp(temperature[col*nlevels+k], p, dewpoint[col*nlevels+k]);
        double buoy = tv_parc_val - tv_env_val;
        if (buoy < 0.0) {
            double dp_ln = fabs(log(pressure[k]) - log(pressure[k+1]));
            dcape += RD * fabs(buoy) * dp_ln;
        }
    }
    dcape_out[col] = dcape;
}
'''
_dcape_raw = cp.RawKernel(_dcape_code, 'downdraft_cape_kernel')

# ============================================================================
# 69. CCL (RawKernel -- convective condensation level per column)
# ============================================================================
_ccl_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void ccl_kernel(
    const double* pressure,    // (nlevels,)
    const double* temperature, // (ncols, nlevels)
    const double* dewpoint,    // (ncols, nlevels)
    double* ccl_p_out,         // (ncols,)
    double* ccl_t_out,         // (ncols,)
    int ncols, int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    int nlev = nlevels < 200 ? nlevels : 200;

    double w_sfc = mixratio_gkg(pressure[0], dewpoint[col*nlevels]);

    for (int k = 1; k < nlev; k++) {
        double ws_prev = mixratio_gkg(pressure[k-1], temperature[col*nlevels+k-1]);
        double ws_curr = mixratio_gkg(pressure[k],   temperature[col*nlevels+k]);
        if (ws_prev >= w_sfc && ws_curr < w_sfc) {
            double frac = (w_sfc - ws_prev) / (ws_curr - ws_prev);
            ccl_p_out[col] = pressure[k-1] + frac * (pressure[k] - pressure[k-1]);
            ccl_t_out[col] = temperature[col*nlevels+k-1] +
                             frac * (temperature[col*nlevels+k] - temperature[col*nlevels+k-1]);
            return;
        }
    }
    ccl_p_out[col] = -9999.0;
    ccl_t_out[col] = -9999.0;
}
'''
_ccl_raw = cp.RawKernel(_ccl_code, 'ccl_kernel')

# ============================================================================
# 70. Wet bulb temperature (RawKernel -- Newton-Raphson iterative)
#     (Separate from #9 which uses satlift; this is a standalone kernel entry)
# ============================================================================
_wet_bulb_temperature_nr_code = _CUDA_CONSTANTS + r'''
extern "C" __global__
void wet_bulb_temperature_kernel(
    const double* pressure, const double* temperature, const double* dewpoint,
    double* wbt_out, int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];
    // Lift to LCL
    double p_lcl, t_lcl;
    drylift(p, t_c, td_c, &p_lcl, &t_lcl);
    // Compute thetam
    double theta_k = (t_lcl + ZEROCNK) * pow(1000.0 / p_lcl, ROCP);
    double theta_c = theta_k - ZEROCNK;
    double thetam = theta_c - wobf(theta_c) + wobf(t_lcl);
    // Descend to original pressure via satlift
    wbt_out[i] = satlift(p, thetam);
}
'''
_wet_bulb_temperature_nr_raw = cp.RawKernel(
    _wet_bulb_temperature_nr_code, 'wet_bulb_temperature_kernel'
)


# ============================================================================
# Helper: launch config for RawKernels
# ============================================================================
def _grid_1d(n, block=256):
    """Return (grid, block) tuple for 1-D launch."""
    return ((n + block - 1) // block,), (block,)


# ============================================================================
# Python wrapper functions
# ============================================================================

def potential_temperature(pressure, temperature):
    """Potential temperature (K) from pressure (hPa) and temperature (C)."""
    p = cp.asarray(pressure, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    return potential_temperature_kernel(p, t)


def temperature_from_potential_temperature(pressure, theta):
    """Temperature (K) from pressure (hPa) and potential temperature (K)."""
    p = cp.asarray(pressure, dtype=cp.float64)
    th = cp.asarray(theta, dtype=cp.float64)
    return temperature_from_potential_temperature_kernel(p, th)


def virtual_temperature(temperature, mixing_ratio):
    """Virtual temperature (C) from T (C) and mixing ratio (kg/kg)."""
    t = cp.asarray(temperature, dtype=cp.float64)
    w = cp.asarray(mixing_ratio, dtype=cp.float64)
    return virtual_temperature_kernel(t, w)


def virtual_temperature_from_dewpoint(pressure, temperature, dewpoint):
    """Virtual temperature (C) from p (hPa), T (C), Td (C)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _vt_dewpoint_raw(grid, block, (t, td, p, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def virtual_potential_temperature(pressure, temperature, mixing_ratio):
    """Virtual potential temperature (K) from p (hPa), T (C), w (kg/kg)."""
    p = cp.asarray(pressure, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    w = cp.asarray(mixing_ratio, dtype=cp.float64)
    return virtual_potential_temperature_kernel(p, t, w)


def equivalent_potential_temperature(pressure, temperature, dewpoint):
    """Equivalent potential temperature (K) from p (hPa), T (C), Td (C). Bolton 1980."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _thetae_raw(grid, block, (p, t, td, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def saturation_equivalent_potential_temperature(pressure, temperature):
    """Saturation equivalent potential temperature (K). Assumes Td=T."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _sat_thetae_raw(grid, block, (p, t, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def wet_bulb_potential_temperature(pressure, temperature, dewpoint):
    """Wet bulb potential temperature (K) from p (hPa), T (C), Td (C)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _wbpt_raw(grid, block, (p, t, td, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def wet_bulb_temperature(pressure, temperature, dewpoint):
    """Wet bulb temperature (C) from p (hPa), T (C), Td (C)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _wbt_raw(grid, block, (p, t, td, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def saturation_vapor_pressure(temperature):
    """Saturation vapor pressure (hPa) from T (C). Ambaum (2020)."""
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _svp_raw(grid, block, (t, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def vapor_pressure(dewpoint):
    """Vapor pressure (hPa) from dewpoint (C). Same as SVP at Td."""
    t = cp.asarray(dewpoint, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _vp_from_td_raw(grid, block, (t, out, np.int32(n)))
    return out.reshape(cp.asarray(dewpoint).shape)


def vapor_pressure_from_mixing_ratio(mixing_ratio, pressure):
    """Vapor pressure (hPa) from mixing ratio (kg/kg) and total pressure (hPa)."""
    w = cp.asarray(mixing_ratio, dtype=cp.float64)
    p = cp.asarray(pressure, dtype=cp.float64)
    return vapor_pressure_from_mixing_ratio_kernel(w, p)


def dewpoint(vapor_pressure_val):
    """Dewpoint (C) from vapor pressure (hPa). Inverse Bolton."""
    e = cp.asarray(vapor_pressure_val, dtype=cp.float64)
    return dewpoint_kernel(e)


def dewpoint_from_relative_humidity(temperature, relative_humidity):
    """Dewpoint (C) from T (C) and RH (%)."""
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    rh = cp.asarray(relative_humidity, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _td_rh_raw(grid, block, (t, rh, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def dewpoint_from_specific_humidity(pressure, specific_humidity):
    """Dewpoint (C) from p (hPa) and q (kg/kg)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    q = cp.asarray(specific_humidity, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _td_q_raw(grid, block, (p, q, out, np.int32(n)))
    return out.reshape(cp.asarray(pressure).shape)


def mixing_ratio(vapor_pres, total_pres):
    """Mixing ratio (kg/kg) from vapor pressure and total pressure (both hPa)."""
    e = cp.asarray(vapor_pres, dtype=cp.float64)
    p = cp.asarray(total_pres, dtype=cp.float64)
    return mixing_ratio_kernel(e, p)


def saturation_mixing_ratio(pressure, temperature):
    """Saturation mixing ratio (kg/kg) from p (hPa) and T (C)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _sat_mr_raw(grid, block, (p, t, out, np.int32(n)))
    return out.reshape(cp.asarray(pressure).shape)


def mixing_ratio_from_relative_humidity(pressure, temperature, relative_humidity):
    """Mixing ratio (kg/kg) from p (hPa), T (C), RH (%)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    rh = cp.asarray(relative_humidity, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _mr_rh_raw(grid, block, (p, t, rh, out, np.int32(n)))
    return out.reshape(cp.asarray(pressure).shape)


def mixing_ratio_from_specific_humidity(specific_humidity):
    """Mixing ratio (kg/kg) from specific humidity (kg/kg)."""
    q = cp.asarray(specific_humidity, dtype=cp.float64)
    return mixing_ratio_from_specific_humidity_kernel(q)


def specific_humidity_from_dewpoint(pressure, dewpoint_val):
    """Specific humidity (kg/kg) from p (hPa) and Td (C)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint_val, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _q_td_raw(grid, block, (p, td, out, np.int32(n)))
    return out.reshape(cp.asarray(pressure).shape)


def specific_humidity_from_mixing_ratio(mixing_ratio_val):
    """Specific humidity (kg/kg) from mixing ratio (kg/kg)."""
    w = cp.asarray(mixing_ratio_val, dtype=cp.float64)
    return specific_humidity_from_mixing_ratio_kernel(w)


def relative_humidity_from_dewpoint(temperature, dewpoint_val):
    """Relative humidity (%) from T (C) and Td (C)."""
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint_val, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _rh_td_raw(grid, block, (t, td, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def relative_humidity_from_mixing_ratio(pressure, temperature, mixing_ratio_val):
    """Relative humidity (%) from p (hPa), T (C), w (kg/kg)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    w = cp.asarray(mixing_ratio_val, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _rh_mr_raw(grid, block, (p, t, w, out, np.int32(n)))
    return out.reshape(cp.asarray(pressure).shape)


def relative_humidity_from_specific_humidity(pressure, temperature, specific_humidity):
    """Relative humidity (%) from p (hPa), T (C), q (kg/kg)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    q = cp.asarray(specific_humidity, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _rh_q_raw(grid, block, (p, t, q, out, np.int32(n)))
    return out.reshape(cp.asarray(pressure).shape)


def density(pressure, temperature, mixing_ratio_val):
    """Air density (kg/m^3) from p (hPa), T (C), w (kg/kg)."""
    p = cp.asarray(pressure, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    w = cp.asarray(mixing_ratio_val, dtype=cp.float64)
    return density_kernel(p, t, w)


def dry_static_energy(height, temperature):
    """Dry static energy (J/kg) from z (m) and T (K)."""
    z = cp.asarray(height, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    return dry_static_energy_kernel(z, t)


def moist_static_energy(height, temperature, specific_humidity):
    """Moist static energy (J/kg) from z (m), T (K), q (kg/kg)."""
    z = cp.asarray(height, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    q = cp.asarray(specific_humidity, dtype=cp.float64)
    return moist_static_energy_kernel(z, t, q)


def exner_function(pressure):
    """Exner function (dimensionless) from p (hPa)."""
    p = cp.asarray(pressure, dtype=cp.float64)
    return exner_function_kernel(p)


def dry_lapse(pressure, reference_pressure, t_surface):
    """Dry adiabatic temperature (C) at pressure level."""
    p = cp.asarray(pressure, dtype=cp.float64)
    p_ref = cp.asarray(reference_pressure, dtype=cp.float64)
    t = cp.asarray(t_surface, dtype=cp.float64)
    return dry_lapse_kernel(p, p_ref, t)


def height_to_pressure_std(height):
    """Pressure (hPa) from height (m) using standard atmosphere."""
    h = cp.asarray(height, dtype=cp.float64)
    return height_to_pressure_std_kernel(h)


def pressure_to_height_std(pressure):
    """Height (m) from pressure (hPa) using standard atmosphere."""
    p = cp.asarray(pressure, dtype=cp.float64)
    return pressure_to_height_std_kernel(p)


def add_height_to_pressure(pressure, delta_height):
    """New pressure (hPa) after ascending by delta_height (m)."""
    p = cp.asarray(pressure, dtype=cp.float64)
    dh = cp.asarray(delta_height, dtype=cp.float64)
    return add_height_to_pressure_kernel(p, dh)


def add_pressure_to_height(height, delta_pressure):
    """New height (m) after pressure change delta_pressure (hPa)."""
    h = cp.asarray(height, dtype=cp.float64)
    dp = cp.asarray(delta_pressure, dtype=cp.float64)
    return add_pressure_to_height_kernel(h, dp)


def altimeter_to_sea_level_pressure(altimeter, elevation, temperature):
    """Sea level pressure (hPa) from altimeter (hPa), elevation (m), T (C)."""
    a = cp.asarray(altimeter, dtype=cp.float64)
    e = cp.asarray(elevation, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    return altimeter_to_sea_level_pressure_kernel(a, e, t)


def altimeter_to_station_pressure(altimeter, elevation):
    """Station pressure (hPa) from altimeter (hPa) and elevation (m)."""
    a = cp.asarray(altimeter, dtype=cp.float64)
    e = cp.asarray(elevation, dtype=cp.float64)
    return altimeter_to_station_pressure_kernel(a, e)


def station_to_altimeter_pressure(station_pressure, elevation):
    """Altimeter setting (hPa) from station pressure (hPa) and elevation (m)."""
    s = cp.asarray(station_pressure, dtype=cp.float64)
    e = cp.asarray(elevation, dtype=cp.float64)
    return station_to_altimeter_pressure_kernel(s, e)


def sigma_to_pressure(sigma, psfc, ptop):
    """Pressure (hPa) from sigma coordinate, surface pressure, model top."""
    sig = cp.asarray(sigma, dtype=cp.float64)
    ps = cp.asarray(psfc, dtype=cp.float64)
    pt = cp.asarray(ptop, dtype=cp.float64)
    return sigma_to_pressure_kernel(sig, ps, pt)


def geopotential_to_height(geopotential):
    """Geopotential height (m) from geopotential (m^2/s^2)."""
    gp = cp.asarray(geopotential, dtype=cp.float64)
    return geopotential_to_height_kernel(gp)


def height_to_geopotential(height):
    """Geopotential (m^2/s^2) from height (m)."""
    h = cp.asarray(height, dtype=cp.float64)
    return height_to_geopotential_kernel(h)


def scale_height(temperature):
    """Scale height (m) from temperature (K)."""
    t = cp.asarray(temperature, dtype=cp.float64)
    return scale_height_kernel(t)


def thickness_hydrostatic(p_bottom, p_top, t_mean_k):
    """Hypsometric thickness (m) from p_bottom, p_top (hPa) and T_mean (K)."""
    pb = cp.asarray(p_bottom, dtype=cp.float64)
    pt = cp.asarray(p_top, dtype=cp.float64)
    tm = cp.asarray(t_mean_k, dtype=cp.float64)
    return thickness_hydrostatic_kernel(pb, pt, tm)


def brunt_vaisala_frequency_squared(height, potential_temp):
    """Brunt-Vaisala frequency squared (1/s^2) from z (m) and theta (K)."""
    z = cp.asarray(height, dtype=cp.float64).ravel()
    th = cp.asarray(potential_temp, dtype=cp.float64).ravel()
    n = z.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _bvf2_raw(grid, block, (z, th, out, np.int32(n)))
    return out


def brunt_vaisala_frequency(height, potential_temp):
    """Brunt-Vaisala frequency (1/s) from z (m) and theta (K)."""
    z = cp.asarray(height, dtype=cp.float64).ravel()
    th = cp.asarray(potential_temp, dtype=cp.float64).ravel()
    n = z.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _bvf_raw(grid, block, (z, th, out, np.int32(n)))
    return out


def brunt_vaisala_period(bvf):
    """Brunt-Vaisala period (s) from frequency (1/s)."""
    n = cp.asarray(bvf, dtype=cp.float64)
    return brunt_vaisala_period_kernel(n)


def static_stability(pressure, temperature_k):
    """Static stability (K/Pa) from p (hPa) and T (K)."""
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature_k, dtype=cp.float64).ravel()
    n = p.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _static_stab_raw(grid, block, (p, t, out, np.int32(n)))
    return out


def vertical_velocity(omega, pressure, temperature):
    """Vertical velocity w (m/s) from omega (Pa/s), p (hPa), T (C)."""
    o = cp.asarray(omega, dtype=cp.float64)
    p = cp.asarray(pressure, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    return vertical_velocity_kernel(o, p, t)


def vertical_velocity_pressure(w, pressure, temperature):
    """Omega (Pa/s) from w (m/s), p (hPa), T (C)."""
    ww = cp.asarray(w, dtype=cp.float64)
    p = cp.asarray(pressure, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    return vertical_velocity_pressure_kernel(ww, p, t)


def montgomery_streamfunction(height, temperature):
    """Montgomery streamfunction (J/kg) from z (m) and T (K)."""
    z = cp.asarray(height, dtype=cp.float64)
    t = cp.asarray(temperature, dtype=cp.float64)
    return montgomery_streamfunction_kernel(z, t)


def heat_index(temperature, relative_humidity):
    """Heat index (C) from T (C) and RH (%). NWS Rothfusz."""
    t = cp.asarray(temperature, dtype=cp.float64)
    rh = cp.asarray(relative_humidity, dtype=cp.float64)
    return heat_index_kernel(t, rh)


def windchill(temperature, wind_speed):
    """Wind chill (C) from T (C) and wind speed (m/s). NWS formula."""
    t = cp.asarray(temperature, dtype=cp.float64)
    ws = cp.asarray(wind_speed, dtype=cp.float64)
    return windchill_kernel(t, ws)


def apparent_temperature(temperature, relative_humidity, wind_speed):
    """Apparent temperature (C) from T (C), RH (%), wind (m/s)."""
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    rh = cp.asarray(relative_humidity, dtype=cp.float64).ravel()
    ws = cp.asarray(wind_speed, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _apparent_temp_raw(grid, block, (t, rh, ws, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def frost_point(temperature, relative_humidity):
    """Frost point (C) from T (C) and RH (%)."""
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    rh = cp.asarray(relative_humidity, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _frost_point_raw(grid, block, (t, rh, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def psychrometric_vapor_pressure(temperature, wet_bulb, pressure):
    """Psychrometric vapor pressure (hPa) from T (C), Tw (C), p (hPa)."""
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    tw = cp.asarray(wet_bulb, dtype=cp.float64).ravel()
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    n = t.size
    out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _psychro_vp_raw(grid, block, (t, tw, p, out, np.int32(n)))
    return out.reshape(cp.asarray(temperature).shape)


def water_latent_heat_vaporization(temperature):
    """Latent heat of vaporization (J/kg) from T (C)."""
    t = cp.asarray(temperature, dtype=cp.float64)
    return water_latent_heat_vaporization_kernel(t)


def water_latent_heat_sublimation(temperature):
    """Latent heat of sublimation (J/kg) from T (C)."""
    t = cp.asarray(temperature, dtype=cp.float64)
    return water_latent_heat_sublimation_kernel(t)


def water_latent_heat_melting(temperature):
    """Latent heat of melting (J/kg) from T (C)."""
    t = cp.asarray(temperature, dtype=cp.float64)
    return water_latent_heat_melting_kernel(t)


def moist_air_gas_constant(mixing_ratio_val):
    """Gas constant for moist air (J/(kg*K)) from w (kg/kg)."""
    w = cp.asarray(mixing_ratio_val, dtype=cp.float64)
    return moist_air_gas_constant_kernel(w)


def moist_air_specific_heat_pressure(mixing_ratio_val):
    """Cp for moist air (J/(kg*K)) from w (kg/kg)."""
    w = cp.asarray(mixing_ratio_val, dtype=cp.float64)
    return moist_air_specific_heat_pressure_kernel(w)


def moist_air_poisson_exponent(mixing_ratio_val):
    """Poisson exponent (kappa) for moist air from w (kg/kg)."""
    w = cp.asarray(mixing_ratio_val, dtype=cp.float64)
    return moist_air_poisson_exponent_kernel(w)


# --- Column kernels ---

def moist_lapse(pressure, t_start):
    """Moist adiabatic temperature profile (C).

    Parameters
    ----------
    pressure : 1-D array (nlevels,) hPa, surface first
    t_start : scalar or 1-D array (ncols,) starting T in C

    Returns
    -------
    2-D array (ncols, nlevels) or 1-D (nlevels,) of T in C
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    ts = cp.asarray(t_start, dtype=cp.float64).ravel()
    nlevels = p.size
    ncols = ts.size
    out = cp.empty((ncols, nlevels), dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _moist_lapse_raw(grid, block, (p, ts, out, np.int32(ncols), np.int32(nlevels)))
    if ncols == 1:
        return out.ravel()
    return out


def parcel_profile(pressure, t_surface, td_surface):
    """Parcel temperature profile (C): dry below LCL, moist above.

    Parameters
    ----------
    pressure : 1-D array (nlevels,) hPa
    t_surface : scalar or 1-D (ncols,) surface T in C
    td_surface : scalar or 1-D (ncols,) surface Td in C

    Returns
    -------
    2-D (ncols, nlevels) or 1-D (nlevels,) of parcel T in C
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    ts = cp.asarray(t_surface, dtype=cp.float64).ravel()
    tds = cp.asarray(td_surface, dtype=cp.float64).ravel()
    nlevels = p.size
    ncols = ts.size
    out = cp.empty((ncols, nlevels), dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _parcel_profile_raw(grid, block, (p, ts, tds, out, np.int32(ncols), np.int32(nlevels)))
    if ncols == 1:
        return out.ravel()
    return out


def cape_cin(pressure, temperature, dewpoint):
    """CAPE, CIN, LCL, LFC, EL for multiple sounding columns.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature : 1-D (nlevels,) or 2-D (ncols, nlevels) in C
    dewpoint : 1-D (nlevels,) or 2-D (ncols, nlevels) in C

    Returns
    -------
    tuple of (cape, cin, lcl_p, lfc_p, el_p) -- all 1-D (ncols,)
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    # Kernel indexes raw memory as t[col * nlevels + k], so must be C-contiguous
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    cape_out = cp.empty(ncols, dtype=cp.float64)
    cin_out = cp.empty(ncols, dtype=cp.float64)
    lcl_out = cp.empty(ncols, dtype=cp.float64)
    lfc_out = cp.empty(ncols, dtype=cp.float64)
    el_out = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _cape_cin_raw(grid, block, (
        p, t, td, cape_out, cin_out, lcl_out, lfc_out, el_out,
        np.int32(ncols), np.int32(nlevels)
    ))
    return cape_out, cin_out, lcl_out, lfc_out, el_out


def lcl(pressure, temperature, dewpoint):
    """LCL pressure (hPa) and temperature (C).

    Parameters
    ----------
    pressure, temperature, dewpoint : arrays of same shape (hPa, C, C)

    Returns
    -------
    tuple of (p_lcl, t_lcl) arrays
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint, dtype=cp.float64).ravel()
    n = p.size
    p_out = cp.empty(n, dtype=cp.float64)
    t_out = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _lcl_raw(grid, block, (p, t, td, p_out, t_out, np.int32(n)))
    shape = cp.asarray(pressure).shape
    return p_out.reshape(shape), t_out.reshape(shape)


def lfc(pressure, temperature, dewpoint):
    """LFC pressure (hPa) and temperature (C) per column.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature, dewpoint : 1-D or 2-D (ncols, nlevels) C

    Returns
    -------
    tuple of (lfc_p, lfc_t) -- 1-D (ncols,)
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    lfc_p = cp.empty(ncols, dtype=cp.float64)
    lfc_t = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _lfc_raw(grid, block, (p, t, td, lfc_p, lfc_t, np.int32(ncols), np.int32(nlevels)))
    return lfc_p, lfc_t


def el(pressure, temperature, dewpoint):
    """EL pressure (hPa) and temperature (C) per column.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature, dewpoint : 1-D or 2-D (ncols, nlevels) C

    Returns
    -------
    tuple of (el_p, el_t) -- 1-D (ncols,)
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    el_p = cp.empty(ncols, dtype=cp.float64)
    el_t = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _el_raw(grid, block, (p, t, td, el_p, el_t, np.int32(ncols), np.int32(nlevels)))
    return el_p, el_t


def lifted_index(pressure, temperature, dewpoint):
    """Lifted index (K) per column.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature, dewpoint : 1-D or 2-D (ncols, nlevels) C

    Returns
    -------
    1-D array (ncols,) of LI values
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    out = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _li_raw(grid, block, (p, t, td, out, np.int32(ncols), np.int32(nlevels)))
    return out


def precipitable_water(pressure, dewpoint_val):
    """Precipitable water (mm) per column.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    dewpoint : 1-D or 2-D (ncols, nlevels) C

    Returns
    -------
    1-D array (ncols,) of PW values in mm
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    td = cp.asarray(dewpoint_val, dtype=cp.float64)
    nlevels = p.size
    if td.ndim == 1:
        td = td.reshape(1, -1)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = td.shape[0]
    out = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _pw_raw(grid, block, (p, td, out, np.int32(ncols), np.int32(nlevels)))
    return out


def mixed_layer(pressure, temperature, dewpoint, depth=100.0):
    """Mixed-layer mean T and Td in lowest depth hPa.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature, dewpoint : 1-D or 2-D (ncols, nlevels) C
    depth : float, mixing depth in hPa

    Returns
    -------
    tuple of (t_ml, td_ml) -- 1-D (ncols,) in C
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    t_out = cp.empty(ncols, dtype=cp.float64)
    td_out = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _mixed_layer_raw(grid, block, (
        p, t, td, t_out, td_out, cp.float64(depth),
        np.int32(ncols), np.int32(nlevels)
    ))
    return t_out, td_out


def downdraft_cape(pressure, temperature, dewpoint):
    """DCAPE (J/kg) per column.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature, dewpoint : 1-D or 2-D (ncols, nlevels) C

    Returns
    -------
    1-D array (ncols,) of DCAPE values
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    out = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _dcape_raw(grid, block, (p, t, td, out, np.int32(ncols), np.int32(nlevels)))
    return out


def ccl(pressure, temperature, dewpoint):
    """CCL pressure (hPa) and temperature (C) per column.

    Parameters
    ----------
    pressure : 1-D (nlevels,) hPa
    temperature, dewpoint : 1-D or 2-D (ncols, nlevels) C

    Returns
    -------
    tuple of (ccl_p, ccl_t) -- 1-D (ncols,)
    """
    p = cp.asarray(pressure, dtype=cp.float64).ravel()
    t = cp.asarray(temperature, dtype=cp.float64)
    td = cp.asarray(dewpoint, dtype=cp.float64)
    nlevels = p.size
    if t.ndim == 1:
        t = t.reshape(1, -1)
        td = td.reshape(1, -1)
    if not t.flags['C_CONTIGUOUS']:
        t = cp.ascontiguousarray(t)
    if not td.flags['C_CONTIGUOUS']:
        td = cp.ascontiguousarray(td)
    ncols = t.shape[0]
    ccl_p = cp.empty(ncols, dtype=cp.float64)
    ccl_t = cp.empty(ncols, dtype=cp.float64)
    grid, block = _grid_1d(ncols)
    _ccl_raw(grid, block, (p, t, td, ccl_p, ccl_t, np.int32(ncols), np.int32(nlevels)))
    return ccl_p, ccl_t
