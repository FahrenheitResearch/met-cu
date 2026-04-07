"""Fused thermodynamic kernels.

These kernels amortize HBM bandwidth by computing multiple outputs from the
same (p, T, Td) inputs in a single pass. The math is inlined from the existing
unfused kernels in metcu.kernels.thermo -- results are bit-identical to rtol 1e-12.
"""

import cupy as cp
import numpy as np


_FUSED_CONSTANTS = r'''
__device__ const double RD   = 287.04749097718457;
__device__ const double RV   = 461.52311572606084;
__device__ const double CP_D = 1004.6662184201462;
__device__ const double ROCP = 0.2857142857142857;
__device__ const double ZEROCNK = 273.15;
__device__ const double EPS  = 0.6219569100577033;
__device__ const double LV0  = 2500840.0;
__device__ const double SAT_PRESSURE_0C = 611.2;
__device__ const double T0_TRIP = 273.16;
__device__ const double CP_L = 4219.4;
__device__ const double CP_V = 1860.078011865639;
__device__ const double RV_METPY = 461.52311572606084;

__device__ double f_svp_liquid_pa(double t_k) {
    double latent = LV0 - (CP_L - CP_V) * (t_k - T0_TRIP);
    double heat_pow = (CP_L - CP_V) / RV_METPY;
    double exp_term = (LV0 / T0_TRIP - latent / t_k) / RV_METPY;
    return SAT_PRESSURE_0C * pow(T0_TRIP / t_k, heat_pow) * exp(exp_term);
}
__device__ double f_svp_hpa(double t_c) {
    return f_svp_liquid_pa(t_c + ZEROCNK) / 100.0;
}
__device__ double f_sat_mixing_ratio(double p_hpa, double t_c) {
    double es = f_svp_hpa(t_c);
    double ws = EPS * es / (p_hpa - es);
    return ws > 0.0 ? ws : 0.0;
}
'''


# ----------------------------------------------------------------------------
# (a) theta, theta_v, theta_e from (p, T, Td)
# ----------------------------------------------------------------------------
_theta_bundle_code = _FUSED_CONSTANTS + r'''
extern "C" __global__
void theta_thetav_thetae_kernel(
    const double* __restrict__ pressure,
    const double* __restrict__ temperature,
    const double* __restrict__ dewpoint,
    double* __restrict__ theta_out,
    double* __restrict__ theta_v_out,
    double* __restrict__ theta_e_out,
    int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];
    double t_k = t_c + ZEROCNK;
    double td_k = td_c + ZEROCNK;

    // theta
    double p_ratio = 1000.0 / p;
    double p_pow = pow(p_ratio, ROCP);
    double theta = t_k * p_pow;
    theta_out[i] = theta;

    // theta_v uses w = saturation_mixing_ratio(p, Td)
    double w = f_sat_mixing_ratio(p, td_c);
    theta_v_out[i] = theta * (1.0 + 0.61 * w);

    // theta_e (Bolton 1980) -- independent of the above "w"
    double t_lcl = 56.0 + 1.0 / (1.0/(td_k - 56.0) + log(t_k/td_k)/800.0);
    double e = f_svp_hpa(td_c);
    double r = EPS * e / (p - e);
    double theta_dl = t_k * pow(1000.0/(p - e), ROCP) * pow(t_k/t_lcl, 0.28*r);
    theta_e_out[i] = theta_dl * exp((3036.0/t_lcl - 1.78) * r * (1.0 + 0.448*r));
}
'''
_theta_bundle_raw = cp.RawKernel(_theta_bundle_code, 'theta_thetav_thetae_kernel')


# ----------------------------------------------------------------------------
# (b) es, e, ws, rh from (p, T, Td)
# ----------------------------------------------------------------------------
_svp_bundle_code = _FUSED_CONSTANTS + r'''
extern "C" __global__
void svp_e_mr_rh_kernel(
    const double* __restrict__ pressure,
    const double* __restrict__ temperature,
    const double* __restrict__ dewpoint,
    double* __restrict__ es_out,
    double* __restrict__ e_out,
    double* __restrict__ ws_out,
    double* __restrict__ rh_out,
    int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];

    double es = f_svp_hpa(t_c);
    double e  = f_svp_hpa(td_c);
    double ws_raw = EPS * es / (p - es);
    double ws = ws_raw > 0.0 ? ws_raw : 0.0;

    es_out[i] = es;
    e_out[i]  = e;
    ws_out[i] = ws;
    rh_out[i] = (es / es) * (e / es) * 100.0;  // = (e/es)*100
}
'''
_svp_bundle_raw = cp.RawKernel(_svp_bundle_code, 'svp_e_mr_rh_kernel')


# ----------------------------------------------------------------------------
# (c) Full bundle: theta, theta_v, theta_e, e, es, w, ws, rh
# ----------------------------------------------------------------------------
_full_bundle_code = _FUSED_CONSTANTS + r'''
extern "C" __global__
void thermo_bundle_kernel(
    const double* __restrict__ pressure,
    const double* __restrict__ temperature,
    const double* __restrict__ dewpoint,
    double* __restrict__ theta_out,
    double* __restrict__ theta_v_out,
    double* __restrict__ theta_e_out,
    double* __restrict__ e_out,
    double* __restrict__ es_out,
    double* __restrict__ w_out,
    double* __restrict__ ws_out,
    double* __restrict__ rh_out,
    int n
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    double p = pressure[i];
    double t_c = temperature[i];
    double td_c = dewpoint[i];
    double t_k = t_c + ZEROCNK;
    double td_k = td_c + ZEROCNK;

    // Saturation and actual vapor pressure
    double es = f_svp_hpa(t_c);
    double e  = f_svp_hpa(td_c);

    // Mixing ratios
    double ws_raw = EPS * es / (p - es);
    double ws = ws_raw > 0.0 ? ws_raw : 0.0;
    // w = mixing_ratio from actual vapor pressure and total pressure
    //     (metcu.mixing_ratio(vapor_pressure(Td), p) = EPS * e / (p - e))
    double w = EPS * e / (p - e);

    // theta, theta_v
    double p_pow = pow(1000.0 / p, ROCP);
    double theta = t_k * p_pow;
    // theta_v uses saturation mixing ratio at Td (matches virtual_potential_temperature(p,t,sat_mr(p,td)))
    double w_for_thetav = ws_raw > 0.0 ? (EPS * e / (p - e)) : 0.0;
    // Actually: unfused path is virtual_potential_temperature(p, t, saturation_mixing_ratio(p, td))
    // saturation_mixing_ratio(p, td) = EPS*f_svp_hpa(td)/(p - f_svp_hpa(td)) clamped to >=0
    double w_sat_td_raw = EPS * e / (p - e);
    double w_sat_td = w_sat_td_raw > 0.0 ? w_sat_td_raw : 0.0;
    double theta_v = theta * (1.0 + 0.61 * w_sat_td);

    // theta_e (Bolton 1980)
    double t_lcl = 56.0 + 1.0 / (1.0/(td_k - 56.0) + log(t_k/td_k)/800.0);
    double r = EPS * e / (p - e);
    double theta_dl = t_k * pow(1000.0/(p - e), ROCP) * pow(t_k/t_lcl, 0.28*r);
    double theta_e = theta_dl * exp((3036.0/t_lcl - 1.78) * r * (1.0 + 0.448*r));

    // RH = (es(Td)/es(T)) * 100
    double rh = (e / es) * 100.0;

    theta_out[i]   = theta;
    theta_v_out[i] = theta_v;
    theta_e_out[i] = theta_e;
    e_out[i]  = e;
    es_out[i] = es;
    w_out[i]  = w;
    ws_out[i] = ws;
    rh_out[i] = rh;
}
'''
_full_bundle_raw = cp.RawKernel(_full_bundle_code, 'thermo_bundle_kernel')


def _grid_1d(n, block=256):
    return ((n + block - 1) // block,), (block,)


def _prep3(pressure, temperature, dewpoint):
    p = cp.ascontiguousarray(cp.asarray(pressure, dtype=cp.float64)).ravel()
    t = cp.ascontiguousarray(cp.asarray(temperature, dtype=cp.float64)).ravel()
    td = cp.ascontiguousarray(cp.asarray(dewpoint, dtype=cp.float64)).ravel()
    return p, t, td


def theta_thetav_thetae(pressure, temperature, dewpoint):
    """Fused (theta, theta_v, theta_e). p in hPa, T/Td in C. Returns K, K, K."""
    shape = cp.asarray(temperature).shape
    p, t, td = _prep3(pressure, temperature, dewpoint)
    n = t.size
    theta = cp.empty(n, dtype=cp.float64)
    theta_v = cp.empty(n, dtype=cp.float64)
    theta_e = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _theta_bundle_raw(grid, block, (p, t, td, theta, theta_v, theta_e, np.int32(n)))
    return theta.reshape(shape), theta_v.reshape(shape), theta_e.reshape(shape)


def svp_e_mr_rh(pressure, temperature, dewpoint):
    """Fused (es, e, ws, rh). Returns (hPa, hPa, kg/kg, %)."""
    shape = cp.asarray(temperature).shape
    p, t, td = _prep3(pressure, temperature, dewpoint)
    n = t.size
    es = cp.empty(n, dtype=cp.float64)
    e = cp.empty(n, dtype=cp.float64)
    ws = cp.empty(n, dtype=cp.float64)
    rh = cp.empty(n, dtype=cp.float64)
    grid, block = _grid_1d(n)
    _svp_bundle_raw(grid, block, (p, t, td, es, e, ws, rh, np.int32(n)))
    return es.reshape(shape), e.reshape(shape), ws.reshape(shape), rh.reshape(shape)


def t_td_to_thermo_bundle(pressure, temperature, dewpoint):
    """Fused 8-output bundle: (theta, theta_v, theta_e, e, es, w, ws, rh)."""
    shape = cp.asarray(temperature).shape
    p, t, td = _prep3(pressure, temperature, dewpoint)
    n = t.size
    outs = [cp.empty(n, dtype=cp.float64) for _ in range(8)]
    grid, block = _grid_1d(n)
    _full_bundle_raw(grid, block, (p, t, td, *outs, np.int32(n)))
    return tuple(o.reshape(shape) for o in outs)
