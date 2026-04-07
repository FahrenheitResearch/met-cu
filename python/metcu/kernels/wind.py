"""Wind, kinematics, and severe weather CUDA kernels for met-cu.

All kernels use CuPy's ElementwiseKernel (per-element) or RawKernel
(per-column / per-gridpoint) to run on the GPU.  Python wrapper
functions mirror the metrust API signatures (plain arrays in, plain
arrays out — no pint Quantity).
"""

import math
import cupy as cp
import numpy as np

from ..constants import OMEGA, g, Rd, Cp_d, Lv, ZEROCNK, epsilon

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BLOCK = 256


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_cp(arr, dtype=cp.float64):
    """Ensure *arr* is a contiguous CuPy array."""
    if isinstance(arr, cp.ndarray):
        return cp.ascontiguousarray(arr, dtype=dtype)
    return cp.ascontiguousarray(cp.asarray(np.asarray(arr, dtype=np.float64)), dtype=dtype)


def _scalar(val):
    return float(val)


# ===================================================================
# 1–9  Per-element wind kernels (ElementwiseKernel)
# ===================================================================

# 1. wind_speed ----------------------------------------------------------
_wind_speed_ek = cp.ElementwiseKernel(
    "float64 u, float64 v",
    "float64 speed",
    "speed = sqrt(u * u + v * v)",
    "wind_speed_kernel",
)


def wind_speed(u, v):
    """Wind speed from u, v components (m/s)."""
    return _wind_speed_ek(_to_cp(u), _to_cp(v))


# 2. wind_direction -------------------------------------------------------
_wind_direction_ek = cp.ElementwiseKernel(
    "float64 u, float64 v",
    "float64 wdir",
    r"""
    double rad = atan2(-u, -v);
    wdir = rad * 180.0 / M_PI;
    if (wdir < 0.0) wdir += 360.0;
    // Calm winds -> 0
    if (u == 0.0 && v == 0.0) wdir = 0.0;
    """,
    "wind_direction_kernel",
)


def wind_direction(u, v):
    """Meteorological wind direction (degrees) from u, v."""
    return _wind_direction_ek(_to_cp(u), _to_cp(v))


# 3. wind_components -------------------------------------------------------
_wind_components_ek = cp.ElementwiseKernel(
    "float64 speed, float64 direction",
    "float64 u, float64 v",
    r"""
    double rad = direction * M_PI / 180.0;
    u = -speed * sin(rad);
    v = -speed * cos(rad);
    """,
    "wind_components_kernel",
)


def wind_components(speed, direction):
    """u, v components from speed (m/s) and direction (degrees)."""
    u, v = _wind_components_ek(_to_cp(speed), _to_cp(direction))
    return u, v


# 4. coriolis_parameter ----------------------------------------------------
_coriolis_ek = cp.ElementwiseKernel(
    "float64 lat",
    "float64 f",
    f"f = 2.0 * {OMEGA} * sin(lat * M_PI / 180.0)",
    "coriolis_parameter_kernel",
)


def coriolis_parameter(latitude):
    """Coriolis parameter f = 2*Omega*sin(lat).  lat in degrees."""
    return _coriolis_ek(_to_cp(latitude))


# 5. angle_to_direction ----------------------------------------------------
# Returns a float code: 0=N, 1=NNE, 2=NE, ..., 15=NNW (for level=16)
_angle_to_direction_ek = cp.ElementwiseKernel(
    "float64 deg, float64 n_dirs",
    "float64 code",
    r"""
    double d = fmod(deg, 360.0);
    if (d < 0.0) d += 360.0;
    double step = 360.0 / n_dirs;
    code = floor((d + step / 2.0) / step);
    if (code >= n_dirs) code = 0.0;
    """,
    "angle_to_direction_kernel",
)


def angle_to_direction(degrees, level=16):
    """Convert angle (degrees) to cardinal direction float code."""
    return _angle_to_direction_ek(_to_cp(degrees), cp.float64(level))


# 6. normal_component ------------------------------------------------------
_normal_component_ek = cp.ElementwiseKernel(
    "float64 u, float64 v, float64 nx, float64 ny",
    "float64 comp",
    "comp = u * nx + v * ny",
    "normal_component_kernel",
)


def normal_component(u, v, start, end):
    """Component of wind normal to a cross-section from start to end.

    start, end : (lat, lon) tuples in degrees.
    """
    lat1, lon1 = start
    lat2, lon2 = end
    dx = lon2 - lon1
    dy = lat2 - lat1
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-12:
        return cp.zeros_like(_to_cp(u))
    # Normal is perpendicular to tangent: tangent=(dx,dy) -> normal=(-dy,dx)
    nx = -dy / mag
    ny = dx / mag
    return _normal_component_ek(_to_cp(u), _to_cp(v), cp.float64(nx), cp.float64(ny))


# 7. tangential_component --------------------------------------------------
def tangential_component(u, v, start, end):
    """Component of wind tangential (parallel) to a cross-section."""
    lat1, lon1 = start
    lat2, lon2 = end
    dx = lon2 - lon1
    dy = lat2 - lat1
    mag = math.sqrt(dx * dx + dy * dy)
    if mag < 1e-12:
        return cp.zeros_like(_to_cp(u))
    tx = dx / mag
    ty = dy / mag
    return _normal_component_ek(_to_cp(u), _to_cp(v), cp.float64(tx), cp.float64(ty))


# 8. friction_velocity -----------------------------------------------------
_friction_velocity_code = r"""
extern "C" __global__
void friction_velocity_kernel(
    const double* u,
    const double* w,
    double* ustar_out,
    int n
) {
    // Single-output kernel: compute u* = (|mean(u'w')|)^0.5
    // Only thread 0 does work.
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid != 0) return;

    double mean_u = 0.0, mean_w = 0.0;
    for (int i = 0; i < n; i++) {
        mean_u += u[i];
        mean_w += w[i];
    }
    mean_u /= (double)n;
    mean_w /= (double)n;

    double cov = 0.0;
    for (int i = 0; i < n; i++) {
        cov += (u[i] - mean_u) * (w[i] - mean_w);
    }
    cov /= (double)n;
    ustar_out[0] = sqrt(fabs(cov));
}
"""
_friction_velocity_mod = cp.RawModule(code=_friction_velocity_code)
_friction_velocity_kern = _friction_velocity_mod.get_function("friction_velocity_kernel")


def friction_velocity(u, w):
    """Friction velocity u* from u and w time series (m/s)."""
    u_d = _to_cp(u).ravel()
    w_d = _to_cp(w).ravel()
    n = u_d.size
    out = cp.empty(1, dtype=cp.float64)
    _friction_velocity_kern((1,), (1,), (u_d, w_d, out, np.int32(n)))
    return float(out[0])


# 9. tke -------------------------------------------------------------------
_tke_code = r"""
extern "C" __global__
void tke_kernel(
    const double* u,
    const double* v,
    const double* w,
    double* tke_out,
    int n
) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid != 0) return;

    double mu = 0.0, mv = 0.0, mw = 0.0;
    for (int i = 0; i < n; i++) {
        mu += u[i]; mv += v[i]; mw += w[i];
    }
    mu /= (double)n; mv /= (double)n; mw /= (double)n;

    double var_u = 0.0, var_v = 0.0, var_w = 0.0;
    for (int i = 0; i < n; i++) {
        double du = u[i] - mu;
        double dv = v[i] - mv;
        double dw = w[i] - mw;
        var_u += du * du;
        var_v += dv * dv;
        var_w += dw * dw;
    }
    var_u /= (double)n; var_v /= (double)n; var_w /= (double)n;
    tke_out[0] = 0.5 * (var_u + var_v + var_w);
}
"""
_tke_mod = cp.RawModule(code=_tke_code)
_tke_kern = _tke_mod.get_function("tke_kernel")


def tke(u, v, w):
    """Turbulent kinetic energy from u, v, w time series (m^2/s^2)."""
    u_d = _to_cp(u).ravel()
    v_d = _to_cp(v).ravel()
    w_d = _to_cp(w).ravel()
    n = u_d.size
    out = cp.empty(1, dtype=cp.float64)
    _tke_kern((1,), (1,), (u_d, v_d, w_d, out, np.int32(n)))
    return float(out[0])


# ===================================================================
# 10–16  Column wind kernels (RawKernel)
# ===================================================================

# 10. bulk_shear_kernel ----------------------------------------------------
_bulk_shear_code = r"""
extern "C" __global__
void bulk_shear_kernel(
    const double* u,        // (ncols, nlevels) m/s
    const double* v,        // (ncols, nlevels) m/s
    const double* heights,  // (ncols, nlevels) meters AGL
    double bottom,          // bottom of layer (m)
    double top,             // top of layer (m)
    double* shear_u_out,    // (ncols,)
    double* shear_v_out,    // (ncols,)
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Interpolate u, v at bottom
    double u_bot = u[off], v_bot = v[off];
    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= bottom) {
            double h0 = heights[off + k - 1];
            double h1 = heights[off + k];
            if (h1 - h0 > 1e-6) {
                double frac = (bottom - h0) / (h1 - h0);
                u_bot = u[off + k - 1] + frac * (u[off + k] - u[off + k - 1]);
                v_bot = v[off + k - 1] + frac * (v[off + k] - v[off + k - 1]);
            } else {
                u_bot = u[off + k];
                v_bot = v[off + k];
            }
            break;
        }
    }

    // Interpolate u, v at top
    double u_top = u[off + nlevels - 1], v_top = v[off + nlevels - 1];
    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= top) {
            double h0 = heights[off + k - 1];
            double h1 = heights[off + k];
            if (h1 - h0 > 1e-6) {
                double frac = (top - h0) / (h1 - h0);
                u_top = u[off + k - 1] + frac * (u[off + k] - u[off + k - 1]);
                v_top = v[off + k - 1] + frac * (v[off + k] - v[off + k - 1]);
            } else {
                u_top = u[off + k];
                v_top = v[off + k];
            }
            break;
        }
    }

    shear_u_out[col] = u_top - u_bot;
    shear_v_out[col] = v_top - v_bot;
}
"""
_bulk_shear_mod = cp.RawModule(code=_bulk_shear_code)
_bulk_shear_kern = _bulk_shear_mod.get_function("bulk_shear_kernel")


def bulk_shear(u, v, height, bottom, top):
    """Bulk wind shear (u_shear, v_shear) over a height layer.

    For 1-D profile inputs (nlevels,) -> scalar pair.
    For 2-D column inputs (ncols, nlevels) -> arrays of size ncols.
    """
    u_d = _to_cp(u)
    v_d = _to_cp(v)
    h_d = _to_cp(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    su = cp.empty(ncols, dtype=cp.float64)
    sv = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _bulk_shear_kern(grid, block, (
        u_d, v_d, h_d,
        cp.float64(bottom), cp.float64(top),
        su, sv,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(su[0]), float(sv[0])
    return su, sv


# 11. mean_wind_kernel -----------------------------------------------------
_mean_wind_code = r"""
extern "C" __global__
void mean_wind_kernel(
    const double* u,
    const double* v,
    const double* heights,
    double bottom,
    double top,
    double* mean_u_out,
    double* mean_v_out,
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;
    double sum_u = 0.0, sum_v = 0.0, sum_dh = 0.0;

    for (int k = 0; k < nlevels; k++) {
        double h = heights[off + k];
        if (h < bottom) continue;
        if (h > top) break;

        double dh;
        if (k == 0 || heights[off + k - 1] < bottom) {
            // First level in layer
            if (k + 1 < nlevels && heights[off + k + 1] <= top) {
                dh = (heights[off + k + 1] - h) / 2.0;
            } else {
                dh = 1.0;
            }
        } else if (k + 1 >= nlevels || heights[off + k + 1] > top) {
            // Last level in layer
            dh = (h - heights[off + k - 1]) / 2.0;
        } else {
            dh = (heights[off + k + 1] - heights[off + k - 1]) / 2.0;
        }
        if (dh < 0.0) dh = 0.0;

        sum_u += u[off + k] * dh;
        sum_v += v[off + k] * dh;
        sum_dh += dh;
    }

    if (sum_dh > 0.0) {
        mean_u_out[col] = sum_u / sum_dh;
        mean_v_out[col] = sum_v / sum_dh;
    } else {
        mean_u_out[col] = 0.0;
        mean_v_out[col] = 0.0;
    }
}
"""
_mean_wind_mod = cp.RawModule(code=_mean_wind_code)
_mean_wind_kern = _mean_wind_mod.get_function("mean_wind_kernel")


def mean_wind(u, v, height, bottom, top):
    """Height-weighted mean wind in a layer.

    Returns (mean_u, mean_v) in m/s.
    """
    u_d = _to_cp(u)
    v_d = _to_cp(v)
    h_d = _to_cp(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    mu = cp.empty(ncols, dtype=cp.float64)
    mv = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _mean_wind_kern(grid, block, (
        u_d, v_d, h_d,
        cp.float64(bottom), cp.float64(top),
        mu, mv,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(mu[0]), float(mv[0])
    return mu, mv


# 12. storm_relative_helicity_kernel --------------------------------------
_srh_code = r"""
extern "C" __global__
void srh_kernel(
    const double* u,           // (ncols, nlevels) m/s
    const double* v,           // (ncols, nlevels) m/s
    const double* heights,     // (ncols, nlevels) meters AGL
    double storm_u,            // storm motion u component
    double storm_v,            // storm motion v component
    double depth,              // integration depth in meters
    double* srh_pos_out,       // (ncols,) positive SRH
    double* srh_neg_out,       // (ncols,) negative SRH
    double* srh_total_out,     // (ncols,)
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    if (nlevels < 2) {
        srh_pos_out[col] = 0.0;
        srh_neg_out[col] = 0.0;
        srh_total_out[col] = 0.0;
        return;
    }

    double pos = 0.0, neg = 0.0;
    int offset = col * nlevels;

    double h_start = heights[offset];
    double h_end = h_start + depth;
    double prev_h = h_start;
    double prev_u = u[offset];
    double prev_v = v[offset];
    bool integrated = false;

    for (int k = 1; k < nlevels; k++) {
        double curr_h = heights[offset + k];
        double curr_u = u[offset + k];
        double curr_v = v[offset + k];
        if (curr_h <= prev_h) {
            prev_h = curr_h;
            prev_u = curr_u;
            prev_v = curr_v;
            continue;
        }

        double next_h = curr_h;
        double next_u = curr_u;
        double next_v = curr_v;

        if (curr_h >= h_end) {
            double frac = (h_end - prev_h) / (curr_h - prev_h);
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            next_h = h_end;
            next_u = prev_u + frac * (curr_u - prev_u);
            next_v = prev_v + frac * (curr_v - prev_v);
        }

        double sru0 = prev_u - storm_u;
        double srv0 = prev_v - storm_v;
        double sru1 = next_u - storm_u;
        double srv1 = next_v - storm_v;
        double val = (sru1 * srv0) - (sru0 * srv1);

        if (val > 0.0) pos += val;
        else neg += val;
        integrated = true;

        if (curr_h >= h_end) break;
        prev_h = curr_h;
        prev_u = curr_u;
        prev_v = curr_v;
    }

    if (!integrated) {
        pos = 0.0;
        neg = 0.0;
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_srh_mod = cp.RawModule(code=_srh_code)
_srh_kern = _srh_mod.get_function("srh_kernel")


def storm_relative_helicity(u, v, height, depth, storm_u, storm_v):
    """Storm-relative helicity (positive, negative, total) in m^2/s^2.

    Parameters
    ----------
    u, v : array (m/s)  — shape (nlevels,) or (ncols, nlevels)
    height : array (m AGL)
    depth : float (m)
    storm_u, storm_v : float (m/s)

    Returns
    -------
    (pos, neg, total) — scalars or arrays
    """
    u_d = _to_cp(u)
    v_d = _to_cp(v)
    h_d = _to_cp(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    pos = cp.empty(ncols, dtype=cp.float64)
    neg = cp.empty(ncols, dtype=cp.float64)
    tot = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _srh_kern(grid, block, (
        u_d, v_d, h_d,
        cp.float64(storm_u), cp.float64(storm_v), cp.float64(depth),
        pos, neg, tot,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(pos[0]), float(neg[0]), float(tot[0])
    return pos, neg, tot


# 13. bunkers_storm_motion_kernel ------------------------------------------
_bunkers_code = r"""
extern "C" __global__
void bunkers_storm_motion_kernel(
    const double* u,
    const double* v,
    const double* heights,
    double* rm_u_out,       // right mover u
    double* rm_v_out,
    double* lm_u_out,       // left mover u
    double* lm_v_out,
    double* mw_u_out,       // mean wind u
    double* mw_v_out,
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;
    double D = 7.5;  // deviation magnitude m/s

    // Mean wind 0-6 km
    double su6 = 0.0, sv6 = 0.0, sdh6 = 0.0;
    for (int k = 0; k < nlevels; k++) {
        double h = heights[off + k];
        if (h > 6000.0) break;
        double dh;
        if (k == 0) dh = (k + 1 < nlevels) ? (heights[off + k + 1] - h) / 2.0 : 1.0;
        else if (k + 1 >= nlevels || heights[off + k + 1] > 6000.0)
            dh = (h - heights[off + k - 1]) / 2.0;
        else
            dh = (heights[off + k + 1] - heights[off + k - 1]) / 2.0;
        if (dh < 0.0) dh = 0.0;
        su6 += u[off + k] * dh;
        sv6 += v[off + k] * dh;
        sdh6 += dh;
    }
    double mu6 = (sdh6 > 0.0) ? su6 / sdh6 : 0.0;
    double mv6 = (sdh6 > 0.0) ? sv6 / sdh6 : 0.0;

    // Shear vector: 0-6 km bulk shear
    // Interpolate at 0 and 6000
    double u_bot = u[off], v_bot = v[off];
    double u_top = u[off + nlevels - 1], v_top = v[off + nlevels - 1];
    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= 6000.0) {
            double h0 = heights[off + k - 1];
            double h1 = heights[off + k];
            if (h1 - h0 > 1e-6) {
                double frac = (6000.0 - h0) / (h1 - h0);
                u_top = u[off + k - 1] + frac * (u[off + k] - u[off + k - 1]);
                v_top = v[off + k - 1] + frac * (v[off + k] - v[off + k - 1]);
            } else {
                u_top = u[off + k];
                v_top = v[off + k];
            }
            break;
        }
    }
    double shear_u = u_top - u_bot;
    double shear_v = v_top - v_bot;

    // Normalize shear vector
    double shear_mag = sqrt(shear_u * shear_u + shear_v * shear_v);
    double shear_norm_u = 0.0, shear_norm_v = 0.0;
    if (shear_mag > 1e-6) {
        shear_norm_u = shear_u / shear_mag;
        shear_norm_v = shear_v / shear_mag;
    }

    // Perpendicular (cross-product in 2D): rotate 90 degrees CW for RM
    double perp_u = shear_norm_v;
    double perp_v = -shear_norm_u;

    // Right mover: mean + D * perp
    rm_u_out[col] = mu6 + D * perp_u;
    rm_v_out[col] = mv6 + D * perp_v;

    // Left mover: mean - D * perp
    lm_u_out[col] = mu6 - D * perp_u;
    lm_v_out[col] = mv6 - D * perp_v;

    // Mean wind
    mw_u_out[col] = mu6;
    mw_v_out[col] = mv6;
}
"""
_bunkers_mod = cp.RawModule(code=_bunkers_code)
_bunkers_kern = _bunkers_mod.get_function("bunkers_storm_motion_kernel")


def bunkers_storm_motion(u, v, height):
    """Bunkers storm motion: (right, left, mean) each as (u, v).

    Returns ((rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v)).
    """
    u_d = _to_cp(u)
    v_d = _to_cp(v)
    h_d = _to_cp(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    rm_u = cp.empty(ncols, dtype=cp.float64)
    rm_v = cp.empty(ncols, dtype=cp.float64)
    lm_u = cp.empty(ncols, dtype=cp.float64)
    lm_v = cp.empty(ncols, dtype=cp.float64)
    mw_u = cp.empty(ncols, dtype=cp.float64)
    mw_v = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _bunkers_kern(grid, block, (
        u_d, v_d, h_d,
        rm_u, rm_v, lm_u, lm_v, mw_u, mw_v,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return (
            (float(rm_u[0]), float(rm_v[0])),
            (float(lm_u[0]), float(lm_v[0])),
            (float(mw_u[0]), float(mw_v[0])),
        )
    return (rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v)


# 14. corfidi_storm_motion_kernel ------------------------------------------
_corfidi_code = r"""
extern "C" __global__
void corfidi_storm_motion_kernel(
    const double* u,
    const double* v,
    const double* heights,
    double u_llj,           // low-level jet u (850 hPa wind, m/s)
    double v_llj,           // low-level jet v
    double* upwind_u_out,
    double* upwind_v_out,
    double* downwind_u_out,
    double* downwind_v_out,
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Mean wind 0-6 km (cloud-layer mean wind)
    double su = 0.0, sv = 0.0, sdh = 0.0;
    for (int k = 0; k < nlevels; k++) {
        double h = heights[off + k];
        if (h > 6000.0) break;
        double dh;
        if (k == 0) dh = (k + 1 < nlevels) ? (heights[off + k + 1] - h) / 2.0 : 1.0;
        else if (k + 1 >= nlevels || heights[off + k + 1] > 6000.0)
            dh = (h - heights[off + k - 1]) / 2.0;
        else
            dh = (heights[off + k + 1] - heights[off + k - 1]) / 2.0;
        if (dh < 0.0) dh = 0.0;
        su += u[off + k] * dh;
        sv += v[off + k] * dh;
        sdh += dh;
    }
    double mw_u = (sdh > 0.0) ? su / sdh : 0.0;
    double mw_v = (sdh > 0.0) ? sv / sdh : 0.0;

    // Propagation vector = mean_wind - LLJ (opposite of LLJ relative to mean)
    double prop_u = mw_u - u_llj;
    double prop_v = mw_v - v_llj;

    // Corfidi upwind = mean_wind - LLJ (propagation only)
    upwind_u_out[col] = prop_u;
    upwind_v_out[col] = prop_v;

    // Corfidi downwind = mean_wind + propagation = 2*mean_wind - LLJ
    downwind_u_out[col] = prop_u + mw_u;
    downwind_v_out[col] = prop_v + mw_v;
}
"""
_corfidi_mod = cp.RawModule(code=_corfidi_code)
_corfidi_kern = _corfidi_mod.get_function("corfidi_storm_motion_kernel")


def corfidi_storm_motion(u, v, height, u_llj, v_llj):
    """Corfidi MCS motion vectors.

    Returns ((upwind_u, upwind_v), (downwind_u, downwind_v)).
    """
    u_d = _to_cp(u)
    v_d = _to_cp(v)
    h_d = _to_cp(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    uu = cp.empty(ncols, dtype=cp.float64)
    uv = cp.empty(ncols, dtype=cp.float64)
    du = cp.empty(ncols, dtype=cp.float64)
    dv = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _corfidi_kern(grid, block, (
        u_d, v_d, h_d,
        cp.float64(u_llj), cp.float64(v_llj),
        uu, uv, du, dv,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return (float(uu[0]), float(uv[0])), (float(du[0]), float(dv[0]))
    return (uu, uv), (du, dv)


# 15. critical_angle_kernel ------------------------------------------------
_critical_angle_ek = cp.ElementwiseKernel(
    "float64 storm_u, float64 storm_v, float64 u_sfc, float64 v_sfc, "
    "float64 u_500, float64 v_500",
    "float64 angle",
    r"""
    // Shear vector: 0-500m
    double shr_u = u_500 - u_sfc;
    double shr_v = v_500 - v_sfc;
    // Inflow vector: storm-relative surface wind
    double inf_u = storm_u - u_sfc;
    double inf_v = storm_v - v_sfc;

    double mag_shr = sqrt(shr_u * shr_u + shr_v * shr_v);
    double mag_inf = sqrt(inf_u * inf_u + inf_v * inf_v);
    double denom = mag_shr * mag_inf;

    if (denom < 1e-10) {
        angle = 0.0;
    } else {
        double cosang = (shr_u * inf_u + shr_v * inf_v) / denom;
        if (cosang > 1.0) cosang = 1.0;
        if (cosang < -1.0) cosang = -1.0;
        angle = acos(cosang) * 180.0 / M_PI;
    }
    """,
    "critical_angle_kernel",
)


def critical_angle(storm_u, storm_v, u_sfc, v_sfc, u_500, v_500):
    """Critical angle (degrees) between low-level shear and storm-relative inflow."""
    return _critical_angle_ek(
        _to_cp(storm_u), _to_cp(storm_v),
        _to_cp(u_sfc), _to_cp(v_sfc),
        _to_cp(u_500), _to_cp(v_500),
    )


# 16. get_layer_kernel -----------------------------------------------------
_get_layer_code = r"""
extern "C" __global__
void get_layer_kernel(
    const double* pressure,    // (ncols, nlevels) hPa, descending
    const double* values,      // (ncols, nlevels)
    double p_bottom,           // hPa (higher pressure = lower altitude)
    double p_top,              // hPa (lower pressure = higher altitude)
    double* p_out,             // (ncols, nlevels) — padded with NaN
    double* v_out,             // (ncols, nlevels) — padded with NaN
    int* count_out,            // (ncols,) — number of valid levels per col
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;
    int cnt = 0;

    for (int k = 0; k < nlevels; k++) {
        double p = pressure[off + k];
        // Pressure is descending: p_bottom >= p >= p_top
        if (p <= p_bottom && p >= p_top) {
            p_out[off + cnt] = p;
            v_out[off + cnt] = values[off + k];
            cnt++;
        }
    }

    // Fill remaining with NaN
    for (int k = cnt; k < nlevels; k++) {
        p_out[off + k] = __longlong_as_double(0x7FF8000000000000LL);
        v_out[off + k] = __longlong_as_double(0x7FF8000000000000LL);
    }
    count_out[col] = cnt;
}
"""
_get_layer_mod = cp.RawModule(code=_get_layer_code)
_get_layer_kern = _get_layer_mod.get_function("get_layer_kernel")


def get_layer(pressure, values, p_bottom, p_top):
    """Extract data between two pressure levels.

    Returns (p_layer, v_layer) — 1-D arrays trimmed to valid levels
    (for single-column input) or 2-D arrays padded with NaN.
    """
    p_d = _to_cp(pressure)
    v_d = _to_cp(values)

    if p_d.ndim == 1:
        p_d = p_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = p_d.shape
    p_out = cp.empty_like(p_d)
    v_out = cp.empty_like(v_d)
    cnt = cp.empty(ncols, dtype=cp.int32)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _get_layer_kern(grid, block, (
        p_d, v_d,
        cp.float64(p_bottom), cp.float64(p_top),
        p_out, v_out, cnt,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        n = int(cnt[0])
        return p_out[0, :n], v_out[0, :n]
    return p_out, v_out


# ===================================================================
# 17–40  Severe weather parameter kernels
# ===================================================================

# 17. significant_tornado_parameter ----------------------------------------
_stp_ek = cp.ElementwiseKernel(
    "float64 cape, float64 lcl, float64 srh, float64 shear",
    "float64 stp",
    r"""
    // Match metrust fixed-layer STP thresholds.
    double cape_term = cape > 0.0 ? cape / 1500.0 : 0.0;
    double srh_term  = srh > 0.0 ? srh / 150.0 : 0.0;
    double shear_term;
    if (shear < 12.5) {
        shear_term = 0.0;
    } else {
        double shear_capped = shear > 30.0 ? 30.0 : shear;
        shear_term = shear_capped / 20.0;
    }

    // LCL term: clamped between 0 and 1
    double lcl_term;
    if (lcl <= 1000.0) {
        lcl_term = 1.0;
    } else {
        lcl_term = (2000.0 - lcl) / 1000.0;
        if (lcl_term < 0.0) lcl_term = 0.0;
        if (lcl_term > 1.0) lcl_term = 1.0;
    }

    stp = cape_term * srh_term * shear_term * lcl_term;
    if (stp < 0.0) stp = 0.0;
    """,
    "significant_tornado_parameter_kernel",
)


def significant_tornado_parameter(sbcape, lcl_height, srh_0_1km, bulk_shear_0_6km):
    """Significant Tornado Parameter (fixed-layer STP).

    Parameters: CAPE (J/kg), LCL height (m), SRH (m^2/s^2), shear (m/s).
    """
    return _stp_ek(
        _to_cp(sbcape), _to_cp(lcl_height),
        _to_cp(srh_0_1km), _to_cp(bulk_shear_0_6km),
    )


# 18. supercell_composite_parameter ----------------------------------------
_scp_ek = cp.ElementwiseKernel(
    "float64 mucape, float64 srh, float64 shear",
    "float64 scp",
    r"""
    double cape_term = mucape > 0.0 ? mucape / 1000.0 : 0.0;
    double srh_term = srh > 0.0 ? srh / 50.0 : 0.0;
    double shear_term;
    if (shear < 10.0) {
        shear_term = 0.0;
    } else {
        double shear_capped = shear > 20.0 ? 20.0 : shear;
        shear_term = shear_capped / 20.0;
    }
    scp = cape_term * srh_term * shear_term;
    """,
    "supercell_composite_parameter_kernel",
)


def supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff):
    """Supercell Composite Parameter (SCP)."""
    return _scp_ek(
        _to_cp(mucape), _to_cp(srh_eff), _to_cp(bulk_shear_eff),
    )


# 19. compute_ship ---------------------------------------------------------
_ship_ek = cp.ElementwiseKernel(
    "float64 cape, float64 shear, float64 t500, float64 lr, float64 mr",
    "float64 ship",
    r"""
    // SHIP = (MUCAPE * MR * LR * (-T500) * SHEAR) / 42000000
    // Clamp components
    double cape_c = (cape < 0.0) ? 0.0 : cape;
    double mr_c   = (mr < 0.0) ? 0.0 : mr;
    double lr_c   = (lr < 0.0) ? 0.0 : lr;
    double t500_c = (t500 > 0.0) ? 0.0 : -t500;  // want positive value
    double shear_c = (shear < 0.0) ? 0.0 : shear;

    ship = (cape_c * mr_c * lr_c * t500_c * shear_c) / 42000000.0;
    if (ship < 0.0) ship = 0.0;
    // Cap SHIP for low CAPE
    if (cape_c < 1300.0) ship *= (cape_c / 1300.0);
    """,
    "compute_ship_kernel",
)


def compute_ship(cape, shear06, t500, lr_700_500, mixing_ratio_gkg):
    """Significant Hail Parameter (SHIP)."""
    return _ship_ek(
        _to_cp(cape), _to_cp(shear06), _to_cp(t500),
        _to_cp(lr_700_500), _to_cp(mixing_ratio_gkg),
    )


# 20. compute_ehi ----------------------------------------------------------
_ehi_ek = cp.ElementwiseKernel(
    "float64 cape, float64 srh",
    "float64 ehi",
    "ehi = (cape * srh) / 160000.0",
    "compute_ehi_kernel",
)


def compute_ehi(cape, srh):
    """Energy-Helicity Index: EHI = CAPE * SRH / 160000."""
    return _ehi_ek(_to_cp(cape), _to_cp(srh))


# 21. compute_dcp ----------------------------------------------------------
_dcp_ek = cp.ElementwiseKernel(
    "float64 dcape, float64 cape, float64 shear, float64 mu_mixing_ratio",
    "float64 dcp",
    r"""
    double dcape_term = dcape > 0.0 ? dcape / 980.0 : 0.0;
    double cape_term = cape > 0.0 ? cape / 2000.0 : 0.0;
    double shear_term = shear > 0.0 ? shear / 20.0 : 0.0;
    double mr_term = mu_mixing_ratio > 0.0 ? mu_mixing_ratio / 11.0 : 0.0;
    dcp = dcape_term * cape_term * shear_term * mr_term;
    """,
    "compute_dcp_kernel",
)


def compute_dcp(dcape, mu_cape, shear06, mu_mixing_ratio):
    """Derecho Composite Parameter.

    DCP = (DCAPE/980) * (MUCAPE/2000) * (SHEAR/20) * (MU_MIXING_RATIO/11).
    """
    return _dcp_ek(
        _to_cp(dcape), _to_cp(mu_cape),
        _to_cp(shear06), _to_cp(mu_mixing_ratio),
    )


# 22. bulk_richardson_number ------------------------------------------------
_brn_ek = cp.ElementwiseKernel(
    "float64 cape, float64 shear",
    "float64 brn",
    r"""
    double shear2 = shear * shear;
    double denom = 0.5 * shear2;
    brn = (denom > 1e-6) ? cape / denom : 0.0;
    """,
    "bulk_richardson_number_kernel",
)


def bulk_richardson_number(cape, shear_0_6km):
    """Bulk Richardson Number: BRN = CAPE / (0.5 * shear^2)."""
    return _brn_ek(_to_cp(cape), _to_cp(shear_0_6km))


# 23. k_index ---------------------------------------------------------------
_k_index_ek = cp.ElementwiseKernel(
    "float64 t850, float64 t700, float64 t500, float64 td850, float64 td700",
    "float64 ki",
    "ki = (t850 - t500) + td850 - (t700 - td700)",
    "k_index_kernel",
)


def k_index(t850, t700, t500, td850, td700):
    """K-Index: KI = (T850-T500) + Td850 - (T700-Td700).  All in degC."""
    return _k_index_ek(
        _to_cp(t850), _to_cp(t700), _to_cp(t500),
        _to_cp(td850), _to_cp(td700),
    )


# 24. total_totals ----------------------------------------------------------
_total_totals_ek = cp.ElementwiseKernel(
    "float64 t850, float64 t500, float64 td850",
    "float64 tt",
    "tt = (t850 - t500) + (td850 - t500)",
    "total_totals_kernel",
)


def total_totals(t850, t500, td850):
    """Total Totals Index: TT = VT + CT = (T850-T500) + (Td850-T500)."""
    return _total_totals_ek(_to_cp(t850), _to_cp(t500), _to_cp(td850))


# 25. cross_totals -----------------------------------------------------------
_cross_totals_ek = cp.ElementwiseKernel(
    "float64 td850, float64 t500",
    "float64 ct",
    "ct = td850 - t500",
    "cross_totals_kernel",
)


def cross_totals(td850, t500):
    """Cross Totals: CT = Td850 - T500."""
    return _cross_totals_ek(_to_cp(td850), _to_cp(t500))


# 26. vertical_totals -------------------------------------------------------
_vertical_totals_ek = cp.ElementwiseKernel(
    "float64 t850, float64 t500",
    "float64 vt",
    "vt = t850 - t500",
    "vertical_totals_kernel",
)


def vertical_totals(t850, t500):
    """Vertical Totals: VT = T850 - T500."""
    return _vertical_totals_ek(_to_cp(t850), _to_cp(t500))


# 27. sweat_index -----------------------------------------------------------
_sweat_ek = cp.ElementwiseKernel(
    "float64 td850, float64 tt, float64 f850, float64 f500, "
    "float64 dd850, float64 dd500",
    "float64 sweat",
    r"""
    // SWEAT = 12*Td850 + 20*(TT-49) + 2*f850 + f500 + 125*(sin(dd500-dd850)+0.2)
    double term1 = 12.0 * td850;
    if (term1 < 0.0) term1 = 0.0;

    double term2 = 20.0 * (tt - 49.0);
    if (term2 < 0.0) term2 = 0.0;

    double term3 = 2.0 * f850;
    double term4 = f500;

    // Shear term: only if certain conditions on wind directions are met
    double dd_diff = (dd500 - dd850) * M_PI / 180.0;
    double term5 = 125.0 * (sin(dd_diff) + 0.2);
    if (term5 < 0.0) term5 = 0.0;

    // Direction criteria: 130<=dd850<=250, 210<=dd500<=310,
    // dd500>dd850, f850>=15kt, f500>=15kt
    if (dd850 < 130.0 || dd850 > 250.0 ||
        dd500 < 210.0 || dd500 > 310.0 ||
        dd500 <= dd850 ||
        f850 < 15.0 || f500 < 15.0) {
        term5 = 0.0;
    }

    sweat = term1 + term2 + term3 + term4 + term5;
    """,
    "sweat_index_kernel",
)


def sweat_index(td850, tt_or_t850=None, f850=None, f500=None,
                dd850=None, dd500=None, t500=None):
    """SWEAT Index.

    Can be called as:
      sweat_index(td850, tt, f850, f500, dd850, dd500)
    or with the 7-param metrust form:
      sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500)
    """
    # Detect 7-param form: sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500)
    if t500 is not None:
        # Keyword form
        t850_v = _to_cp(td850)  # actually t850
        td850_v = _to_cp(tt_or_t850)  # actually td850
        t500_v = _to_cp(t500)
        tt_v = (t850_v - t500_v) + (td850_v - t500_v)
        dd850_v = _to_cp(dd850)
        dd500_v = _to_cp(dd500)
        f850_v = _to_cp(f850)
        f500_v = _to_cp(f500)
    elif dd500 is not None:
        # 6-param form: (td850, tt, f850, f500, dd850, dd500)
        td850_v = _to_cp(td850)
        tt_v = _to_cp(tt_or_t850)
        f850_v = _to_cp(f850)
        f500_v = _to_cp(f500)
        dd850_v = _to_cp(dd850)
        dd500_v = _to_cp(dd500)
    else:
        raise TypeError("sweat_index requires at least 6 parameters")
    return _sweat_ek(td850_v, tt_v, f850_v, f500_v, dd850_v, dd500_v)


def sweat_index_direct(t850, td850, t500, dd850, dd500, ff850, ff500):
    """SWEAT Index (direct 7-parameter form matching metrust).

    t850, td850, t500 in degC; dd850, dd500 in degrees; ff850, ff500 in knots.
    """
    t850_v = _to_cp(t850)
    td850_v = _to_cp(td850)
    t500_v = _to_cp(t500)
    tt_v = (t850_v - t500_v) + (td850_v - t500_v)
    return _sweat_ek(
        td850_v, tt_v,
        _to_cp(ff850), _to_cp(ff500),
        _to_cp(dd850), _to_cp(dd500),
    )


# 28. showalter_index -------------------------------------------------------
_showalter_code = r"""
extern "C" __global__
void showalter_index_kernel(
    const double* pressure,     // (ncols, nlevels) hPa descending
    const double* temperature,  // (ncols, nlevels) degC
    const double* dewpoint,     // (ncols, nlevels) degC
    double* si_out,             // (ncols,)
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Find 850 hPa T and Td by interpolation
    double t850 = temperature[off];
    double td850 = dewpoint[off];
    for (int k = 1; k < nlevels; k++) {
        double p0 = pressure[off + k - 1];
        double p1 = pressure[off + k];
        if (p0 >= 850.0 && p1 <= 850.0) {
            double frac = (850.0 - p0) / (p1 - p0);
            t850 = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            td850 = dewpoint[off + k - 1] + frac * (dewpoint[off + k] - dewpoint[off + k - 1]);
            break;
        }
    }

    // Find 500 hPa T
    double t500_env = temperature[off + nlevels - 1];
    for (int k = 1; k < nlevels; k++) {
        double p0 = pressure[off + k - 1];
        double p1 = pressure[off + k];
        if (p0 >= 500.0 && p1 <= 500.0) {
            double frac = (500.0 - p0) / (p1 - p0);
            t500_env = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            break;
        }
    }

    // Lift parcel from 850 to LCL (dry adiabatic), then to 500 (moist adiabatic)
    // Simplified: use iterative approach
    double Rd = 287.04;
    double Cp = 1004.0;
    double Lv = 2.501e6;
    double eps = 0.622;
    double kappa = Rd / Cp;

    double t_k = t850 + 273.15;
    double td_k = td850 + 273.15;

    // LCL temperature estimate (Bolton 1980)
    double tlcl = 1.0 / (1.0 / (td_k - 56.0) + log(t_k / td_k) / 800.0) + 56.0;
    double plcl = 850.0 * pow(tlcl / t_k, Cp / Rd);

    // Moist adiabatic ascent from LCL to 500 hPa
    double t_parcel = tlcl;
    double p_curr = plcl;
    double dp = 5.0;  // hPa steps
    while (p_curr > 500.0) {
        double next_p = p_curr - dp;
        if (next_p < 500.0) {
            dp = p_curr - 500.0;
            next_p = 500.0;
        }
        // Saturation mixing ratio
        double es = 6.112 * exp(17.67 * (t_parcel - 273.15) / (t_parcel - 29.65));
        double ws = eps * es / (p_curr - es);
        // Moist adiabatic lapse rate (dT/dp)
        double gamma = (Rd * t_parcel / Cp + Lv * ws / Cp) /
                       (p_curr * (1.0 + Lv * Lv * ws * eps / (Cp * Rd * t_parcel * t_parcel)));
        t_parcel -= gamma * dp;
        p_curr = next_p;
    }

    double t500_parcel = t_parcel - 273.15;
    si_out[col] = t500_env - t500_parcel;
}
"""
_showalter_mod = cp.RawModule(code=_showalter_code)
_showalter_kern = _showalter_mod.get_function("showalter_index_kernel")


def showalter_index(pressure, temperature, dewpoint):
    """Showalter Index: SI = T500_env - T500_parcel (lifted from 850).

    Profile inputs: pressure (hPa), temperature (degC), dewpoint (degC).
    """
    p_d = _to_cp(pressure)
    t_d = _to_cp(temperature)
    td_d = _to_cp(dewpoint)

    if p_d.ndim == 1:
        p_d = p_d.reshape(1, -1)
        t_d = t_d.reshape(1, -1)
        td_d = td_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = p_d.shape
    out = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _showalter_kern(grid, block, (
        p_d, t_d, td_d, out,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(out[0])
    return out


# 29. boyden_index ----------------------------------------------------------
_boyden_ek = cp.ElementwiseKernel(
    "float64 z1000, float64 z700, float64 t700",
    "float64 bi",
    r"""
    // BI = (Z700 - Z1000)/10 - T700 - 200
    // Z in decameters: (z700 - z1000) / 10
    bi = (z700 - z1000) / 10.0 - t700 - 200.0;
    """,
    "boyden_index_kernel",
)


def boyden_index(z1000, z700, t700):
    """Boyden Index.

    z1000, z700 in meters; t700 in degC.
    """
    return _boyden_ek(_to_cp(z1000), _to_cp(z700), _to_cp(t700))


# 30. galvez_davison_index --------------------------------------------------
_gdi_ek = cp.ElementwiseKernel(
    "float64 t950, float64 t850, float64 t700, float64 t500, "
    "float64 td950, float64 td850, float64 td700, float64 sst",
    "float64 gdi",
    r"""
    // Galvez-Davison Index for tropical thunderstorm potential
    // Simplified GDI formulation

    // Column buoyancy index (CBI): Low-level theta-e minus mid-level theta-e
    double Lv = 2.501e6;
    double Cp = 1004.0;

    // Approximate theta-e at 950 hPa
    double es950 = 6.112 * exp(17.67 * td950 / (td950 + 243.5));
    double w950 = 0.622 * es950 / (950.0 - es950);
    double theta950 = (t950 + 273.15) * pow(1000.0 / 950.0, 0.286);
    double thetae_950 = theta950 * exp(Lv * w950 / (Cp * (t950 + 273.15)));

    // Approximate theta-e at 850 hPa
    double es850 = 6.112 * exp(17.67 * td850 / (td850 + 243.5));
    double w850 = 0.622 * es850 / (850.0 - es850);
    double theta850 = (t850 + 273.15) * pow(1000.0 / 850.0, 0.286);
    double thetae_850 = theta850 * exp(Lv * w850 / (Cp * (t850 + 273.15)));

    // Approximate theta-e at 700 hPa
    double es700 = 6.112 * exp(17.67 * td700 / (td700 + 243.5));
    double w700 = 0.622 * es700 / (700.0 - es700);
    double theta700 = (t700 + 273.15) * pow(1000.0 / 700.0, 0.286);
    double thetae_700 = theta700 * exp(Lv * w700 / (Cp * (t700 + 273.15)));

    // Theta-e at 500 hPa (assume Td not available, use T for dry)
    double theta500 = (t500 + 273.15) * pow(1000.0 / 500.0, 0.286);

    // Column Buoyancy Index
    double alpha = (thetae_950 + thetae_850) / 2.0;
    double beta  = theta500;

    // Inversion index (II): check for trade wind inversion
    double gamma_term = theta700 - theta950;

    // Mid-level warming index (MWI)
    double mu_term = theta500 - theta700;

    // GDI = CBI + II + MWI (simplified)
    // Empirical: GDI ~ A*(alpha - beta) + B*gamma_term + C*(sst - 273.15 - 26.5)
    double cbi = alpha - beta;
    double sst_term = sst - 26.5;  // assume sst in degC

    gdi = cbi / 20.0 + sst_term * 5.0 - gamma_term * 2.0;
    """,
    "galvez_davison_index_kernel",
)


def galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst):
    """Galvez-Davison Index (tropical thunderstorm potential).

    All temperatures in degC, SST in degC.
    """
    return _gdi_ek(
        _to_cp(t950), _to_cp(t850), _to_cp(t700), _to_cp(t500),
        _to_cp(td950), _to_cp(td850), _to_cp(td700), _to_cp(sst),
    )


# 31. fosberg_fire_weather_index -------------------------------------------
_ffwi_ek = cp.ElementwiseKernel(
    "float64 temp_f, float64 rh, float64 wind_mph",
    "float64 ffwi",
    r"""
    // Fosberg Fire Weather Index
    // temp_f: temperature in Fahrenheit
    // rh: relative humidity in percent
    // wind_mph: wind speed in mph

    // Equilibrium moisture content (EMC)
    double m;
    if (rh <= 10.0) {
        m = 0.03229 + 0.281073 * rh - 0.000578 * rh * temp_f;
    } else if (rh <= 50.0) {
        m = 2.22749 + 0.160107 * rh - 0.01478 * temp_f;
    } else {
        m = 21.0606 + 0.005565 * rh * rh - 0.00035 * rh * temp_f
            - 0.483199 * rh;
    }
    if (m < 0.0) m = 0.0;
    if (m > 30.0) m = 30.0;

    // Moisture damping coefficient
    double eta = 1.0 - 2.0 * (m / 30.0) + 1.5 * pow(m / 30.0, 2.0)
                 - 0.5 * pow(m / 30.0, 3.0);

    // Wind effect
    double fw = sqrt(1.0 + wind_mph * wind_mph);

    ffwi = eta * fw / 0.3002;
    if (ffwi < 0.0) ffwi = 0.0;
    """,
    "fosberg_fire_weather_index_kernel",
)


def fosberg_fire_weather_index(temperature, relative_humidity, wind_speed_val):
    """Fosberg Fire Weather Index.

    temperature in degF, relative_humidity in percent, wind_speed in mph.
    """
    return _ffwi_ek(
        _to_cp(temperature), _to_cp(relative_humidity), _to_cp(wind_speed_val),
    )


# 32. haines_index ----------------------------------------------------------
_haines_ek = cp.ElementwiseKernel(
    "float64 t950, float64 t850, float64 td850",
    "float64 haines",
    r"""
    // Haines Index (Low Elevation version using 950-850-850)
    // Stability term (A): T950 - T850
    double delta_t = t950 - t850;
    double a;
    if (delta_t <= 3.0) a = 1.0;
    else if (delta_t <= 7.0) a = 2.0;
    else a = 3.0;

    // Moisture term (B): T850 - Td850
    double delta_td = t850 - td850;
    double b;
    if (delta_td <= 5.0) b = 1.0;
    else if (delta_td <= 9.0) b = 2.0;
    else b = 3.0;

    haines = a + b;
    """,
    "haines_index_kernel",
)


def haines_index(t_950, t_850, td_850):
    """Haines Index (fire weather stability/moisture).

    All temperatures in degC.
    """
    return _haines_ek(_to_cp(t_950), _to_cp(t_850), _to_cp(td_850))


# 33. hot_dry_windy ---------------------------------------------------------
_hdw_ek = cp.ElementwiseKernel(
    "float64 temp_c, float64 rh, float64 wind_ms, float64 vpd_in",
    "float64 hdw",
    r"""
    // Hot-Dry-Windy Index = VPD * wind
    double vpd;
    if (vpd_in > 0.0) {
        vpd = vpd_in;
    } else {
        // Compute VPD from T and RH using the SHARPpy/Wexler SVP polynomial.
        double pol = temp_c * (1.1112018e-17 + (temp_c * -3.0994571e-20));
        pol = temp_c * (2.1874425e-13 + (temp_c * (-1.789232e-15 + pol)));
        pol = temp_c * (4.3884180e-09 + (temp_c * (-2.988388e-11 + pol)));
        pol = temp_c * (7.8736169e-05 + (temp_c * (-6.111796e-07 + pol)));
        pol = 0.99999683 + (temp_c * (-9.082695e-03 + pol));
        double es = 6.1078 / pow(pol, 8.0);
        double ea = es * rh / 100.0;
        vpd = es - ea;
    }
    hdw = vpd * wind_ms;
    if (hdw < 0.0) hdw = 0.0;
    """,
    "hot_dry_windy_kernel",
)


def hot_dry_windy(temperature, relative_humidity, wind_speed_val, vpd=0.0):
    """Hot-Dry-Windy Index.

    temperature in degC, relative_humidity in percent, wind_speed in m/s.
    vpd: vapor pressure deficit (hPa), 0 = compute internally.
    """
    return _hdw_ek(
        _to_cp(temperature), _to_cp(relative_humidity),
        _to_cp(wind_speed_val), _to_cp(vpd),
    )


# 34. significant_tornado (alternative formulation) -------------------------
_sig_tor_ek = cp.ElementwiseKernel(
    "float64 cape, float64 cin, float64 lcl, float64 srh, float64 shear",
    "float64 stp",
    r"""
    // Full STP with CIN term:
    // STP = (CAPE/1500) * ((200+CIN)/150) * (SRH/150) * (shear/20) * LCL_term
    double cape_term = cape / 1500.0;

    double cin_term;
    if (cin > -50.0) {
        cin_term = 1.0;
    } else if (cin < -200.0) {
        cin_term = 0.0;
    } else {
        cin_term = (200.0 + cin) / 150.0;
    }

    double srh_term  = srh / 150.0;
    double shear_term = shear / 20.0;

    double lcl_term;
    if (lcl < 1000.0) lcl_term = 1.0;
    else if (lcl > 2000.0) lcl_term = 0.0;
    else lcl_term = (2000.0 - lcl) / 1000.0;

    stp = cape_term * cin_term * srh_term * shear_term * lcl_term;
    if (stp < 0.0) stp = 0.0;
    """,
    "significant_tornado_kernel",
)


def significant_tornado(cape, cin, lcl_height, srh_0_1km, bulk_shear_0_6km):
    """Significant Tornado Parameter with CIN term.

    CAPE (J/kg), CIN (J/kg, negative), LCL (m), SRH (m^2/s^2), shear (m/s).
    """
    return _sig_tor_ek(
        _to_cp(cape), _to_cp(cin), _to_cp(lcl_height),
        _to_cp(srh_0_1km), _to_cp(bulk_shear_0_6km),
    )


# 35. freezing_rain_composite -----------------------------------------------
_fzra_code = r"""
extern "C" __global__
void freezing_rain_composite_kernel(
    const double* temperature,  // (ncols, nlevels) degC
    const double* pressure,     // (ncols, nlevels) hPa
    int precip_type,            // precipitation type flag
    double* frc_out,            // (ncols,)
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Count warm layers (T > 0) and cold layers (T < 0) above surface
    double warm_depth = 0.0;
    double cold_depth_sfc = 0.0;
    int found_warm = 0;

    // Surface is first level; scan upward
    for (int k = 1; k < nlevels; k++) {
        double t = temperature[off + k];
        double dp = pressure[off + k - 1] - pressure[off + k];
        if (dp < 0.0) dp = -dp;

        if (t > 0.0) {
            warm_depth += dp;
            found_warm = 1;
        } else if (!found_warm) {
            cold_depth_sfc += dp;
        }
    }

    // Composite: needs warm nose aloft, cold surface layer, precipitation
    double frc = 0.0;
    if (temperature[off] < 0.0 && warm_depth > 50.0 && precip_type > 0) {
        frc = warm_depth / 100.0 * cold_depth_sfc / 50.0;
    }
    frc_out[col] = frc;
}
"""
_fzra_mod = cp.RawModule(code=_fzra_code)
_fzra_kern = _fzra_mod.get_function("freezing_rain_composite_kernel")


def freezing_rain_composite(temperature, pressure, precip_type):
    """Freezing rain composite index.

    temperature (degC), pressure (hPa) as profile arrays, precip_type as int flag.
    """
    t_d = _to_cp(temperature)
    p_d = _to_cp(pressure)

    if t_d.ndim == 1:
        t_d = t_d.reshape(1, -1)
        p_d = p_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    out = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _fzra_kern(grid, block, (
        t_d, p_d, np.int32(int(precip_type)),
        out,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(out[0])
    return out


# 36. warm_nose_check -------------------------------------------------------
_warm_nose_code = r"""
extern "C" __global__
void warm_nose_check_kernel(
    const double* temperature,  // (ncols, nlevels) degC
    const double* pressure,     // (ncols, nlevels) hPa
    int* result_out,            // (ncols,) 1 if warm nose found, 0 otherwise
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Warm nose: surface T < 0, then T > 0 aloft, then T < 0 again higher up
    int state = 0;  // 0: looking for surface cold, 1: found cold, looking for warm,
                     // 2: found warm nose
    int found = 0;

    if (temperature[off] >= 0.0) {
        // Surface not below freezing — no freezing rain warm nose
        result_out[col] = 0;
        return;
    }

    state = 1;  // surface is cold
    for (int k = 1; k < nlevels; k++) {
        double t = temperature[off + k];
        if (state == 1 && t > 0.0) {
            state = 2;  // entered warm layer
        } else if (state == 2 && t <= 0.0) {
            found = 1;  // returned to cold above warm layer
            break;
        }
    }
    // Also flag if we found warm aloft without returning to cold
    // (warm nose extending to top of profile)
    if (state == 2) found = 1;

    result_out[col] = found;
}
"""
_warm_nose_mod = cp.RawModule(code=_warm_nose_code)
_warm_nose_kern = _warm_nose_mod.get_function("warm_nose_check_kernel")


def warm_nose_check(temperature, pressure):
    """Detect warm nose (elevated warm layer above freezing surface).

    Returns bool for single profile, or int array for batch.
    """
    t_d = _to_cp(temperature)
    p_d = _to_cp(pressure)

    if t_d.ndim == 1:
        t_d = t_d.reshape(1, -1)
        p_d = p_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    out = cp.empty(ncols, dtype=cp.int32)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _warm_nose_kern(grid, block, (
        t_d, p_d, out,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return bool(out[0])
    return out


# 37. dendritic_growth_zone -------------------------------------------------
_dgz_code = r"""
extern "C" __global__
void dendritic_growth_zone_kernel(
    const double* temperature,  // (ncols, nlevels) degC
    const double* pressure,     // (ncols, nlevels) hPa
    double* p_bot_out,          // (ncols,) bottom pressure of DGZ
    double* p_top_out,          // (ncols,) top pressure of DGZ
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // DGZ: -12C to -18C layer (best dendritic ice crystal growth)
    double p_bot = -1.0, p_top = -1.0;  // NaN sentinel

    for (int k = 0; k < nlevels; k++) {
        double t = temperature[off + k];
        double p = pressure[off + k];

        // Scan for -12 to -18 range
        if (t <= -12.0 && t >= -18.0) {
            if (p_bot < 0.0) p_bot = p;  // first (highest pressure) level
            p_top = p;  // keep updating (lowest pressure)
        }
    }

    // If not found, try interpolation at boundaries
    if (p_bot < 0.0) {
        for (int k = 1; k < nlevels; k++) {
            double t0 = temperature[off + k - 1];
            double t1 = temperature[off + k];
            double p0 = pressure[off + k - 1];
            double p1 = pressure[off + k];
            // Crossing -12
            if ((t0 > -12.0 && t1 <= -12.0) || (t0 <= -12.0 && t1 > -12.0)) {
                double frac = (-12.0 - t0) / (t1 - t0);
                double p_interp = p0 + frac * (p1 - p0);
                if (p_bot < 0.0) p_bot = p_interp;
            }
            // Crossing -18
            if ((t0 > -18.0 && t1 <= -18.0) || (t0 <= -18.0 && t1 > -18.0)) {
                double frac = (-18.0 - t0) / (t1 - t0);
                double p_interp = p0 + frac * (p1 - p0);
                p_top = p_interp;
            }
        }
    }

    p_bot_out[col] = (p_bot > 0.0) ? p_bot : 0.0;
    p_top_out[col] = (p_top > 0.0) ? p_top : 0.0;
}
"""
_dgz_mod = cp.RawModule(code=_dgz_code)
_dgz_kern = _dgz_mod.get_function("dendritic_growth_zone_kernel")


def dendritic_growth_zone(temperature, pressure):
    """Dendritic growth zone bounds (bottom, top) in hPa.

    temperature (degC), pressure (hPa) as profile arrays.
    """
    t_d = _to_cp(temperature)
    p_d = _to_cp(pressure)

    if t_d.ndim == 1:
        t_d = t_d.reshape(1, -1)
        p_d = p_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    bot = cp.empty(ncols, dtype=cp.float64)
    top = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _dgz_kern(grid, block, (
        t_d, p_d, bot, top,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(bot[0]), float(top[0])
    return bot, top


# 38. compute_lapse_rate ----------------------------------------------------
_lapse_rate_code = r"""
extern "C" __global__
void compute_lapse_rate_kernel(
    const double* temperature,  // (ncols, nlevels) degC
    const double* heights,      // (ncols, nlevels) meters AGL
    double bottom_m,            // bottom of layer (m)
    double top_m,               // top of layer (m)
    double* lr_out,             // (ncols,) lapse rate in C/km
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Interpolate temperature at bottom and top heights
    double t_bot = temperature[off];
    double t_top = temperature[off + nlevels - 1];

    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= bottom_m) {
            double h0 = heights[off + k - 1];
            double h1 = heights[off + k];
            if (h1 - h0 > 1e-6) {
                double frac = (bottom_m - h0) / (h1 - h0);
                t_bot = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            } else {
                t_bot = temperature[off + k];
            }
            break;
        }
    }

    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= top_m) {
            double h0 = heights[off + k - 1];
            double h1 = heights[off + k];
            if (h1 - h0 > 1e-6) {
                double frac = (top_m - h0) / (h1 - h0);
                t_top = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            } else {
                t_top = temperature[off + k];
            }
            break;
        }
    }

    // Lapse rate = -(dT/dz) in C/km
    double dz = top_m - bottom_m;
    if (dz > 1e-6) {
        lr_out[col] = -(t_top - t_bot) / (dz / 1000.0);
    } else {
        lr_out[col] = 0.0;
    }
}
"""
_lapse_rate_mod = cp.RawModule(code=_lapse_rate_code)
_lapse_rate_kern = _lapse_rate_mod.get_function("compute_lapse_rate_kernel")


def compute_lapse_rate(temperature, heights, bottom_m=0.0, top_m=3000.0):
    """Lapse rate (C/km) between two height levels.

    temperature (degC), heights (m AGL).
    """
    t_d = _to_cp(temperature)
    h_d = _to_cp(heights)

    if t_d.ndim == 1:
        t_d = t_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    out = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _lapse_rate_kern(grid, block, (
        t_d, h_d,
        cp.float64(bottom_m), cp.float64(top_m),
        out,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(out[0])
    return out


# 39. convective_inhibition_depth -------------------------------------------
_cid_code = r"""
extern "C" __global__
void convective_inhibition_depth_kernel(
    const double* pressure,     // (ncols, nlevels) hPa, descending
    const double* temperature,  // (ncols, nlevels) degC
    const double* dewpoint,     // (ncols, nlevels) degC
    double* cid_out,            // (ncols,) CIN depth in hPa
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;

    // Surface parcel: lift dry adiabatically then moist
    double t_sfc = temperature[off] + 273.15;  // K
    double td_sfc = dewpoint[off] + 273.15;    // K
    double p_sfc = pressure[off];

    // LCL estimate (Bolton 1980)
    double tlcl = 1.0 / (1.0 / (td_sfc - 56.0) + log(t_sfc / td_sfc) / 800.0) + 56.0;
    double plcl = p_sfc * pow(tlcl / t_sfc, 1004.0 / 287.04);

    // Track CIN: negative buoyancy depth (in hPa from surface to LFC)
    double Rd = 287.04;
    double Cp = 1004.0;
    double kappa = Rd / Cp;
    double cin_depth = 0.0;

    // Lift parcel through levels
    double t_parcel = t_sfc;
    for (int k = 1; k < nlevels; k++) {
        double p_k = pressure[off + k];
        double t_env_k = temperature[off + k] + 273.15;

        // Dry or moist adiabatic parcel temperature
        if (p_k > plcl) {
            // Dry adiabatic
            t_parcel = t_sfc * pow(p_k / p_sfc, kappa);
        } else {
            // Simplified moist: use pseudoadiabatic with rough Lv/Cp correction
            double Lv = 2.501e6;
            double es = 6.112 * exp(17.67 * (t_parcel - 273.15) / (t_parcel - 29.65));
            double ws = 0.622 * es / (p_k - es);
            double gamma = (Rd * t_parcel / Cp + Lv * ws / Cp) /
                           (p_k * (1.0 + Lv * Lv * ws * 0.622 / (Cp * Rd * t_parcel * t_parcel)));
            double dp = pressure[off + k - 1] - p_k;
            t_parcel -= gamma * dp;
        }

        // If parcel is colder than environment, accumulate CIN depth
        if (t_parcel < t_env_k) {
            cin_depth += pressure[off + k - 1] - p_k;
        } else {
            // LFC reached
            break;
        }
    }

    cid_out[col] = cin_depth;
}
"""
_cid_mod = cp.RawModule(code=_cid_code)
_cid_kern = _cid_mod.get_function("convective_inhibition_depth_kernel")


def convective_inhibition_depth(pressure, temperature, dewpoint):
    """CIN depth (hPa) from surface to LFC.

    pressure (hPa, descending), temperature (degC), dewpoint (degC).
    """
    p_d = _to_cp(pressure)
    t_d = _to_cp(temperature)
    td_d = _to_cp(dewpoint)

    if p_d.ndim == 1:
        p_d = p_d.reshape(1, -1)
        t_d = t_d.reshape(1, -1)
        td_d = td_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = p_d.shape
    out = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _cid_kern(grid, block, (
        p_d, t_d, td_d, out,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return float(out[0])
    return out


# 40. gradient_richardson_number --------------------------------------------
_gri_code = r"""
extern "C" __global__
void gradient_richardson_number_kernel(
    const double* height,       // (ncols, nlevels) m
    const double* theta,        // (ncols, nlevels) K (potential temperature)
    const double* u,            // (ncols, nlevels) m/s
    const double* v,            // (ncols, nlevels) m/s
    double* ri_out,             // (ncols, nlevels) — NaN at boundaries
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    int off = col * nlevels;
    double g = 9.80665;
    double nan_val = __longlong_as_double(0x7FF8000000000000LL);

    // First and last levels get NaN
    ri_out[off] = nan_val;
    ri_out[off + nlevels - 1] = nan_val;

    for (int k = 1; k < nlevels - 1; k++) {
        double dz = height[off + k + 1] - height[off + k - 1];
        if (dz < 1e-6) {
            ri_out[off + k] = nan_val;
            continue;
        }

        // Central differences
        double dtheta = theta[off + k + 1] - theta[off + k - 1];
        double du = u[off + k + 1] - u[off + k - 1];
        double dv = v[off + k + 1] - v[off + k - 1];

        double theta_mean = theta[off + k];
        double shear_sq = (du * du + dv * dv) / (dz * dz);

        if (shear_sq < 1e-12) {
            // Very weak shear — Ri is effectively infinite
            ri_out[off + k] = 1e6;
        } else {
            double n2 = (g / theta_mean) * (dtheta / dz);
            ri_out[off + k] = n2 / shear_sq;
        }
    }
}
"""
_gri_mod = cp.RawModule(code=_gri_code)
_gri_kern = _gri_mod.get_function("gradient_richardson_number_kernel")


def gradient_richardson_number(height, potential_temperature, u, v):
    """Gradient Richardson number at each level.

    height (m), potential_temperature (K), u, v (m/s).
    Returns array of same shape as input.
    """
    h_d = _to_cp(height)
    theta_d = _to_cp(potential_temperature)
    u_d = _to_cp(u)
    v_d = _to_cp(v)

    if h_d.ndim == 1:
        h_d = h_d.reshape(1, -1)
        theta_d = theta_d.reshape(1, -1)
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = h_d.shape
    out = cp.empty((ncols, nlevels), dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    _gri_kern(grid, block, (
        h_d, theta_d, u_d, v_d, out,
        np.int32(ncols), np.int32(nlevels),
    ))
    if squeeze:
        return out[0]
    return out


# ===================================================================
# Grid-scale SRH (per-column storm motion arrays)
# ===================================================================

_grid_srh_code = r"""
extern "C" __global__
void grid_srh_kernel(
    const double* u,              // (ncols, nlevels) m/s
    const double* v,              // (ncols, nlevels) m/s
    const double* heights,        // (ncols, nlevels) meters AGL
    const double* storm_u_arr,    // (ncols,) per-column storm motion u
    const double* storm_v_arr,    // (ncols,) per-column storm motion v
    double depth,                 // integration depth in meters
    double* srh_pos_out,          // (ncols,) positive SRH
    double* srh_neg_out,          // (ncols,) negative SRH
    double* srh_total_out,        // (ncols,)
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    if (nlevels < 2) {
        srh_pos_out[col] = 0.0;
        srh_neg_out[col] = 0.0;
        srh_total_out[col] = 0.0;
        return;
    }

    double su = storm_u_arr[col];
    double sv = storm_v_arr[col];

    double pos = 0.0, neg = 0.0;
    int offset = col * nlevels;

    double h_start = heights[offset];
    double h_end = h_start + depth;
    double prev_h = h_start;
    double prev_u = u[offset];
    double prev_v = v[offset];
    bool integrated = false;

    for (int k = 1; k < nlevels; k++) {
        double curr_h = heights[offset + k];
        double curr_u = u[offset + k];
        double curr_v = v[offset + k];
        if (curr_h <= prev_h) {
            prev_h = curr_h;
            prev_u = curr_u;
            prev_v = curr_v;
            continue;
        }

        double next_h = curr_h;
        double next_u = curr_u;
        double next_v = curr_v;

        if (curr_h >= h_end) {
            double frac = (h_end - prev_h) / (curr_h - prev_h);
            if (frac < 0.0) frac = 0.0;
            if (frac > 1.0) frac = 1.0;
            next_h = h_end;
            next_u = prev_u + frac * (curr_u - prev_u);
            next_v = prev_v + frac * (curr_v - prev_v);
        }

        double sru0 = prev_u - su;
        double srv0 = prev_v - sv;
        double sru1 = next_u - su;
        double srv1 = next_v - sv;
        double val = (sru1 * srv0) - (sru0 * srv1);

        if (val > 0.0) pos += val;
        else neg += val;
        integrated = true;

        if (curr_h >= h_end) break;
        prev_h = curr_h;
        prev_u = curr_u;
        prev_v = curr_v;
    }

    if (!integrated) {
        pos = 0.0;
        neg = 0.0;
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_grid_srh_mod = cp.RawModule(code=_grid_srh_code)
_grid_srh_kern = _grid_srh_mod.get_function("grid_srh_kernel")


_grid_srh_exact_code = r"""
__device__ int ordered_idx(int k, int nlevels, int reverse) {
    return reverse ? (nlevels - 1 - k) : k;
}

__device__ double interp_at_height_ordered(
    double target_h,
    const double* heights,
    const double* values,
    int offset,
    int nlevels,
    int reverse
) {
    int first = ordered_idx(0, nlevels, reverse);
    int last = ordered_idx(nlevels - 1, nlevels, reverse);
    double h_first = heights[offset + first];
    double h_last = heights[offset + last];
    if (target_h <= h_first) {
        return values[offset + first];
    }
    if (target_h >= h_last) {
        return values[offset + last];
    }
    for (int k = 0; k < nlevels - 1; k++) {
        int i0 = ordered_idx(k, nlevels, reverse);
        int i1 = ordered_idx(k + 1, nlevels, reverse);
        double h0 = heights[offset + i0];
        double h1 = heights[offset + i1];
        if (h0 <= target_h && h1 >= target_h && h1 > h0) {
            double frac = (target_h - h0) / (h1 - h0);
            return values[offset + i0] + frac * (values[offset + i1] - values[offset + i0]);
        }
    }
    return values[offset + last];
}

extern "C" __global__
void grid_srh_exact_kernel(
    const double* u,              // (ncols, nlevels) m/s
    const double* v,              // (ncols, nlevels) m/s
    const double* heights,        // (ncols, nlevels) meters AGL
    double depth,                 // integration top in meters AGL
    double* srh_pos_out,          // (ncols,)
    double* srh_neg_out,          // (ncols,)
    double* srh_total_out,        // (ncols,)
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;
    if (nlevels < 2) {
        srh_pos_out[col] = 0.0;
        srh_neg_out[col] = 0.0;
        srh_total_out[col] = 0.0;
        return;
    }

    int offset = col * nlevels;
    int reverse = heights[offset] > heights[offset + nlevels - 1];

    // 1. Mean wind in the 0-6 km layer using the exact grid-column logic.
    double sum_u = 0.0;
    double sum_v = 0.0;
    double sum_dz = 0.0;
    for (int k = 0; k < nlevels - 1; k++) {
        int i0 = ordered_idx(k, nlevels, reverse);
        int i1 = ordered_idx(k + 1, nlevels, reverse);
        double h_bot = heights[offset + i0];
        double h_next = heights[offset + i1];
        if (h_bot >= 6000.0) {
            break;
        }
        double h_top = h_next < 6000.0 ? h_next : 6000.0;
        double dz = h_top - h_bot;
        if (dz <= 0.0) {
            continue;
        }
        double u_mid = 0.5 * (u[offset + i0] + u[offset + i1]);
        double v_mid = 0.5 * (v[offset + i0] + v[offset + i1]);
        sum_u += u_mid * dz;
        sum_v += v_mid * dz;
        sum_dz += dz;
    }

    if (sum_dz <= 0.0) {
        srh_pos_out[col] = 0.0;
        srh_neg_out[col] = 0.0;
        srh_total_out[col] = 0.0;
        return;
    }

    double mean_u = sum_u / sum_dz;
    double mean_v = sum_v / sum_dz;

    // 2. 0-6 km shear vector from the surface to an interpolated 6 km wind.
    int first = ordered_idx(0, nlevels, reverse);
    double u_sfc = u[offset + first];
    double v_sfc = v[offset + first];
    double u_6km = interp_at_height_ordered(6000.0, heights, u, offset, nlevels, reverse);
    double v_6km = interp_at_height_ordered(6000.0, heights, v, offset, nlevels, reverse);
    double shear_u = u_6km - u_sfc;
    double shear_v = v_6km - v_sfc;
    double shear_mag = sqrt(shear_u * shear_u + shear_v * shear_v);

    double dev_u = 0.0;
    double dev_v = 0.0;
    if (shear_mag > 0.1) {
        double scale = 7.5 / shear_mag;
        dev_u = shear_v * scale;
        dev_v = -shear_u * scale;
    }

    double storm_u = mean_u + dev_u;
    double storm_v = mean_v + dev_v;

    // 3. Integrate SRH from the surface to the requested top.
    double pos = 0.0;
    double neg = 0.0;
    for (int k = 0; k < nlevels - 1; k++) {
        int i0 = ordered_idx(k, nlevels, reverse);
        int i1 = ordered_idx(k + 1, nlevels, reverse);
        double h_bot = heights[offset + i0];
        double h_next = heights[offset + i1];
        if (h_bot >= depth) {
            break;
        }

        double h_top = h_next < depth ? h_next : depth;
        if (h_top <= h_bot) {
            continue;
        }

        double u_bot = u[offset + i0];
        double v_bot = v[offset + i0];
        double u_top_val = u[offset + i1];
        double v_top_val = v[offset + i1];
        if (h_top < h_next && h_next > h_bot) {
            double frac = (h_top - h_bot) / (h_next - h_bot);
            u_top_val = u_bot + frac * (u[offset + i1] - u_bot);
            v_top_val = v_bot + frac * (v[offset + i1] - v_bot);
        }

        double sr_u_bot = u_bot - storm_u;
        double sr_v_bot = v_bot - storm_v;
        double sr_u_top = u_top_val - storm_u;
        double sr_v_top = v_top_val - storm_v;
        double val = sr_u_top * sr_v_bot - sr_u_bot * sr_v_top;

        if (val > 0.0) {
            pos += val;
        } else {
            neg += val;
        }
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_grid_srh_exact_mod = cp.RawModule(code=_grid_srh_exact_code)
_grid_srh_exact_kern = _grid_srh_exact_mod.get_function("grid_srh_exact_kernel")


def grid_storm_relative_helicity(u, v, height, depth, storm_u_arr=None, storm_v_arr=None):
    """Storm-relative helicity with optional per-column storm motion arrays.

    Parameters
    ----------
    u, v : array (m/s) — shape (ncols, nlevels)
    height : array (m AGL) — shape (ncols, nlevels)
    depth : float (m)
    storm_u_arr, storm_v_arr : array (m/s) — shape (ncols,), optional
        When omitted, the kernel computes the exact grid-column Bunkers storm
        motion internally, matching metrust.compute_srh.

    Returns
    -------
    (pos, neg, total) — each (ncols,)
    """
    u_d = _to_cp(u)
    v_d = _to_cp(v)
    h_d = _to_cp(height)
    su_d = _to_cp(storm_u_arr)
    sv_d = _to_cp(storm_v_arr)

    if u_d.ndim == 1:
        u_d = u_d.reshape(1, -1)
        v_d = v_d.reshape(1, -1)
        h_d = h_d.reshape(1, -1)
        su_d = su_d.reshape(1)
        sv_d = sv_d.reshape(1)
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    pos = cp.empty(ncols, dtype=cp.float64)
    neg = cp.empty(ncols, dtype=cp.float64)
    tot = cp.empty(ncols, dtype=cp.float64)
    grid = (_ceil_div(ncols, _BLOCK),)
    block = (min(ncols, _BLOCK),)
    if storm_u_arr is None and storm_v_arr is None:
        _grid_srh_exact_kern(grid, block, (
            u_d, v_d, h_d, cp.float64(depth),
            pos, neg, tot,
            np.int32(ncols), np.int32(nlevels),
        ))
    else:
        if storm_u_arr is None or storm_v_arr is None:
            raise ValueError("storm_u_arr and storm_v_arr must be provided together")
        su_d = _to_cp(storm_u_arr)
        sv_d = _to_cp(storm_v_arr)
        if su_d.ndim == 0:
            su_d = su_d.reshape(1)
            sv_d = sv_d.reshape(1)
        _grid_srh_kern(grid, block, (
            u_d, v_d, h_d,
            su_d, sv_d, cp.float64(depth),
            pos, neg, tot,
            np.int32(ncols), np.int32(nlevels),
        ))
    if squeeze:
        return float(pos[0]), float(neg[0]), float(tot[0])
    return pos, neg, tot
