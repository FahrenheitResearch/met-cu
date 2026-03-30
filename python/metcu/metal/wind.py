"""Wind, kinematics, and severe weather Metal compute kernels for met-cu.

All kernels use Metal Shading Language (MSL) compute shaders dispatched via
the Metal runtime.  Python wrapper functions mirror the CUDA (CuPy) API
signatures (plain arrays in, plain arrays out).

Metal on Apple Silicon does NOT support float64 in compute shaders, so all
GPU work is done in float32.  Python wrappers accept float64 and convert
transparently at the boundary.
"""

import math
import struct
import numpy as np

from .runtime import MetalArray, MetalDevice, MetalKernel, metal_device, to_metal, to_numpy
from metcu.constants import OMEGA, g, Rd, Cp_d, Lv, ZEROCNK, epsilon

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_BLOCK = 256


def _ceil_div(a, b):
    return (a + b - 1) // b


def _to_metal(arr):
    """Ensure *arr* is a MetalArray (float32 on GPU)."""
    if isinstance(arr, MetalArray):
        return arr
    return to_metal(arr)


def _scalar(val):
    return float(val)


def _pack_int(val):
    """Pack an integer as a 4-byte little-endian buffer."""
    return struct.pack("i", int(val))


def _pack_float(val):
    """Pack a float as a 4-byte little-endian float32 buffer."""
    return struct.pack("f", float(val))


# ===================================================================
# 1-9  Per-element wind kernels
# ===================================================================

# 1. wind_speed ----------------------------------------------------------
_wind_speed_source = """
#include <metal_stdlib>
using namespace metal;

kernel void wind_speed_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* speed [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    speed[i] = sqrt(u[i] * u[i] + v[i] * v[i]);
}
"""
_wind_speed_compiled = None


def wind_speed(u, v):
    """Wind speed from u, v components (m/s)."""
    global _wind_speed_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    n = u_d.size
    out = MetalArray(shape=u_d.shape, _device=dev)
    if _wind_speed_compiled is None:
        _wind_speed_compiled = dev.compile(_wind_speed_source, "wind_speed_kernel")
    _wind_speed_compiled.dispatch(
        [u_d, v_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 2. wind_direction -------------------------------------------------------
_wind_direction_source = """
#include <metal_stdlib>
using namespace metal;

kernel void wind_direction_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device float* wdir [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float rad = atan2(-u[i], -v[i]);
    float d = rad * 180.0f / M_PI_F;
    if (d < 0.0f) d += 360.0f;
    if (u[i] == 0.0f && v[i] == 0.0f) d = 0.0f;
    wdir[i] = d;
}
"""
_wind_direction_compiled = None


def wind_direction(u, v):
    """Meteorological wind direction (degrees) from u, v."""
    global _wind_direction_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    n = u_d.size
    out = MetalArray(shape=u_d.shape, _device=dev)
    if _wind_direction_compiled is None:
        _wind_direction_compiled = dev.compile(_wind_direction_source, "wind_direction_kernel")
    _wind_direction_compiled.dispatch(
        [u_d, v_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 3. wind_components -------------------------------------------------------
_wind_components_source = """
#include <metal_stdlib>
using namespace metal;

kernel void wind_components_kernel(
    device const float* speed [[buffer(0)]],
    device const float* direction [[buffer(1)]],
    device float* u_out [[buffer(2)]],
    device float* v_out [[buffer(3)]],
    device const int* n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float rad = direction[i] * M_PI_F / 180.0f;
    u_out[i] = -speed[i] * sin(rad);
    v_out[i] = -speed[i] * cos(rad);
}
"""
_wind_components_compiled = None


def wind_components(speed, direction):
    """u, v components from speed (m/s) and direction (degrees)."""
    global _wind_components_compiled
    dev = metal_device()
    s_d = _to_metal(speed)
    d_d = _to_metal(direction)
    n = s_d.size
    u_out = MetalArray(shape=s_d.shape, _device=dev)
    v_out = MetalArray(shape=s_d.shape, _device=dev)
    if _wind_components_compiled is None:
        _wind_components_compiled = dev.compile(_wind_components_source, "wind_components_kernel")
    _wind_components_compiled.dispatch(
        [s_d, d_d, u_out, v_out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return u_out, v_out


# 4. coriolis_parameter ----------------------------------------------------
_coriolis_source = f"""
#include <metal_stdlib>
using namespace metal;

kernel void coriolis_parameter_kernel(
    device const float* lat [[buffer(0)]],
    device float* f_out [[buffer(1)]],
    device const int* n [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {{
    if (i >= uint(*n)) return;
    f_out[i] = 2.0f * {float(OMEGA)}f * sin(lat[i] * M_PI_F / 180.0f);
}}
"""
_coriolis_compiled = None


def coriolis_parameter(latitude):
    """Coriolis parameter f = 2*Omega*sin(lat).  lat in degrees."""
    global _coriolis_compiled
    dev = metal_device()
    lat_d = _to_metal(latitude)
    n = lat_d.size
    out = MetalArray(shape=lat_d.shape, _device=dev)
    if _coriolis_compiled is None:
        _coriolis_compiled = dev.compile(_coriolis_source, "coriolis_parameter_kernel")
    _coriolis_compiled.dispatch(
        [lat_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 5. angle_to_direction ----------------------------------------------------
_angle_to_direction_source = """
#include <metal_stdlib>
using namespace metal;

kernel void angle_to_direction_kernel(
    device const float* deg [[buffer(0)]],
    device const float* n_dirs_buf [[buffer(1)]],
    device float* code [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float n_dirs = n_dirs_buf[i];
    float d = fmod(deg[i], 360.0f);
    if (d < 0.0f) d += 360.0f;
    float step = 360.0f / n_dirs;
    float c = floor((d + step / 2.0f) / step);
    if (c >= n_dirs) c = 0.0f;
    code[i] = c;
}
"""
_angle_to_direction_compiled = None


def angle_to_direction(degrees, level=16):
    """Convert angle (degrees) to cardinal direction float code."""
    global _angle_to_direction_compiled
    dev = metal_device()
    deg_d = _to_metal(degrees)
    n = deg_d.size
    # Broadcast level to match shape
    level_arr = np.full(deg_d.shape, float(level), dtype=np.float32)
    level_d = to_metal(level_arr)
    out = MetalArray(shape=deg_d.shape, _device=dev)
    if _angle_to_direction_compiled is None:
        _angle_to_direction_compiled = dev.compile(_angle_to_direction_source, "angle_to_direction_kernel")
    _angle_to_direction_compiled.dispatch(
        [deg_d, level_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 6. normal_component ------------------------------------------------------
_normal_component_source = """
#include <metal_stdlib>
using namespace metal;

kernel void normal_component_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* nx_buf [[buffer(2)]],
    device const float* ny_buf [[buffer(3)]],
    device float* comp [[buffer(4)]],
    device const int* n [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    comp[i] = u[i] * nx_buf[i] + v[i] * ny_buf[i];
}
"""
_normal_component_compiled = None


def _dispatch_component(u_d, v_d, cx, cy):
    """Shared dispatch for normal/tangential component."""
    global _normal_component_compiled
    dev = metal_device()
    n = u_d.size
    cx_arr = np.full(u_d.shape, float(cx), dtype=np.float32)
    cy_arr = np.full(u_d.shape, float(cy), dtype=np.float32)
    cx_d = to_metal(cx_arr)
    cy_d = to_metal(cy_arr)
    out = MetalArray(shape=u_d.shape, _device=dev)
    if _normal_component_compiled is None:
        _normal_component_compiled = dev.compile(_normal_component_source, "normal_component_kernel")
    _normal_component_compiled.dispatch(
        [u_d, v_d, cx_d, cy_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


def normal_component(u, v, start, end):
    """Component of wind normal to a cross-section from start to end.

    start, end : (lat, lon) tuples in degrees.
    """
    lat1, lon1 = start
    lat2, lon2 = end
    dx = lon2 - lon1
    dy = lat2 - lat1
    mag = math.sqrt(dx * dx + dy * dy)
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    if mag < 1e-12:
        return MetalArray(shape=u_d.shape, _device=metal_device())
    nx = -dy / mag
    ny = dx / mag
    return _dispatch_component(u_d, v_d, nx, ny)


# 7. tangential_component --------------------------------------------------
def tangential_component(u, v, start, end):
    """Component of wind tangential (parallel) to a cross-section."""
    lat1, lon1 = start
    lat2, lon2 = end
    dx = lon2 - lon1
    dy = lat2 - lat1
    mag = math.sqrt(dx * dx + dy * dy)
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    if mag < 1e-12:
        return MetalArray(shape=u_d.shape, _device=metal_device())
    tx = dx / mag
    ty = dy / mag
    return _dispatch_component(u_d, v_d, tx, ty)


# 8. friction_velocity -----------------------------------------------------
_friction_velocity_source = """
#include <metal_stdlib>
using namespace metal;

kernel void friction_velocity_kernel(
    device const float* u [[buffer(0)]],
    device const float* w [[buffer(1)]],
    device float* ustar_out [[buffer(2)]],
    device const int* n_buf [[buffer(3)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    int n = *n_buf;

    float mean_u = 0.0f, mean_w = 0.0f;
    for (int i = 0; i < n; i++) {
        mean_u += u[i];
        mean_w += w[i];
    }
    mean_u /= float(n);
    mean_w /= float(n);

    float cov = 0.0f;
    for (int i = 0; i < n; i++) {
        cov += (u[i] - mean_u) * (w[i] - mean_w);
    }
    cov /= float(n);
    ustar_out[0] = sqrt(abs(cov));
}
"""
_friction_velocity_compiled = None


def friction_velocity(u, w):
    """Friction velocity u* from u and w time series (m/s)."""
    global _friction_velocity_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    if u_d.ndim > 1:
        u_d = u_d.reshape((u_d.size,))
    w_d = _to_metal(w)
    if w_d.ndim > 1:
        w_d = w_d.reshape((w_d.size,))
    n = u_d.size
    out = MetalArray(shape=(1,), _device=dev)
    if _friction_velocity_compiled is None:
        _friction_velocity_compiled = dev.compile(_friction_velocity_source, "friction_velocity_kernel")
    _friction_velocity_compiled.dispatch(
        [u_d, w_d, out, _pack_int(n)],
        grid_size=(1,), threadgroup_size=(1,),
    )
    return float(out.numpy().flat[0])


# 9. tke -------------------------------------------------------------------
_tke_source = """
#include <metal_stdlib>
using namespace metal;

kernel void tke_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* w [[buffer(2)]],
    device float* tke_out [[buffer(3)]],
    device const int* n_buf [[buffer(4)]],
    uint tid [[thread_position_in_grid]]
) {
    if (tid != 0) return;
    int n = *n_buf;

    float mu = 0.0f, mv = 0.0f, mw = 0.0f;
    for (int i = 0; i < n; i++) {
        mu += u[i]; mv += v[i]; mw += w[i];
    }
    mu /= float(n); mv /= float(n); mw /= float(n);

    float var_u = 0.0f, var_v = 0.0f, var_w = 0.0f;
    for (int i = 0; i < n; i++) {
        float du = u[i] - mu;
        float dv = v[i] - mv;
        float dw = w[i] - mw;
        var_u += du * du;
        var_v += dv * dv;
        var_w += dw * dw;
    }
    var_u /= float(n); var_v /= float(n); var_w /= float(n);
    tke_out[0] = 0.5f * (var_u + var_v + var_w);
}
"""
_tke_compiled = None


def tke(u, v, w):
    """Turbulent kinetic energy from u, v, w time series (m^2/s^2)."""
    global _tke_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    if u_d.ndim > 1:
        u_d = u_d.reshape((u_d.size,))
    v_d = _to_metal(v)
    if v_d.ndim > 1:
        v_d = v_d.reshape((v_d.size,))
    w_d = _to_metal(w)
    if w_d.ndim > 1:
        w_d = w_d.reshape((w_d.size,))
    n = u_d.size
    out = MetalArray(shape=(1,), _device=dev)
    if _tke_compiled is None:
        _tke_compiled = dev.compile(_tke_source, "tke_kernel")
    _tke_compiled.dispatch(
        [u_d, v_d, w_d, out, _pack_int(n)],
        grid_size=(1,), threadgroup_size=(1,),
    )
    return float(out.numpy().flat[0])


# ===================================================================
# 10-16  Column wind kernels
# ===================================================================

# 10. bulk_shear_kernel ----------------------------------------------------
_bulk_shear_source = """
#include <metal_stdlib>
using namespace metal;

kernel void bulk_shear_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device const float* bottom_buf [[buffer(3)]],
    device const float* top_buf [[buffer(4)]],
    device float* shear_u_out [[buffer(5)]],
    device float* shear_v_out [[buffer(6)]],
    device const int* ncols_buf [[buffer(7)]],
    device const int* nlevels_buf [[buffer(8)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    float bottom = *bottom_buf;
    float top = *top_buf;

    int off = int(col) * nlevels;

    // Interpolate u, v at bottom
    float u_bot = u[off], v_bot = v[off];
    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= bottom) {
            float h0 = heights[off + k - 1];
            float h1 = heights[off + k];
            if (h1 - h0 > 1e-6f) {
                float frac = (bottom - h0) / (h1 - h0);
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
    float u_top = u[off + nlevels - 1], v_top = v[off + nlevels - 1];
    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= top) {
            float h0 = heights[off + k - 1];
            float h1 = heights[off + k];
            if (h1 - h0 > 1e-6f) {
                float frac = (top - h0) / (h1 - h0);
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
_bulk_shear_compiled = None


def bulk_shear(u, v, height, bottom, top):
    """Bulk wind shear (u_shear, v_shear) over a height layer.

    For 1-D profile inputs (nlevels,) -> scalar pair.
    For 2-D column inputs (ncols, nlevels) -> arrays of size ncols.
    """
    global _bulk_shear_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    h_d = _to_metal(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    su = MetalArray(shape=(ncols,), _device=dev)
    sv = MetalArray(shape=(ncols,), _device=dev)
    if _bulk_shear_compiled is None:
        _bulk_shear_compiled = dev.compile(_bulk_shear_source, "bulk_shear_kernel")
    _bulk_shear_compiled.dispatch(
        [u_d, v_d, h_d, _pack_float(bottom), _pack_float(top),
         su, sv, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(su.numpy().flat[0]), float(sv.numpy().flat[0])
    return su, sv


# 11. mean_wind_kernel -----------------------------------------------------
_mean_wind_source = """
#include <metal_stdlib>
using namespace metal;

kernel void mean_wind_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device const float* bottom_buf [[buffer(3)]],
    device const float* top_buf [[buffer(4)]],
    device float* mean_u_out [[buffer(5)]],
    device float* mean_v_out [[buffer(6)]],
    device const int* ncols_buf [[buffer(7)]],
    device const int* nlevels_buf [[buffer(8)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    float bottom = *bottom_buf;
    float top = *top_buf;

    int off = int(col) * nlevels;
    float sum_u = 0.0f, sum_v = 0.0f, sum_dh = 0.0f;

    for (int k = 0; k < nlevels; k++) {
        float h = heights[off + k];
        if (h < bottom) continue;
        if (h > top) break;

        float dh;
        if (k == 0 || heights[off + k - 1] < bottom) {
            if (k + 1 < nlevels && heights[off + k + 1] <= top) {
                dh = (heights[off + k + 1] - h) / 2.0f;
            } else {
                dh = 1.0f;
            }
        } else if (k + 1 >= nlevels || heights[off + k + 1] > top) {
            dh = (h - heights[off + k - 1]) / 2.0f;
        } else {
            dh = (heights[off + k + 1] - heights[off + k - 1]) / 2.0f;
        }
        if (dh < 0.0f) dh = 0.0f;

        sum_u += u[off + k] * dh;
        sum_v += v[off + k] * dh;
        sum_dh += dh;
    }

    if (sum_dh > 0.0f) {
        mean_u_out[col] = sum_u / sum_dh;
        mean_v_out[col] = sum_v / sum_dh;
    } else {
        mean_u_out[col] = 0.0f;
        mean_v_out[col] = 0.0f;
    }
}
"""
_mean_wind_compiled = None


def mean_wind(u, v, height, bottom, top):
    """Height-weighted mean wind in a layer.

    Returns (mean_u, mean_v) in m/s.
    """
    global _mean_wind_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    h_d = _to_metal(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    mu = MetalArray(shape=(ncols,), _device=dev)
    mv = MetalArray(shape=(ncols,), _device=dev)
    if _mean_wind_compiled is None:
        _mean_wind_compiled = dev.compile(_mean_wind_source, "mean_wind_kernel")
    _mean_wind_compiled.dispatch(
        [u_d, v_d, h_d, _pack_float(bottom), _pack_float(top),
         mu, mv, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(mu.numpy().flat[0]), float(mv.numpy().flat[0])
    return mu, mv


# 12. storm_relative_helicity_kernel ----------------------------------------
_srh_source = """
#include <metal_stdlib>
using namespace metal;

kernel void srh_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device const float* storm_u_buf [[buffer(3)]],
    device const float* storm_v_buf [[buffer(4)]],
    device const float* depth_buf [[buffer(5)]],
    device float* srh_pos_out [[buffer(6)]],
    device float* srh_neg_out [[buffer(7)]],
    device float* srh_total_out [[buffer(8)]],
    device const int* ncols_buf [[buffer(9)]],
    device const int* nlevels_buf [[buffer(10)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    if (nlevels < 2) {
        srh_pos_out[col] = 0.0f;
        srh_neg_out[col] = 0.0f;
        srh_total_out[col] = 0.0f;
        return;
    }

    float storm_u = *storm_u_buf;
    float storm_v = *storm_v_buf;
    float depth = *depth_buf;

    float pos = 0.0f, neg = 0.0f;
    int offset = int(col) * nlevels;

    float h_start = heights[offset];
    float h_end = h_start + depth;
    float prev_h = h_start;
    float prev_u = u[offset];
    float prev_v = v[offset];
    bool integrated = false;

    for (int k = 1; k < nlevels; k++) {
        float curr_h = heights[offset + k];
        float curr_u = u[offset + k];
        float curr_v = v[offset + k];
        if (curr_h <= prev_h) {
            prev_h = curr_h;
            prev_u = curr_u;
            prev_v = curr_v;
            continue;
        }

        float next_h = curr_h;
        float next_u = curr_u;
        float next_v = curr_v;

        if (curr_h >= h_end) {
            float frac = (h_end - prev_h) / (curr_h - prev_h);
            if (frac < 0.0f) frac = 0.0f;
            if (frac > 1.0f) frac = 1.0f;
            next_h = h_end;
            next_u = prev_u + frac * (curr_u - prev_u);
            next_v = prev_v + frac * (curr_v - prev_v);
        }

        float sru0 = prev_u - storm_u;
        float srv0 = prev_v - storm_v;
        float sru1 = next_u - storm_u;
        float srv1 = next_v - storm_v;
        float val = (sru1 * srv0) - (sru0 * srv1);

        if (val > 0.0f) pos += val;
        else neg += val;
        integrated = true;

        if (curr_h >= h_end) break;
        prev_h = curr_h;
        prev_u = curr_u;
        prev_v = curr_v;
    }

    if (!integrated) {
        pos = 0.0f;
        neg = 0.0f;
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_srh_compiled = None


def storm_relative_helicity(u, v, height, depth, storm_u, storm_v):
    """Storm-relative helicity (positive, negative, total) in m^2/s^2.

    Parameters
    ----------
    u, v : array (m/s)  -- shape (nlevels,) or (ncols, nlevels)
    height : array (m AGL)
    depth : float (m)
    storm_u, storm_v : float (m/s)

    Returns
    -------
    (pos, neg, total) -- scalars or arrays
    """
    global _srh_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    h_d = _to_metal(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    pos = MetalArray(shape=(ncols,), _device=dev)
    neg = MetalArray(shape=(ncols,), _device=dev)
    tot = MetalArray(shape=(ncols,), _device=dev)
    if _srh_compiled is None:
        _srh_compiled = dev.compile(_srh_source, "srh_kernel")
    _srh_compiled.dispatch(
        [u_d, v_d, h_d,
         _pack_float(storm_u), _pack_float(storm_v), _pack_float(depth),
         pos, neg, tot,
         _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(pos.numpy().flat[0]), float(neg.numpy().flat[0]), float(tot.numpy().flat[0])
    return pos, neg, tot


# 13. bunkers_storm_motion_kernel ------------------------------------------
_bunkers_source = """
#include <metal_stdlib>
using namespace metal;

kernel void bunkers_storm_motion_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device float* rm_u_out [[buffer(3)]],
    device float* rm_v_out [[buffer(4)]],
    device float* lm_u_out [[buffer(5)]],
    device float* lm_v_out [[buffer(6)]],
    device float* mw_u_out [[buffer(7)]],
    device float* mw_v_out [[buffer(8)]],
    device const int* ncols_buf [[buffer(9)]],
    device const int* nlevels_buf [[buffer(10)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;
    float D = 7.5f;

    // Mean wind 0-6 km
    float su6 = 0.0f, sv6 = 0.0f, sdh6 = 0.0f;
    for (int k = 0; k < nlevels; k++) {
        float h = heights[off + k];
        if (h > 6000.0f) break;
        float dh;
        if (k == 0) dh = (k + 1 < nlevels) ? (heights[off + k + 1] - h) / 2.0f : 1.0f;
        else if (k + 1 >= nlevels || heights[off + k + 1] > 6000.0f)
            dh = (h - heights[off + k - 1]) / 2.0f;
        else
            dh = (heights[off + k + 1] - heights[off + k - 1]) / 2.0f;
        if (dh < 0.0f) dh = 0.0f;
        su6 += u[off + k] * dh;
        sv6 += v[off + k] * dh;
        sdh6 += dh;
    }
    float mu6 = (sdh6 > 0.0f) ? su6 / sdh6 : 0.0f;
    float mv6 = (sdh6 > 0.0f) ? sv6 / sdh6 : 0.0f;

    // Shear vector: 0-6 km bulk shear
    float u_bot = u[off], v_bot = v[off];
    float u_top = u[off + nlevels - 1], v_top = v[off + nlevels - 1];
    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= 6000.0f) {
            float h0 = heights[off + k - 1];
            float h1 = heights[off + k];
            if (h1 - h0 > 1e-6f) {
                float frac = (6000.0f - h0) / (h1 - h0);
                u_top = u[off + k - 1] + frac * (u[off + k] - u[off + k - 1]);
                v_top = v[off + k - 1] + frac * (v[off + k] - v[off + k - 1]);
            } else {
                u_top = u[off + k];
                v_top = v[off + k];
            }
            break;
        }
    }
    float shear_u = u_top - u_bot;
    float shear_v = v_top - v_bot;

    float shear_mag = sqrt(shear_u * shear_u + shear_v * shear_v);
    float shear_norm_u = 0.0f, shear_norm_v = 0.0f;
    if (shear_mag > 1e-6f) {
        shear_norm_u = shear_u / shear_mag;
        shear_norm_v = shear_v / shear_mag;
    }

    float perp_u = shear_norm_v;
    float perp_v = -shear_norm_u;

    rm_u_out[col] = mu6 + D * perp_u;
    rm_v_out[col] = mv6 + D * perp_v;
    lm_u_out[col] = mu6 - D * perp_u;
    lm_v_out[col] = mv6 - D * perp_v;
    mw_u_out[col] = mu6;
    mw_v_out[col] = mv6;
}
"""
_bunkers_compiled = None


def bunkers_storm_motion(u, v, height):
    """Bunkers storm motion: (right, left, mean) each as (u, v).

    Returns ((rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v)).
    """
    global _bunkers_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    h_d = _to_metal(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    rm_u = MetalArray(shape=(ncols,), _device=dev)
    rm_v = MetalArray(shape=(ncols,), _device=dev)
    lm_u = MetalArray(shape=(ncols,), _device=dev)
    lm_v = MetalArray(shape=(ncols,), _device=dev)
    mw_u = MetalArray(shape=(ncols,), _device=dev)
    mw_v = MetalArray(shape=(ncols,), _device=dev)
    if _bunkers_compiled is None:
        _bunkers_compiled = dev.compile(_bunkers_source, "bunkers_storm_motion_kernel")
    _bunkers_compiled.dispatch(
        [u_d, v_d, h_d,
         rm_u, rm_v, lm_u, lm_v, mw_u, mw_v,
         _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return (
            (float(rm_u.numpy().flat[0]), float(rm_v.numpy().flat[0])),
            (float(lm_u.numpy().flat[0]), float(lm_v.numpy().flat[0])),
            (float(mw_u.numpy().flat[0]), float(mw_v.numpy().flat[0])),
        )
    return (rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v)


# 14. corfidi_storm_motion_kernel ------------------------------------------
_corfidi_source = """
#include <metal_stdlib>
using namespace metal;

kernel void corfidi_storm_motion_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device const float* u_llj_buf [[buffer(3)]],
    device const float* v_llj_buf [[buffer(4)]],
    device float* upwind_u_out [[buffer(5)]],
    device float* upwind_v_out [[buffer(6)]],
    device float* downwind_u_out [[buffer(7)]],
    device float* downwind_v_out [[buffer(8)]],
    device const int* ncols_buf [[buffer(9)]],
    device const int* nlevels_buf [[buffer(10)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    float u_llj = *u_llj_buf;
    float v_llj = *v_llj_buf;
    int off = int(col) * nlevels;

    // Mean wind 0-6 km
    float su = 0.0f, sv = 0.0f, sdh = 0.0f;
    for (int k = 0; k < nlevels; k++) {
        float h = heights[off + k];
        if (h > 6000.0f) break;
        float dh;
        if (k == 0) dh = (k + 1 < nlevels) ? (heights[off + k + 1] - h) / 2.0f : 1.0f;
        else if (k + 1 >= nlevels || heights[off + k + 1] > 6000.0f)
            dh = (h - heights[off + k - 1]) / 2.0f;
        else
            dh = (heights[off + k + 1] - heights[off + k - 1]) / 2.0f;
        if (dh < 0.0f) dh = 0.0f;
        su += u[off + k] * dh;
        sv += v[off + k] * dh;
        sdh += dh;
    }
    float mw_u = (sdh > 0.0f) ? su / sdh : 0.0f;
    float mw_v = (sdh > 0.0f) ? sv / sdh : 0.0f;

    float prop_u = mw_u - u_llj;
    float prop_v = mw_v - v_llj;

    upwind_u_out[col] = prop_u;
    upwind_v_out[col] = prop_v;
    downwind_u_out[col] = prop_u + mw_u;
    downwind_v_out[col] = prop_v + mw_v;
}
"""
_corfidi_compiled = None


def corfidi_storm_motion(u, v, height, u_llj, v_llj):
    """Corfidi MCS motion vectors.

    Returns ((upwind_u, upwind_v), (downwind_u, downwind_v)).
    """
    global _corfidi_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    h_d = _to_metal(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    uu = MetalArray(shape=(ncols,), _device=dev)
    uv = MetalArray(shape=(ncols,), _device=dev)
    du = MetalArray(shape=(ncols,), _device=dev)
    dv = MetalArray(shape=(ncols,), _device=dev)
    if _corfidi_compiled is None:
        _corfidi_compiled = dev.compile(_corfidi_source, "corfidi_storm_motion_kernel")
    _corfidi_compiled.dispatch(
        [u_d, v_d, h_d,
         _pack_float(u_llj), _pack_float(v_llj),
         uu, uv, du, dv,
         _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return (float(uu.numpy().flat[0]), float(uv.numpy().flat[0])), \
               (float(du.numpy().flat[0]), float(dv.numpy().flat[0]))
    return (uu, uv), (du, dv)


# 15. critical_angle_kernel ------------------------------------------------
_critical_angle_source = """
#include <metal_stdlib>
using namespace metal;

kernel void critical_angle_kernel(
    device const float* storm_u [[buffer(0)]],
    device const float* storm_v [[buffer(1)]],
    device const float* u_sfc [[buffer(2)]],
    device const float* v_sfc [[buffer(3)]],
    device const float* u_500 [[buffer(4)]],
    device const float* v_500 [[buffer(5)]],
    device float* angle_out [[buffer(6)]],
    device const int* n [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float shr_u = u_500[i] - u_sfc[i];
    float shr_v = v_500[i] - v_sfc[i];
    float inf_u = storm_u[i] - u_sfc[i];
    float inf_v = storm_v[i] - v_sfc[i];

    float mag_shr = sqrt(shr_u * shr_u + shr_v * shr_v);
    float mag_inf = sqrt(inf_u * inf_u + inf_v * inf_v);
    float denom = mag_shr * mag_inf;

    if (denom < 1e-10f) {
        angle_out[i] = 0.0f;
    } else {
        float cosang = (shr_u * inf_u + shr_v * inf_v) / denom;
        if (cosang > 1.0f) cosang = 1.0f;
        if (cosang < -1.0f) cosang = -1.0f;
        angle_out[i] = acos(cosang) * 180.0f / M_PI_F;
    }
}
"""
_critical_angle_compiled = None


def critical_angle(storm_u, storm_v, u_sfc, v_sfc, u_500, v_500):
    """Critical angle (degrees) between low-level shear and storm-relative inflow."""
    global _critical_angle_compiled
    dev = metal_device()
    su_d = _to_metal(storm_u)
    sv_d = _to_metal(storm_v)
    us_d = _to_metal(u_sfc)
    vs_d = _to_metal(v_sfc)
    u5_d = _to_metal(u_500)
    v5_d = _to_metal(v_500)
    n = su_d.size
    out = MetalArray(shape=su_d.shape, _device=dev)
    if _critical_angle_compiled is None:
        _critical_angle_compiled = dev.compile(_critical_angle_source, "critical_angle_kernel")
    _critical_angle_compiled.dispatch(
        [su_d, sv_d, us_d, vs_d, u5_d, v5_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 16. get_layer_kernel -----------------------------------------------------
_get_layer_source = """
#include <metal_stdlib>
using namespace metal;

kernel void get_layer_kernel(
    device const float* pressure [[buffer(0)]],
    device const float* values [[buffer(1)]],
    device const float* p_bottom_buf [[buffer(2)]],
    device const float* p_top_buf [[buffer(3)]],
    device float* p_out [[buffer(4)]],
    device float* v_out [[buffer(5)]],
    device int* count_out [[buffer(6)]],
    device const int* ncols_buf [[buffer(7)]],
    device const int* nlevels_buf [[buffer(8)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    float p_bottom = *p_bottom_buf;
    float p_top = *p_top_buf;

    int off = int(col) * nlevels;
    int cnt = 0;

    for (int k = 0; k < nlevels; k++) {
        float p = pressure[off + k];
        if (p <= p_bottom && p >= p_top) {
            p_out[off + cnt] = p;
            v_out[off + cnt] = values[off + k];
            cnt++;
        }
    }

    // Fill remaining with NaN
    for (int k = cnt; k < nlevels; k++) {
        p_out[off + k] = NAN;
        v_out[off + k] = NAN;
    }
    count_out[col] = cnt;
}
"""
_get_layer_compiled = None


def get_layer(pressure, values, p_bottom, p_top):
    """Extract data between two pressure levels.

    Returns (p_layer, v_layer) -- 1-D arrays trimmed to valid levels
    (for single-column input) or 2-D arrays padded with NaN.
    """
    global _get_layer_compiled
    dev = metal_device()
    p_d = _to_metal(pressure)
    v_d = _to_metal(values)

    if p_d.ndim == 1:
        p_d = p_d.reshape((1, p_d.size))
        v_d = v_d.reshape((1, v_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = p_d.shape
    p_out = MetalArray(shape=(ncols, nlevels), _device=dev)
    v_out = MetalArray(shape=(ncols, nlevels), _device=dev)
    # count_out needs int32 buffer
    cnt_np = np.zeros(ncols, dtype=np.int32)
    cnt_d = to_metal(cnt_np)
    if _get_layer_compiled is None:
        _get_layer_compiled = dev.compile(_get_layer_source, "get_layer_kernel")
    _get_layer_compiled.dispatch(
        [p_d, v_d, _pack_float(p_bottom), _pack_float(p_top),
         p_out, v_out, cnt_d,
         _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        n = int(cnt_d.numpy().flat[0])
        p_np = p_out.numpy()
        v_np = v_out.numpy()
        return to_metal(p_np[0, :n]), to_metal(v_np[0, :n])
    return p_out, v_out


# ===================================================================
# 17-40  Severe weather parameter kernels
# ===================================================================

# 17. significant_tornado_parameter ----------------------------------------
_stp_source = """
#include <metal_stdlib>
using namespace metal;

kernel void significant_tornado_parameter_kernel(
    device const float* cape [[buffer(0)]],
    device const float* lcl [[buffer(1)]],
    device const float* srh [[buffer(2)]],
    device const float* shear [[buffer(3)]],
    device float* stp [[buffer(4)]],
    device const int* n [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float cape_term = cape[i] / 1500.0f;
    float srh_term  = srh[i] / 150.0f;
    float shear_term = shear[i] / 20.0f;

    float lcl_term;
    if (lcl[i] < 1000.0f) {
        lcl_term = 1.0f;
    } else if (lcl[i] > 2000.0f) {
        lcl_term = 0.0f;
    } else {
        lcl_term = (2000.0f - lcl[i]) / 1000.0f;
    }

    float val = cape_term * srh_term * shear_term * lcl_term;
    if (val < 0.0f) val = 0.0f;
    stp[i] = val;
}
"""
_stp_compiled = None


def significant_tornado_parameter(sbcape, lcl_height, srh_0_1km, bulk_shear_0_6km):
    """Significant Tornado Parameter (fixed-layer STP).

    Parameters: CAPE (J/kg), LCL height (m), SRH (m^2/s^2), shear (m/s).
    """
    global _stp_compiled
    dev = metal_device()
    cape_d = _to_metal(sbcape)
    lcl_d = _to_metal(lcl_height)
    srh_d = _to_metal(srh_0_1km)
    shear_d = _to_metal(bulk_shear_0_6km)
    n = cape_d.size
    out = MetalArray(shape=cape_d.shape, _device=dev)
    if _stp_compiled is None:
        _stp_compiled = dev.compile(_stp_source, "significant_tornado_parameter_kernel")
    _stp_compiled.dispatch(
        [cape_d, lcl_d, srh_d, shear_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 18. supercell_composite_parameter ----------------------------------------
_scp_source = """
#include <metal_stdlib>
using namespace metal;

kernel void supercell_composite_parameter_kernel(
    device const float* mucape [[buffer(0)]],
    device const float* srh [[buffer(1)]],
    device const float* shear [[buffer(2)]],
    device float* scp [[buffer(3)]],
    device const int* n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float val = (mucape[i] / 1000.0f) * (srh[i] / 50.0f) * (shear[i] / 30.0f);
    if (val < 0.0f) val = 0.0f;
    scp[i] = val;
}
"""
_scp_compiled = None


def supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff):
    """Supercell Composite Parameter (SCP)."""
    global _scp_compiled
    dev = metal_device()
    mc_d = _to_metal(mucape)
    sr_d = _to_metal(srh_eff)
    sh_d = _to_metal(bulk_shear_eff)
    n = mc_d.size
    out = MetalArray(shape=mc_d.shape, _device=dev)
    if _scp_compiled is None:
        _scp_compiled = dev.compile(_scp_source, "supercell_composite_parameter_kernel")
    _scp_compiled.dispatch(
        [mc_d, sr_d, sh_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 19. compute_ship ---------------------------------------------------------
_ship_source = """
#include <metal_stdlib>
using namespace metal;

kernel void compute_ship_kernel(
    device const float* cape [[buffer(0)]],
    device const float* shear [[buffer(1)]],
    device const float* t500 [[buffer(2)]],
    device const float* lr [[buffer(3)]],
    device const float* mr [[buffer(4)]],
    device float* ship [[buffer(5)]],
    device const int* n [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float cape_c = (cape[i] < 0.0f) ? 0.0f : cape[i];
    float mr_c   = (mr[i] < 0.0f) ? 0.0f : mr[i];
    float lr_c   = (lr[i] < 0.0f) ? 0.0f : lr[i];
    float t500_c = (t500[i] > 0.0f) ? 0.0f : -t500[i];
    float shear_c = (shear[i] < 0.0f) ? 0.0f : shear[i];

    float val = (cape_c * mr_c * lr_c * t500_c * shear_c) / 42000000.0f;
    if (val < 0.0f) val = 0.0f;
    if (cape_c < 1300.0f) val *= (cape_c / 1300.0f);
    ship[i] = val;
}
"""
_ship_compiled = None


def compute_ship(cape, shear06, t500, lr_700_500, mixing_ratio_gkg):
    """Significant Hail Parameter (SHIP)."""
    global _ship_compiled
    dev = metal_device()
    c_d = _to_metal(cape)
    s_d = _to_metal(shear06)
    t_d = _to_metal(t500)
    l_d = _to_metal(lr_700_500)
    m_d = _to_metal(mixing_ratio_gkg)
    n = c_d.size
    out = MetalArray(shape=c_d.shape, _device=dev)
    if _ship_compiled is None:
        _ship_compiled = dev.compile(_ship_source, "compute_ship_kernel")
    _ship_compiled.dispatch(
        [c_d, s_d, t_d, l_d, m_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 20. compute_ehi ----------------------------------------------------------
_ehi_source = """
#include <metal_stdlib>
using namespace metal;

kernel void compute_ehi_kernel(
    device const float* cape [[buffer(0)]],
    device const float* srh [[buffer(1)]],
    device float* ehi [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    ehi[i] = (cape[i] * srh[i]) / 160000.0f;
}
"""
_ehi_compiled = None


def compute_ehi(cape, srh):
    """Energy-Helicity Index: EHI = CAPE * SRH / 160000."""
    global _ehi_compiled
    dev = metal_device()
    c_d = _to_metal(cape)
    s_d = _to_metal(srh)
    n = c_d.size
    out = MetalArray(shape=c_d.shape, _device=dev)
    if _ehi_compiled is None:
        _ehi_compiled = dev.compile(_ehi_source, "compute_ehi_kernel")
    _ehi_compiled.dispatch(
        [c_d, s_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 21. compute_dcp ----------------------------------------------------------
_dcp_source = """
#include <metal_stdlib>
using namespace metal;

kernel void compute_dcp_kernel(
    device const float* dcape [[buffer(0)]],
    device const float* cape [[buffer(1)]],
    device const float* shear [[buffer(2)]],
    device const float* mean_wind_val [[buffer(3)]],
    device float* dcp [[buffer(4)]],
    device const int* n [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float val = (dcape[i] / 980.0f) * (cape[i] / 2000.0f) * (shear[i] / 20.0f) * (mean_wind_val[i] / 16.0f);
    if (val < 0.0f) val = 0.0f;
    dcp[i] = val;
}
"""
_dcp_compiled = None


def compute_dcp(dcape, mu_cape, shear06, mean_wind_val):
    """Derecho Composite Parameter.

    DCP = (DCAPE/980) * (CAPE/2000) * (shear/20) * (mean_wind/16).
    """
    global _dcp_compiled
    dev = metal_device()
    dc_d = _to_metal(dcape)
    mc_d = _to_metal(mu_cape)
    sh_d = _to_metal(shear06)
    mw_d = _to_metal(mean_wind_val)
    n = dc_d.size
    out = MetalArray(shape=dc_d.shape, _device=dev)
    if _dcp_compiled is None:
        _dcp_compiled = dev.compile(_dcp_source, "compute_dcp_kernel")
    _dcp_compiled.dispatch(
        [dc_d, mc_d, sh_d, mw_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 22. bulk_richardson_number ------------------------------------------------
_brn_source = """
#include <metal_stdlib>
using namespace metal;

kernel void bulk_richardson_number_kernel(
    device const float* cape [[buffer(0)]],
    device const float* shear [[buffer(1)]],
    device float* brn [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    float shear2 = shear[i] * shear[i];
    float denom = 0.5f * shear2;
    brn[i] = (denom > 1e-6f) ? cape[i] / denom : 0.0f;
}
"""
_brn_compiled = None


def bulk_richardson_number(cape, shear_0_6km):
    """Bulk Richardson Number: BRN = CAPE / (0.5 * shear^2)."""
    global _brn_compiled
    dev = metal_device()
    c_d = _to_metal(cape)
    s_d = _to_metal(shear_0_6km)
    n = c_d.size
    out = MetalArray(shape=c_d.shape, _device=dev)
    if _brn_compiled is None:
        _brn_compiled = dev.compile(_brn_source, "bulk_richardson_number_kernel")
    _brn_compiled.dispatch(
        [c_d, s_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 23. k_index ---------------------------------------------------------------
_k_index_source = """
#include <metal_stdlib>
using namespace metal;

kernel void k_index_kernel(
    device const float* t850 [[buffer(0)]],
    device const float* t700 [[buffer(1)]],
    device const float* t500 [[buffer(2)]],
    device const float* td850 [[buffer(3)]],
    device const float* td700 [[buffer(4)]],
    device float* ki [[buffer(5)]],
    device const int* n [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    ki[i] = (t850[i] - t500[i]) + td850[i] - (t700[i] - td700[i]);
}
"""
_k_index_compiled = None


def k_index(t850, t700, t500, td850, td700):
    """K-Index: KI = (T850-T500) + Td850 - (T700-Td700).  All in degC."""
    global _k_index_compiled
    dev = metal_device()
    t850_d = _to_metal(t850)
    t700_d = _to_metal(t700)
    t500_d = _to_metal(t500)
    td850_d = _to_metal(td850)
    td700_d = _to_metal(td700)
    n = t850_d.size
    out = MetalArray(shape=t850_d.shape, _device=dev)
    if _k_index_compiled is None:
        _k_index_compiled = dev.compile(_k_index_source, "k_index_kernel")
    _k_index_compiled.dispatch(
        [t850_d, t700_d, t500_d, td850_d, td700_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 24. total_totals ----------------------------------------------------------
_total_totals_source = """
#include <metal_stdlib>
using namespace metal;

kernel void total_totals_kernel(
    device const float* t850 [[buffer(0)]],
    device const float* t500 [[buffer(1)]],
    device const float* td850 [[buffer(2)]],
    device float* tt [[buffer(3)]],
    device const int* n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    tt[i] = (t850[i] - t500[i]) + (td850[i] - t500[i]);
}
"""
_total_totals_compiled = None


def total_totals(t850, t500, td850):
    """Total Totals Index: TT = VT + CT = (T850-T500) + (Td850-T500)."""
    global _total_totals_compiled
    dev = metal_device()
    t850_d = _to_metal(t850)
    t500_d = _to_metal(t500)
    td850_d = _to_metal(td850)
    n = t850_d.size
    out = MetalArray(shape=t850_d.shape, _device=dev)
    if _total_totals_compiled is None:
        _total_totals_compiled = dev.compile(_total_totals_source, "total_totals_kernel")
    _total_totals_compiled.dispatch(
        [t850_d, t500_d, td850_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 25. cross_totals -----------------------------------------------------------
_cross_totals_source = """
#include <metal_stdlib>
using namespace metal;

kernel void cross_totals_kernel(
    device const float* td850 [[buffer(0)]],
    device const float* t500 [[buffer(1)]],
    device float* ct [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    ct[i] = td850[i] - t500[i];
}
"""
_cross_totals_compiled = None


def cross_totals(td850, t500):
    """Cross Totals: CT = Td850 - T500."""
    global _cross_totals_compiled
    dev = metal_device()
    td850_d = _to_metal(td850)
    t500_d = _to_metal(t500)
    n = td850_d.size
    out = MetalArray(shape=td850_d.shape, _device=dev)
    if _cross_totals_compiled is None:
        _cross_totals_compiled = dev.compile(_cross_totals_source, "cross_totals_kernel")
    _cross_totals_compiled.dispatch(
        [td850_d, t500_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 26. vertical_totals -------------------------------------------------------
_vertical_totals_source = """
#include <metal_stdlib>
using namespace metal;

kernel void vertical_totals_kernel(
    device const float* t850 [[buffer(0)]],
    device const float* t500 [[buffer(1)]],
    device float* vt [[buffer(2)]],
    device const int* n [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    vt[i] = t850[i] - t500[i];
}
"""
_vertical_totals_compiled = None


def vertical_totals(t850, t500):
    """Vertical Totals: VT = T850 - T500."""
    global _vertical_totals_compiled
    dev = metal_device()
    t850_d = _to_metal(t850)
    t500_d = _to_metal(t500)
    n = t850_d.size
    out = MetalArray(shape=t850_d.shape, _device=dev)
    if _vertical_totals_compiled is None:
        _vertical_totals_compiled = dev.compile(_vertical_totals_source, "vertical_totals_kernel")
    _vertical_totals_compiled.dispatch(
        [t850_d, t500_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 27. sweat_index -----------------------------------------------------------
_sweat_source = """
#include <metal_stdlib>
using namespace metal;

kernel void sweat_index_kernel(
    device const float* td850 [[buffer(0)]],
    device const float* tt [[buffer(1)]],
    device const float* f850 [[buffer(2)]],
    device const float* f500 [[buffer(3)]],
    device const float* dd850 [[buffer(4)]],
    device const float* dd500 [[buffer(5)]],
    device float* sweat [[buffer(6)]],
    device const int* n [[buffer(7)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float term1 = 12.0f * td850[i];
    if (term1 < 0.0f) term1 = 0.0f;

    float term2 = 20.0f * (tt[i] - 49.0f);
    if (term2 < 0.0f) term2 = 0.0f;

    float term3 = 2.0f * f850[i];
    float term4 = f500[i];

    float dd_diff = (dd500[i] - dd850[i]) * M_PI_F / 180.0f;
    float term5 = 125.0f * (sin(dd_diff) + 0.2f);
    if (term5 < 0.0f) term5 = 0.0f;

    if (dd850[i] < 130.0f || dd850[i] > 250.0f ||
        dd500[i] < 210.0f || dd500[i] > 310.0f ||
        dd500[i] <= dd850[i] ||
        f850[i] < 15.0f || f500[i] < 15.0f) {
        term5 = 0.0f;
    }

    sweat[i] = term1 + term2 + term3 + term4 + term5;
}
"""
_sweat_compiled = None


def _dispatch_sweat(td850_d, tt_d, f850_d, f500_d, dd850_d, dd500_d):
    """Shared dispatch for sweat index variants."""
    global _sweat_compiled
    dev = metal_device()
    n = td850_d.size
    out = MetalArray(shape=td850_d.shape, _device=dev)
    if _sweat_compiled is None:
        _sweat_compiled = dev.compile(_sweat_source, "sweat_index_kernel")
    _sweat_compiled.dispatch(
        [td850_d, tt_d, f850_d, f500_d, dd850_d, dd500_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


def sweat_index(td850, tt_or_t850=None, f850=None, f500=None,
                dd850=None, dd500=None, t500=None):
    """SWEAT Index.

    Can be called as:
      sweat_index(td850, tt, f850, f500, dd850, dd500)
    or with the 7-param metrust form:
      sweat_index(t850, td850, t500, dd850, dd500, ff850, ff500)
    """
    if t500 is not None:
        t850_v = _to_metal(td850)  # actually t850
        td850_v = _to_metal(tt_or_t850)  # actually td850
        t500_v = _to_metal(t500)
        # tt = (t850 - t500) + (td850 - t500) -- compute on CPU via numpy
        tt_np = t850_v.numpy() - t500_v.numpy() + td850_v.numpy() - t500_v.numpy()
        tt_v = _to_metal(tt_np)
        dd850_v = _to_metal(dd850)
        dd500_v = _to_metal(dd500)
        f850_v = _to_metal(f850)
        f500_v = _to_metal(f500)
    elif dd500 is not None:
        td850_v = _to_metal(td850)
        tt_v = _to_metal(tt_or_t850)
        f850_v = _to_metal(f850)
        f500_v = _to_metal(f500)
        dd850_v = _to_metal(dd850)
        dd500_v = _to_metal(dd500)
    else:
        raise TypeError("sweat_index requires at least 6 parameters")
    return _dispatch_sweat(td850_v, tt_v, f850_v, f500_v, dd850_v, dd500_v)


def sweat_index_direct(t850, td850, t500, dd850, dd500, ff850, ff500):
    """SWEAT Index (direct 7-parameter form matching metrust).

    t850, td850, t500 in degC; dd850, dd500 in degrees; ff850, ff500 in knots.
    """
    t850_v = _to_metal(t850)
    td850_v = _to_metal(td850)
    t500_v = _to_metal(t500)
    tt_np = t850_v.numpy() - t500_v.numpy() + td850_v.numpy() - t500_v.numpy()
    tt_v = _to_metal(tt_np)
    return _dispatch_sweat(
        td850_v, tt_v,
        _to_metal(ff850), _to_metal(ff500),
        _to_metal(dd850), _to_metal(dd500),
    )


# 28. showalter_index -------------------------------------------------------
_showalter_source = """
#include <metal_stdlib>
using namespace metal;

kernel void showalter_index_kernel(
    device const float* pressure [[buffer(0)]],
    device const float* temperature [[buffer(1)]],
    device const float* dewpoint [[buffer(2)]],
    device float* si_out [[buffer(3)]],
    device const int* ncols_buf [[buffer(4)]],
    device const int* nlevels_buf [[buffer(5)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;

    // Find 850 hPa T and Td by interpolation
    float t850 = temperature[off];
    float td850 = dewpoint[off];
    for (int k = 1; k < nlevels; k++) {
        float p0 = pressure[off + k - 1];
        float p1 = pressure[off + k];
        if (p0 >= 850.0f && p1 <= 850.0f) {
            float frac = (850.0f - p0) / (p1 - p0);
            t850 = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            td850 = dewpoint[off + k - 1] + frac * (dewpoint[off + k] - dewpoint[off + k - 1]);
            break;
        }
    }

    // Find 500 hPa T
    float t500_env = temperature[off + nlevels - 1];
    for (int k = 1; k < nlevels; k++) {
        float p0 = pressure[off + k - 1];
        float p1 = pressure[off + k];
        if (p0 >= 500.0f && p1 <= 500.0f) {
            float frac = (500.0f - p0) / (p1 - p0);
            t500_env = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            break;
        }
    }

    float Rd = 287.04f;
    float Cp = 1004.0f;
    float Lv = 2.501e6f;
    float eps = 0.622f;

    float t_k = t850 + 273.15f;
    float td_k = td850 + 273.15f;

    // LCL temperature estimate (Bolton 1980)
    float tlcl = 1.0f / (1.0f / (td_k - 56.0f) + log(t_k / td_k) / 800.0f) + 56.0f;
    float plcl = 850.0f * pow(tlcl / t_k, Cp / Rd);

    // Moist adiabatic ascent from LCL to 500 hPa
    float t_parcel = tlcl;
    float p_curr = plcl;
    float dp = 5.0f;
    while (p_curr > 500.0f) {
        float next_p = p_curr - dp;
        if (next_p < 500.0f) {
            dp = p_curr - 500.0f;
            next_p = 500.0f;
        }
        float es = 6.112f * exp(17.67f * (t_parcel - 273.15f) / (t_parcel - 29.65f));
        float ws = eps * es / (p_curr - es);
        float gamma = (Rd * t_parcel / Cp + Lv * ws / Cp) /
                       (p_curr * (1.0f + Lv * Lv * ws * eps / (Cp * Rd * t_parcel * t_parcel)));
        t_parcel -= gamma * dp;
        p_curr = next_p;
    }

    float t500_parcel = t_parcel - 273.15f;
    si_out[col] = t500_env - t500_parcel;
}
"""
_showalter_compiled = None


def showalter_index(pressure, temperature, dewpoint):
    """Showalter Index: SI = T500_env - T500_parcel (lifted from 850).

    Profile inputs: pressure (hPa), temperature (degC), dewpoint (degC).
    """
    global _showalter_compiled
    dev = metal_device()
    p_d = _to_metal(pressure)
    t_d = _to_metal(temperature)
    td_d = _to_metal(dewpoint)

    if p_d.ndim == 1:
        p_d = p_d.reshape((1, p_d.size))
        t_d = t_d.reshape((1, t_d.size))
        td_d = td_d.reshape((1, td_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = p_d.shape
    out = MetalArray(shape=(ncols,), _device=dev)
    if _showalter_compiled is None:
        _showalter_compiled = dev.compile(_showalter_source, "showalter_index_kernel")
    _showalter_compiled.dispatch(
        [p_d, t_d, td_d, out, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(out.numpy().flat[0])
    return out


# 29. boyden_index ----------------------------------------------------------
_boyden_source = """
#include <metal_stdlib>
using namespace metal;

kernel void boyden_index_kernel(
    device const float* z1000 [[buffer(0)]],
    device const float* z700 [[buffer(1)]],
    device const float* t700 [[buffer(2)]],
    device float* bi [[buffer(3)]],
    device const int* n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;
    bi[i] = (z700[i] - z1000[i]) / 10.0f - t700[i] - 200.0f;
}
"""
_boyden_compiled = None


def boyden_index(z1000, z700, t700):
    """Boyden Index.

    z1000, z700 in meters; t700 in degC.
    """
    global _boyden_compiled
    dev = metal_device()
    z1000_d = _to_metal(z1000)
    z700_d = _to_metal(z700)
    t700_d = _to_metal(t700)
    n = z1000_d.size
    out = MetalArray(shape=z1000_d.shape, _device=dev)
    if _boyden_compiled is None:
        _boyden_compiled = dev.compile(_boyden_source, "boyden_index_kernel")
    _boyden_compiled.dispatch(
        [z1000_d, z700_d, t700_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 30. galvez_davison_index --------------------------------------------------
_gdi_source = """
#include <metal_stdlib>
using namespace metal;

kernel void galvez_davison_index_kernel(
    device const float* t950 [[buffer(0)]],
    device const float* t850 [[buffer(1)]],
    device const float* t700 [[buffer(2)]],
    device const float* t500 [[buffer(3)]],
    device const float* td950 [[buffer(4)]],
    device const float* td850 [[buffer(5)]],
    device const float* td700 [[buffer(6)]],
    device const float* sst [[buffer(7)]],
    device float* gdi [[buffer(8)]],
    device const int* n [[buffer(9)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float Lv = 2.501e6f;
    float Cp = 1004.0f;

    float es950 = 6.112f * exp(17.67f * td950[i] / (td950[i] + 243.5f));
    float w950 = 0.622f * es950 / (950.0f - es950);
    float theta950 = (t950[i] + 273.15f) * pow(1000.0f / 950.0f, 0.286f);
    float thetae_950 = theta950 * exp(Lv * w950 / (Cp * (t950[i] + 273.15f)));

    float es850 = 6.112f * exp(17.67f * td850[i] / (td850[i] + 243.5f));
    float w850 = 0.622f * es850 / (850.0f - es850);
    float theta850 = (t850[i] + 273.15f) * pow(1000.0f / 850.0f, 0.286f);
    float thetae_850 = theta850 * exp(Lv * w850 / (Cp * (t850[i] + 273.15f)));

    float es700 = 6.112f * exp(17.67f * td700[i] / (td700[i] + 243.5f));
    float w700 = 0.622f * es700 / (700.0f - es700);
    float theta700 = (t700[i] + 273.15f) * pow(1000.0f / 700.0f, 0.286f);
    float thetae_700 = theta700 * exp(Lv * w700 / (Cp * (t700[i] + 273.15f)));

    float theta500 = (t500[i] + 273.15f) * pow(1000.0f / 500.0f, 0.286f);

    float alpha = (thetae_950 + thetae_850) / 2.0f;
    float beta  = theta500;
    float gamma_term = theta700 - theta950;
    float cbi = alpha - beta;
    float sst_term = sst[i] - 26.5f;

    gdi[i] = cbi / 20.0f + sst_term * 5.0f - gamma_term * 2.0f;
}
"""
_gdi_compiled = None


def galvez_davison_index(t950, t850, t700, t500, td950, td850, td700, sst):
    """Galvez-Davison Index (tropical thunderstorm potential).

    All temperatures in degC, SST in degC.
    """
    global _gdi_compiled
    dev = metal_device()
    t950_d = _to_metal(t950)
    t850_d = _to_metal(t850)
    t700_d = _to_metal(t700)
    t500_d = _to_metal(t500)
    td950_d = _to_metal(td950)
    td850_d = _to_metal(td850)
    td700_d = _to_metal(td700)
    sst_d = _to_metal(sst)
    n = t950_d.size
    out = MetalArray(shape=t950_d.shape, _device=dev)
    if _gdi_compiled is None:
        _gdi_compiled = dev.compile(_gdi_source, "galvez_davison_index_kernel")
    _gdi_compiled.dispatch(
        [t950_d, t850_d, t700_d, t500_d, td950_d, td850_d, td700_d, sst_d,
         out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 31. fosberg_fire_weather_index -------------------------------------------
_ffwi_source = """
#include <metal_stdlib>
using namespace metal;

kernel void fosberg_fire_weather_index_kernel(
    device const float* temp_f [[buffer(0)]],
    device const float* rh [[buffer(1)]],
    device const float* wind_mph [[buffer(2)]],
    device float* ffwi [[buffer(3)]],
    device const int* n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float m;
    if (rh[i] <= 10.0f) {
        m = 0.03229f + 0.281073f * rh[i] - 0.000578f * rh[i] * temp_f[i];
    } else if (rh[i] <= 50.0f) {
        m = 2.22749f + 0.160107f * rh[i] - 0.01478f * temp_f[i];
    } else {
        m = 21.0606f + 0.005565f * rh[i] * rh[i] - 0.00035f * rh[i] * temp_f[i]
            - 0.483199f * rh[i];
    }
    if (m < 0.0f) m = 0.0f;
    if (m > 30.0f) m = 30.0f;

    float eta = 1.0f - 2.0f * (m / 30.0f) + 1.5f * pow(m / 30.0f, 2.0f)
                 - 0.5f * pow(m / 30.0f, 3.0f);

    float fw = sqrt(1.0f + wind_mph[i] * wind_mph[i]);

    float val = eta * fw / 0.3002f;
    if (val < 0.0f) val = 0.0f;
    ffwi[i] = val;
}
"""
_ffwi_compiled = None


def fosberg_fire_weather_index(temperature, relative_humidity, wind_speed_val):
    """Fosberg Fire Weather Index.

    temperature in degF, relative_humidity in percent, wind_speed in mph.
    """
    global _ffwi_compiled
    dev = metal_device()
    t_d = _to_metal(temperature)
    rh_d = _to_metal(relative_humidity)
    w_d = _to_metal(wind_speed_val)
    n = t_d.size
    out = MetalArray(shape=t_d.shape, _device=dev)
    if _ffwi_compiled is None:
        _ffwi_compiled = dev.compile(_ffwi_source, "fosberg_fire_weather_index_kernel")
    _ffwi_compiled.dispatch(
        [t_d, rh_d, w_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 32. haines_index ----------------------------------------------------------
_haines_source = """
#include <metal_stdlib>
using namespace metal;

kernel void haines_index_kernel(
    device const float* t950 [[buffer(0)]],
    device const float* t850 [[buffer(1)]],
    device const float* td850 [[buffer(2)]],
    device float* haines [[buffer(3)]],
    device const int* n [[buffer(4)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float delta_t = t950[i] - t850[i];
    float a;
    if (delta_t < 4.0f) a = 1.0f;
    else if (delta_t < 8.0f) a = 2.0f;
    else a = 3.0f;

    float delta_td = t850[i] - td850[i];
    float b;
    if (delta_td < 6.0f) b = 1.0f;
    else if (delta_td < 10.0f) b = 2.0f;
    else b = 3.0f;

    haines[i] = a + b;
}
"""
_haines_compiled = None


def haines_index(t_950, t_850, td_850):
    """Haines Index (fire weather stability/moisture).

    All temperatures in degC.
    """
    global _haines_compiled
    dev = metal_device()
    t950_d = _to_metal(t_950)
    t850_d = _to_metal(t_850)
    td850_d = _to_metal(td_850)
    n = t950_d.size
    out = MetalArray(shape=t950_d.shape, _device=dev)
    if _haines_compiled is None:
        _haines_compiled = dev.compile(_haines_source, "haines_index_kernel")
    _haines_compiled.dispatch(
        [t950_d, t850_d, td850_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 33. hot_dry_windy ---------------------------------------------------------
_hdw_source = """
#include <metal_stdlib>
using namespace metal;

kernel void hot_dry_windy_kernel(
    device const float* temp_c [[buffer(0)]],
    device const float* rh [[buffer(1)]],
    device const float* wind_ms [[buffer(2)]],
    device const float* vpd_in [[buffer(3)]],
    device float* hdw [[buffer(4)]],
    device const int* n [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float vpd;
    if (vpd_in[i] > 0.0f) {
        vpd = vpd_in[i];
    } else {
        float es = 6.112f * exp(17.67f * temp_c[i] / (temp_c[i] + 243.5f));
        float ea = es * rh[i] / 100.0f;
        vpd = es - ea;
    }
    float val = vpd * wind_ms[i];
    if (val < 0.0f) val = 0.0f;
    hdw[i] = val;
}
"""
_hdw_compiled = None


def hot_dry_windy(temperature, relative_humidity, wind_speed_val, vpd=0.0):
    """Hot-Dry-Windy Index.

    temperature in degC, relative_humidity in percent, wind_speed in m/s.
    vpd: vapor pressure deficit (hPa), 0 = compute internally.
    """
    global _hdw_compiled
    dev = metal_device()
    t_d = _to_metal(temperature)
    rh_d = _to_metal(relative_humidity)
    w_d = _to_metal(wind_speed_val)
    vpd_d = _to_metal(vpd)
    n = t_d.size
    out = MetalArray(shape=t_d.shape, _device=dev)
    if _hdw_compiled is None:
        _hdw_compiled = dev.compile(_hdw_source, "hot_dry_windy_kernel")
    _hdw_compiled.dispatch(
        [t_d, rh_d, w_d, vpd_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 34. significant_tornado (alternative formulation) -------------------------
_sig_tor_source = """
#include <metal_stdlib>
using namespace metal;

kernel void significant_tornado_kernel(
    device const float* cape [[buffer(0)]],
    device const float* cin [[buffer(1)]],
    device const float* lcl [[buffer(2)]],
    device const float* srh [[buffer(3)]],
    device const float* shear [[buffer(4)]],
    device float* stp [[buffer(5)]],
    device const int* n [[buffer(6)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= uint(*n)) return;

    float cape_term = cape[i] / 1500.0f;

    float cin_term;
    if (cin[i] > -50.0f) {
        cin_term = 1.0f;
    } else if (cin[i] < -200.0f) {
        cin_term = 0.0f;
    } else {
        cin_term = (200.0f + cin[i]) / 150.0f;
    }

    float srh_term  = srh[i] / 150.0f;
    float shear_term = shear[i] / 20.0f;

    float lcl_term;
    if (lcl[i] < 1000.0f) lcl_term = 1.0f;
    else if (lcl[i] > 2000.0f) lcl_term = 0.0f;
    else lcl_term = (2000.0f - lcl[i]) / 1000.0f;

    float val = cape_term * cin_term * srh_term * shear_term * lcl_term;
    if (val < 0.0f) val = 0.0f;
    stp[i] = val;
}
"""
_sig_tor_compiled = None


def significant_tornado(cape, cin, lcl_height, srh_0_1km, bulk_shear_0_6km):
    """Significant Tornado Parameter with CIN term.

    CAPE (J/kg), CIN (J/kg, negative), LCL (m), SRH (m^2/s^2), shear (m/s).
    """
    global _sig_tor_compiled
    dev = metal_device()
    c_d = _to_metal(cape)
    ci_d = _to_metal(cin)
    l_d = _to_metal(lcl_height)
    s_d = _to_metal(srh_0_1km)
    sh_d = _to_metal(bulk_shear_0_6km)
    n = c_d.size
    out = MetalArray(shape=c_d.shape, _device=dev)
    if _sig_tor_compiled is None:
        _sig_tor_compiled = dev.compile(_sig_tor_source, "significant_tornado_kernel")
    _sig_tor_compiled.dispatch(
        [c_d, ci_d, l_d, s_d, sh_d, out, _pack_int(n)],
        grid_size=(n,), threadgroup_size=(min(_BLOCK, n),),
    )
    return out


# 35. freezing_rain_composite -----------------------------------------------
_fzra_source = """
#include <metal_stdlib>
using namespace metal;

kernel void freezing_rain_composite_kernel(
    device const float* temperature [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device const int* precip_type_buf [[buffer(2)]],
    device float* frc_out [[buffer(3)]],
    device const int* ncols_buf [[buffer(4)]],
    device const int* nlevels_buf [[buffer(5)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    int precip_type = *precip_type_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;

    float warm_depth = 0.0f;
    float cold_depth_sfc = 0.0f;
    int found_warm = 0;

    for (int k = 1; k < nlevels; k++) {
        float t = temperature[off + k];
        float dp = pressure[off + k - 1] - pressure[off + k];
        if (dp < 0.0f) dp = -dp;

        if (t > 0.0f) {
            warm_depth += dp;
            found_warm = 1;
        } else if (!found_warm) {
            cold_depth_sfc += dp;
        }
    }

    float frc = 0.0f;
    if (temperature[off] < 0.0f && warm_depth > 50.0f && precip_type > 0) {
        frc = warm_depth / 100.0f * cold_depth_sfc / 50.0f;
    }
    frc_out[col] = frc;
}
"""
_fzra_compiled = None


def freezing_rain_composite(temperature, pressure, precip_type):
    """Freezing rain composite index.

    temperature (degC), pressure (hPa) as profile arrays, precip_type as int flag.
    """
    global _fzra_compiled
    dev = metal_device()
    t_d = _to_metal(temperature)
    p_d = _to_metal(pressure)

    if t_d.ndim == 1:
        t_d = t_d.reshape((1, t_d.size))
        p_d = p_d.reshape((1, p_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    out = MetalArray(shape=(ncols,), _device=dev)
    if _fzra_compiled is None:
        _fzra_compiled = dev.compile(_fzra_source, "freezing_rain_composite_kernel")
    _fzra_compiled.dispatch(
        [t_d, p_d, _pack_int(int(precip_type)),
         out, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(out.numpy().flat[0])
    return out


# 36. warm_nose_check -------------------------------------------------------
_warm_nose_source = """
#include <metal_stdlib>
using namespace metal;

kernel void warm_nose_check_kernel(
    device const float* temperature [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device int* result_out [[buffer(2)]],
    device const int* ncols_buf [[buffer(3)]],
    device const int* nlevels_buf [[buffer(4)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;

    if (temperature[off] >= 0.0f) {
        result_out[col] = 0;
        return;
    }

    int state = 1;
    int found = 0;

    for (int k = 1; k < nlevels; k++) {
        float t = temperature[off + k];
        if (state == 1 && t > 0.0f) {
            state = 2;
        } else if (state == 2 && t <= 0.0f) {
            found = 1;
            break;
        }
    }
    if (state == 2) found = 1;

    result_out[col] = found;
}
"""
_warm_nose_compiled = None


def warm_nose_check(temperature, pressure):
    """Detect warm nose (elevated warm layer above freezing surface).

    Returns bool for single profile, or int array for batch.
    """
    global _warm_nose_compiled
    dev = metal_device()
    t_d = _to_metal(temperature)
    p_d = _to_metal(pressure)

    if t_d.ndim == 1:
        t_d = t_d.reshape((1, t_d.size))
        p_d = p_d.reshape((1, p_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    # int32 output
    out_np = np.zeros(ncols, dtype=np.int32)
    out_d = to_metal(out_np)
    if _warm_nose_compiled is None:
        _warm_nose_compiled = dev.compile(_warm_nose_source, "warm_nose_check_kernel")
    _warm_nose_compiled.dispatch(
        [t_d, p_d, out_d, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return bool(int(out_d.numpy().flat[0]))
    return out_d


# 37. dendritic_growth_zone -------------------------------------------------
_dgz_source = """
#include <metal_stdlib>
using namespace metal;

kernel void dendritic_growth_zone_kernel(
    device const float* temperature [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device float* p_bot_out [[buffer(2)]],
    device float* p_top_out [[buffer(3)]],
    device const int* ncols_buf [[buffer(4)]],
    device const int* nlevels_buf [[buffer(5)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;

    float p_bot = -1.0f, p_top = -1.0f;

    for (int k = 0; k < nlevels; k++) {
        float t = temperature[off + k];
        float p = pressure[off + k];
        if (t <= -12.0f && t >= -18.0f) {
            if (p_bot < 0.0f) p_bot = p;
            p_top = p;
        }
    }

    if (p_bot < 0.0f) {
        for (int k = 1; k < nlevels; k++) {
            float t0 = temperature[off + k - 1];
            float t1 = temperature[off + k];
            float p0 = pressure[off + k - 1];
            float p1 = pressure[off + k];
            if ((t0 > -12.0f && t1 <= -12.0f) || (t0 <= -12.0f && t1 > -12.0f)) {
                float frac = (-12.0f - t0) / (t1 - t0);
                float p_interp = p0 + frac * (p1 - p0);
                if (p_bot < 0.0f) p_bot = p_interp;
            }
            if ((t0 > -18.0f && t1 <= -18.0f) || (t0 <= -18.0f && t1 > -18.0f)) {
                float frac = (-18.0f - t0) / (t1 - t0);
                float p_interp = p0 + frac * (p1 - p0);
                p_top = p_interp;
            }
        }
    }

    p_bot_out[col] = (p_bot > 0.0f) ? p_bot : 0.0f;
    p_top_out[col] = (p_top > 0.0f) ? p_top : 0.0f;
}
"""
_dgz_compiled = None


def dendritic_growth_zone(temperature, pressure):
    """Dendritic growth zone bounds (bottom, top) in hPa.

    temperature (degC), pressure (hPa) as profile arrays.
    """
    global _dgz_compiled
    dev = metal_device()
    t_d = _to_metal(temperature)
    p_d = _to_metal(pressure)

    if t_d.ndim == 1:
        t_d = t_d.reshape((1, t_d.size))
        p_d = p_d.reshape((1, p_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    bot = MetalArray(shape=(ncols,), _device=dev)
    top = MetalArray(shape=(ncols,), _device=dev)
    if _dgz_compiled is None:
        _dgz_compiled = dev.compile(_dgz_source, "dendritic_growth_zone_kernel")
    _dgz_compiled.dispatch(
        [t_d, p_d, bot, top, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(bot.numpy().flat[0]), float(top.numpy().flat[0])
    return bot, top


# 38. compute_lapse_rate ----------------------------------------------------
_lapse_rate_source = """
#include <metal_stdlib>
using namespace metal;

kernel void compute_lapse_rate_kernel(
    device const float* temperature [[buffer(0)]],
    device const float* heights [[buffer(1)]],
    device const float* bottom_m_buf [[buffer(2)]],
    device const float* top_m_buf [[buffer(3)]],
    device float* lr_out [[buffer(4)]],
    device const int* ncols_buf [[buffer(5)]],
    device const int* nlevels_buf [[buffer(6)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    float bottom_m = *bottom_m_buf;
    float top_m = *top_m_buf;

    int off = int(col) * nlevels;

    float t_bot = temperature[off];
    float t_top = temperature[off + nlevels - 1];

    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= bottom_m) {
            float h0 = heights[off + k - 1];
            float h1 = heights[off + k];
            if (h1 - h0 > 1e-6f) {
                float frac = (bottom_m - h0) / (h1 - h0);
                t_bot = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            } else {
                t_bot = temperature[off + k];
            }
            break;
        }
    }

    for (int k = 1; k < nlevels; k++) {
        if (heights[off + k] >= top_m) {
            float h0 = heights[off + k - 1];
            float h1 = heights[off + k];
            if (h1 - h0 > 1e-6f) {
                float frac = (top_m - h0) / (h1 - h0);
                t_top = temperature[off + k - 1] + frac * (temperature[off + k] - temperature[off + k - 1]);
            } else {
                t_top = temperature[off + k];
            }
            break;
        }
    }

    float dz = top_m - bottom_m;
    if (dz > 1e-6f) {
        lr_out[col] = -(t_top - t_bot) / (dz / 1000.0f);
    } else {
        lr_out[col] = 0.0f;
    }
}
"""
_lapse_rate_compiled = None


def compute_lapse_rate(temperature, heights, bottom_m=0.0, top_m=3000.0):
    """Lapse rate (C/km) between two height levels.

    temperature (degC), heights (m AGL).
    """
    global _lapse_rate_compiled
    dev = metal_device()
    t_d = _to_metal(temperature)
    h_d = _to_metal(heights)

    if t_d.ndim == 1:
        t_d = t_d.reshape((1, t_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = t_d.shape
    out = MetalArray(shape=(ncols,), _device=dev)
    if _lapse_rate_compiled is None:
        _lapse_rate_compiled = dev.compile(_lapse_rate_source, "compute_lapse_rate_kernel")
    _lapse_rate_compiled.dispatch(
        [t_d, h_d, _pack_float(bottom_m), _pack_float(top_m),
         out, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(out.numpy().flat[0])
    return out


# 39. convective_inhibition_depth -------------------------------------------
_cid_source = """
#include <metal_stdlib>
using namespace metal;

kernel void convective_inhibition_depth_kernel(
    device const float* pressure [[buffer(0)]],
    device const float* temperature [[buffer(1)]],
    device const float* dewpoint [[buffer(2)]],
    device float* cid_out [[buffer(3)]],
    device const int* ncols_buf [[buffer(4)]],
    device const int* nlevels_buf [[buffer(5)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;

    float t_sfc = temperature[off] + 273.15f;
    float td_sfc = dewpoint[off] + 273.15f;
    float p_sfc = pressure[off];

    float Rd = 287.04f;
    float Cp = 1004.0f;
    float kappa = Rd / Cp;

    // LCL estimate (Bolton 1980)
    float tlcl = 1.0f / (1.0f / (td_sfc - 56.0f) + log(t_sfc / td_sfc) / 800.0f) + 56.0f;
    float plcl = p_sfc * pow(tlcl / t_sfc, 1004.0f / 287.04f);

    float cin_depth = 0.0f;
    float t_parcel = t_sfc;

    for (int k = 1; k < nlevels; k++) {
        float p_k = pressure[off + k];
        float t_env_k = temperature[off + k] + 273.15f;

        if (p_k > plcl) {
            t_parcel = t_sfc * pow(p_k / p_sfc, kappa);
        } else {
            float Lv = 2.501e6f;
            float es = 6.112f * exp(17.67f * (t_parcel - 273.15f) / (t_parcel - 29.65f));
            float ws = 0.622f * es / (p_k - es);
            float gamma = (Rd * t_parcel / Cp + Lv * ws / Cp) /
                           (p_k * (1.0f + Lv * Lv * ws * 0.622f / (Cp * Rd * t_parcel * t_parcel)));
            float dp = pressure[off + k - 1] - p_k;
            t_parcel -= gamma * dp;
        }

        if (t_parcel < t_env_k) {
            cin_depth += pressure[off + k - 1] - p_k;
        } else {
            break;
        }
    }

    cid_out[col] = cin_depth;
}
"""
_cid_compiled = None


def convective_inhibition_depth(pressure, temperature, dewpoint):
    """CIN depth (hPa) from surface to LFC.

    pressure (hPa, descending), temperature (degC), dewpoint (degC).
    """
    global _cid_compiled
    dev = metal_device()
    p_d = _to_metal(pressure)
    t_d = _to_metal(temperature)
    td_d = _to_metal(dewpoint)

    if p_d.ndim == 1:
        p_d = p_d.reshape((1, p_d.size))
        t_d = t_d.reshape((1, t_d.size))
        td_d = td_d.reshape((1, td_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = p_d.shape
    out = MetalArray(shape=(ncols,), _device=dev)
    if _cid_compiled is None:
        _cid_compiled = dev.compile(_cid_source, "convective_inhibition_depth_kernel")
    _cid_compiled.dispatch(
        [p_d, t_d, td_d, out, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return float(out.numpy().flat[0])
    return out


# 40. gradient_richardson_number --------------------------------------------
_gri_source = """
#include <metal_stdlib>
using namespace metal;

kernel void gradient_richardson_number_kernel(
    device const float* height [[buffer(0)]],
    device const float* theta [[buffer(1)]],
    device const float* u [[buffer(2)]],
    device const float* v [[buffer(3)]],
    device float* ri_out [[buffer(4)]],
    device const int* ncols_buf [[buffer(5)]],
    device const int* nlevels_buf [[buffer(6)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;

    int off = int(col) * nlevels;
    float g = 9.80665f;

    // First and last levels get NaN
    ri_out[off] = NAN;
    ri_out[off + nlevels - 1] = NAN;

    for (int k = 1; k < nlevels - 1; k++) {
        float dz = height[off + k + 1] - height[off + k - 1];
        if (dz < 1e-6f) {
            ri_out[off + k] = NAN;
            continue;
        }

        float dtheta = theta[off + k + 1] - theta[off + k - 1];
        float du = u[off + k + 1] - u[off + k - 1];
        float dv = v[off + k + 1] - v[off + k - 1];

        float theta_mean = theta[off + k];
        float shear_sq = (du * du + dv * dv) / (dz * dz);

        if (shear_sq < 1e-12f) {
            ri_out[off + k] = 1e6f;
        } else {
            float n2 = (g / theta_mean) * (dtheta / dz);
            ri_out[off + k] = n2 / shear_sq;
        }
    }
}
"""
_gri_compiled = None


def gradient_richardson_number(height, potential_temperature, u, v):
    """Gradient Richardson number at each level.

    height (m), potential_temperature (K), u, v (m/s).
    Returns array of same shape as input.
    """
    global _gri_compiled
    dev = metal_device()
    h_d = _to_metal(height)
    theta_d = _to_metal(potential_temperature)
    u_d = _to_metal(u)
    v_d = _to_metal(v)

    if h_d.ndim == 1:
        h_d = h_d.reshape((1, h_d.size))
        theta_d = theta_d.reshape((1, theta_d.size))
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = h_d.shape
    out = MetalArray(shape=(ncols, nlevels), _device=dev)
    if _gri_compiled is None:
        _gri_compiled = dev.compile(_gri_source, "gradient_richardson_number_kernel")
    _gri_compiled.dispatch(
        [h_d, theta_d, u_d, v_d, out, _pack_int(ncols), _pack_int(nlevels)],
        grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
    )
    if squeeze:
        return out[0] if hasattr(out, '__getitem__') else out
    return out


# ===================================================================
# Grid-scale SRH (per-column storm motion arrays)
# ===================================================================

_grid_srh_source = """
#include <metal_stdlib>
using namespace metal;

kernel void grid_srh_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device const float* storm_u_arr [[buffer(3)]],
    device const float* storm_v_arr [[buffer(4)]],
    device const float* depth_buf [[buffer(5)]],
    device float* srh_pos_out [[buffer(6)]],
    device float* srh_neg_out [[buffer(7)]],
    device float* srh_total_out [[buffer(8)]],
    device const int* ncols_buf [[buffer(9)]],
    device const int* nlevels_buf [[buffer(10)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    if (nlevels < 2) {
        srh_pos_out[col] = 0.0f;
        srh_neg_out[col] = 0.0f;
        srh_total_out[col] = 0.0f;
        return;
    }

    float su = storm_u_arr[col];
    float sv = storm_v_arr[col];
    float depth = *depth_buf;

    float pos = 0.0f, neg = 0.0f;
    int offset = int(col) * nlevels;

    float h_start = heights[offset];
    float h_end = h_start + depth;
    float prev_h = h_start;
    float prev_u = u[offset];
    float prev_v = v[offset];
    bool integrated = false;

    for (int k = 1; k < nlevels; k++) {
        float curr_h = heights[offset + k];
        float curr_u = u[offset + k];
        float curr_v = v[offset + k];
        if (curr_h <= prev_h) {
            prev_h = curr_h;
            prev_u = curr_u;
            prev_v = curr_v;
            continue;
        }

        float next_h = curr_h;
        float next_u = curr_u;
        float next_v = curr_v;

        if (curr_h >= h_end) {
            float frac = (h_end - prev_h) / (curr_h - prev_h);
            if (frac < 0.0f) frac = 0.0f;
            if (frac > 1.0f) frac = 1.0f;
            next_h = h_end;
            next_u = prev_u + frac * (curr_u - prev_u);
            next_v = prev_v + frac * (curr_v - prev_v);
        }

        float sru0 = prev_u - su;
        float srv0 = prev_v - sv;
        float sru1 = next_u - su;
        float srv1 = next_v - sv;
        float val = (sru1 * srv0) - (sru0 * srv1);

        if (val > 0.0f) pos += val;
        else neg += val;
        integrated = true;

        if (curr_h >= h_end) break;
        prev_h = curr_h;
        prev_u = curr_u;
        prev_v = curr_v;
    }

    if (!integrated) {
        pos = 0.0f;
        neg = 0.0f;
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_grid_srh_compiled = None


_grid_srh_exact_source = """
#include <metal_stdlib>
using namespace metal;

inline int ordered_idx(int k, int nlevels, int reverse) {
    return reverse ? (nlevels - 1 - k) : k;
}

inline float interp_at_height_ordered(
    float target_h,
    device const float* heights,
    device const float* values,
    int offset,
    int nlevels,
    int reverse
) {
    int first = ordered_idx(0, nlevels, reverse);
    int last = ordered_idx(nlevels - 1, nlevels, reverse);
    float h_first = heights[offset + first];
    float h_last = heights[offset + last];
    if (target_h <= h_first) {
        return values[offset + first];
    }
    if (target_h >= h_last) {
        return values[offset + last];
    }
    for (int k = 0; k < nlevels - 1; k++) {
        int i0 = ordered_idx(k, nlevels, reverse);
        int i1 = ordered_idx(k + 1, nlevels, reverse);
        float h0 = heights[offset + i0];
        float h1 = heights[offset + i1];
        if (h0 <= target_h && h1 >= target_h && h1 > h0) {
            float frac = (target_h - h0) / (h1 - h0);
            return values[offset + i0] + frac * (values[offset + i1] - values[offset + i0]);
        }
    }
    return values[offset + last];
}

kernel void grid_srh_exact_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* heights [[buffer(2)]],
    device const float* depth_buf [[buffer(3)]],
    device float* srh_pos_out [[buffer(4)]],
    device float* srh_neg_out [[buffer(5)]],
    device float* srh_total_out [[buffer(6)]],
    device const int* ncols_buf [[buffer(7)]],
    device const int* nlevels_buf [[buffer(8)]],
    uint col [[thread_position_in_grid]]
) {
    int ncols = *ncols_buf;
    int nlevels = *nlevels_buf;
    if (int(col) >= ncols) return;
    if (nlevels < 2) {
        srh_pos_out[col] = 0.0f;
        srh_neg_out[col] = 0.0f;
        srh_total_out[col] = 0.0f;
        return;
    }

    float depth = *depth_buf;
    int offset = int(col) * nlevels;
    int reverse = heights[offset] > heights[offset + nlevels - 1] ? 1 : 0;

    // 1. Mean wind in the 0-6 km layer
    float sum_u = 0.0f;
    float sum_v = 0.0f;
    float sum_dz = 0.0f;
    for (int k = 0; k < nlevels - 1; k++) {
        int i0 = ordered_idx(k, nlevels, reverse);
        int i1 = ordered_idx(k + 1, nlevels, reverse);
        float h_bot = heights[offset + i0];
        float h_next = heights[offset + i1];
        if (h_bot >= 6000.0f) break;
        float h_top = h_next < 6000.0f ? h_next : 6000.0f;
        float dz = h_top - h_bot;
        if (dz <= 0.0f) continue;
        float u_mid = 0.5f * (u[offset + i0] + u[offset + i1]);
        float v_mid = 0.5f * (v[offset + i0] + v[offset + i1]);
        sum_u += u_mid * dz;
        sum_v += v_mid * dz;
        sum_dz += dz;
    }

    if (sum_dz <= 0.0f) {
        srh_pos_out[col] = 0.0f;
        srh_neg_out[col] = 0.0f;
        srh_total_out[col] = 0.0f;
        return;
    }

    float mean_u = sum_u / sum_dz;
    float mean_v = sum_v / sum_dz;

    // 2. 0-6 km shear vector
    int first = ordered_idx(0, nlevels, reverse);
    float u_sfc = u[offset + first];
    float v_sfc = v[offset + first];
    float u_6km = interp_at_height_ordered(6000.0f, heights, u, offset, nlevels, reverse);
    float v_6km = interp_at_height_ordered(6000.0f, heights, v, offset, nlevels, reverse);
    float shear_u = u_6km - u_sfc;
    float shear_v = v_6km - v_sfc;
    float shear_mag = sqrt(shear_u * shear_u + shear_v * shear_v);

    float dev_u = 0.0f;
    float dev_v = 0.0f;
    if (shear_mag > 0.1f) {
        float scale = 7.5f / shear_mag;
        dev_u = shear_v * scale;
        dev_v = -shear_u * scale;
    }

    float storm_u = mean_u + dev_u;
    float storm_v = mean_v + dev_v;

    // 3. Integrate SRH
    float pos = 0.0f;
    float neg = 0.0f;
    for (int k = 0; k < nlevels - 1; k++) {
        int i0 = ordered_idx(k, nlevels, reverse);
        int i1 = ordered_idx(k + 1, nlevels, reverse);
        float h_bot = heights[offset + i0];
        float h_next = heights[offset + i1];
        if (h_bot >= depth) break;

        float h_top = h_next < depth ? h_next : depth;
        if (h_top <= h_bot) continue;

        float u_bot_v = u[offset + i0];
        float v_bot_v = v[offset + i0];
        float u_top_val = u[offset + i1];
        float v_top_val = v[offset + i1];
        if (h_top < h_next && h_next > h_bot) {
            float frac = (h_top - h_bot) / (h_next - h_bot);
            u_top_val = u_bot_v + frac * (u[offset + i1] - u_bot_v);
            v_top_val = v_bot_v + frac * (v[offset + i1] - v_bot_v);
        }

        float sr_u_bot = u_bot_v - storm_u;
        float sr_v_bot = v_bot_v - storm_v;
        float sr_u_top = u_top_val - storm_u;
        float sr_v_top = v_top_val - storm_v;
        float val = sr_u_top * sr_v_bot - sr_u_bot * sr_v_top;

        if (val > 0.0f) pos += val;
        else neg += val;
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_grid_srh_exact_compiled = None


def grid_storm_relative_helicity(u, v, height, depth, storm_u_arr=None, storm_v_arr=None):
    """Storm-relative helicity with optional per-column storm motion arrays.

    Parameters
    ----------
    u, v : array (m/s) -- shape (ncols, nlevels)
    height : array (m AGL) -- shape (ncols, nlevels)
    depth : float (m)
    storm_u_arr, storm_v_arr : array (m/s) -- shape (ncols,), optional
        When omitted, the kernel computes the exact grid-column Bunkers storm
        motion internally, matching metrust.compute_srh.

    Returns
    -------
    (pos, neg, total) -- each (ncols,)
    """
    global _grid_srh_compiled, _grid_srh_exact_compiled
    dev = metal_device()
    u_d = _to_metal(u)
    v_d = _to_metal(v)
    h_d = _to_metal(height)

    if u_d.ndim == 1:
        u_d = u_d.reshape((1, u_d.size))
        v_d = v_d.reshape((1, v_d.size))
        h_d = h_d.reshape((1, h_d.size))
        squeeze = True
    else:
        squeeze = False

    ncols, nlevels = u_d.shape
    pos = MetalArray(shape=(ncols,), _device=dev)
    neg = MetalArray(shape=(ncols,), _device=dev)
    tot = MetalArray(shape=(ncols,), _device=dev)

    if storm_u_arr is None and storm_v_arr is None:
        if _grid_srh_exact_compiled is None:
            _grid_srh_exact_compiled = dev.compile(_grid_srh_exact_source, "grid_srh_exact_kernel")
        _grid_srh_exact_compiled.dispatch(
            [u_d, v_d, h_d, _pack_float(depth),
             pos, neg, tot,
             _pack_int(ncols), _pack_int(nlevels)],
            grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
        )
    else:
        if storm_u_arr is None or storm_v_arr is None:
            raise ValueError("storm_u_arr and storm_v_arr must be provided together")
        su_d = _to_metal(storm_u_arr)
        sv_d = _to_metal(storm_v_arr)
        if su_d.ndim == 0:
            su_d = su_d.reshape((1,))
            sv_d = sv_d.reshape((1,))
        if _grid_srh_compiled is None:
            _grid_srh_compiled = dev.compile(_grid_srh_source, "grid_srh_kernel")
        _grid_srh_compiled.dispatch(
            [u_d, v_d, h_d,
             su_d, sv_d, _pack_float(depth),
             pos, neg, tot,
             _pack_int(ncols), _pack_int(nlevels)],
            grid_size=(ncols,), threadgroup_size=(min(ncols, _BLOCK),),
        )

    if squeeze:
        return float(pos.numpy().flat[0]), float(neg.numpy().flat[0]), float(tot.numpy().flat[0])
    return pos, neg, tot
