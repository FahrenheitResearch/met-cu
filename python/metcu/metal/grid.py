"""
Grid operations and stencil Metal compute kernels for met-cu (Apple Silicon).

2D stencil computations on (ny, nx) fields, smoothing/convolution kernels,
interpolation kernels, and grid utility kernels.  Each kernel launches one
GPU thread per output grid point using 2D (16, 16) thread groups.

All GPU arrays are float32 (Metal on M1 does not support float64 in compute
shaders).  Python wrapper functions accept float64 and convert transparently
via the MetalArray/to_metal helpers.
"""

import math
import struct
import numpy as np

from .runtime import MetalArray, MetalDevice, metal_device, to_metal, to_numpy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gpu(arr):
    """Accept numpy, MetalArray, or scalar, return MetalArray on the device."""
    if isinstance(arr, MetalArray):
        return arr
    if isinstance(arr, (int, float)):
        return MetalArray(data=np.array(arr, dtype=np.float32))
    if isinstance(arr, np.ndarray):
        return MetalArray(data=arr)
    return MetalArray(data=np.asarray(arr, dtype=np.float32))


def _broadcast_spacing(val, shape):
    """Broadcast a scalar or 0-d array to a full 2-D MetalArray matching shape."""
    if isinstance(val, MetalArray):
        arr = val.numpy()
    else:
        arr = np.asarray(val, dtype=np.float64)
    if arr.ndim == 0:
        return MetalArray(data=np.full(shape, float(arr), dtype=np.float32))
    if arr.shape != shape:
        return MetalArray(data=np.broadcast_to(arr, shape).copy().astype(np.float32))
    return MetalArray(data=arr.astype(np.float32))


def _dims_buf(ny, nx):
    """Pack (ny, nx) into a bytes buffer for kernel dims argument."""
    return struct.pack("ii", ny, nx)


def _dims3_buf(ny, nx, nz):
    """Pack (ny, nx, nz) into a bytes buffer for kernel dims argument."""
    return struct.pack("iii", ny, nx, nz)


def _tg(nx, ny):
    """Compute threadgroup size capped at 16."""
    return (min(16, nx), min(16, ny))


# ===================================================================
# Shared MSL device functions for boundary-aware finite differences.
# ===================================================================
_deriv_metal_funcs = """
#include <metal_stdlib>
using namespace metal;

/* ---- df/dx at (j, i) with boundary-aware stencil ---- */
inline float ddx(device const float* f, device const float* dx,
                 int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dx[idx];
    if (i == 0)
        return (-3.0f*f[idx] + 4.0f*f[idx+1] - f[idx+2]) / (2.0f*h);
    if (i == nx - 1)
        return (3.0f*f[idx] - 4.0f*f[idx-1] + f[idx-2]) / (2.0f*h);
    return (f[idx+1] - f[idx-1]) / (2.0f*h);
}

/* ---- df/dy at (j, i) with boundary-aware stencil ---- */
inline float ddy(device const float* f, device const float* dy,
                 int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dy[idx];
    if (j == 0)
        return (-3.0f*f[idx] + 4.0f*f[idx+nx] - f[idx+2*nx]) / (2.0f*h);
    if (j == ny - 1)
        return (3.0f*f[idx] - 4.0f*f[idx-nx] + f[idx-2*nx]) / (2.0f*h);
    return (f[idx+nx] - f[idx-nx]) / (2.0f*h);
}

/* ---- d2f/dx2 at (j, i) with boundary-aware stencil ---- */
inline float d2dx2(device const float* f, device const float* dx,
                   int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dx[idx];
    if (i == 0)
        return (f[idx] - 2.0f*f[idx+1] + f[idx+2]) / (h*h);
    if (i == nx - 1)
        return (f[idx] - 2.0f*f[idx-1] + f[idx-2]) / (h*h);
    return (f[idx+1] - 2.0f*f[idx] + f[idx-1]) / (h*h);
}

/* ---- d2f/dy2 at (j, i) with boundary-aware stencil ---- */
inline float d2dy2(device const float* f, device const float* dy,
                   int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dy[idx];
    if (j == 0)
        return (f[idx] - 2.0f*f[idx+nx] + f[idx+2*nx]) / (h*h);
    if (j == ny - 1)
        return (f[idx] - 2.0f*f[idx-nx] + f[idx-2*nx]) / (h*h);
    return (f[idx+nx] - 2.0f*f[idx] + f[idx-nx]) / (h*h);
}
"""


# ===================================================================
# 1. Differential-operator stencil kernels
# ===================================================================

# ------------------------------------------------------------------
# 1. vorticity  dv/dx - du/dy
# ------------------------------------------------------------------
_vorticity_source = _deriv_metal_funcs + """
kernel void vorticity_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dvdx = ddx(v, dx, j, i, ny, nx);
    float dudy = ddy(u, dy, j, i, ny, nx);
    out[idx] = dvdx - dudy;
}
"""
_vorticity_compiled = None


def vorticity(u, v, dx, dy):
    """Relative vorticity  dv/dx - du/dy."""
    global _vorticity_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _vorticity_compiled is None:
        _vorticity_compiled = dev.compile(_vorticity_source, "vorticity_kernel")
    _vorticity_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 2. divergence  du/dx + dv/dy
# ------------------------------------------------------------------
_divergence_source = _deriv_metal_funcs + """
kernel void divergence_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dudx = ddx(u, dx, j, i, ny, nx);
    float dvdy = ddy(v, dy, j, i, ny, nx);
    out[idx] = dudx + dvdy;
}
"""
_divergence_compiled = None


def divergence(u, v, dx, dy):
    """Horizontal divergence  du/dx + dv/dy."""
    global _divergence_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _divergence_compiled is None:
        _divergence_compiled = dev.compile(_divergence_source, "divergence_kernel")
    _divergence_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 3. absolute_vorticity  relative_vort + f
# ------------------------------------------------------------------
_absolute_vorticity_source = _deriv_metal_funcs + """
kernel void absolute_vorticity_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device const float* f [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const int* dims [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dvdx = ddx(v, dx, j, i, ny, nx);
    float dudy = ddy(u, dy, j, i, ny, nx);
    out[idx] = dvdx - dudy + f[idx];
}
"""
_absolute_vorticity_compiled = None


def absolute_vorticity(u, v, dx, dy, f):
    """Absolute vorticity  (dv/dx - du/dy) + f."""
    global _absolute_vorticity_compiled
    u, v, f = _to_gpu(u), _to_gpu(v), _to_gpu(f)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _absolute_vorticity_compiled is None:
        _absolute_vorticity_compiled = dev.compile(_absolute_vorticity_source, "absolute_vorticity_kernel")
    _absolute_vorticity_compiled.dispatch(
        [u, v, dx_m, dy_m, f, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 4. shearing_deformation  dv/dx + du/dy
# ------------------------------------------------------------------
_shearing_deformation_source = _deriv_metal_funcs + """
kernel void shearing_deformation_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dvdx = ddx(v, dx, j, i, ny, nx);
    float dudy = ddy(u, dy, j, i, ny, nx);
    out[idx] = dvdx + dudy;
}
"""
_shearing_deformation_compiled = None


def shearing_deformation(u, v, dx, dy):
    """Shearing deformation  dv/dx + du/dy."""
    global _shearing_deformation_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _shearing_deformation_compiled is None:
        _shearing_deformation_compiled = dev.compile(_shearing_deformation_source, "shearing_deformation_kernel")
    _shearing_deformation_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 5. stretching_deformation  du/dx - dv/dy
# ------------------------------------------------------------------
_stretching_deformation_source = _deriv_metal_funcs + """
kernel void stretching_deformation_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dudx = ddx(u, dx, j, i, ny, nx);
    float dvdy = ddy(v, dy, j, i, ny, nx);
    out[idx] = dudx - dvdy;
}
"""
_stretching_deformation_compiled = None


def stretching_deformation(u, v, dx, dy):
    """Stretching deformation  du/dx - dv/dy."""
    global _stretching_deformation_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _stretching_deformation_compiled is None:
        _stretching_deformation_compiled = dev.compile(_stretching_deformation_source, "stretching_deformation_kernel")
    _stretching_deformation_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 6. total_deformation  sqrt(shearing^2 + stretching^2)
# ------------------------------------------------------------------
_total_deformation_source = _deriv_metal_funcs + """
kernel void total_deformation_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dudx = ddx(u, dx, j, i, ny, nx);
    float dvdy = ddy(v, dy, j, i, ny, nx);
    float dvdx = ddx(v, dx, j, i, ny, nx);
    float dudy = ddy(u, dy, j, i, ny, nx);
    float shear = dvdx + dudy;
    float stretch = dudx - dvdy;
    out[idx] = sqrt(shear * shear + stretch * stretch);
}
"""
_total_deformation_compiled = None


def total_deformation(u, v, dx, dy):
    """Total deformation  sqrt(shearing^2 + stretching^2)."""
    global _total_deformation_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _total_deformation_compiled is None:
        _total_deformation_compiled = dev.compile(_total_deformation_source, "total_deformation_kernel")
    _total_deformation_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 7. curvature_vorticity
# ------------------------------------------------------------------
_curvature_vorticity_source = _deriv_metal_funcs + """
kernel void curvature_vorticity_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float uc = u[idx], vc = v[idx];
    float spd2 = uc * uc + vc * vc;
    if (spd2 < 1e-20f) { out[idx] = 0.0f; return; }
    float dudx_v = ddx(u, dx, j, i, ny, nx);
    float dudy_v = ddy(u, dy, j, i, ny, nx);
    float dvdx_v = ddx(v, dx, j, i, ny, nx);
    float dvdy_v = ddy(v, dy, j, i, ny, nx);
    out[idx] = (uc * uc * dvdx_v - vc * vc * dudy_v
                + uc * vc * (dvdy_v - dudx_v)) / spd2;
}
"""
_curvature_vorticity_compiled = None


def curvature_vorticity(u, v, dx, dy):
    """Curvature vorticity component of relative vorticity."""
    global _curvature_vorticity_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _curvature_vorticity_compiled is None:
        _curvature_vorticity_compiled = dev.compile(_curvature_vorticity_source, "curvature_vorticity_kernel")
    _curvature_vorticity_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 8. shear_vorticity
# ------------------------------------------------------------------
_shear_vorticity_source = _deriv_metal_funcs + """
kernel void shear_vorticity_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float uc = u[idx], vc = v[idx];
    float spd2 = uc * uc + vc * vc;
    if (spd2 < 1e-20f) { out[idx] = 0.0f; return; }
    float dudx_v = ddx(u, dx, j, i, ny, nx);
    float dudy_v = ddy(u, dy, j, i, ny, nx);
    float dvdx_v = ddx(v, dx, j, i, ny, nx);
    float dvdy_v = ddy(v, dy, j, i, ny, nx);
    out[idx] = -(vc * vc * dudx_v + uc * uc * dvdy_v
                 - uc * vc * (dvdx_v + dudy_v)) / spd2;
}
"""
_shear_vorticity_compiled = None


def shear_vorticity(u, v, dx, dy):
    """Shear vorticity component of relative vorticity."""
    global _shear_vorticity_compiled
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = MetalArray(shape=(ny, nx))
    dev = metal_device()
    if _shear_vorticity_compiled is None:
        _shear_vorticity_compiled = dev.compile(_shear_vorticity_source, "shear_vorticity_kernel")
    _shear_vorticity_compiled.dispatch(
        [u, v, dx_m, dy_m, out, _dims_buf(ny, nx)],
        grid_size=(nx, ny),
        threadgroup_size=_tg(nx, ny),
    )
    return out


# ------------------------------------------------------------------
# 9. first_derivative_x  df/dx
# ------------------------------------------------------------------
_first_derivative_x_source = _deriv_metal_funcs + """
kernel void first_derivative_x_kernel(
    device const float* f [[buffer(0)]],
    device const float* dx [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = ddx(f, dx, j, i, ny, nx);
}
"""
_first_derivative_x_compiled = None


def first_derivative_x(f, dx):
    """df/dx via centered finite differences."""
    global _first_derivative_x_compiled
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx_m = _broadcast_spacing(dx, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _first_derivative_x_compiled is None:
        _first_derivative_x_compiled = dev.compile(_first_derivative_x_source, "first_derivative_x_kernel")
    _first_derivative_x_compiled.dispatch(
        [f, dx_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 10. first_derivative_y  df/dy
# ------------------------------------------------------------------
_first_derivative_y_source = _deriv_metal_funcs + """
kernel void first_derivative_y_kernel(
    device const float* f [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = ddy(f, dy, j, i, ny, nx);
}
"""
_first_derivative_y_compiled = None


def first_derivative_y(f, dy):
    """df/dy via centered finite differences."""
    global _first_derivative_y_compiled
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dy_m = _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _first_derivative_y_compiled is None:
        _first_derivative_y_compiled = dev.compile(_first_derivative_y_source, "first_derivative_y_kernel")
    _first_derivative_y_compiled.dispatch(
        [f, dy_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 11. second_derivative_x  d2f/dx2
# ------------------------------------------------------------------
_second_derivative_x_source = _deriv_metal_funcs + """
kernel void second_derivative_x_kernel(
    device const float* f [[buffer(0)]],
    device const float* dx [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = d2dx2(f, dx, j, i, ny, nx);
}
"""
_second_derivative_x_compiled = None


def second_derivative_x(f, dx):
    """d2f/dx2 via centered finite differences."""
    global _second_derivative_x_compiled
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx_m = _broadcast_spacing(dx, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _second_derivative_x_compiled is None:
        _second_derivative_x_compiled = dev.compile(_second_derivative_x_source, "second_derivative_x_kernel")
    _second_derivative_x_compiled.dispatch(
        [f, dx_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 12. second_derivative_y  d2f/dy2
# ------------------------------------------------------------------
_second_derivative_y_source = _deriv_metal_funcs + """
kernel void second_derivative_y_kernel(
    device const float* f [[buffer(0)]],
    device const float* dy [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* dims [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = d2dy2(f, dy, j, i, ny, nx);
}
"""
_second_derivative_y_compiled = None


def second_derivative_y(f, dy):
    """d2f/dy2 via centered finite differences."""
    global _second_derivative_y_compiled
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dy_m = _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _second_derivative_y_compiled is None:
        _second_derivative_y_compiled = dev.compile(_second_derivative_y_source, "second_derivative_y_kernel")
    _second_derivative_y_compiled.dispatch(
        [f, dy_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 13. laplacian  d2f/dx2 + d2f/dy2
# ------------------------------------------------------------------
_laplacian_source = _deriv_metal_funcs + """
kernel void laplacian_kernel(
    device const float* f [[buffer(0)]],
    device const float* dx [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const int* dims [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = d2dx2(f, dx, j, i, ny, nx) + d2dy2(f, dy, j, i, ny, nx);
}
"""
_laplacian_compiled = None


def laplacian(f, dx, dy):
    """Laplacian  d2f/dx2 + d2f/dy2."""
    global _laplacian_compiled
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _laplacian_compiled is None:
        _laplacian_compiled = dev.compile(_laplacian_source, "laplacian_kernel")
    _laplacian_compiled.dispatch(
        [f, dx_m, dy_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 14. gradient  (df/dx, df/dy)
# ------------------------------------------------------------------
_gradient_source = _deriv_metal_funcs + """
kernel void gradient_kernel(
    device const float* f [[buffer(0)]],
    device const float* dx [[buffer(1)]],
    device const float* dy [[buffer(2)]],
    device float* dfdx [[buffer(3)]],
    device float* dfdy [[buffer(4)]],
    device const int* dims [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    dfdx[idx] = ddx(f, dx, j, i, ny, nx);
    dfdy[idx] = ddy(f, dy, j, i, ny, nx);
}
"""
_gradient_compiled = None


def gradient(f, dx, dy):
    """Horizontal gradient  returns (df/dx, df/dy)."""
    global _gradient_compiled
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    dfdx = MetalArray(shape=(ny, nx_))
    dfdy = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _gradient_compiled is None:
        _gradient_compiled = dev.compile(_gradient_source, "gradient_kernel")
    _gradient_compiled.dispatch(
        [f, dx_m, dy_m, dfdx, dfdy, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return dfdx, dfdy


# ------------------------------------------------------------------
# 15. advection  -u*df/dx - v*df/dy
# ------------------------------------------------------------------
_advection_source = _deriv_metal_funcs + """
kernel void advection_kernel(
    device const float* field [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* dx [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const int* dims [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dfdx = ddx(field, dx, j, i, ny, nx);
    float dfdy = ddy(field, dy, j, i, ny, nx);
    out[idx] = -(u[idx] * dfdx + v[idx] * dfdy);
}
"""
_advection_compiled = None


def advection(field, u, v, dx, dy):
    """Horizontal advection  -u*df/dx - v*df/dy."""
    global _advection_compiled
    field, u, v = _to_gpu(field), _to_gpu(u), _to_gpu(v)
    ny, nx_ = field.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _advection_compiled is None:
        _advection_compiled = dev.compile(_advection_source, "advection_kernel")
    _advection_compiled.dispatch(
        [field, u, v, dx_m, dy_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 16. frontogenesis  (Petterssen)
# ------------------------------------------------------------------
_frontogenesis_source = _deriv_metal_funcs + """
kernel void frontogenesis_kernel(
    device const float* theta [[buffer(0)]],
    device const float* u [[buffer(1)]],
    device const float* v [[buffer(2)]],
    device const float* dx [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* out [[buffer(5)]],
    device const int* dims [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    float dtdx = ddx(theta, dx, j, i, ny, nx);
    float dtdy = ddy(theta, dy, j, i, ny, nx);
    float mag = sqrt(dtdx * dtdx + dtdy * dtdy);
    if (mag < 1e-20f) { out[idx] = 0.0f; return; }

    float dudx_v = ddx(u, dx, j, i, ny, nx);
    float dudy_v = ddy(u, dy, j, i, ny, nx);
    float dvdx_v = ddx(v, dx, j, i, ny, nx);
    float dvdy_v = ddy(v, dy, j, i, ny, nx);

    float F = (dtdx * dtdx * dudx_v
             + dtdy * dtdy * dvdy_v
             + dtdx * dtdy * (dvdx_v + dudy_v));
    out[idx] = -F / mag;
}
"""
_frontogenesis_compiled = None


def frontogenesis(theta, u, v, dx, dy):
    """Petterssen frontogenesis function (scalar, 2D)."""
    global _frontogenesis_compiled
    theta, u, v = _to_gpu(theta), _to_gpu(u), _to_gpu(v)
    ny, nx_ = theta.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _frontogenesis_compiled is None:
        _frontogenesis_compiled = dev.compile(_frontogenesis_source, "frontogenesis_kernel")
    _frontogenesis_compiled.dispatch(
        [theta, u, v, dx_m, dy_m, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 17. q_vector  Q1, Q2
# ------------------------------------------------------------------
_q_vector_source = _deriv_metal_funcs + """
kernel void q_vector_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* temperature [[buffer(2)]],
    device const float* dx [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device const float* params [[buffer(5)]],
    device float* q1_out [[buffer(6)]],
    device float* q2_out [[buffer(7)]],
    device const int* dims [[buffer(8)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    float pressure = params[0];
    float Rd = params[1];

    float dTdx = ddx(temperature, dx, j, i, ny, nx);
    float dTdy = ddy(temperature, dy, j, i, ny, nx);

    float dudx_v = ddx(u, dx, j, i, ny, nx);
    float dudy_v = ddy(u, dy, j, i, ny, nx);
    float dvdx_v = ddx(v, dx, j, i, ny, nx);
    float dvdy_v = ddy(v, dy, j, i, ny, nx);

    float coeff = -Rd / pressure;
    q1_out[idx] = coeff * (dudx_v * dTdx + dvdx_v * dTdy);
    q2_out[idx] = coeff * (dudy_v * dTdx + dvdy_v * dTdy);
}
"""
_q_vector_compiled = None


def q_vector(u, v, temperature, dx, dy, pressure, Rd=287.04):
    """Q-vector components (Q1, Q2).  *pressure* is a scalar in Pa."""
    global _q_vector_compiled
    u, v, temperature = _to_gpu(u), _to_gpu(v), _to_gpu(temperature)
    ny, nx_ = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    q1 = MetalArray(shape=(ny, nx_))
    q2 = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _q_vector_compiled is None:
        _q_vector_compiled = dev.compile(_q_vector_source, "q_vector_kernel")
    params = struct.pack("ff", float(pressure), float(Rd))
    _q_vector_compiled.dispatch(
        [u, v, temperature, dx_m, dy_m, params, q1, q2, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return q1, q2


# ------------------------------------------------------------------
# 18. geostrophic_wind  ug = -(g/f)*dZ/dy,  vg = (g/f)*dZ/dx
# ------------------------------------------------------------------
_geostrophic_wind_source = _deriv_metal_funcs + """
kernel void geostrophic_wind_kernel(
    device const float* Z [[buffer(0)]],
    device const float* f [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device const float* params [[buffer(4)]],
    device float* ug [[buffer(5)]],
    device float* vg [[buffer(6)]],
    device const int* dims [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float grav = params[0];
    float fc = f[idx];
    if (abs(fc) < 1e-20f) { ug[idx] = 0.0f; vg[idx] = 0.0f; return; }
    float dZdx = ddx(Z, dx, j, i, ny, nx);
    float dZdy = ddy(Z, dy, j, i, ny, nx);
    ug[idx] = -(grav / fc) * dZdy;
    vg[idx] =  (grav / fc) * dZdx;
}
"""
_geostrophic_wind_compiled = None


def geostrophic_wind(Z, f, dx, dy, g=9.80665):
    """Geostrophic wind from geopotential height.  Returns (ug, vg)."""
    global _geostrophic_wind_compiled
    Z, f = _to_gpu(Z), _to_gpu(f)
    ny, nx_ = Z.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    ug = MetalArray(shape=(ny, nx_))
    vg = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _geostrophic_wind_compiled is None:
        _geostrophic_wind_compiled = dev.compile(_geostrophic_wind_source, "geostrophic_wind_kernel")
    params = struct.pack("f", float(g))
    _geostrophic_wind_compiled.dispatch(
        [Z, f, dx_m, dy_m, params, ug, vg, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return ug, vg


# ------------------------------------------------------------------
# 19. ageostrophic_wind  ua = u - ug,  va = v - vg
# ------------------------------------------------------------------
_ageostrophic_wind_source = _deriv_metal_funcs + """
kernel void ageostrophic_wind_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* Z [[buffer(2)]],
    device const float* f [[buffer(3)]],
    device const float* dx [[buffer(4)]],
    device const float* dy [[buffer(5)]],
    device const float* params [[buffer(6)]],
    device float* ua [[buffer(7)]],
    device float* va [[buffer(8)]],
    device const int* dims [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float grav = params[0];
    float fc = f[idx];
    if (abs(fc) < 1e-20f) { ua[idx] = u[idx]; va[idx] = v[idx]; return; }
    float dZdx = ddx(Z, dx, j, i, ny, nx);
    float dZdy = ddy(Z, dy, j, i, ny, nx);
    float ugc = -(grav / fc) * dZdy;
    float vgc =  (grav / fc) * dZdx;
    ua[idx] = u[idx] - ugc;
    va[idx] = v[idx] - vgc;
}
"""
_ageostrophic_wind_compiled = None


def ageostrophic_wind(u, v, Z, f, dx, dy, g=9.80665):
    """Ageostrophic wind.  Returns (ua, va) = (u - ug, v - vg)."""
    global _ageostrophic_wind_compiled
    u, v, Z, f = _to_gpu(u), _to_gpu(v), _to_gpu(Z), _to_gpu(f)
    ny, nx_ = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    ua = MetalArray(shape=(ny, nx_))
    va = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _ageostrophic_wind_compiled is None:
        _ageostrophic_wind_compiled = dev.compile(_ageostrophic_wind_source, "ageostrophic_wind_kernel")
    params = struct.pack("f", float(g))
    _ageostrophic_wind_compiled.dispatch(
        [u, v, Z, f, dx_m, dy_m, params, ua, va, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return ua, va


# ------------------------------------------------------------------
# 20. potential_vorticity_baroclinic
# ------------------------------------------------------------------
_pv_baroclinic_source = _deriv_metal_funcs + """
kernel void potential_vorticity_baroclinic_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* theta [[buffer(2)]],
    device const float* pressure [[buffer(3)]],
    device const float* dx [[buffer(4)]],
    device const float* dy [[buffer(5)]],
    device const float* f [[buffer(6)]],
    device const float* params [[buffer(7)]],
    device float* out [[buffer(8)]],
    device const int* dims [[buffer(9)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = dims[0];
    int nx = dims[1];
    int nz = dims[2];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    float grav = params[0];
    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int k = 1; k < nz - 1; k++) {
        int idx3d = k * nxy + idx2d;
        float dvdx_v = ddx(&v[k * nxy], dx, j, i, ny, nx);
        float dudy_v = ddy(&u[k * nxy], dy, j, i, ny, nx);
        float zeta = dvdx_v - dudy_v;

        float dp = pressure[(k + 1) * nxy + idx2d] - pressure[(k - 1) * nxy + idx2d];
        float dtheta = theta[(k + 1) * nxy + idx2d] - theta[(k - 1) * nxy + idx2d];
        float dthetadp = (abs(dp) > 1e-10f) ? dtheta / dp : 0.0f;

        out[idx3d] = -grav * (f[idx2d] + zeta) * dthetadp;
    }
}
"""
_pv_baroclinic_compiled = None


def potential_vorticity_baroclinic(u, v, theta, pressure, dx, dy, f, g=9.80665):
    """Baroclinic (Ertel) potential vorticity on 3D (nz, ny, nx) arrays.

    Parameters
    ----------
    u, v : (nz, ny, nx) wind components
    theta : (nz, ny, nx) potential temperature
    pressure : (nz, ny, nx) pressure in Pa
    dx, dy : (ny, nx) grid spacing in meters
    f : (ny, nx) Coriolis parameter

    Returns
    -------
    PV : (nz, ny, nx)
    """
    global _pv_baroclinic_compiled
    u, v, theta, pressure, f = _to_gpu(u), _to_gpu(v), _to_gpu(theta), _to_gpu(pressure), _to_gpu(f)
    nz, ny, nx_ = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(nz, ny, nx_))
    dev = metal_device()
    if _pv_baroclinic_compiled is None:
        _pv_baroclinic_compiled = dev.compile(_pv_baroclinic_source, "potential_vorticity_baroclinic_kernel")
    params = struct.pack("f", float(g))
    _pv_baroclinic_compiled.dispatch(
        [u, v, theta, pressure, dx_m, dy_m, f, params, out, _dims3_buf(ny, nx_, nz)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 21. potential_vorticity_barotropic
# ------------------------------------------------------------------
_pv_barotropic_source = _deriv_metal_funcs + """
kernel void potential_vorticity_barotropic_kernel(
    device const float* u [[buffer(0)]],
    device const float* v [[buffer(1)]],
    device const float* dx [[buffer(2)]],
    device const float* dy [[buffer(3)]],
    device const float* f [[buffer(4)]],
    device const float* depth [[buffer(5)]],
    device float* out [[buffer(6)]],
    device const int* dims [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float dvdx_v = ddx(v, dx, j, i, ny, nx);
    float dudy_v = ddy(u, dy, j, i, ny, nx);
    float zeta = dvdx_v - dudy_v;
    float h = depth[idx];
    out[idx] = (abs(h) > 1e-10f) ? (f[idx] + zeta) / h : 0.0f;
}
"""
_pv_barotropic_compiled = None


def potential_vorticity_barotropic(u, v, dx, dy, f, depth):
    """Barotropic potential vorticity  (f + zeta) / depth."""
    global _pv_barotropic_compiled
    u, v, f, depth = _to_gpu(u), _to_gpu(v), _to_gpu(f), _to_gpu(depth)
    ny, nx_ = u.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _pv_barotropic_compiled is None:
        _pv_barotropic_compiled = dev.compile(_pv_barotropic_source, "potential_vorticity_barotropic_kernel")
    _pv_barotropic_compiled.dispatch(
        [u, v, dx_m, dy_m, f, depth, out, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 22. inertial_advective_wind
# ------------------------------------------------------------------
_inertial_advective_wind_source = _deriv_metal_funcs + """
kernel void inertial_advective_wind_kernel(
    device const float* ug [[buffer(0)]],
    device const float* vg [[buffer(1)]],
    device const float* f [[buffer(2)]],
    device const float* dx [[buffer(3)]],
    device const float* dy [[buffer(4)]],
    device float* u_ia [[buffer(5)]],
    device float* v_ia [[buffer(6)]],
    device const int* dims [[buffer(7)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nx = dims[1];
    int ny = dims[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    float fc = f[idx];
    if (abs(fc) < 1e-20f) { u_ia[idx] = ug[idx]; v_ia[idx] = vg[idx]; return; }

    float dugdx = ddx(ug, dx, j, i, ny, nx);
    float dugdy = ddy(ug, dy, j, i, ny, nx);
    float dvgdx = ddx(vg, dx, j, i, ny, nx);
    float dvgdy = ddy(vg, dy, j, i, ny, nx);

    float ugc = ug[idx], vgc = vg[idx];
    u_ia[idx] = ugc + (1.0f / fc) * (ugc * dugdx + vgc * dugdy);
    v_ia[idx] = vgc + (1.0f / fc) * (ugc * dvgdx + vgc * dvgdy);
}
"""
_inertial_advective_wind_compiled = None


def inertial_advective_wind(ug, vg, f, dx, dy):
    """Inertial-advective wind from geostrophic wind.  Returns (u_ia, v_ia)."""
    global _inertial_advective_wind_compiled
    ug, vg, f = _to_gpu(ug), _to_gpu(vg), _to_gpu(f)
    ny, nx_ = ug.shape
    dx_m, dy_m = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    u_ia = MetalArray(shape=(ny, nx_))
    v_ia = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _inertial_advective_wind_compiled is None:
        _inertial_advective_wind_compiled = dev.compile(_inertial_advective_wind_source, "inertial_advective_wind_kernel")
    _inertial_advective_wind_compiled.dispatch(
        [ug, vg, f, dx_m, dy_m, u_ia, v_ia, _dims_buf(ny, nx_)],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return u_ia, v_ia


# ===================================================================
# 2. Smoothing kernels
# ===================================================================

# ------------------------------------------------------------------
# 23. smooth_gaussian  --- Gaussian filter
# ------------------------------------------------------------------
_smooth_gaussian_source = """
#include <metal_stdlib>
using namespace metal;

kernel void smooth_gaussian_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* params [[buffer(2)]],
    device const float* fparams [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = params[0];
    int nx = params[1];
    int radius = params[2];
    float sigma = fparams[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    float sum = 0.0f, wsum = 0.0f;
    float inv2s2 = 1.0f / (2.0f * sigma * sigma);
    for (int dj = -radius; dj <= radius; dj++) {
        for (int di = -radius; di <= radius; di++) {
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            float r2 = float(di * di + dj * dj);
            float w = exp(-r2 * inv2s2);
            sum += w * input[jj * nx + ii];
            wsum += w;
        }
    }
    output[j * nx + i] = sum / wsum;
}
"""
_smooth_gaussian_compiled = None


def smooth_gaussian(field, sigma=1.0, radius=None):
    """Gaussian smoothing filter.

    Parameters
    ----------
    field : (ny, nx) array
    sigma : float, Gaussian sigma in grid units (default 1.0)
    radius : int, filter half-width (default: ceil(3*sigma))
    """
    global _smooth_gaussian_compiled
    field = _to_gpu(field)
    ny, nx_ = field.shape
    if radius is None:
        radius = int(math.ceil(3.0 * sigma))
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _smooth_gaussian_compiled is None:
        _smooth_gaussian_compiled = dev.compile(_smooth_gaussian_source, "smooth_gaussian_kernel")
    iparams = struct.pack("iii", ny, nx_, radius)
    fparams = struct.pack("f", float(sigma))
    _smooth_gaussian_compiled.dispatch(
        [field, out, iparams, fparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 24. smooth_n_point  --- 5-point or 9-point smoother
# ------------------------------------------------------------------
_smooth_n_point_source = """
#include <metal_stdlib>
using namespace metal;

kernel void smooth_n_point_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = params[0];
    int nx = params[1];
    int n_point = params[2];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1) return;
    int idx = j * nx + i;

    if (n_point == 5) {
        output[idx] = (4.0f * input[idx]
                       + input[idx - 1] + input[idx + 1]
                       + input[idx - nx] + input[idx + nx]) / 8.0f;
    } else {
        output[idx] = (4.0f * input[idx]
                       + input[idx - 1] + input[idx + 1]
                       + input[idx - nx] + input[idx + nx]
                       + 0.5f * (input[idx - nx - 1] + input[idx - nx + 1]
                              + input[idx + nx - 1] + input[idx + nx + 1])) / 10.0f;
    }
}
"""
_smooth_n_point_compiled = None


def smooth_n_point(field, n_point=5, passes=1):
    """N-point smoother (5-point or 9-point).

    Parameters
    ----------
    field : (ny, nx) array
    n_point : int, 5 or 9 (default 5)
    passes : int, number of smoothing passes (default 1)
    """
    global _smooth_n_point_compiled
    field = _to_gpu(field)
    ny, nx_ = field.shape
    dev = metal_device()
    if _smooth_n_point_compiled is None:
        _smooth_n_point_compiled = dev.compile(_smooth_n_point_source, "smooth_n_point_kernel")
    iparams = struct.pack("iii", ny, nx_, n_point)
    src = field.copy()
    dst = MetalArray(shape=(ny, nx_))
    for _ in range(passes):
        _smooth_n_point_compiled.dispatch(
            [src, dst, iparams],
            grid_size=(nx_, ny),
            threadgroup_size=_tg(nx_, ny),
        )
        src, dst = dst, src
    return src


# ------------------------------------------------------------------
# 25. smooth_rectangular  --- box filter
# ------------------------------------------------------------------
_smooth_rectangular_source = """
#include <metal_stdlib>
using namespace metal;

kernel void smooth_rectangular_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = params[0];
    int nx = params[1];
    int radius_x = params[2];
    int radius_y = params[3];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    float sum = 0.0f;
    int count = 0;
    for (int dj = -radius_y; dj <= radius_y; dj++) {
        for (int di = -radius_x; di <= radius_x; di++) {
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            sum += input[jj * nx + ii];
            count++;
        }
    }
    output[j * nx + i] = sum / float(count);
}
"""
_smooth_rectangular_compiled = None


def smooth_rectangular(field, radius_x=1, radius_y=1):
    """Rectangular (box) filter.

    Parameters
    ----------
    field : (ny, nx)
    radius_x, radius_y : int, half-widths in x and y
    """
    global _smooth_rectangular_compiled
    field = _to_gpu(field)
    ny, nx_ = field.shape
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _smooth_rectangular_compiled is None:
        _smooth_rectangular_compiled = dev.compile(_smooth_rectangular_source, "smooth_rectangular_kernel")
    iparams = struct.pack("iiii", ny, nx_, radius_x, radius_y)
    _smooth_rectangular_compiled.dispatch(
        [field, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 26. smooth_circular  --- circular window filter
# ------------------------------------------------------------------
_smooth_circular_source = """
#include <metal_stdlib>
using namespace metal;

kernel void smooth_circular_kernel(
    device const float* input [[buffer(0)]],
    device float* output [[buffer(1)]],
    device const int* params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = params[0];
    int nx = params[1];
    int radius = params[2];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    float sum = 0.0f;
    int count = 0;
    float r2max = float(radius * radius);
    for (int dj = -radius; dj <= radius; dj++) {
        for (int di = -radius; di <= radius; di++) {
            if (float(di * di + dj * dj) > r2max) continue;
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            sum += input[jj * nx + ii];
            count++;
        }
    }
    output[j * nx + i] = sum / float(count);
}
"""
_smooth_circular_compiled = None


def smooth_circular(field, radius=2):
    """Circular window filter.

    Parameters
    ----------
    field : (ny, nx)
    radius : int, circle radius in grid points
    """
    global _smooth_circular_compiled
    field = _to_gpu(field)
    ny, nx_ = field.shape
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _smooth_circular_compiled is None:
        _smooth_circular_compiled = dev.compile(_smooth_circular_source, "smooth_circular_kernel")
    iparams = struct.pack("iii", ny, nx_, radius)
    _smooth_circular_compiled.dispatch(
        [field, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 27. smooth_window  --- generic window function (weights passed in)
# ------------------------------------------------------------------
_smooth_window_source = """
#include <metal_stdlib>
using namespace metal;

kernel void smooth_window_kernel(
    device const float* input [[buffer(0)]],
    device const float* weights [[buffer(1)]],
    device float* output [[buffer(2)]],
    device const int* params [[buffer(3)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = params[0];
    int nx = params[1];
    int win_h = params[2];
    int win_w = params[3];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int rh = win_h / 2, rw = win_w / 2;
    float sum = 0.0f, wsum = 0.0f;
    for (int dj = -rh; dj <= rh; dj++) {
        for (int di = -rw; di <= rw; di++) {
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            float w = weights[(dj + rh) * win_w + (di + rw)];
            sum += w * input[jj * nx + ii];
            wsum += w;
        }
    }
    output[j * nx + i] = (wsum > 0.0f) ? sum / wsum : input[j * nx + i];
}
"""
_smooth_window_compiled = None


def smooth_window(field, weights):
    """Smooth with an arbitrary 2D weight array.

    Parameters
    ----------
    field : (ny, nx)
    weights : (win_h, win_w)  weight kernel (need not be normalized)
    """
    global _smooth_window_compiled
    field = _to_gpu(field)
    weights_m = _to_gpu(weights)
    ny, nx_ = field.shape
    win_h, win_w = weights_m.shape
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _smooth_window_compiled is None:
        _smooth_window_compiled = dev.compile(_smooth_window_source, "smooth_window_kernel")
    iparams = struct.pack("iiii", ny, nx_, win_h, win_w)
    _smooth_window_compiled.dispatch(
        [field, weights_m, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ===================================================================
# 3. Interpolation kernels
# ===================================================================

# ------------------------------------------------------------------
# 28. interpolate_1d  --- per-column 1D linear interpolation
# ------------------------------------------------------------------
_interpolate_1d_source = """
#include <metal_stdlib>
using namespace metal;

kernel void interpolate_1d_kernel(
    device const float* field [[buffer(0)]],
    device const float* levels_in [[buffer(1)]],
    device const float* levels_out [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz_in = params[0];
    int nz_out = params[1];
    int ny = params[2];
    int nx = params[3];
    int ascending = params[4];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int ko = 0; ko < nz_out; ko++) {
        float target = levels_out[ko];
        int found = 0;
        for (int k = 0; k < nz_in - 1; k++) {
            float lo = levels_in[k * nxy + idx2d];
            float hi = levels_in[(k + 1) * nxy + idx2d];
            int bracket;
            if (ascending)
                bracket = (lo <= target && target <= hi) || (hi <= target && target <= lo);
            else
                bracket = (lo >= target && target >= hi) || (hi >= target && target >= lo);
            if (bracket) {
                float denom = hi - lo;
                float frac = (abs(denom) > 1e-30f) ? (target - lo) / denom : 0.0f;
                float f0 = field[k * nxy + idx2d];
                float f1 = field[(k + 1) * nxy + idx2d];
                out[ko * nxy + idx2d] = f0 + frac * (f1 - f0);
                found = 1;
                break;
            }
        }
        if (!found) {
            out[ko * nxy + idx2d] = NAN;
        }
    }
}
"""
_interpolate_1d_compiled = None


def interpolate_1d(field, levels_in, levels_out, ascending=True):
    """Linear interpolation per column to new vertical levels.

    Parameters
    ----------
    field : (nz_in, ny, nx)
    levels_in : (nz_in, ny, nx)  coordinate values at input levels
    levels_out : scalar list/array of target levels -> broadcast to (nz_out,)
    ascending : bool, whether coordinate increases with index

    Returns
    -------
    (nz_out, ny, nx) interpolated field
    """
    global _interpolate_1d_compiled
    field, levels_in = _to_gpu(field), _to_gpu(levels_in)
    nz_in, ny, nx_ = field.shape
    lo_1d = np.asarray(levels_out, dtype=np.float32).ravel()
    nz_out = lo_1d.size
    lo_m = MetalArray(data=lo_1d)
    out = MetalArray(shape=(nz_out, ny, nx_))
    dev = metal_device()
    if _interpolate_1d_compiled is None:
        _interpolate_1d_compiled = dev.compile(_interpolate_1d_source, "interpolate_1d_kernel")
    iparams = struct.pack("iiiii", nz_in, nz_out, ny, nx_, 1 if ascending else 0)
    _interpolate_1d_compiled.dispatch(
        [field, levels_in, lo_m, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 29. log_interpolate_1d  --- log-pressure interpolation per column
# ------------------------------------------------------------------
_log_interpolate_1d_source = """
#include <metal_stdlib>
using namespace metal;

kernel void log_interpolate_1d_kernel(
    device const float* field [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device const float* p_target [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz_in = params[0];
    int nz_out = params[1];
    int ny = params[2];
    int nx = params[3];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int ko = 0; ko < nz_out; ko++) {
        float lnpt = log(p_target[ko]);
        int found = 0;
        for (int k = 0; k < nz_in - 1; k++) {
            float lnp0 = log(pressure[k * nxy + idx2d]);
            float lnp1 = log(pressure[(k + 1) * nxy + idx2d]);
            if ((lnp0 >= lnpt && lnpt >= lnp1) || (lnp1 >= lnpt && lnpt >= lnp0)) {
                float denom = lnp1 - lnp0;
                float frac = (abs(denom) > 1e-30f) ? (lnpt - lnp0) / denom : 0.0f;
                float f0 = field[k * nxy + idx2d];
                float f1 = field[(k + 1) * nxy + idx2d];
                out[ko * nxy + idx2d] = f0 + frac * (f1 - f0);
                found = 1;
                break;
            }
        }
        if (!found) out[ko * nxy + idx2d] = NAN;
    }
}
"""
_log_interpolate_1d_compiled = None


def log_interpolate_1d(field, pressure, p_target):
    """Log-pressure interpolation per column.

    Parameters
    ----------
    field : (nz_in, ny, nx)
    pressure : (nz_in, ny, nx)  pressure in Pa
    p_target : 1D array of target pressures in Pa

    Returns
    -------
    (nz_out, ny, nx) interpolated field
    """
    global _log_interpolate_1d_compiled
    field, pressure = _to_gpu(field), _to_gpu(pressure)
    nz_in, ny, nx_ = field.shape
    pt_1d = np.asarray(p_target, dtype=np.float32).ravel()
    nz_out = pt_1d.size
    pt_m = MetalArray(data=pt_1d)
    out = MetalArray(shape=(nz_out, ny, nx_))
    dev = metal_device()
    if _log_interpolate_1d_compiled is None:
        _log_interpolate_1d_compiled = dev.compile(_log_interpolate_1d_source, "log_interpolate_1d_kernel")
    iparams = struct.pack("iiii", nz_in, nz_out, ny, nx_)
    _log_interpolate_1d_compiled.dispatch(
        [field, pressure, pt_m, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 30. interpolate_to_isosurface
# ------------------------------------------------------------------
_interpolate_to_isosurface_source = """
#include <metal_stdlib>
using namespace metal;

kernel void interpolate_to_isosurface_kernel(
    device const float* field [[buffer(0)]],
    device const float* coord [[buffer(1)]],
    device const float* params_f [[buffer(2)]],
    device float* out [[buffer(3)]],
    device const int* params [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz = params[0];
    int ny = params[1];
    int nx = params[2];
    float target_value = params_f[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;
    float result = NAN;

    for (int k = 0; k < nz - 1; k++) {
        float f0 = field[k * nxy + idx2d];
        float f1 = field[(k + 1) * nxy + idx2d];
        if ((f0 <= target_value && target_value <= f1) ||
            (f1 <= target_value && target_value <= f0)) {
            float denom = f1 - f0;
            float frac = (abs(denom) > 1e-30f) ? (target_value - f0) / denom : 0.0f;
            float c0 = coord[k * nxy + idx2d];
            float c1 = coord[(k + 1) * nxy + idx2d];
            result = c0 + frac * (c1 - c0);
            break;
        }
    }
    out[idx2d] = result;
}
"""
_interpolate_to_isosurface_compiled = None


def interpolate_to_isosurface(field, coord, target_value):
    """Find the coordinate value where *field* equals *target_value* per column.

    Parameters
    ----------
    field : (nz, ny, nx)  the field to search (e.g., temperature)
    coord : (nz, ny, nx)  the coordinate to interpolate (e.g., pressure)
    target_value : float

    Returns
    -------
    (ny, nx) interpolated coordinate values
    """
    global _interpolate_to_isosurface_compiled
    field, coord = _to_gpu(field), _to_gpu(coord)
    nz, ny, nx_ = field.shape
    # Initialize output with NaN
    out = MetalArray(data=np.full((ny, nx_), np.nan, dtype=np.float32))
    dev = metal_device()
    if _interpolate_to_isosurface_compiled is None:
        _interpolate_to_isosurface_compiled = dev.compile(_interpolate_to_isosurface_source, "interpolate_to_isosurface_kernel")
    params_f = struct.pack("f", float(target_value))
    iparams = struct.pack("iii", nz, ny, nx_)
    _interpolate_to_isosurface_compiled.dispatch(
        [field, coord, params_f, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 31. isentropic_interpolation
# ------------------------------------------------------------------
_isentropic_interpolation_source = """
#include <metal_stdlib>
using namespace metal;

kernel void isentropic_interpolation_kernel(
    device const float* theta [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device const float* field [[buffer(2)]],
    device const float* theta_targets [[buffer(3)]],
    device float* out [[buffer(4)]],
    device const int* params [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz = params[0];
    int n_theta = params[1];
    int ny = params[2];
    int nx = params[3];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int kt = 0; kt < n_theta; kt++) {
        float target = theta_targets[kt];
        int found = 0;
        for (int k = 0; k < nz - 1; k++) {
            float t0 = theta[k * nxy + idx2d];
            float t1 = theta[(k + 1) * nxy + idx2d];
            if ((t0 <= target && target <= t1) || (t1 <= target && target <= t0)) {
                float denom = t1 - t0;
                float frac = (abs(denom) > 1e-10f) ? (target - t0) / denom : 0.0f;
                float f0 = field[k * nxy + idx2d];
                float f1 = field[(k + 1) * nxy + idx2d];
                out[kt * nxy + idx2d] = f0 + frac * (f1 - f0);
                found = 1;
                break;
            }
        }
        if (!found) out[kt * nxy + idx2d] = NAN;
    }
}
"""
_isentropic_interpolation_compiled = None


def isentropic_interpolation(theta, pressure, field, theta_targets):
    """Interpolate a field to isentropic (theta) surfaces.

    Parameters
    ----------
    theta : (nz, ny, nx)  potential temperature
    pressure : (nz, ny, nx)  pressure (unused in interpolation itself,
               kept for API consistency -- pass field=pressure to get
               pressure on theta surfaces)
    field : (nz, ny, nx)  field to interpolate
    theta_targets : 1D array of target theta values (K)

    Returns
    -------
    (n_theta, ny, nx) interpolated field
    """
    global _isentropic_interpolation_compiled
    theta, field = _to_gpu(theta), _to_gpu(field)
    nz, ny, nx_ = theta.shape
    tt_1d = np.asarray(theta_targets, dtype=np.float32).ravel()
    n_theta = tt_1d.size
    tt_m = MetalArray(data=tt_1d)
    out = MetalArray(shape=(n_theta, ny, nx_))
    dev = metal_device()
    if _isentropic_interpolation_compiled is None:
        _isentropic_interpolation_compiled = dev.compile(_isentropic_interpolation_source, "isentropic_interpolation_kernel")
    iparams = struct.pack("iiii", nz, n_theta, ny, nx_)
    _isentropic_interpolation_compiled.dispatch(
        [theta, _to_gpu(pressure), field, tt_m, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 32. lat_lon_grid_deltas  --- compute dx, dy from lat/lon via haversine
# ------------------------------------------------------------------
_lat_lon_grid_deltas_source = """
#include <metal_stdlib>
using namespace metal;

kernel void lat_lon_grid_deltas_kernel(
    device const float* lat [[buffer(0)]],
    device const float* lon [[buffer(1)]],
    device float* dx [[buffer(2)]],
    device float* dy [[buffer(3)]],
    device const int* params_i [[buffer(4)]],
    device const float* params_f [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int ny = params_i[0];
    int nx = params_i[1];
    float earth_radius = params_f[0];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    float deg2rad = M_PI_F / 180.0f;

    // dx: distance between (j, i) and (j, i+1) [or i-1 at boundary]
    if (i < nx - 1) {
        float lat1 = lat[idx] * deg2rad;
        float lat2 = lat[idx + 1] * deg2rad;
        float dlon = (lon[idx + 1] - lon[idx]) * deg2rad;
        float dlat = lat2 - lat1;
        float a = sin(dlat / 2.0f) * sin(dlat / 2.0f)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0f) * sin(dlon / 2.0f);
        float c = 2.0f * atan2(sqrt(a), sqrt(1.0f - a));
        dx[idx] = earth_radius * c;
    } else {
        int prev = j * nx + (i - 1);
        float lat1 = lat[prev] * deg2rad;
        float lat2 = lat[idx] * deg2rad;
        float dlon = (lon[idx] - lon[prev]) * deg2rad;
        float dlat = lat2 - lat1;
        float a = sin(dlat / 2.0f) * sin(dlat / 2.0f)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0f) * sin(dlon / 2.0f);
        float c = 2.0f * atan2(sqrt(a), sqrt(1.0f - a));
        dx[idx] = earth_radius * c;
    }

    // dy: distance between (j, i) and (j+1, i) [or j-1 at boundary]
    if (j < ny - 1) {
        float lat1 = lat[idx] * deg2rad;
        float lat2 = lat[idx + nx] * deg2rad;
        float dlon = (lon[idx + nx] - lon[idx]) * deg2rad;
        float dlat = lat2 - lat1;
        float a = sin(dlat / 2.0f) * sin(dlat / 2.0f)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0f) * sin(dlon / 2.0f);
        float c = 2.0f * atan2(sqrt(a), sqrt(1.0f - a));
        dy[idx] = earth_radius * c;
    } else {
        int prev = (j - 1) * nx + i;
        float lat1 = lat[prev] * deg2rad;
        float lat2 = lat[idx] * deg2rad;
        float dlon = (lon[idx] - lon[prev]) * deg2rad;
        float dlat = lat2 - lat1;
        float a = sin(dlat / 2.0f) * sin(dlat / 2.0f)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0f) * sin(dlon / 2.0f);
        float c = 2.0f * atan2(sqrt(a), sqrt(1.0f - a));
        dy[idx] = earth_radius * c;
    }
}
"""
_lat_lon_grid_deltas_compiled = None


def lat_lon_grid_deltas(lat, lon, earth_radius=6371229.0):
    """Compute dx, dy grid spacings from lat/lon arrays using haversine.

    Parameters
    ----------
    lat, lon : (ny, nx) arrays in degrees
    earth_radius : float, meters (default 6371229.0)

    Returns
    -------
    dx, dy : (ny, nx) grid spacings in meters
    """
    global _lat_lon_grid_deltas_compiled
    lat, lon = _to_gpu(lat), _to_gpu(lon)
    ny, nx_ = lat.shape
    dx_out = MetalArray(shape=(ny, nx_))
    dy_out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _lat_lon_grid_deltas_compiled is None:
        _lat_lon_grid_deltas_compiled = dev.compile(_lat_lon_grid_deltas_source, "lat_lon_grid_deltas_kernel")
    iparams = struct.pack("ii", ny, nx_)
    fparams = struct.pack("f", float(earth_radius))
    _lat_lon_grid_deltas_compiled.dispatch(
        [lat, lon, dx_out, dy_out, iparams, fparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return dx_out, dy_out


# ===================================================================
# 4. Grid utility kernels
# ===================================================================

# ------------------------------------------------------------------
# 33. composite_reflectivity  --- max value along vertical axis per column
# ------------------------------------------------------------------
_composite_reflectivity_source = """
#include <metal_stdlib>
using namespace metal;

kernel void composite_reflectivity_kernel(
    device const float* field [[buffer(0)]],
    device float* out [[buffer(1)]],
    device const int* params [[buffer(2)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz = params[0];
    int ny = params[1];
    int nx = params[2];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;
    float maxval = -1e38f;
    for (int k = 0; k < nz; k++) {
        float val = field[k * nxy + idx2d];
        if (val > maxval) maxval = val;
    }
    out[idx2d] = maxval;
}
"""
_composite_reflectivity_compiled = None


def composite_reflectivity(field):
    """Maximum value along the vertical axis (axis=0) per column.

    Parameters
    ----------
    field : (nz, ny, nx)

    Returns
    -------
    (ny, nx) column-max values
    """
    global _composite_reflectivity_compiled
    field = _to_gpu(field)
    nz, ny, nx_ = field.shape
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _composite_reflectivity_compiled is None:
        _composite_reflectivity_compiled = dev.compile(_composite_reflectivity_source, "composite_reflectivity_kernel")
    iparams = struct.pack("iii", nz, ny, nx_)
    _composite_reflectivity_compiled.dispatch(
        [field, out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 34. mean_pressure_weighted  --- pressure-weighted mean in a layer
# ------------------------------------------------------------------
_mean_pressure_weighted_source = """
#include <metal_stdlib>
using namespace metal;

kernel void mean_pressure_weighted_kernel(
    device const float* field [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device float* out [[buffer(2)]],
    device const int* params_i [[buffer(3)]],
    device const float* params_f [[buffer(4)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz = params_i[0];
    int ny = params_i[1];
    int nx = params_i[2];
    float p_bottom = params_f[0];
    float p_top = params_f[1];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;
    float sum = 0.0f, wsum = 0.0f;

    for (int k = 0; k < nz; k++) {
        float p = pressure[k * nxy + idx2d];
        if (p <= p_bottom && p >= p_top) {
            float dp;
            if (k == 0)
                dp = abs(pressure[idx2d] - pressure[nxy + idx2d]) * 0.5f;
            else if (k == nz - 1)
                dp = abs(pressure[(nz - 2) * nxy + idx2d] - pressure[(nz - 1) * nxy + idx2d]) * 0.5f;
            else
                dp = abs(pressure[(k - 1) * nxy + idx2d] - pressure[(k + 1) * nxy + idx2d]) * 0.5f;
            sum += field[k * nxy + idx2d] * dp;
            wsum += dp;
        }
    }
    out[idx2d] = (wsum > 0.0f) ? sum / wsum : NAN;
}
"""
_mean_pressure_weighted_compiled = None


def mean_pressure_weighted(field, pressure, p_bottom, p_top):
    """Pressure-weighted mean of a field within a layer.

    Parameters
    ----------
    field : (nz, ny, nx)
    pressure : (nz, ny, nx) in Pa
    p_bottom, p_top : float, layer bounds in Pa  (p_bottom > p_top)

    Returns
    -------
    (ny, nx) pressure-weighted mean
    """
    global _mean_pressure_weighted_compiled
    field, pressure = _to_gpu(field), _to_gpu(pressure)
    nz, ny, nx_ = field.shape
    out = MetalArray(shape=(ny, nx_))
    dev = metal_device()
    if _mean_pressure_weighted_compiled is None:
        _mean_pressure_weighted_compiled = dev.compile(_mean_pressure_weighted_source, "mean_pressure_weighted_kernel")
    iparams = struct.pack("iii", nz, ny, nx_)
    fparams = struct.pack("ff", float(p_bottom), float(p_top))
    _mean_pressure_weighted_compiled.dispatch(
        [field, pressure, out, iparams, fparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return out


# ------------------------------------------------------------------
# 35. get_layer_heights  --- extract heights within a pressure layer
# ------------------------------------------------------------------
_get_layer_heights_source = """
#include <metal_stdlib>
using namespace metal;

kernel void get_layer_heights_kernel(
    device const float* heights [[buffer(0)]],
    device const float* pressure [[buffer(1)]],
    device const float* params_f [[buffer(2)]],
    device float* h_bottom [[buffer(3)]],
    device float* h_top [[buffer(4)]],
    device const int* params_i [[buffer(5)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz = params_i[0];
    int ny = params_i[1];
    int nx = params_i[2];
    float p_bottom = params_f[0];
    float p_top = params_f[1];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    float hbot = NAN;
    float htop = NAN;

    for (int k = 0; k < nz - 1; k++) {
        float p0 = pressure[k * nxy + idx2d];
        float p1 = pressure[(k + 1) * nxy + idx2d];
        // p_bottom
        if ((p0 >= p_bottom && p_bottom >= p1) || (p1 >= p_bottom && p_bottom >= p0)) {
            float denom = p1 - p0;
            float frac = (abs(denom) > 1e-10f) ? (p_bottom - p0) / denom : 0.0f;
            float h0 = heights[k * nxy + idx2d];
            float h1 = heights[(k + 1) * nxy + idx2d];
            hbot = h0 + frac * (h1 - h0);
        }
        // p_top
        if ((p0 >= p_top && p_top >= p1) || (p1 >= p_top && p_top >= p0)) {
            float denom = p1 - p0;
            float frac = (abs(denom) > 1e-10f) ? (p_top - p0) / denom : 0.0f;
            float h0 = heights[k * nxy + idx2d];
            float h1 = heights[(k + 1) * nxy + idx2d];
            htop = h0 + frac * (h1 - h0);
        }
    }
    h_bottom[idx2d] = hbot;
    h_top[idx2d] = htop;
}
"""
_get_layer_heights_compiled = None


def get_layer_heights(heights, pressure, p_bottom, p_top):
    """Get interpolated heights at the top and bottom of a pressure layer.

    Parameters
    ----------
    heights : (nz, ny, nx) in meters
    pressure : (nz, ny, nx) in Pa
    p_bottom, p_top : float, Pa

    Returns
    -------
    h_bottom, h_top : (ny, nx) height arrays in meters
    """
    global _get_layer_heights_compiled
    heights, pressure = _to_gpu(heights), _to_gpu(pressure)
    nz, ny, nx_ = heights.shape
    h_bottom = MetalArray(data=np.full((ny, nx_), np.nan, dtype=np.float32))
    h_top = MetalArray(data=np.full((ny, nx_), np.nan, dtype=np.float32))
    dev = metal_device()
    if _get_layer_heights_compiled is None:
        _get_layer_heights_compiled = dev.compile(_get_layer_heights_source, "get_layer_heights_kernel")
    fparams = struct.pack("ff", float(p_bottom), float(p_top))
    iparams = struct.pack("iii", nz, ny, nx_)
    _get_layer_heights_compiled.dispatch(
        [heights, pressure, fparams, h_bottom, h_top, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return h_bottom, h_top


# ------------------------------------------------------------------
# 36. composite_reflectivity_from_hydrometeors
# ------------------------------------------------------------------
_composite_reflectivity_hydro_source = """
#include <metal_stdlib>
using namespace metal;

kernel void composite_reflectivity_hydro_kernel(
    device const float* pressure_3d [[buffer(0)]],
    device const float* temperature_3d [[buffer(1)]],
    device const float* qrain_3d [[buffer(2)]],
    device const float* qsnow_3d [[buffer(3)]],
    device const float* qgraup_3d [[buffer(4)]],
    device float* refl_out [[buffer(5)]],
    device const int* params [[buffer(6)]],
    uint2 gid [[thread_position_in_grid]]
) {
    int nz = params[0];
    int ny = params[1];
    int nx = params[2];
    int i = int(gid.x);
    int j = int(gid.y);
    if (i >= nx || j >= ny) return;

    int col = j * nx + i;
    float max_dbz = -999.0f;

    for (int k = 0; k < nz; k++) {
        int idx = k * ny * nx + col;
        float p = pressure_3d[idx];
        float t_c = temperature_3d[idx];
        float t_k = t_c + 273.15f;
        float rho = p / (287.05f * t_k);

        float qr = qrain_3d[idx];
        float qs = qsnow_3d[idx];
        float qg = qgraup_3d[idx];
        if (qr < 0.0f) qr = 0.0f;
        if (qs < 0.0f) qs = 0.0f;
        if (qg < 0.0f) qg = 0.0f;

        float rho_qr = rho * qr;
        float rho_qs = rho * qs;
        float rho_qg = rho * qg;

        float z_rain = 3.63e9f * pow(rho_qr, 1.75f);
        float z_snow = 9.80e8f * pow(rho_qs, 1.75f);
        float z_graup = 4.33e9f * pow(rho_qg, 1.75f);

        float z_total = z_rain + z_snow + z_graup;
        float dbz = (z_total > 1e-6f) ? 10.0f * log10(z_total) : -30.0f;

        if (dbz > max_dbz) max_dbz = dbz;
    }

    refl_out[col] = max_dbz;
}
"""
_composite_reflectivity_hydro_compiled = None


def composite_reflectivity_from_hydrometeors(
    pressure_3d, temperature_c_3d, qrain_3d, qsnow_3d, qgraup_3d
):
    """Composite reflectivity from hydrometeor mixing ratios.

    Uses the Koch et al. (2005) / Thompson microphysics reflectivity
    formulation (Smith 1984 framework) to compute equivalent reflectivity
    factor from rain, snow, and graupel mixing ratios, then takes the
    column maximum.

    Parameters
    ----------
    pressure_3d : (nz, ny, nx) array, Pa
    temperature_c_3d : (nz, ny, nx) array, degrees Celsius
    qrain_3d : (nz, ny, nx) array, kg/kg
    qsnow_3d : (nz, ny, nx) array, kg/kg
    qgraup_3d : (nz, ny, nx) array, kg/kg

    Returns
    -------
    refl_out : (ny, nx) array, composite reflectivity in dBZ
    """
    global _composite_reflectivity_hydro_compiled
    pressure_3d = _to_gpu(pressure_3d)
    temperature_c_3d = _to_gpu(temperature_c_3d)
    qrain_3d = _to_gpu(qrain_3d)
    qsnow_3d = _to_gpu(qsnow_3d)
    qgraup_3d = _to_gpu(qgraup_3d)

    nz, ny, nx_ = pressure_3d.shape
    refl_out = MetalArray(data=np.full((ny, nx_), -999.0, dtype=np.float32))
    dev = metal_device()
    if _composite_reflectivity_hydro_compiled is None:
        _composite_reflectivity_hydro_compiled = dev.compile(
            _composite_reflectivity_hydro_source, "composite_reflectivity_hydro_kernel"
        )
    iparams = struct.pack("iii", nz, ny, nx_)
    _composite_reflectivity_hydro_compiled.dispatch(
        [pressure_3d, temperature_c_3d, qrain_3d, qsnow_3d, qgraup_3d,
         refl_out, iparams],
        grid_size=(nx_, ny),
        threadgroup_size=_tg(nx_, ny),
    )
    return refl_out
