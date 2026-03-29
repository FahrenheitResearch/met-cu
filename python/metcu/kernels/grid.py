"""
Grid operations and stencil CUDA kernels for met-cu.

2D stencil computations on (ny, nx) fields, smoothing/convolution kernels,
interpolation kernels, and grid utility kernels.  Each kernel launches one
GPU thread per output grid point using 2D (16, 16) thread blocks.
"""

import math
import cupy as cp
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_gpu(arr):
    """Accept numpy, cupy, or scalar, return cupy on the current device."""
    if isinstance(arr, (int, float)):
        return cp.asarray(arr, dtype=cp.float64)
    if isinstance(arr, np.ndarray):
        return cp.asarray(arr, dtype=cp.float64)
    if isinstance(arr, cp.ndarray):
        return cp.ascontiguousarray(arr, dtype=cp.float64)
    return cp.asarray(arr, dtype=cp.float64)


def _grid_block(ny, nx):
    """Standard 2D block/grid dims."""
    block = (16, 16, 1)
    grid = ((nx + 15) // 16, (ny + 15) // 16, 1)
    return grid, block


def _broadcast_spacing(val, shape):
    """Broadcast a scalar or 0-d array to a full 2-D array matching shape."""
    val = _to_gpu(val)
    if val.ndim == 0:
        return cp.full(shape, float(val), dtype=cp.float64)
    if val.shape != shape:
        return cp.broadcast_to(val, shape).copy()
    return val


# ===================================================================
# Shared CUDA device functions for boundary-aware finite differences.
# Every stencil kernel prepends this preamble so that derivative
# computations use one-sided (second-order) stencils at domain edges
# instead of skipping those points entirely.
# ===================================================================
_deriv_device_funcs = r'''
/* ---- df/dx at (j, i) with boundary-aware stencil ---- */
__device__ double ddx(const double* f, const double* dx,
                      int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    double h = dx[idx];
    if (i == 0)
        return (-3.0*f[idx] + 4.0*f[idx+1] - f[idx+2]) / (2.0*h);
    if (i == nx - 1)
        return (3.0*f[idx] - 4.0*f[idx-1] + f[idx-2]) / (2.0*h);
    return (f[idx+1] - f[idx-1]) / (2.0*h);
}

/* ---- df/dy at (j, i) with boundary-aware stencil ---- */
__device__ double ddy(const double* f, const double* dy,
                      int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    double h = dy[idx];
    if (j == 0)
        return (-3.0*f[idx] + 4.0*f[idx+nx] - f[idx+2*nx]) / (2.0*h);
    if (j == ny - 1)
        return (3.0*f[idx] - 4.0*f[idx-nx] + f[idx-2*nx]) / (2.0*h);
    return (f[idx+nx] - f[idx-nx]) / (2.0*h);
}

/* ---- d2f/dx2 at (j, i) with boundary-aware stencil ---- */
__device__ double d2dx2(const double* f, const double* dx,
                        int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    double h = dx[idx];
    if (i == 0)
        return (f[idx] - 2.0*f[idx+1] + f[idx+2]) / (h*h);
    if (i == nx - 1)
        return (f[idx] - 2.0*f[idx-1] + f[idx-2]) / (h*h);
    return (f[idx+1] - 2.0*f[idx] + f[idx-1]) / (h*h);
}

/* ---- d2f/dy2 at (j, i) with boundary-aware stencil ---- */
__device__ double d2dy2(const double* f, const double* dy,
                        int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    double h = dy[idx];
    if (j == 0)
        return (f[idx] - 2.0*f[idx+nx] + f[idx+2*nx]) / (h*h);
    if (j == ny - 1)
        return (f[idx] - 2.0*f[idx-nx] + f[idx-2*nx]) / (h*h);
    return (f[idx+nx] - 2.0*f[idx] + f[idx-nx]) / (h*h);
}
'''


# ===================================================================
# 1. Differential-operator stencil kernels
# ===================================================================

# ------------------------------------------------------------------
# 1. vorticity  dv/dx - du/dy
# ------------------------------------------------------------------
_vorticity_code = _deriv_device_funcs + r'''
extern "C" __global__
void vorticity_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dvdx = ddx(v, dx, j, i, ny, nx);
    double dudy = ddy(u, dy, j, i, ny, nx);
    out[idx] = dvdx - dudy;
}
'''
_vorticity_kern = cp.RawKernel(_vorticity_code, 'vorticity_kernel')


def vorticity(u, v, dx, dy):
    """Relative vorticity  dv/dx - du/dy."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _vorticity_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 2. divergence  du/dx + dv/dy
# ------------------------------------------------------------------
_divergence_code = _deriv_device_funcs + r'''
extern "C" __global__
void divergence_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dudx = ddx(u, dx, j, i, ny, nx);
    double dvdy = ddy(v, dy, j, i, ny, nx);
    out[idx] = dudx + dvdy;
}
'''
_divergence_kern = cp.RawKernel(_divergence_code, 'divergence_kernel')


def divergence(u, v, dx, dy):
    """Horizontal divergence  du/dx + dv/dy."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _divergence_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 3. absolute_vorticity  relative_vort + f
# ------------------------------------------------------------------
_absolute_vorticity_code = _deriv_device_funcs + r'''
extern "C" __global__
void absolute_vorticity_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    const double* f,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dvdx = ddx(v, dx, j, i, ny, nx);
    double dudy = ddy(u, dy, j, i, ny, nx);
    out[idx] = dvdx - dudy + f[idx];
}
'''
_absolute_vorticity_kern = cp.RawKernel(_absolute_vorticity_code, 'absolute_vorticity_kernel')


def absolute_vorticity(u, v, dx, dy, f):
    """Absolute vorticity  (dv/dx - du/dy) + f."""
    u, v, f = _to_gpu(u), _to_gpu(v), _to_gpu(f)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _absolute_vorticity_kern(grid, block, (u, v, dx, dy, f, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 4. shearing_deformation  dv/dx + du/dy
# ------------------------------------------------------------------
_shearing_deformation_code = _deriv_device_funcs + r'''
extern "C" __global__
void shearing_deformation_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dvdx = ddx(v, dx, j, i, ny, nx);
    double dudy = ddy(u, dy, j, i, ny, nx);
    out[idx] = dvdx + dudy;
}
'''
_shearing_deformation_kern = cp.RawKernel(_shearing_deformation_code, 'shearing_deformation_kernel')


def shearing_deformation(u, v, dx, dy):
    """Shearing deformation  dv/dx + du/dy."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _shearing_deformation_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 5. stretching_deformation  du/dx - dv/dy
# ------------------------------------------------------------------
_stretching_deformation_code = _deriv_device_funcs + r'''
extern "C" __global__
void stretching_deformation_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dudx = ddx(u, dx, j, i, ny, nx);
    double dvdy = ddy(v, dy, j, i, ny, nx);
    out[idx] = dudx - dvdy;
}
'''
_stretching_deformation_kern = cp.RawKernel(_stretching_deformation_code, 'stretching_deformation_kernel')


def stretching_deformation(u, v, dx, dy):
    """Stretching deformation  du/dx - dv/dy."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _stretching_deformation_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 6. total_deformation  sqrt(shearing^2 + stretching^2)
# ------------------------------------------------------------------
_total_deformation_code = _deriv_device_funcs + r'''
extern "C" __global__
void total_deformation_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dudx = ddx(u, dx, j, i, ny, nx);
    double dvdy = ddy(v, dy, j, i, ny, nx);
    double dvdx = ddx(v, dx, j, i, ny, nx);
    double dudy = ddy(u, dy, j, i, ny, nx);
    double shear = dvdx + dudy;
    double stretch = dudx - dvdy;
    out[idx] = sqrt(shear * shear + stretch * stretch);
}
'''
_total_deformation_kern = cp.RawKernel(_total_deformation_code, 'total_deformation_kernel')


def total_deformation(u, v, dx, dy):
    """Total deformation  sqrt(shearing^2 + stretching^2)."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _total_deformation_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 7. curvature_vorticity
#    zeta_c = (V dot grad(wind_dir)) * |V|
#    Using natural-coordinate form:
#    zeta_c = (u^2 * dv/dx - v^2 * du/dy - u*v*(du/dx - dv/dy)) / (u^2+v^2)
# ------------------------------------------------------------------
_curvature_vorticity_code = _deriv_device_funcs + r'''
extern "C" __global__
void curvature_vorticity_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double uc = u[idx], vc = v[idx];
    double spd2 = uc * uc + vc * vc;
    if (spd2 < 1e-20) { out[idx] = 0.0; return; }
    double dudx_v = ddx(u, dx, j, i, ny, nx);
    double dudy_v = ddy(u, dy, j, i, ny, nx);
    double dvdx_v = ddx(v, dx, j, i, ny, nx);
    double dvdy_v = ddy(v, dy, j, i, ny, nx);
    // curvature vorticity = (u^2 dvdx - v^2 dudy + uv(dvdy - dudx)) / V^2
    out[idx] = (uc * uc * dvdx_v - vc * vc * dudy_v
                + uc * vc * (dvdy_v - dudx_v)) / spd2;
}
'''
_curvature_vorticity_kern = cp.RawKernel(_curvature_vorticity_code, 'curvature_vorticity_kernel')


def curvature_vorticity(u, v, dx, dy):
    """Curvature vorticity component of relative vorticity."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _curvature_vorticity_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 8. shear_vorticity
#    zeta_s = relative_vort - curvature_vort
#    Direct formula: (v^2 dudx + u^2 dvdy - uv(dvdx + dudy)) / V^2
#    but we can also just subtract, here we compute directly.
# ------------------------------------------------------------------
_shear_vorticity_code = _deriv_device_funcs + r'''
extern "C" __global__
void shear_vorticity_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double uc = u[idx], vc = v[idx];
    double spd2 = uc * uc + vc * vc;
    if (spd2 < 1e-20) { out[idx] = 0.0; return; }
    double dudx_v = ddx(u, dx, j, i, ny, nx);
    double dudy_v = ddy(u, dy, j, i, ny, nx);
    double dvdx_v = ddx(v, dx, j, i, ny, nx);
    double dvdy_v = ddy(v, dy, j, i, ny, nx);
    // shear vorticity = -(v^2 dudx + u^2 dvdy - uv(dvdx + dudy)) / V^2
    // Derived from zeta_s = zeta - zeta_c
    out[idx] = -(vc * vc * dudx_v + uc * uc * dvdy_v
                 - uc * vc * (dvdx_v + dudy_v)) / spd2;
}
'''
_shear_vorticity_kern = cp.RawKernel(_shear_vorticity_code, 'shear_vorticity_kernel')


def shear_vorticity(u, v, dx, dy):
    """Shear vorticity component of relative vorticity."""
    u, v = _to_gpu(u), _to_gpu(v)
    ny, nx = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx)), _broadcast_spacing(dy, (ny, nx))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx)
    _shear_vorticity_kern(grid, block, (u, v, dx, dy, out, np.int32(ny), np.int32(nx)))
    return out


# ------------------------------------------------------------------
# 9. first_derivative_x  df/dx  (centered)
# ------------------------------------------------------------------
_first_derivative_x_code = _deriv_device_funcs + r'''
extern "C" __global__
void first_derivative_x_kernel(
    const double* f,
    const double* dx,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = ddx(f, dx, j, i, ny, nx);
}
'''
_first_derivative_x_kern = cp.RawKernel(_first_derivative_x_code, 'first_derivative_x_kernel')


def first_derivative_x(f, dx):
    """df/dx via centered finite differences."""
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx = _broadcast_spacing(dx, (ny, nx_))
    out = cp.zeros_like(f)
    grid, block = _grid_block(ny, nx_)
    _first_derivative_x_kern(grid, block, (f, dx, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 10. first_derivative_y  df/dy
# ------------------------------------------------------------------
_first_derivative_y_code = _deriv_device_funcs + r'''
extern "C" __global__
void first_derivative_y_kernel(
    const double* f,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = ddy(f, dy, j, i, ny, nx);
}
'''
_first_derivative_y_kern = cp.RawKernel(_first_derivative_y_code, 'first_derivative_y_kernel')


def first_derivative_y(f, dy):
    """df/dy via centered finite differences."""
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dy = _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(f)
    grid, block = _grid_block(ny, nx_)
    _first_derivative_y_kern(grid, block, (f, dy, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 11. second_derivative_x  d2f/dx2
# ------------------------------------------------------------------
_second_derivative_x_code = _deriv_device_funcs + r'''
extern "C" __global__
void second_derivative_x_kernel(
    const double* f,
    const double* dx,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = d2dx2(f, dx, j, i, ny, nx);
}
'''
_second_derivative_x_kern = cp.RawKernel(_second_derivative_x_code, 'second_derivative_x_kernel')


def second_derivative_x(f, dx):
    """d2f/dx2 via centered finite differences."""
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx = _broadcast_spacing(dx, (ny, nx_))
    out = cp.zeros_like(f)
    grid, block = _grid_block(ny, nx_)
    _second_derivative_x_kern(grid, block, (f, dx, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 12. second_derivative_y  d2f/dy2
# ------------------------------------------------------------------
_second_derivative_y_code = _deriv_device_funcs + r'''
extern "C" __global__
void second_derivative_y_kernel(
    const double* f,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = d2dy2(f, dy, j, i, ny, nx);
}
'''
_second_derivative_y_kern = cp.RawKernel(_second_derivative_y_code, 'second_derivative_y_kernel')


def second_derivative_y(f, dy):
    """d2f/dy2 via centered finite differences."""
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dy = _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(f)
    grid, block = _grid_block(ny, nx_)
    _second_derivative_y_kern(grid, block, (f, dy, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 13. laplacian  d2f/dx2 + d2f/dy2
# ------------------------------------------------------------------
_laplacian_code = _deriv_device_funcs + r'''
extern "C" __global__
void laplacian_kernel(
    const double* f,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    out[j * nx + i] = d2dx2(f, dx, j, i, ny, nx) + d2dy2(f, dy, j, i, ny, nx);
}
'''
_laplacian_kern = cp.RawKernel(_laplacian_code, 'laplacian_kernel')


def laplacian(f, dx, dy):
    """Laplacian  d2f/dx2 + d2f/dy2."""
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(f)
    grid, block = _grid_block(ny, nx_)
    _laplacian_kern(grid, block, (f, dx, dy, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 14. gradient  (df/dx, df/dy)
# ------------------------------------------------------------------
_gradient_code = _deriv_device_funcs + r'''
extern "C" __global__
void gradient_kernel(
    const double* f,
    const double* dx,
    const double* dy,
    double* dfdx,
    double* dfdy,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    dfdx[idx] = ddx(f, dx, j, i, ny, nx);
    dfdy[idx] = ddy(f, dy, j, i, ny, nx);
}
'''
_gradient_kern = cp.RawKernel(_gradient_code, 'gradient_kernel')


def gradient(f, dx, dy):
    """Horizontal gradient  returns (df/dx, df/dy)."""
    f = _to_gpu(f)
    ny, nx_ = f.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    dfdx = cp.zeros_like(f)
    dfdy = cp.zeros_like(f)
    grid, block = _grid_block(ny, nx_)
    _gradient_kern(grid, block, (f, dx, dy, dfdx, dfdy, np.int32(ny), np.int32(nx_)))
    return dfdx, dfdy


# ------------------------------------------------------------------
# 15. advection  -u*df/dx - v*df/dy
# ------------------------------------------------------------------
_advection_code = _deriv_device_funcs + r'''
extern "C" __global__
void advection_kernel(
    const double* field,
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dfdx = ddx(field, dx, j, i, ny, nx);
    double dfdy = ddy(field, dy, j, i, ny, nx);
    out[idx] = -(u[idx] * dfdx + v[idx] * dfdy);
}
'''
_advection_kern = cp.RawKernel(_advection_code, 'advection_kernel')


def advection(field, u, v, dx, dy):
    """Horizontal advection  -u*df/dx - v*df/dy."""
    field, u, v = _to_gpu(field), _to_gpu(u), _to_gpu(v)
    ny, nx_ = field.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(field)
    grid, block = _grid_block(ny, nx_)
    _advection_kern(grid, block, (field, u, v, dx, dy, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 16. frontogenesis  (Petterssen)
#     F = (1/|grad(theta)|) * [ (dtheta/dx)^2 * dudx
#         + (dtheta/dy)^2 * dvdy
#         + (dtheta/dx)*(dtheta/dy) * (dvdx + dudy) ]
#     Scalar frontogenesis (kinematic, 2D):
#     F = -0.5 * |grad(theta)| * (D - E*cos(2*beta))
#     We use the component form below (Petterssen 1936).
# ------------------------------------------------------------------
_frontogenesis_code = _deriv_device_funcs + r'''
extern "C" __global__
void frontogenesis_kernel(
    const double* theta,
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    // First derivatives of theta
    double dtdx = ddx(theta, dx, j, i, ny, nx);
    double dtdy = ddy(theta, dy, j, i, ny, nx);
    double mag = sqrt(dtdx * dtdx + dtdy * dtdy);
    if (mag < 1e-20) { out[idx] = 0.0; return; }

    // Wind derivatives
    double dudx_v = ddx(u, dx, j, i, ny, nx);
    double dudy_v = ddy(u, dy, j, i, ny, nx);
    double dvdx_v = ddx(v, dx, j, i, ny, nx);
    double dvdy_v = ddy(v, dy, j, i, ny, nx);

    // Petterssen frontogenesis:
    // F = (1/|grad_theta|) * (dtdx^2*dudx + dtdy^2*dvdy + dtdx*dtdy*(dvdx+dudy))
    // Sign convention: positive = frontogenesis
    double F = (dtdx * dtdx * dudx_v
              + dtdy * dtdy * dvdy_v
              + dtdx * dtdy * (dvdx_v + dudy_v));
    // Multiply by -1 for standard convention (confluence -> positive)
    out[idx] = -F / mag;
}
'''
_frontogenesis_kern = cp.RawKernel(_frontogenesis_code, 'frontogenesis_kernel')


def frontogenesis(theta, u, v, dx, dy):
    """Petterssen frontogenesis function (scalar, 2D)."""
    theta, u, v = _to_gpu(theta), _to_gpu(u), _to_gpu(v)
    ny, nx_ = theta.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(theta)
    grid, block = _grid_block(ny, nx_)
    _frontogenesis_kern(grid, block, (theta, u, v, dx, dy, out, np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 17. q_vector  Q1, Q2
#     Q1 = -(R/p)*(dug/dx * dT/dx + dvg/dx * dT/dy)
#     Q2 = -(R/p)*(dug/dy * dT/dx + dvg/dy * dT/dy)
#     For simplicity we accept (u, v, T, p_scalar) and compute
#     Q-vectors using the full wind (caller passes geostrophic wind).
# ------------------------------------------------------------------
_q_vector_code = _deriv_device_funcs + r'''
extern "C" __global__
void q_vector_kernel(
    const double* u,
    const double* v,
    const double* temperature,
    const double* dx,
    const double* dy,
    double pressure,
    double Rd,
    double* q1_out,
    double* q2_out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    double dTdx = ddx(temperature, dx, j, i, ny, nx);
    double dTdy = ddy(temperature, dy, j, i, ny, nx);

    double dudx_v = ddx(u, dx, j, i, ny, nx);
    double dudy_v = ddy(u, dy, j, i, ny, nx);
    double dvdx_v = ddx(v, dx, j, i, ny, nx);
    double dvdy_v = ddy(v, dy, j, i, ny, nx);

    double coeff = -Rd / pressure;
    q1_out[idx] = coeff * (dudx_v * dTdx + dvdx_v * dTdy);
    q2_out[idx] = coeff * (dudy_v * dTdx + dvdy_v * dTdy);
}
'''
_q_vector_kern = cp.RawKernel(_q_vector_code, 'q_vector_kernel')


def q_vector(u, v, temperature, dx, dy, pressure, Rd=287.04):
    """Q-vector components (Q1, Q2).  *pressure* is a scalar in Pa."""
    u, v, temperature = _to_gpu(u), _to_gpu(v), _to_gpu(temperature)
    ny, nx_ = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    q1 = cp.zeros_like(u)
    q2 = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx_)
    _q_vector_kern(grid, block, (u, v, temperature, dx, dy,
                                 np.float64(pressure), np.float64(Rd),
                                 q1, q2, np.int32(ny), np.int32(nx_)))
    return q1, q2


# ------------------------------------------------------------------
# 18. geostrophic_wind  ug = -(g/f)*dZ/dy,  vg = (g/f)*dZ/dx
#     Z is geopotential height (m), f is Coriolis (rad/s).
# ------------------------------------------------------------------
_geostrophic_wind_code = _deriv_device_funcs + r'''
extern "C" __global__
void geostrophic_wind_kernel(
    const double* Z,
    const double* f,
    const double* dx,
    const double* dy,
    double grav,
    double* ug,
    double* vg,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double fc = f[idx];
    if (fabs(fc) < 1e-20) { ug[idx] = 0.0; vg[idx] = 0.0; return; }
    double dZdx = ddx(Z, dx, j, i, ny, nx);
    double dZdy = ddy(Z, dy, j, i, ny, nx);
    ug[idx] = -(grav / fc) * dZdy;
    vg[idx] =  (grav / fc) * dZdx;
}
'''
_geostrophic_wind_kern = cp.RawKernel(_geostrophic_wind_code, 'geostrophic_wind_kernel')


def geostrophic_wind(Z, f, dx, dy, g=9.80665):
    """Geostrophic wind from geopotential height.  Returns (ug, vg)."""
    Z, f = _to_gpu(Z), _to_gpu(f)
    ny, nx_ = Z.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    ug = cp.zeros_like(Z)
    vg = cp.zeros_like(Z)
    grid, block = _grid_block(ny, nx_)
    _geostrophic_wind_kern(grid, block, (Z, f, dx, dy, np.float64(g),
                                         ug, vg, np.int32(ny), np.int32(nx_)))
    return ug, vg


# ------------------------------------------------------------------
# 19. ageostrophic_wind  ua = u - ug,  va = v - vg
# ------------------------------------------------------------------
_ageostrophic_wind_code = _deriv_device_funcs + r'''
extern "C" __global__
void ageostrophic_wind_kernel(
    const double* u,
    const double* v,
    const double* Z,
    const double* f,
    const double* dx,
    const double* dy,
    double grav,
    double* ua,
    double* va,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double fc = f[idx];
    if (fabs(fc) < 1e-20) { ua[idx] = u[idx]; va[idx] = v[idx]; return; }
    double dZdx = ddx(Z, dx, j, i, ny, nx);
    double dZdy = ddy(Z, dy, j, i, ny, nx);
    double ug = -(grav / fc) * dZdy;
    double vg =  (grav / fc) * dZdx;
    ua[idx] = u[idx] - ug;
    va[idx] = v[idx] - vg;
}
'''
_ageostrophic_wind_kern = cp.RawKernel(_ageostrophic_wind_code, 'ageostrophic_wind_kernel')


def ageostrophic_wind(u, v, Z, f, dx, dy, g=9.80665):
    """Ageostrophic wind.  Returns (ua, va) = (u - ug, v - vg)."""
    u, v, Z, f = _to_gpu(u), _to_gpu(v), _to_gpu(Z), _to_gpu(f)
    ny, nx_ = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    ua = cp.zeros_like(u)
    va = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx_)
    _ageostrophic_wind_kern(grid, block, (u, v, Z, f, dx, dy, np.float64(g),
                                          ua, va, np.int32(ny), np.int32(nx_)))
    return ua, va


# ------------------------------------------------------------------
# 20. potential_vorticity_baroclinic
#     PV = -g * (f + zeta) * dtheta/dp
#     On isentropic surfaces, zeta and dtheta/dp are at each (j, i).
#     pressure is a 2D field on the isentropic surface.
# ------------------------------------------------------------------
_pv_baroclinic_code = _deriv_device_funcs + r'''
extern "C" __global__
void potential_vorticity_baroclinic_kernel(
    const double* u,
    const double* v,
    const double* theta,
    const double* pressure,
    const double* dx,
    const double* dy,
    const double* f,
    double grav,
    double* out,
    int ny, int nx, int nz
) {
    // Per-column, per-level: thread covers (j, i, k)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int k = 1; k < nz - 1; k++) {
        int idx3d = k * nxy + idx2d;
        // Relative vorticity at this level (pass pointer to level-k slice)
        double dvdx_v = ddx(&v[k * nxy], dx, j, i, ny, nx);
        double dudy_v = ddy(&u[k * nxy], dy, j, i, ny, nx);
        double zeta = dvdx_v - dudy_v;

        // dtheta/dp centered in vertical
        double dp = pressure[(k + 1) * nxy + idx2d] - pressure[(k - 1) * nxy + idx2d];
        double dtheta = theta[(k + 1) * nxy + idx2d] - theta[(k - 1) * nxy + idx2d];
        double dthetadp = (fabs(dp) > 1e-10) ? dtheta / dp : 0.0;

        out[idx3d] = -grav * (f[idx2d] + zeta) * dthetadp;
    }
}
'''
_pv_baroclinic_kern = cp.RawKernel(_pv_baroclinic_code, 'potential_vorticity_baroclinic_kernel')


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
    u, v, theta, pressure, f = _to_gpu(u), _to_gpu(v), _to_gpu(theta), _to_gpu(pressure), _to_gpu(f)
    nz, ny, nx_ = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx_)
    _pv_baroclinic_kern(grid, block, (u, v, theta, pressure, dx, dy, f,
                                      np.float64(g), out,
                                      np.int32(ny), np.int32(nx_), np.int32(nz)))
    return out


# ------------------------------------------------------------------
# 21. potential_vorticity_barotropic
#     PV_bt = (f + zeta) / depth
#     depth is a 2D layer depth field.
# ------------------------------------------------------------------
_pv_barotropic_code = _deriv_device_funcs + r'''
extern "C" __global__
void potential_vorticity_barotropic_kernel(
    const double* u,
    const double* v,
    const double* dx,
    const double* dy,
    const double* f,
    const double* depth,
    double* out,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double dvdx_v = ddx(v, dx, j, i, ny, nx);
    double dudy_v = ddy(u, dy, j, i, ny, nx);
    double zeta = dvdx_v - dudy_v;
    double h = depth[idx];
    out[idx] = (fabs(h) > 1e-10) ? (f[idx] + zeta) / h : 0.0;
}
'''
_pv_barotropic_kern = cp.RawKernel(_pv_barotropic_code, 'potential_vorticity_barotropic_kernel')


def potential_vorticity_barotropic(u, v, dx, dy, f, depth):
    """Barotropic potential vorticity  (f + zeta) / depth."""
    u, v, f, depth = _to_gpu(u), _to_gpu(v), _to_gpu(f), _to_gpu(depth)
    ny, nx_ = u.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    out = cp.zeros_like(u)
    grid, block = _grid_block(ny, nx_)
    _pv_barotropic_kern(grid, block, (u, v, dx, dy, f, depth, out,
                                      np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 22. inertial_advective_wind
#     The inertial-advective wind is the wind that would result from
#     pure inertial advection: V_ia = V + (1/f) * k x (dV/dt)
#     Approximated as: u_ia = u + (v/f)(dvdx + dvdy*v/u)
#     Here we use the simpler steady-state form:
#     u_ia = ug + (1/f)(ug*dug/dx + vg*dug/dy)
#     v_ia = vg + (1/f)(ug*dvg/dx + vg*dvg/dy)
# ------------------------------------------------------------------
_inertial_advective_wind_code = _deriv_device_funcs + r'''
extern "C" __global__
void inertial_advective_wind_kernel(
    const double* ug,
    const double* vg,
    const double* f,
    const double* dx,
    const double* dy,
    double* u_ia,
    double* v_ia,
    int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;
    double fc = f[idx];
    if (fabs(fc) < 1e-20) { u_ia[idx] = ug[idx]; v_ia[idx] = vg[idx]; return; }

    double dugdx = ddx(ug, dx, j, i, ny, nx);
    double dugdy = ddy(ug, dy, j, i, ny, nx);
    double dvgdx = ddx(vg, dx, j, i, ny, nx);
    double dvgdy = ddy(vg, dy, j, i, ny, nx);

    double ugc = ug[idx], vgc = vg[idx];
    u_ia[idx] = ugc + (1.0 / fc) * (ugc * dugdx + vgc * dugdy);
    v_ia[idx] = vgc + (1.0 / fc) * (ugc * dvgdx + vgc * dvgdy);
}
'''
_inertial_advective_wind_kern = cp.RawKernel(_inertial_advective_wind_code,
                                              'inertial_advective_wind_kernel')


def inertial_advective_wind(ug, vg, f, dx, dy):
    """Inertial-advective wind from geostrophic wind.  Returns (u_ia, v_ia)."""
    ug, vg, f = _to_gpu(ug), _to_gpu(vg), _to_gpu(f)
    ny, nx_ = ug.shape
    dx, dy = _broadcast_spacing(dx, (ny, nx_)), _broadcast_spacing(dy, (ny, nx_))
    u_ia = cp.zeros_like(ug)
    v_ia = cp.zeros_like(ug)
    grid, block = _grid_block(ny, nx_)
    _inertial_advective_wind_kern(grid, block, (ug, vg, f, dx, dy,
                                                u_ia, v_ia,
                                                np.int32(ny), np.int32(nx_)))
    return u_ia, v_ia


# ===================================================================
# 2. Smoothing kernels
# ===================================================================

# ------------------------------------------------------------------
# 23. smooth_gaussian  — Gaussian filter
# ------------------------------------------------------------------
_smooth_gaussian_code = r'''
extern "C" __global__
void smooth_gaussian_kernel(
    const double* input,
    double* output,
    int ny, int nx,
    int radius,
    double sigma
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    double sum = 0.0, wsum = 0.0;
    double inv2s2 = 1.0 / (2.0 * sigma * sigma);
    for (int dj = -radius; dj <= radius; dj++) {
        for (int di = -radius; di <= radius; di++) {
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            double r2 = (double)(di * di + dj * dj);
            double w = exp(-r2 * inv2s2);
            sum += w * input[jj * nx + ii];
            wsum += w;
        }
    }
    output[j * nx + i] = sum / wsum;
}
'''
_smooth_gaussian_kern = cp.RawKernel(_smooth_gaussian_code, 'smooth_gaussian_kernel')


def smooth_gaussian(field, sigma=1.0, radius=None):
    """Gaussian smoothing filter.

    Parameters
    ----------
    field : (ny, nx) array
    sigma : float, Gaussian sigma in grid units (default 1.0)
    radius : int, filter half-width (default: ceil(3*sigma))
    """
    field = _to_gpu(field)
    ny, nx_ = field.shape
    if radius is None:
        radius = int(math.ceil(3.0 * sigma))
    out = cp.zeros_like(field)
    grid, block = _grid_block(ny, nx_)
    _smooth_gaussian_kern(grid, block, (field, out,
                                        np.int32(ny), np.int32(nx_),
                                        np.int32(radius), np.float64(sigma)))
    return out


# ------------------------------------------------------------------
# 24. smooth_n_point  — 5-point or 9-point smoother
# ------------------------------------------------------------------
_smooth_n_point_code = r'''
extern "C" __global__
void smooth_n_point_kernel(
    const double* input,
    double* output,
    int ny, int nx,
    int n_point
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1) return;
    int idx = j * nx + i;

    if (n_point == 5) {
        // 5-point: center + 4 cardinal neighbors
        output[idx] = (4.0 * input[idx]
                       + input[idx - 1] + input[idx + 1]
                       + input[idx - nx] + input[idx + nx]) / 8.0;
    } else {
        // 9-point: center + 4 cardinal + 4 diagonal
        output[idx] = (4.0 * input[idx]
                       + input[idx - 1] + input[idx + 1]
                       + input[idx - nx] + input[idx + nx]
                       + 0.5 * (input[idx - nx - 1] + input[idx - nx + 1]
                              + input[idx + nx - 1] + input[idx + nx + 1])) / 10.0;
    }
}
'''
_smooth_n_point_kern = cp.RawKernel(_smooth_n_point_code, 'smooth_n_point_kernel')


def smooth_n_point(field, n_point=5, passes=1):
    """N-point smoother (5-point or 9-point).

    Parameters
    ----------
    field : (ny, nx) array
    n_point : int, 5 or 9 (default 5)
    passes : int, number of smoothing passes (default 1)
    """
    field = _to_gpu(field)
    ny, nx_ = field.shape
    grid, block = _grid_block(ny, nx_)
    src = field.copy()
    dst = cp.zeros_like(field)
    for _ in range(passes):
        _smooth_n_point_kern(grid, block, (src, dst,
                                           np.int32(ny), np.int32(nx_),
                                           np.int32(n_point)))
        src, dst = dst, src
    return src


# ------------------------------------------------------------------
# 25. smooth_rectangular  — box filter
# ------------------------------------------------------------------
_smooth_rectangular_code = r'''
extern "C" __global__
void smooth_rectangular_kernel(
    const double* input,
    double* output,
    int ny, int nx,
    int radius_x,
    int radius_y
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    double sum = 0.0;
    int count = 0;
    for (int dj = -radius_y; dj <= radius_y; dj++) {
        for (int di = -radius_x; di <= radius_x; di++) {
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            sum += input[jj * nx + ii];
            count++;
        }
    }
    output[j * nx + i] = sum / (double)count;
}
'''
_smooth_rectangular_kern = cp.RawKernel(_smooth_rectangular_code, 'smooth_rectangular_kernel')


def smooth_rectangular(field, radius_x=1, radius_y=1):
    """Rectangular (box) filter.

    Parameters
    ----------
    field : (ny, nx)
    radius_x, radius_y : int, half-widths in x and y
    """
    field = _to_gpu(field)
    ny, nx_ = field.shape
    out = cp.zeros_like(field)
    grid, block = _grid_block(ny, nx_)
    _smooth_rectangular_kern(grid, block, (field, out,
                                           np.int32(ny), np.int32(nx_),
                                           np.int32(radius_x), np.int32(radius_y)))
    return out


# ------------------------------------------------------------------
# 26. smooth_circular  — circular window filter
# ------------------------------------------------------------------
_smooth_circular_code = r'''
extern "C" __global__
void smooth_circular_kernel(
    const double* input,
    double* output,
    int ny, int nx,
    int radius
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    double sum = 0.0;
    int count = 0;
    double r2max = (double)(radius * radius);
    for (int dj = -radius; dj <= radius; dj++) {
        for (int di = -radius; di <= radius; di++) {
            if ((double)(di * di + dj * dj) > r2max) continue;
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            sum += input[jj * nx + ii];
            count++;
        }
    }
    output[j * nx + i] = sum / (double)count;
}
'''
_smooth_circular_kern = cp.RawKernel(_smooth_circular_code, 'smooth_circular_kernel')


def smooth_circular(field, radius=2):
    """Circular window filter.

    Parameters
    ----------
    field : (ny, nx)
    radius : int, circle radius in grid points
    """
    field = _to_gpu(field)
    ny, nx_ = field.shape
    out = cp.zeros_like(field)
    grid, block = _grid_block(ny, nx_)
    _smooth_circular_kern(grid, block, (field, out,
                                        np.int32(ny), np.int32(nx_),
                                        np.int32(radius)))
    return out


# ------------------------------------------------------------------
# 27. smooth_window  — generic window function (weights passed in)
# ------------------------------------------------------------------
_smooth_window_code = r'''
extern "C" __global__
void smooth_window_kernel(
    const double* input,
    const double* weights,
    double* output,
    int ny, int nx,
    int win_h, int win_w
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int rh = win_h / 2, rw = win_w / 2;
    double sum = 0.0, wsum = 0.0;
    for (int dj = -rh; dj <= rh; dj++) {
        for (int di = -rw; di <= rw; di++) {
            int jj = j + dj, ii = i + di;
            if (jj < 0 || jj >= ny || ii < 0 || ii >= nx) continue;
            double w = weights[(dj + rh) * win_w + (di + rw)];
            sum += w * input[jj * nx + ii];
            wsum += w;
        }
    }
    output[j * nx + i] = (wsum > 0.0) ? sum / wsum : input[j * nx + i];
}
'''
_smooth_window_kern = cp.RawKernel(_smooth_window_code, 'smooth_window_kernel')


def smooth_window(field, weights):
    """Smooth with an arbitrary 2D weight array.

    Parameters
    ----------
    field : (ny, nx)
    weights : (win_h, win_w)  weight kernel (need not be normalized)
    """
    field = _to_gpu(field)
    weights = _to_gpu(weights)
    ny, nx_ = field.shape
    win_h, win_w = weights.shape
    out = cp.zeros_like(field)
    grid, block = _grid_block(ny, nx_)
    _smooth_window_kern(grid, block, (field, weights, out,
                                      np.int32(ny), np.int32(nx_),
                                      np.int32(win_h), np.int32(win_w)))
    return out


# ===================================================================
# 3. Interpolation kernels
# ===================================================================

# ------------------------------------------------------------------
# 28. interpolate_1d  — per-column 1D linear interpolation to new levels
# ------------------------------------------------------------------
_interpolate_1d_code = r'''
extern "C" __global__
void interpolate_1d_kernel(
    const double* field,
    const double* levels_in,
    const double* levels_out,
    double* out,
    int nz_in, int nz_out, int ny, int nx,
    int ascending
) {
    // One thread per output column (j, i)
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int ko = 0; ko < nz_out; ko++) {
        double target = levels_out[ko];
        // Find bracketing levels
        int found = 0;
        for (int k = 0; k < nz_in - 1; k++) {
            double lo = levels_in[k * nxy + idx2d];
            double hi = levels_in[(k + 1) * nxy + idx2d];
            int bracket;
            if (ascending)
                bracket = (lo <= target && target <= hi) || (hi <= target && target <= lo);
            else
                bracket = (lo >= target && target >= hi) || (hi >= target && target >= lo);
            if (bracket) {
                double denom = hi - lo;
                double frac = (fabs(denom) > 1e-30) ? (target - lo) / denom : 0.0;
                double f0 = field[k * nxy + idx2d];
                double f1 = field[(k + 1) * nxy + idx2d];
                out[ko * nxy + idx2d] = f0 + frac * (f1 - f0);
                found = 1;
                break;
            }
        }
        if (!found) {
            out[ko * nxy + idx2d] = 0.0 / 0.0;  // NaN
        }
    }
}
'''
_interpolate_1d_kern = cp.RawKernel(_interpolate_1d_code, 'interpolate_1d_kernel')


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
    field, levels_in = _to_gpu(field), _to_gpu(levels_in)
    nz_in, ny, nx_ = field.shape
    levels_out_1d = cp.asarray(levels_out, dtype=cp.float64).ravel()
    nz_out = levels_out_1d.size
    # Broadcast levels_out to (nz_out, ny, nx) for the kernel
    lo_3d = cp.broadcast_to(levels_out_1d[:, None, None],
                            (nz_out, ny, nx_)).astype(cp.float64).copy()
    # But kernel reads levels_out as 1D (indexed [ko]), so we pass scalar array
    # Actually we index levels_out[ko] — so just pass the 1D array
    # Re-read kernel: levels_out[ko] — it's a 1D array.
    out = cp.empty((nz_out, ny, nx_), dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _interpolate_1d_kern(grid, block, (field, levels_in, levels_out_1d, out,
                                       np.int32(nz_in), np.int32(nz_out),
                                       np.int32(ny), np.int32(nx_),
                                       np.int32(1 if ascending else 0)))
    return out


# ------------------------------------------------------------------
# 29. log_interpolate_1d  — log-pressure interpolation per column
# ------------------------------------------------------------------
_log_interpolate_1d_code = r'''
extern "C" __global__
void log_interpolate_1d_kernel(
    const double* field,
    const double* pressure,
    const double* p_target,
    double* out,
    int nz_in, int nz_out, int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int ko = 0; ko < nz_out; ko++) {
        double lnpt = log(p_target[ko]);
        int found = 0;
        for (int k = 0; k < nz_in - 1; k++) {
            double lnp0 = log(pressure[k * nxy + idx2d]);
            double lnp1 = log(pressure[(k + 1) * nxy + idx2d]);
            // Check bracket (pressure usually decreasing with height)
            if ((lnp0 >= lnpt && lnpt >= lnp1) || (lnp1 >= lnpt && lnpt >= lnp0)) {
                double denom = lnp1 - lnp0;
                double frac = (fabs(denom) > 1e-30) ? (lnpt - lnp0) / denom : 0.0;
                double f0 = field[k * nxy + idx2d];
                double f1 = field[(k + 1) * nxy + idx2d];
                out[ko * nxy + idx2d] = f0 + frac * (f1 - f0);
                found = 1;
                break;
            }
        }
        if (!found) out[ko * nxy + idx2d] = 0.0 / 0.0;
    }
}
'''
_log_interpolate_1d_kern = cp.RawKernel(_log_interpolate_1d_code, 'log_interpolate_1d_kernel')


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
    field, pressure = _to_gpu(field), _to_gpu(pressure)
    nz_in, ny, nx_ = field.shape
    p_target_arr = cp.asarray(p_target, dtype=cp.float64).ravel()
    nz_out = p_target_arr.size
    out = cp.empty((nz_out, ny, nx_), dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _log_interpolate_1d_kern(grid, block, (field, pressure, p_target_arr, out,
                                           np.int32(nz_in), np.int32(nz_out),
                                           np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 30. interpolate_to_isosurface
#     Find pressure/height where field == value, per column.
# ------------------------------------------------------------------
_interpolate_to_isosurface_code = r'''
extern "C" __global__
void interpolate_to_isosurface_kernel(
    const double* field,
    const double* coord,
    double target_value,
    double* out,
    int nz, int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;
    double result = 0.0 / 0.0;  // NaN default

    for (int k = 0; k < nz - 1; k++) {
        double f0 = field[k * nxy + idx2d];
        double f1 = field[(k + 1) * nxy + idx2d];
        // Check if target is bracketed
        if ((f0 <= target_value && target_value <= f1) ||
            (f1 <= target_value && target_value <= f0)) {
            double denom = f1 - f0;
            double frac = (fabs(denom) > 1e-30) ? (target_value - f0) / denom : 0.0;
            double c0 = coord[k * nxy + idx2d];
            double c1 = coord[(k + 1) * nxy + idx2d];
            result = c0 + frac * (c1 - c0);
            break;
        }
    }
    out[idx2d] = result;
}
'''
_interpolate_to_isosurface_kern = cp.RawKernel(_interpolate_to_isosurface_code,
                                                'interpolate_to_isosurface_kernel')


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
    field, coord = _to_gpu(field), _to_gpu(coord)
    nz, ny, nx_ = field.shape
    out = cp.full((ny, nx_), cp.nan, dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _interpolate_to_isosurface_kern(grid, block, (field, coord,
                                                   np.float64(target_value), out,
                                                   np.int32(nz), np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 31. isentropic_interpolation
#     Interpolate fields to theta surfaces per column.
# ------------------------------------------------------------------
_isentropic_interpolation_code = r'''
extern "C" __global__
void isentropic_interpolation_kernel(
    const double* theta,
    const double* pressure,
    const double* field,
    const double* theta_targets,
    double* out,
    int nz, int n_theta, int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    for (int kt = 0; kt < n_theta; kt++) {
        double target = theta_targets[kt];
        int found = 0;
        for (int k = 0; k < nz - 1; k++) {
            double t0 = theta[k * nxy + idx2d];
            double t1 = theta[(k + 1) * nxy + idx2d];
            if ((t0 <= target && target <= t1) || (t1 <= target && target <= t0)) {
                double denom = t1 - t0;
                double frac = (fabs(denom) > 1e-10) ? (target - t0) / denom : 0.0;
                double f0 = field[k * nxy + idx2d];
                double f1 = field[(k + 1) * nxy + idx2d];
                out[kt * nxy + idx2d] = f0 + frac * (f1 - f0);
                found = 1;
                break;
            }
        }
        if (!found) out[kt * nxy + idx2d] = 0.0 / 0.0;
    }
}
'''
_isentropic_interpolation_kern = cp.RawKernel(_isentropic_interpolation_code,
                                               'isentropic_interpolation_kernel')


def isentropic_interpolation(theta, pressure, field, theta_targets):
    """Interpolate a field to isentropic (theta) surfaces.

    Parameters
    ----------
    theta : (nz, ny, nx)  potential temperature
    pressure : (nz, ny, nx)  pressure (unused in interpolation itself,
               kept for API consistency — pass field=pressure to get
               pressure on theta surfaces)
    field : (nz, ny, nx)  field to interpolate
    theta_targets : 1D array of target theta values (K)

    Returns
    -------
    (n_theta, ny, nx) interpolated field
    """
    theta, field = _to_gpu(theta), _to_gpu(field)
    nz, ny, nx_ = theta.shape
    theta_t = cp.asarray(theta_targets, dtype=cp.float64).ravel()
    n_theta = theta_t.size
    out = cp.empty((n_theta, ny, nx_), dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _isentropic_interpolation_kern(grid, block, (theta, _to_gpu(pressure), field,
                                                  theta_t, out,
                                                  np.int32(nz), np.int32(n_theta),
                                                  np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 32. lat_lon_grid_deltas  — compute dx, dy from lat/lon via haversine
# ------------------------------------------------------------------
_lat_lon_grid_deltas_code = r'''
extern "C" __global__
void lat_lon_grid_deltas_kernel(
    const double* lat,
    const double* lon,
    double* dx,
    double* dy,
    int ny, int nx,
    double earth_radius
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;
    int idx = j * nx + i;

    double deg2rad = 3.14159265358979323846 / 180.0;

    // dx: distance between (j, i) and (j, i+1)  [or i-1 at boundary]
    if (i < nx - 1) {
        double lat1 = lat[idx] * deg2rad;
        double lat2 = lat[idx + 1] * deg2rad;
        double dlon = (lon[idx + 1] - lon[idx]) * deg2rad;
        double dlat = lat2 - lat1;
        double a = sin(dlat / 2.0) * sin(dlat / 2.0)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0);
        double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
        dx[idx] = earth_radius * c;
    } else {
        // Copy from i-1
        int prev = j * nx + (i - 1);
        double lat1 = lat[prev] * deg2rad;
        double lat2 = lat[idx] * deg2rad;
        double dlon = (lon[idx] - lon[prev]) * deg2rad;
        double dlat = lat2 - lat1;
        double a = sin(dlat / 2.0) * sin(dlat / 2.0)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0);
        double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
        dx[idx] = earth_radius * c;
    }

    // dy: distance between (j, i) and (j+1, i)  [or j-1 at boundary]
    if (j < ny - 1) {
        double lat1 = lat[idx] * deg2rad;
        double lat2 = lat[idx + nx] * deg2rad;
        double dlon = (lon[idx + nx] - lon[idx]) * deg2rad;
        double dlat = lat2 - lat1;
        double a = sin(dlat / 2.0) * sin(dlat / 2.0)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0);
        double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
        dy[idx] = earth_radius * c;
    } else {
        int prev = (j - 1) * nx + i;
        double lat1 = lat[prev] * deg2rad;
        double lat2 = lat[idx] * deg2rad;
        double dlon = (lon[idx] - lon[prev]) * deg2rad;
        double dlat = lat2 - lat1;
        double a = sin(dlat / 2.0) * sin(dlat / 2.0)
                 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) * sin(dlon / 2.0);
        double c = 2.0 * atan2(sqrt(a), sqrt(1.0 - a));
        dy[idx] = earth_radius * c;
    }
}
'''
_lat_lon_grid_deltas_kern = cp.RawKernel(_lat_lon_grid_deltas_code,
                                          'lat_lon_grid_deltas_kernel')


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
    lat, lon = _to_gpu(lat), _to_gpu(lon)
    ny, nx_ = lat.shape
    dx = cp.zeros_like(lat)
    dy = cp.zeros_like(lat)
    grid, block = _grid_block(ny, nx_)
    _lat_lon_grid_deltas_kern(grid, block, (lat, lon, dx, dy,
                                            np.int32(ny), np.int32(nx_),
                                            np.float64(earth_radius)))
    return dx, dy


# ===================================================================
# 4. Grid utility kernels
# ===================================================================

# ------------------------------------------------------------------
# 33. composite_reflectivity  — max value along vertical axis per column
# ------------------------------------------------------------------
_composite_reflectivity_code = r'''
extern "C" __global__
void composite_reflectivity_kernel(
    const double* field,
    double* out,
    int nz, int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;
    double maxval = -1e308;
    for (int k = 0; k < nz; k++) {
        double val = field[k * nxy + idx2d];
        if (val > maxval) maxval = val;
    }
    out[idx2d] = maxval;
}
'''
_composite_reflectivity_kern = cp.RawKernel(_composite_reflectivity_code,
                                             'composite_reflectivity_kernel')


def composite_reflectivity(field):
    """Maximum value along the vertical axis (axis=0) per column.

    Parameters
    ----------
    field : (nz, ny, nx)

    Returns
    -------
    (ny, nx) column-max values
    """
    field = _to_gpu(field)
    nz, ny, nx_ = field.shape
    out = cp.empty((ny, nx_), dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _composite_reflectivity_kern(grid, block, (field, out,
                                               np.int32(nz), np.int32(ny), np.int32(nx_)))
    return out


# ------------------------------------------------------------------
# 34. mean_pressure_weighted  — pressure-weighted mean in a layer
# ------------------------------------------------------------------
_mean_pressure_weighted_code = r'''
extern "C" __global__
void mean_pressure_weighted_kernel(
    const double* field,
    const double* pressure,
    double* out,
    int nz, int ny, int nx,
    double p_bottom, double p_top
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;
    double sum = 0.0, wsum = 0.0;

    for (int k = 0; k < nz; k++) {
        double p = pressure[k * nxy + idx2d];
        if (p <= p_bottom && p >= p_top) {
            // Use dp as weight (trapezoidal-ish)
            double dp;
            if (k == 0)
                dp = fabs(pressure[idx2d] - pressure[nxy + idx2d]) * 0.5;
            else if (k == nz - 1)
                dp = fabs(pressure[(nz - 2) * nxy + idx2d] - pressure[(nz - 1) * nxy + idx2d]) * 0.5;
            else
                dp = fabs(pressure[(k - 1) * nxy + idx2d] - pressure[(k + 1) * nxy + idx2d]) * 0.5;
            sum += field[k * nxy + idx2d] * dp;
            wsum += dp;
        }
    }
    out[idx2d] = (wsum > 0.0) ? sum / wsum : 0.0 / 0.0;
}
'''
_mean_pressure_weighted_kern = cp.RawKernel(_mean_pressure_weighted_code,
                                             'mean_pressure_weighted_kernel')


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
    field, pressure = _to_gpu(field), _to_gpu(pressure)
    nz, ny, nx_ = field.shape
    out = cp.empty((ny, nx_), dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _mean_pressure_weighted_kern(grid, block, (field, pressure, out,
                                               np.int32(nz), np.int32(ny), np.int32(nx_),
                                               np.float64(p_bottom), np.float64(p_top)))
    return out


# ------------------------------------------------------------------
# 35. get_layer_heights  — extract heights within a pressure layer
# ------------------------------------------------------------------
_get_layer_heights_code = r'''
extern "C" __global__
void get_layer_heights_kernel(
    const double* heights,
    const double* pressure,
    double p_bottom, double p_top,
    double* h_bottom,
    double* h_top,
    int nz, int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int nxy = ny * nx;
    int idx2d = j * nx + i;

    double hbot = 0.0 / 0.0;
    double htop = 0.0 / 0.0;

    // Find height at p_bottom (interpolate)
    for (int k = 0; k < nz - 1; k++) {
        double p0 = pressure[k * nxy + idx2d];
        double p1 = pressure[(k + 1) * nxy + idx2d];
        // p_bottom
        if ((p0 >= p_bottom && p_bottom >= p1) || (p1 >= p_bottom && p_bottom >= p0)) {
            double denom = p1 - p0;
            double frac = (fabs(denom) > 1e-10) ? (p_bottom - p0) / denom : 0.0;
            double h0 = heights[k * nxy + idx2d];
            double h1 = heights[(k + 1) * nxy + idx2d];
            hbot = h0 + frac * (h1 - h0);
        }
        // p_top
        if ((p0 >= p_top && p_top >= p1) || (p1 >= p_top && p_top >= p0)) {
            double denom = p1 - p0;
            double frac = (fabs(denom) > 1e-10) ? (p_top - p0) / denom : 0.0;
            double h0 = heights[k * nxy + idx2d];
            double h1 = heights[(k + 1) * nxy + idx2d];
            htop = h0 + frac * (h1 - h0);
        }
    }
    h_bottom[idx2d] = hbot;
    h_top[idx2d] = htop;
}
'''
_get_layer_heights_kern = cp.RawKernel(_get_layer_heights_code, 'get_layer_heights_kernel')


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
    heights, pressure = _to_gpu(heights), _to_gpu(pressure)
    nz, ny, nx_ = heights.shape
    h_bottom = cp.full((ny, nx_), cp.nan, dtype=cp.float64)
    h_top = cp.full((ny, nx_), cp.nan, dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _get_layer_heights_kern(grid, block, (heights, pressure,
                                          np.float64(p_bottom), np.float64(p_top),
                                          h_bottom, h_top,
                                          np.int32(nz), np.int32(ny), np.int32(nx_)))
    return h_bottom, h_top


# ---------------------------------------------------------------------------
# Composite reflectivity from hydrometeor mixing ratios
# Smith (1984) / Koch et al. (2005) / Thompson microphysics formulation
# ---------------------------------------------------------------------------

_composite_reflectivity_hydro_code = r'''
extern "C" __global__
void composite_reflectivity_hydro_kernel(
    const double* pressure_3d,
    const double* temperature_3d,
    const double* qrain_3d,
    const double* qsnow_3d,
    const double* qgraup_3d,
    double* refl_out,
    int nz, int ny, int nx
) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;
    if (i >= nx || j >= ny) return;

    int col = j * nx + i;
    double max_dbz = -999.0;

    for (int k = 0; k < nz; k++) {
        int idx = k * ny * nx + col;
        double p = pressure_3d[idx];
        double t_c = temperature_3d[idx];
        double t_k = t_c + 273.15;
        double rho = p / (287.05 * t_k);

        double qr = qrain_3d[idx];
        double qs = qsnow_3d[idx];
        double qg = qgraup_3d[idx];
        if (qr < 0.0) qr = 0.0;
        if (qs < 0.0) qs = 0.0;
        if (qg < 0.0) qg = 0.0;

        double rho_qr = rho * qr;
        double rho_qs = rho * qs;
        double rho_qg = rho * qg;

        double z_rain = 3.63e9 * pow(rho_qr, 1.75);
        double z_snow = 9.80e8 * pow(rho_qs, 1.75);
        double z_graup = 4.33e9 * pow(rho_qg, 1.75);

        double z_total = z_rain + z_snow + z_graup;
        double dbz = (z_total > 1e-6) ? 10.0 * log10(z_total) : -30.0;

        if (dbz > max_dbz) max_dbz = dbz;
    }

    refl_out[col] = max_dbz;
}
'''
_composite_reflectivity_hydro_kern = cp.RawKernel(
    _composite_reflectivity_hydro_code, 'composite_reflectivity_hydro_kernel'
)


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
    pressure_3d : (nz, ny, nx) cupy array, Pa
    temperature_c_3d : (nz, ny, nx) cupy array, degrees Celsius
    qrain_3d : (nz, ny, nx) cupy array, kg/kg
    qsnow_3d : (nz, ny, nx) cupy array, kg/kg
    qgraup_3d : (nz, ny, nx) cupy array, kg/kg

    Returns
    -------
    refl_out : (ny, nx) cupy array, composite reflectivity in dBZ
    """
    pressure_3d = _to_gpu(pressure_3d)
    temperature_c_3d = _to_gpu(temperature_c_3d)
    qrain_3d = _to_gpu(qrain_3d)
    qsnow_3d = _to_gpu(qsnow_3d)
    qgraup_3d = _to_gpu(qgraup_3d)

    nz, ny, nx_ = pressure_3d.shape
    refl_out = cp.full((ny, nx_), -999.0, dtype=cp.float64)
    grid, block = _grid_block(ny, nx_)
    _composite_reflectivity_hydro_kern(
        grid, block,
        (pressure_3d, temperature_c_3d, qrain_3d, qsnow_3d, qgraup_3d,
         refl_out,
         np.int32(nz), np.int32(ny), np.int32(nx_))
    )
    return refl_out
