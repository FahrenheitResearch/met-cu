"""
MSL (Metal Shading Language) kernel helpers for met-cu.

Provides ElementwiseKernel and RawKernel equivalents that compile and
dispatch Metal compute shaders, mirroring the CuPy API used by the
CUDA kernels.

IMPORTANT: Metal on M1/M2/M3 does NOT support double-precision (float64)
in compute shaders.  All GPU work uses float32.  Python wrappers accept
float64 and convert at the boundary.
"""

from __future__ import annotations

import struct
import numpy as np
from typing import Optional, Tuple, Union

from .runtime import MetalDevice, MetalArray, MetalKernel, metal_device, to_metal


# --------------------------------------------------------------------------
# MSL preamble — common device functions injected into every kernel
# --------------------------------------------------------------------------
MSL_PREAMBLE = r"""
#include <metal_stdlib>
using namespace metal;

// Physical constants (matching CUDA kernels exactly)
constant float RD       = 287.04749;
constant float RV       = 461.52312;
constant float CP_D     = 1004.6662;
constant float G0       = 9.80665;
constant float ROCP     = 0.28571429;
constant float ZEROCNK  = 273.15;
constant float EPS      = 0.62195691;
constant float LV0      = 2500840.0;
constant float LS0      = 2834540.0;
constant float LAPSE_STD = 0.0065;
constant float P0_STD   = 1013.25;
constant float T0_STD   = 288.15;

// Ambaum (2020) SVP constants
constant float SAT_PRESSURE_0C = 611.2;
constant float T0_TRIP  = 273.16;
constant float CP_L     = 4219.4;
constant float CP_V     = 1860.078;
constant float CP_I     = 2090.0;
constant float RV_METPY = 461.52312;

// SVP over liquid water (Pa) — Ambaum (2020)
inline float svp_liquid_pa(float t_k) {
    float latent = LV0 - (CP_L - CP_V) * (t_k - T0_TRIP);
    float heat_pow = (CP_L - CP_V) / RV_METPY;
    float exp_term = (LV0 / T0_TRIP - latent / t_k) / RV_METPY;
    return SAT_PRESSURE_0C * pow(T0_TRIP / t_k, heat_pow) * exp(exp_term);
}

// SVP in hPa from Celsius
inline float svp_hpa(float t_c) {
    return svp_liquid_pa(t_c + ZEROCNK) / 100.0;
}

// Saturation mixing ratio (kg/kg) from p (hPa) and T (C)
inline float sat_mixing_ratio(float p_hpa, float t_c) {
    float es = svp_hpa(t_c);
    float ws = EPS * es / (p_hpa - es);
    return ws > 0.0 ? ws : 0.0;
}

// SHARPpy mixing ratio (g/kg) with Wexler enhancement
inline float vappres_sharppy(float t) {
    float pol = t * (1.1112018e-17f + (t * -3.0994571e-20f));
    pol = t * (2.1874425e-13f + (t * (-1.789232e-15f + pol)));
    pol = t * (4.3884180e-09f + (t * (-2.988388e-11f + pol)));
    pol = t * (7.8736169e-05f + (t * (-6.111796e-07f + pol)));
    pol = 0.99999683f + (t * (-9.082695e-03f + pol));
    float p8 = pol*pol; p8 *= p8; p8 *= p8;
    return 6.1078f / p8;
}

inline float mixratio_gkg(float p, float t) {
    float x = 0.02f * (t - 12.5f + (7500.0f / p));
    float wfw = 1.0f + (0.0000045f * p) + (0.0014f * x * x);
    float fwesw = wfw * vappres_sharppy(t);
    return 621.97f * (fwesw / (p - fwesw));
}

// Virtual temperature (Celsius)
inline float virtual_temp(float t, float p, float td) {
    float w = mixratio_gkg(p, td) / 1000.0f;
    float tk = t + ZEROCNK;
    return tk * (1.0f + 0.61f * w) - ZEROCNK;
}

// Wobus function for moist adiabat
inline float wobf(float t) {
    float tc = t - 20.0f;
    if (tc <= 0.0f) {
        float npol = 1.0f
            + tc * (-8.841660499999999e-3f
                + tc * (1.4714143e-4f
                    + tc * (-9.671989000000001e-7f
                        + tc * (-3.2607217e-8f + tc * (-3.8598073e-10f)))));
        float n2 = npol * npol;
        return 15.13f / (n2 * n2);
    } else {
        float ppol = tc
            * (4.9618922e-07f
                + tc * (-6.1059365e-09f
                    + tc * (3.9401551e-11f
                        + tc * (-1.2588129e-13f + tc * (1.6688280e-16f)))));
        ppol = 1.0f + tc * (3.6182989e-03f + tc * (-1.3603273e-05f + ppol));
        float p2 = ppol * ppol;
        return (29.93f / (p2 * p2)) + (0.96f * tc) - 14.8f;
    }
}

// Saturated lift
inline float satlift(float p, float thetam) {
    if (p >= 1000.0f) return thetam;
    float pwrp = pow(p / 1000.0f, ROCP);
    float t1 = (thetam + ZEROCNK) * pwrp - ZEROCNK;
    float e1 = wobf(t1) - wobf(thetam);
    float rate = 1.0f;
    for (int iter = 0; iter < 7; iter++) {
        if (abs(e1) < 0.001f) break;
        float t2 = t1 - (e1 * rate);
        float e2 = (t2 + ZEROCNK) / pwrp - ZEROCNK;
        e2 += wobf(t2) - wobf(e2) - thetam;
        rate = (t2 - t1) / (e2 - e1);
        t1 = t2;
        e1 = e2;
    }
    return t1 - e1 * rate;
}

// LCL temperature from T, Td (Celsius)
inline float lcltemp(float t, float td) {
    float s = t - td;
    float dlt = s * (1.2185f + 0.001278f * t + s * (-0.00219f + 1.173e-5f * s - 0.0000052f * t));
    return t - dlt;
}

// Dry lift to LCL
inline void drylift(float p, float t, float td,
                    thread float &p_lcl, thread float &t_lcl) {
    t_lcl = lcltemp(t, td);
    p_lcl = 1000.0f * pow((t_lcl + ZEROCNK) / ((t + ZEROCNK) * pow(1000.0f / p, ROCP)), 1.0f / ROCP);
}

// Dewpoint from vapor pressure (hPa)
inline float dewpoint_from_vp(float e_hpa) {
    if (e_hpa <= 0.0f) return -ZEROCNK;
    float ln_ratio = log(e_hpa / 6.112f);
    return 243.5f * ln_ratio / (17.67f - ln_ratio);
}

// Moist lapse rate dT/dp (K/hPa)
inline float moist_lapse_rate(float p_hpa, float t_c) {
    float t_k = t_c + ZEROCNK;
    float es = svp_hpa(t_c);
    float rs = EPS * es / (p_hpa - es);
    if (rs < 0.0f) rs = 0.0f;
    float num = (RD * t_k + LV0 * rs) / p_hpa;
    float den = CP_D + (LV0 * LV0 * rs * EPS) / (RD * t_k * t_k);
    return num / den;
}

// RK4 step for moist adiabat
inline float moist_rk4_step(float p, float t, float dp) {
    float k1 = dp * moist_lapse_rate(p, t);
    float k2 = dp * moist_lapse_rate(p + dp/2.0f, t + k1/2.0f);
    float k3 = dp * moist_lapse_rate(p + dp/2.0f, t + k2/2.0f);
    float k4 = dp * moist_lapse_rate(p + dp, t + k3);
    return t + (k1 + 2.0f*k2 + 2.0f*k3 + k4) / 6.0f;
}

// Finite-difference stencil helpers for grid operations
inline float ddx(device const float* f, device const float* dx_arr,
                 int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dx_arr[idx];
    if (i == 0)
        return (-3.0f*f[idx] + 4.0f*f[idx+1] - f[idx+2]) / (2.0f*h);
    if (i == nx - 1)
        return (3.0f*f[idx] - 4.0f*f[idx-1] + f[idx-2]) / (2.0f*h);
    return (f[idx+1] - f[idx-1]) / (2.0f*h);
}

inline float ddy(device const float* f, device const float* dy_arr,
                 int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dy_arr[idx];
    if (j == 0)
        return (-3.0f*f[idx] + 4.0f*f[idx+nx] - f[idx+2*nx]) / (2.0f*h);
    if (j == ny - 1)
        return (3.0f*f[idx] - 4.0f*f[idx-nx] + f[idx-2*nx]) / (2.0f*h);
    return (f[idx+nx] - f[idx-nx]) / (2.0f*h);
}

inline float d2dx2(device const float* f, device const float* dx_arr,
                   int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dx_arr[idx];
    if (i == 0)
        return (f[idx] - 2.0f*f[idx+1] + f[idx+2]) / (h*h);
    if (i == nx - 1)
        return (f[idx] - 2.0f*f[idx-1] + f[idx-2]) / (h*h);
    return (f[idx+1] - 2.0f*f[idx] + f[idx-1]) / (h*h);
}

inline float d2dy2(device const float* f, device const float* dy_arr,
                   int j, int i, int ny, int nx) {
    int idx = j * nx + i;
    float h = dy_arr[idx];
    if (j == 0)
        return (f[idx] - 2.0f*f[idx+nx] + f[idx+2*nx]) / (h*h);
    if (j == ny - 1)
        return (f[idx] - 2.0f*f[idx-nx] + f[idx-2*nx]) / (h*h);
    return (f[idx+nx] - 2.0f*f[idx] + f[idx-nx]) / (h*h);
}
"""


# --------------------------------------------------------------------------
# MetalElementwiseKernel — drop-in for cp.ElementwiseKernel
# --------------------------------------------------------------------------
class MetalElementwiseKernel:
    """Mimics CuPy's ElementwiseKernel using Metal compute shaders.

    Parameters
    ----------
    in_params : str
        Comma-separated typed input parameters (e.g. "float64 x, float64 y").
    out_params : str
        Comma-separated typed output parameters (e.g. "float64 z").
    operation : str
        CUDA-style C code for the per-element operation.
    name : str
        Kernel function name.
    """

    def __init__(self, in_params: str, out_params: str, operation: str,
                 name: str = "elementwise"):
        self.name = name
        self._in_params = _parse_params(in_params)
        self._out_params = _parse_params(out_params)
        self._operation = operation
        self._kernel: Optional[MetalKernel] = None
        self._source = self._generate_msl()

    def _generate_msl(self) -> str:
        """Generate MSL kernel source from the CUDA operation."""
        all_params = self._in_params + self._out_params
        # Build argument list
        args = []
        for i, (dtype, pname) in enumerate(self._in_params):
            args.append(
                f"device const float* {pname} [[buffer({i})]]")
        out_offset = len(self._in_params)
        for i, (dtype, pname) in enumerate(self._out_params):
            args.append(
                f"device float* {pname} [[buffer({out_offset + i})]]")
        args.append(
            f"device const int* _n_elem [[buffer({len(all_params)})]]")
        args.append(
            "uint i [[thread_position_in_grid]]")
        args_str = ",\n    ".join(args)

        # Convert CUDA operation to MSL
        op = _cuda_to_msl_op(self._operation, self._in_params, self._out_params)

        src = f"""{MSL_PREAMBLE}

kernel void {self.name}(
    {args_str}
) {{
    if (i >= uint(*_n_elem)) return;
    // Map array params to per-element scalars
"""
        # For input params, read from buffer
        for dtype, pname in self._in_params:
            src += f"    float _{pname}_val = {pname}[i];\n"

        # The operation uses the param names directly for outputs
        # We need to handle the CUDA convention where output names are
        # assigned directly
        src += f"\n    // --- kernel operation ---\n"

        # For simple single-line operations, we can handle them directly
        # For multi-line, wrap in a block
        lines = op.strip().split('\n')
        for line in lines:
            src += f"    {line.strip()}\n"

        # Write outputs
        for dtype, pname in self._out_params:
            src += f"    {pname}[i] = {pname};\n"

        src += "}\n"
        return src

    def __call__(self, *args):
        """Execute the elementwise kernel.

        Accepts MetalArrays or numpy arrays.  Returns MetalArray(s).
        """
        dev = metal_device()

        # The actual MSL source needs to be generated properly
        # based on whether inputs are arrays or scalars
        in_arrays = []
        n_elem = 0
        for a in args:
            ma = to_metal(a)
            in_arrays.append(ma)
            n_elem = max(n_elem, ma.size)

        # Allocate outputs
        out_shape = in_arrays[0].shape if in_arrays else (n_elem,)
        out_arrays = [
            MetalArray(shape=out_shape, _device=dev)
            for _ in self._out_params
        ]

        # Build the proper MSL with array indexing
        source = self._build_proper_msl()

        if self._kernel is None:
            self._kernel = dev.compile(source, self.name)

        # Pack n_elem as int32
        n_buf = struct.pack("i", n_elem)

        buffers = in_arrays + out_arrays + [n_buf]
        self._kernel.dispatch(
            buffers,
            grid_size=(n_elem,),
            threadgroup_size=(min(256, n_elem),) if n_elem > 0 else (1,),
        )

        if len(out_arrays) == 1:
            return out_arrays[0]
        return tuple(out_arrays)

    def _build_proper_msl(self) -> str:
        """Build the final MSL source with proper array indexing."""
        args = []
        for i, (dtype, pname) in enumerate(self._in_params):
            args.append(
                f"device const float* buf_{pname} [[buffer({i})]]")
        out_offset = len(self._in_params)
        for i, (dtype, pname) in enumerate(self._out_params):
            args.append(
                f"device float* buf_{pname} [[buffer({out_offset + i})]]")
        n_idx = len(self._in_params) + len(self._out_params)
        args.append(f"device const int* _n_elem [[buffer({n_idx})]]")
        args.append("uint i [[thread_position_in_grid]]")
        args_str = ",\n    ".join(args)

        op = _cuda_to_msl_op(
            self._operation, self._in_params, self._out_params)

        body = f"{MSL_PREAMBLE}\n\nkernel void {self.name}(\n    {args_str}\n) {{\n"
        body += "    if (i >= uint(*_n_elem)) return;\n\n"

        # Read inputs from buffers
        for dtype, pname in self._in_params:
            body += f"    float {pname} = buf_{pname}[i];\n"

        # Declare outputs
        for dtype, pname in self._out_params:
            body += f"    float {pname};\n"

        body += "\n    // --- operation ---\n"
        for line in op.strip().split('\n'):
            body += f"    {line.strip()}\n"

        # Write outputs to buffers
        body += "\n"
        for dtype, pname in self._out_params:
            body += f"    buf_{pname}[i] = {pname};\n"

        body += "}\n"
        return body


# --------------------------------------------------------------------------
# MetalRawKernel — drop-in for cp.RawKernel / cp.RawModule
# --------------------------------------------------------------------------
class MetalRawKernel:
    """Wraps a Metal compute kernel compiled from MSL source.

    Usage mirrors cp.RawKernel: kernel(grid, block, args).
    """

    def __init__(self, source: str, function_name: str):
        self.function_name = function_name
        self._source = source
        self._kernel: Optional[MetalKernel] = None

    def __call__(self, grid, block, args):
        """Dispatch the kernel.

        Parameters
        ----------
        grid : tuple
            Grid dimensions (matching CUDA convention).
        block : tuple
            Block dimensions (used for threadgroup size).
        args : tuple
            Kernel arguments — MetalArrays, numpy scalars, np.int32, etc.
        """
        dev = metal_device()
        if self._kernel is None:
            self._kernel = dev.compile(self._source, self.function_name)

        # Convert args
        buffers = []
        for a in args:
            if isinstance(a, MetalArray):
                buffers.append(a)
            elif isinstance(a, np.ndarray):
                buffers.append(to_metal(a))
            elif isinstance(a, (np.int32, np.intc)):
                buffers.append(int(a))
            elif isinstance(a, (np.float32, np.float64)):
                buffers.append(float(a))
            elif isinstance(a, int):
                buffers.append(a)
            elif isinstance(a, float):
                buffers.append(a)
            else:
                buffers.append(to_metal(a))

        # Compute total threads from grid * block
        gx = grid[0] if len(grid) > 0 else 1
        gy = grid[1] if len(grid) > 1 else 1
        gz = grid[2] if len(grid) > 2 else 1
        bx = block[0] if len(block) > 0 else 1
        by = block[1] if len(block) > 1 else 1
        bz = block[2] if len(block) > 2 else 1

        total = (gx * bx, gy * by, gz * bz)
        tg = (bx, by, bz)

        self._kernel.dispatch(buffers, grid_size=total, threadgroup_size=tg)


class MetalRawModule:
    """Mimics cp.RawModule — compiles MSL source and extracts kernels by name."""

    def __init__(self, code: str):
        self._source = code
        self._kernels: dict[str, MetalRawKernel] = {}

    def get_function(self, name: str) -> MetalRawKernel:
        if name not in self._kernels:
            self._kernels[name] = MetalRawKernel(self._source, name)
        return self._kernels[name]


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def _parse_params(params_str: str) -> list[tuple[str, str]]:
    """Parse CuPy-style param string like 'float64 x, float64 y'."""
    result = []
    for p in params_str.split(","):
        p = p.strip()
        if not p:
            continue
        parts = p.split()
        if len(parts) == 2:
            result.append((parts[0], parts[1]))
        elif len(parts) == 1:
            result.append(("float", parts[0]))
    return result


def _cuda_to_msl_op(cuda_op: str, in_params, out_params) -> str:
    """Convert CUDA C operation snippet to MSL-compatible code."""
    op = cuda_op
    # Replace CUDA types
    op = op.replace("double", "float")
    # M_PI -> M_PI_F in Metal
    op = op.replace("M_PI", "M_PI_F")
    # fabs -> abs (Metal uses abs for float)
    op = op.replace("fabs(", "abs(")
    # __syncthreads() -> threadgroup_barrier(mem_flags::mem_threadgroup)
    op = op.replace("__syncthreads()",
                     "threadgroup_barrier(mem_flags::mem_threadgroup)")
    return op
