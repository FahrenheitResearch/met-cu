"""Benchmark: naive loop vs CUDA Graph replay for launch-bound stencils.

The metcu Python wrappers call ``_broadcast_spacing`` which does
``float(val)`` on a GPU scalar -- a device->host sync that CUDA forbids
during stream capture. So the graph path here uses the documented
workaround: pre-allocate every output buffer and every broadcast spacing
array OUTSIDE the ``with`` block, then call the raw PTX kernels from
``metcu.kernels.grid`` directly.

Run:
    PYTHONPATH=python python tests/bench_graphs.py
"""

from __future__ import annotations

import time

import cupy as cp
import numpy as np

import metcu
from metcu.graph import graph_capture
from metcu.kernels import grid as gk


N = 2048
ITERS = 100
SMALL_N = 256  # launch-overhead-bound regime


def build_inputs():
    rng = cp.random.default_rng(0)
    u = rng.standard_normal((N, N), dtype=cp.float64)
    v = rng.standard_normal((N, N), dtype=cp.float64)
    f = rng.standard_normal((N, N), dtype=cp.float64)
    return u, v, f


def run_sequence_naive(u, v, f, dx, dy):
    """Five launch-bound grid stencils via the public metcu API."""
    vort = metcu.vorticity(u, v, dx, dy)
    div = metcu.divergence(u, v, dx, dy)
    gy, gx = metcu.gradient(f, deltas=(dy, dx))
    lap = metcu.laplacian(f, dx, dy)
    sm = metcu.smooth_gaussian(f, sigma=1.0)
    return vort, div, gx, gy, lap, sm


def preallocate_raw(u, v, f, dx_scalar, dy_scalar):
    """Pre-allocate outputs + broadcast spacing arrays for the raw path."""
    ny, nx = u.shape
    dx2d = cp.full((ny, nx), dx_scalar, dtype=cp.float64)
    dy2d = cp.full((ny, nx), dy_scalar, dtype=cp.float64)

    out_vort = cp.zeros_like(u)
    out_div = cp.zeros_like(u)
    out_gx = cp.zeros_like(f)
    out_gy = cp.zeros_like(f)
    out_lap = cp.zeros_like(f)
    out_sm = cp.zeros_like(f)

    ny32 = np.int32(ny)
    nx32 = np.int32(nx)
    block = (16, 16, 1)
    grid_ = ((nx + 15) // 16, (ny + 15) // 16, 1)

    def launch_all():
        gk._vorticity_kern(grid_, block, (u, v, dx2d, dy2d, out_vort, ny32, nx32))
        gk._divergence_kern(grid_, block, (u, v, dx2d, dy2d, out_div, ny32, nx32))
        gk._first_derivative_x_kern(grid_, block, (f, dx2d, out_gx, ny32, nx32))
        gk._first_derivative_y_kern(grid_, block, (f, dy2d, out_gy, ny32, nx32))
        gk._laplacian_kern(grid_, block, (f, dx2d, dy2d, out_lap, ny32, nx32))
        gk._smooth_gaussian_kern(grid_, block,
                                 (f, out_sm, ny32, nx32,
                                  np.int32(3), np.float64(1.0)))

    outs = (out_vort, out_div, out_gx, out_gy, out_lap, out_sm)
    return launch_all, outs


def time_ms(fn):
    cp.cuda.get_current_stream().synchronize()
    t0 = time.perf_counter()
    fn()
    cp.cuda.get_current_stream().synchronize()
    return (time.perf_counter() - t0) * 1000.0


def main():
    u, v, f = build_inputs()
    dx_scalar = 1000.0
    dy_scalar = 1000.0

    # Warm-up naive wrappers (triggers JIT / PTX load).
    ref = run_sequence_naive(u, v, f, dx_scalar, dy_scalar)
    cp.cuda.get_current_stream().synchronize()

    def naive():
        for _ in range(ITERS):
            run_sequence_naive(u, v, f, dx_scalar, dy_scalar)

    naive_ms = time_ms(naive)

    # Raw-kernel path (also launch-for-launch equivalent on the default stream).
    launch_all, outs = preallocate_raw(u, v, f, dx_scalar, dy_scalar)
    launch_all()  # warm

    def raw():
        for _ in range(ITERS):
            launch_all()

    raw_ms = time_ms(raw)

    # Graph capture of the raw path on a non-blocking stream.
    capture_stream = cp.cuda.Stream(non_blocking=True)
    with graph_capture(stream=capture_stream) as g:
        launch_all()
    g.launch()
    g.synchronize()

    def replay():
        for _ in range(ITERS):
            g.launch()

    graph_ms = time_ms(lambda: (replay(), g.synchronize()))

    # Parity: raw-path outputs (after last replay) vs naive reference.
    g.launch()
    g.synchronize()
    names = ["vorticity", "divergence", "grad_x", "grad_y", "laplacian", "smooth"]
    print(f"grid: {N}x{N}   iters: {ITERS}")
    print(f"naive wrappers : {naive_ms:8.2f} ms  ({naive_ms/ITERS:.3f} ms/iter)")
    print(f"raw loop       : {raw_ms:8.2f} ms  ({raw_ms/ITERS:.3f} ms/iter)")
    print(f"graph replay   : {graph_ms:8.2f} ms  ({graph_ms/ITERS:.3f} ms/iter)")
    print(f"speedup vs naive: {naive_ms / graph_ms:6.2f}x")
    print(f"speedup vs raw  : {raw_ms   / graph_ms:6.2f}x")
    print("parity (captured vs naive wrapper reference):")
    worst = 0.0
    for name, a, b in zip(names, outs, ref):
        d = float(cp.max(cp.abs(a - b)))
        worst = max(worst, d)
        print(f"  {name:10s} max|diff| = {d:.3e}")
    print(f"bit-equal overall: {worst == 0.0}  (worst {worst:.3e})")


def small_grid_demo():
    """Repeat the comparison on a small grid where launch overhead dominates."""
    global N
    saved = N
    N = SMALL_N
    try:
        print()
        print(f"=== Small grid ({SMALL_N}x{SMALL_N}) launch-bound regime ===")
        main()
    finally:
        N = saved


if __name__ == "__main__":
    main()
    small_grid_demo()
