"""Representative met-cu GPU vs MetPy CPU benchmark subset.

This is intentionally smaller than tests/benchmark_complete.py because MetPy
does not expose a drop-in equivalent for every metrust/metcu benchmarked API.
It covers a representative mix of:

- per-element thermo
- 2-D stencil/grid calculations
- single-sounding profile calculations
"""

import json
import os
import sys
import time

import numpy as np

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

try:
    import cupy as cp
except ImportError as exc:
    raise SystemExit("cupy is required to run this benchmark") from exc

try:
    import metpy
    import metpy.calc as mp
    from metpy.units import units
except ImportError as exc:
    raise SystemExit("metpy is required to run this benchmark") from exc

import metcu.calc as mc


def bench(name, gpu_call, cpu_call, reps=3):
    for _ in range(2):
        try:
            gpu_call()
        except Exception:
            pass
    cp.cuda.Device().synchronize()

    gpu_times = []
    for _ in range(reps):
        cp.cuda.Device().synchronize()
        t0 = time.perf_counter()
        gpu_call()
        cp.cuda.Device().synchronize()
        gpu_times.append((time.perf_counter() - t0) * 1000.0)

    cpu_times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        cpu_call()
        cpu_times.append((time.perf_counter() - t0) * 1000.0)

    gpu_ms = min(gpu_times)
    cpu_ms = min(cpu_times)
    speedup = cpu_ms / gpu_ms if gpu_ms > 0 else float("nan")
    return {
        "name": name,
        "gpu_ms": round(gpu_ms, 2),
        "metpy_ms": round(cpu_ms, 2),
        "speedup": round(speedup, 2),
    }


def main():
    props = cp.cuda.runtime.getDeviceProperties(0)
    gpu_name = props["name"].decode()
    gpu_mem_gb = props["totalGlobalMem"] / 1e9

    np.random.seed(42)

    ny, nx = 1059, 1799
    nlevels = 40
    n = ny * nx

    p_sfc = np.full(n, 1000.0, dtype=np.float64)
    t_2d = np.random.randn(n) * 15 + 25
    td_2d = t_2d - np.abs(np.random.randn(n) * 8)

    u_grid = np.ascontiguousarray(np.random.randn(ny, nx) * 10, dtype=np.float64)
    v_grid = np.ascontiguousarray(np.random.randn(ny, nx) * 10, dtype=np.float64)
    t_grid = np.ascontiguousarray(np.random.randn(ny, nx) * 15 + 25, dtype=np.float64)
    dx_scalar = 3000.0
    dy_scalar = 3000.0

    p_snd = np.linspace(1000, 100, nlevels).astype(np.float64)
    t_snd = np.linspace(25, -60, nlevels).astype(np.float64)
    td_snd = np.linspace(20, -65, nlevels).astype(np.float64)
    u_snd = np.linspace(5, 40, nlevels).astype(np.float64)
    v_snd = np.linspace(0, 20, nlevels).astype(np.float64)
    h_snd = np.linspace(0, 16000, nlevels).astype(np.float64)

    vapor_pressure = mp.saturation_vapor_pressure(td_2d * units.degC)
    parcel_prof = mp.parcel_profile(
        p_snd * units.hPa,
        t_snd[0] * units.degC,
        td_snd[0] * units.degC,
    )

    rows = [
        bench(
            "potential_temperature",
            lambda: mc.potential_temperature(p_sfc, t_2d),
            lambda: mp.potential_temperature(p_sfc * units.hPa, t_2d * units.degC),
        ),
        bench(
            "dewpoint",
            lambda: mc.dewpoint(vapor_pressure.magnitude),
            lambda: mp.dewpoint(vapor_pressure),
        ),
        bench(
            "vorticity",
            lambda: mc.vorticity(u_grid, v_grid, dx=dx_scalar, dy=dy_scalar),
            lambda: mp.vorticity(
                u_grid * units("m/s"),
                v_grid * units("m/s"),
                dx=dx_scalar * units.m,
                dy=dy_scalar * units.m,
            ),
        ),
        bench(
            "frontogenesis",
            lambda: mc.frontogenesis(t_grid + 273.15, u_grid, v_grid, dx=dx_scalar, dy=dy_scalar),
            lambda: mp.frontogenesis(
                (t_grid + 273.15) * units.K,
                u_grid * units("m/s"),
                v_grid * units("m/s"),
                dx=dx_scalar * units.m,
                dy=dy_scalar * units.m,
            ),
        ),
        bench(
            "q_vector",
            lambda: mc.q_vector(u_grid, v_grid, t_grid + 273.15, 850.0, dx=dx_scalar, dy=dy_scalar),
            lambda: mp.q_vector(
                u_grid * units("m/s"),
                v_grid * units("m/s"),
                (t_grid + 273.15) * units.K,
                850.0 * units.hPa,
                dx=dx_scalar * units.m,
                dy=dy_scalar * units.m,
            ),
        ),
        bench(
            "lcl",
            lambda: mc.lcl(1000.0, 25.0, 20.0),
            lambda: mp.lcl(1000.0 * units.hPa, 25.0 * units.degC, 20.0 * units.degC),
        ),
        bench(
            "lfc",
            lambda: mc.lfc(p_snd, t_snd, td_snd),
            lambda: mp.lfc(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
        ),
        bench(
            "el",
            lambda: mc.el(p_snd, t_snd, td_snd),
            lambda: mp.el(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
        ),
        bench(
            "parcel_profile",
            lambda: mc.parcel_profile(p_snd, t_snd[0], td_snd[0]),
            lambda: mp.parcel_profile(p_snd * units.hPa, t_snd[0] * units.degC, td_snd[0] * units.degC),
        ),
        bench(
            "moist_lapse",
            lambda: mc.moist_lapse(p_snd, t_snd[0]),
            lambda: mp.moist_lapse(p_snd * units.hPa, t_snd[0] * units.degC),
        ),
        bench(
            "cape_cin",
            lambda: mc.cape_cin(p_snd, t_snd, td_snd),
            lambda: mp.cape_cin(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC, parcel_prof),
        ),
        bench(
            "surface_based_cape_cin",
            lambda: mc.surface_based_cape_cin(p_snd, t_snd, td_snd),
            lambda: mp.surface_based_cape_cin(p_snd * units.hPa, t_snd * units.degC, td_snd * units.degC),
        ),
        bench(
            "precipitable_water",
            lambda: mc.precipitable_water(p_snd, td_snd),
            lambda: mp.precipitable_water(p_snd * units.hPa, td_snd * units.degC),
        ),
        bench(
            "bulk_shear",
            lambda: mc.bulk_shear(u_snd, v_snd, h_snd, bottom=0.0, top=6000.0),
            lambda: mp.bulk_shear(
                p_snd * units.hPa,
                u_snd * units("m/s"),
                v_snd * units("m/s"),
                height=h_snd * units.m,
                bottom=0.0 * units.m,
                depth=6000.0 * units.m,
            ),
        ),
        bench(
            "storm_relative_helicity",
            lambda: mc.storm_relative_helicity(h_snd, u_snd, v_snd, depth=3000.0, storm_u=5.0, storm_v=5.0),
            lambda: mp.storm_relative_helicity(
                h_snd * units.m,
                u_snd * units("m/s"),
                v_snd * units("m/s"),
                depth=3000.0 * units.m,
                storm_u=5.0 * units("m/s"),
                storm_v=5.0 * units("m/s"),
            ),
        ),
        bench(
            "bunkers_storm_motion",
            lambda: mc.bunkers_storm_motion(u_snd, v_snd, h_snd),
            lambda: mp.bunkers_storm_motion(
                p_snd * units.hPa,
                u_snd * units("m/s"),
                v_snd * units("m/s"),
                height=h_snd * units.m,
            ),
        ),
    ]

    print(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")
    print(f"MetPy: {metpy.__version__}")
    print("")
    print("| Function | GPU ms | MetPy CPU ms | Speedup |")
    print("|---|---:|---:|---:|")
    for row in rows:
        print(
            f"| `{row['name']}` | {row['gpu_ms']:.2f} | {row['metpy_ms']:.2f} | {row['speedup']:.2f}x |"
        )
    print("")
    print(json.dumps(rows))


if __name__ == "__main__":
    main()
