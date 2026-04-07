"""Real-world 3-way benchmark: met-cu (GPU PTX) vs metrust-py (Rust CPU) vs MetPy.

Loads actual RAP isobaric data from ``rap.grib2`` and runs each library on
identical inputs. Reports wall-clock ms (best-of-3 after warmup) and a parity
delta against MetPy as the reference.

Run from repo root:
    PYTHONPATH="python;C:/Users/drew/metrust-py/python" python tests/benchmark_three_way.py
"""

from __future__ import annotations

import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import numpy as np

if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

import cupy as cp
import xarray as xr

import metpy
import metpy.calc as mp
from metpy.units import units

import metcu
import metcu.calc as mc

import metrust
import metrust.calc as mr
from metrust.units import units as mr_units

# ---------------------------------------------------------------------------
# Load real RAP data
# ---------------------------------------------------------------------------
print("Loading rap.grib2 ...")
ds = xr.open_dataset(
    "rap.grib2",
    engine="cfgrib",
    backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
)

levels_raw = ds["isobaricInhPa"].values.astype(np.float64)  # hPa, possibly unsorted
T_full = ds["t"].values.astype(np.float64) - 273.15           # (40, 794802) C
TD_full = ds["dpt"].values.astype(np.float64) - 273.15
U_full = ds["u"].values.astype(np.float64)
V_full = ds["v"].values.astype(np.float64)
GH_full = ds["gh"].values.astype(np.float64)

# Surface-first ordering for sounding ops
order = np.argsort(-levels_raw)
levels = levels_raw[order]
T_full = T_full[order]
TD_full = TD_full[order]
U_full = U_full[order]
V_full = V_full[order]
GH_full = GH_full[order]

NLEV, NPTS = T_full.shape
print(f"  RAP: {NLEV} levels, {NPTS} grid points  ({NLEV*NPTS/1e6:.1f}M elements/level)")
print(f"  P range {levels.min():.0f}-{levels.max():.0f} hPa")

# ---------------------------------------------------------------------------
# Working slices used by each test
# ---------------------------------------------------------------------------
# Elementwise: full surface field flattened (~800k points)
P_sfc = np.full(NPTS, levels[0], dtype=np.float64)
T_sfc = T_full[0].copy()
TD_sfc = TD_full[0].copy()
U_sfc = U_full[0].copy()
V_sfc = V_full[0].copy()

# 2D grid for stencils: fold the 1D points into a square slice
side = int(np.floor(np.sqrt(NPTS)))
side = (side // 32) * 32  # round down to a tidy multiple
T_grid = T_sfc[: side * side].reshape(side, side).copy()
U_grid = U_sfc[: side * side].reshape(side, side).copy()
V_grid = V_sfc[: side * side].reshape(side, side).copy()
DX = 13000.0  # RAP nominal grid spacing m
DY = 13000.0
print(f"  2D stencil slice: {side}x{side} ({side*side/1e3:.0f}k cells)")

# Soundings: 50 random columns
rng = np.random.default_rng(2026)
COL_IDX = rng.choice(NPTS, size=50, replace=False)
T_cols = np.ascontiguousarray(T_full[:, COL_IDX].T)   # (50, NLEV)
TD_cols = np.ascontiguousarray(TD_full[:, COL_IDX].T)
U_cols = np.ascontiguousarray(U_full[:, COL_IDX].T)
V_cols = np.ascontiguousarray(V_full[:, COL_IDX].T)
GH_cols = np.ascontiguousarray(GH_full[:, COL_IDX].T)
print(f"  Soundings: {len(COL_IDX)} columns x {NLEV} levels")

# Pick one representative column for single-sounding tests
SND_T = T_cols[0]
SND_TD = TD_cols[0]
SND_U = U_cols[0]
SND_V = V_cols[0]
SND_H = GH_cols[0] - GH_cols[0, 0]  # heights AGL
SND_P = levels.copy()

# Pre-stage GPU copies once so allocation isn't on the hot path
gpu_P_sfc = cp.asarray(P_sfc); gpu_T_sfc = cp.asarray(T_sfc); gpu_TD_sfc = cp.asarray(TD_sfc)
gpu_U_sfc = cp.asarray(U_sfc); gpu_V_sfc = cp.asarray(V_sfc)
gpu_T_grid = cp.asarray(T_grid); gpu_U_grid = cp.asarray(U_grid); gpu_V_grid = cp.asarray(V_grid)
gpu_levels = cp.asarray(levels); gpu_T_cols = cp.asarray(T_cols); gpu_TD_cols = cp.asarray(TD_cols)


# ---------------------------------------------------------------------------
# Bench harness: best-of-N after 2-iter warmup
# ---------------------------------------------------------------------------
REPS = 5


def _time(fn, sync_gpu=False):
    times = []
    for _ in range(2):
        try:
            fn()
        except Exception:
            return None
    if sync_gpu:
        cp.cuda.Stream.null.synchronize()
    for _ in range(REPS):
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        t0 = time.perf_counter()
        try:
            fn()
        except Exception:
            return None
        if sync_gpu:
            cp.cuda.Stream.null.synchronize()
        times.append((time.perf_counter() - t0) * 1000.0)
    return min(times)


def _to_np(x):
    if x is None:
        return None
    if isinstance(x, tuple):
        return tuple(_to_np(v) for v in x)
    if isinstance(x, cp.ndarray):
        return cp.asnumpy(x)
    if hasattr(x, "magnitude"):
        return np.asarray(x.magnitude)
    return np.asarray(x)


def _maxrel(a, b):
    """Return max relative error between two array-likes (after .magnitude / .get)."""
    try:
        a = _to_np(a)
        b = _to_np(b)
        if isinstance(a, tuple):
            return max(_maxrel(x, y) for x, y in zip(a, b))
        a = np.atleast_1d(np.asarray(a, dtype=np.float64))
        b = np.atleast_1d(np.asarray(b, dtype=np.float64))
        a = a.ravel()
        b = b.ravel()
        n = min(a.size, b.size)
        a = a[:n]; b = b[:n]
        denom = np.maximum(np.abs(b), 1e-12)
        rel = np.abs(a - b) / denom
        rel = rel[np.isfinite(rel)]
        return float(rel.max()) if rel.size else float("nan")
    except Exception:
        return float("nan")


# ---------------------------------------------------------------------------
# Test definitions
# ---------------------------------------------------------------------------
# Each entry: (name, metcu_call, metrust_call, metpy_call)
# All calls are zero-arg lambdas that return their result.

tests = []

# --- Elementwise on 800k surface field ---
tests.append((
    "potential_temperature [800k]",
    lambda: mc.potential_temperature(gpu_P_sfc, gpu_T_sfc + 273.15),
    lambda: mr.potential_temperature(P_sfc * mr_units.hPa, (T_sfc + 273.15) * mr_units.K),
    lambda: mp.potential_temperature(P_sfc * units.hPa, (T_sfc + 273.15) * units.K),
))

tests.append((
    "saturation_vapor_pressure [800k]",
    lambda: mc.saturation_vapor_pressure(gpu_T_sfc + 273.15),
    lambda: mr.saturation_vapor_pressure((T_sfc + 273.15) * mr_units.K),
    lambda: mp.saturation_vapor_pressure((T_sfc + 273.15) * units.K),
))

tests.append((
    "dewpoint_from_relative_humidity [800k]",
    lambda: mc.dewpoint_from_relative_humidity(gpu_T_sfc + 273.15, cp.full_like(gpu_T_sfc, 70.0)),
    lambda: mr.dewpoint_from_relative_humidity((T_sfc + 273.15) * mr_units.K, np.full_like(T_sfc, 70.0) * mr_units.percent),
    lambda: mp.dewpoint_from_relative_humidity((T_sfc + 273.15) * units.K, np.full_like(T_sfc, 70.0) * units.percent),
))

tests.append((
    "wind_speed [800k]",
    lambda: mc.wind_speed(gpu_U_sfc, gpu_V_sfc),
    lambda: mr.wind_speed(U_sfc * mr_units("m/s"), V_sfc * mr_units("m/s")),
    lambda: mp.wind_speed(U_sfc * units("m/s"), V_sfc * units("m/s")),
))

tests.append((
    "equivalent_potential_temperature [800k]",
    lambda: mc.equivalent_potential_temperature(gpu_P_sfc, gpu_T_sfc + 273.15, gpu_TD_sfc + 273.15),
    lambda: mr.equivalent_potential_temperature(P_sfc * mr_units.hPa, (T_sfc + 273.15) * mr_units.K, (TD_sfc + 273.15) * mr_units.K),
    lambda: mp.equivalent_potential_temperature(P_sfc * units.hPa, (T_sfc + 273.15) * units.K, (TD_sfc + 273.15) * units.K),
))

# --- 2D stencils ---
tests.append((
    f"vorticity [{side}x{side}]",
    lambda: mc.vorticity(gpu_U_grid, gpu_V_grid, dx=DX, dy=DY),
    lambda: mr.vorticity(U_grid * mr_units("m/s"), V_grid * mr_units("m/s"), dx=DX * mr_units.m, dy=DY * mr_units.m),
    lambda: mp.vorticity(U_grid * units("m/s"), V_grid * units("m/s"), dx=DX * units.m, dy=DY * units.m),
))

tests.append((
    f"divergence [{side}x{side}]",
    lambda: mc.divergence(gpu_U_grid, gpu_V_grid, dx=DX, dy=DY),
    lambda: mr.divergence(U_grid * mr_units("m/s"), V_grid * mr_units("m/s"), dx=DX * mr_units.m, dy=DY * mr_units.m),
    lambda: mp.divergence(U_grid * units("m/s"), V_grid * units("m/s"), dx=DX * units.m, dy=DY * units.m),
))

tests.append((
    f"laplacian [{side}x{side}]",
    lambda: mc.laplacian(gpu_T_grid + 273.15, dx=DX, dy=DY),
    lambda: mr.laplacian((T_grid + 273.15) * mr_units.K, deltas=[DY * mr_units.m, DX * mr_units.m]),
    lambda: mp.laplacian((T_grid + 273.15) * units.K, deltas=[DY * units.m, DX * units.m]),
))

tests.append((
    f"frontogenesis [{side}x{side}]",
    lambda: mc.frontogenesis(gpu_T_grid + 273.15, gpu_U_grid, gpu_V_grid, dx=DX, dy=DY),
    lambda: mr.frontogenesis((T_grid + 273.15) * mr_units.K, U_grid * mr_units("m/s"), V_grid * mr_units("m/s"), dx=DX * mr_units.m, dy=DY * mr_units.m),
    lambda: mp.frontogenesis((T_grid + 273.15) * units.K, U_grid * units("m/s"), V_grid * units("m/s"), dx=DX * units.m, dy=DY * units.m),
))

# --- Sounding (single column) ---
tests.append((
    "lcl [single sounding]",
    lambda: mc.lcl(SND_P[0], SND_T[0] + 273.15, SND_TD[0] + 273.15),
    lambda: mr.lcl(SND_P[0] * mr_units.hPa, (SND_T[0] + 273.15) * mr_units.K, (SND_TD[0] + 273.15) * mr_units.K),
    lambda: mp.lcl(SND_P[0] * units.hPa, (SND_T[0] + 273.15) * units.K, (SND_TD[0] + 273.15) * units.K),
))

tests.append((
    "parcel_profile [40 lev]",
    lambda: mc.parcel_profile(SND_P, SND_T[0] + 273.15, SND_TD[0] + 273.15),
    lambda: mr.parcel_profile(SND_P * mr_units.hPa, (SND_T[0] + 273.15) * mr_units.K, (SND_TD[0] + 273.15) * mr_units.K),
    lambda: mp.parcel_profile(SND_P * units.hPa, (SND_T[0] + 273.15) * units.K, (SND_TD[0] + 273.15) * units.K),
))

tests.append((
    "surface_based_cape_cin [40 lev]",
    lambda: mc.surface_based_cape_cin(SND_P, SND_T + 273.15, SND_TD + 273.15),
    lambda: mr.surface_based_cape_cin(SND_P * mr_units.hPa, (SND_T + 273.15) * mr_units.K, (SND_TD + 273.15) * mr_units.K),
    lambda: mp.surface_based_cape_cin(SND_P * units.hPa, (SND_T + 273.15) * units.K, (SND_TD + 273.15) * units.K),
))

tests.append((
    "precipitable_water [40 lev]",
    lambda: mc.precipitable_water(SND_P, SND_TD + 273.15),
    lambda: mr.precipitable_water(SND_P * mr_units.hPa, (SND_TD + 273.15) * mr_units.K),
    lambda: mp.precipitable_water(SND_P * units.hPa, (SND_TD + 273.15) * units.K),
))

# --- Sounding (50 columns of real RAP) ---
# metcu can do these in a single batched call; metpy/metrust loop in Python.
def _metpy_50_cape():
    out = np.zeros((50, 2))
    for i in range(50):
        try:
            c, ci = mp.surface_based_cape_cin(
                SND_P * units.hPa,
                (T_cols[i] + 273.15) * units.K,
                (TD_cols[i] + 273.15) * units.K,
            )
            out[i] = (float(c.magnitude), float(ci.magnitude))
        except Exception:
            out[i] = (np.nan, np.nan)
    return out

def _metrust_50_cape():
    out = np.zeros((50, 2))
    for i in range(50):
        try:
            c, ci = mr.surface_based_cape_cin(
                SND_P * mr_units.hPa,
                (T_cols[i] + 273.15) * mr_units.K,
                (TD_cols[i] + 273.15) * mr_units.K,
            )
            out[i] = (float(c.magnitude if hasattr(c, "magnitude") else c),
                      float(ci.magnitude if hasattr(ci, "magnitude") else ci))
        except Exception:
            out[i] = (np.nan, np.nan)
    return out

tests.append((
    "cape_cin x50 cols [batched]",
    lambda: metcu.kernels.cape_cin(gpu_levels, gpu_T_cols + 273.15, gpu_TD_cols + 273.15),
    _metrust_50_cape,
    _metpy_50_cape,
))


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
print()
props = cp.cuda.runtime.getDeviceProperties(0)
gpu_name = props["name"].decode()
print(f"GPU:     {gpu_name}")
print(f"metcu:   PTX runtime ({len(list((__import__('pathlib').Path('python/metcu/ptx').glob('*.ptx'))))} PTX files)")
print(f"metrust: {metrust.__file__}")
print(f"MetPy:   {metpy.__version__}")
print()

# Header
hdr = f"{'function':40s}  {'metcu(ms)':>10s}  {'metrust(ms)':>11s}  {'metpy(ms)':>10s}  {'spd-vs-mp':>10s}  {'spd-vs-mr':>10s}  {'maxrel':>10s}"
print(hdr)
print("-" * len(hdr))

results = []
for name, gpu_fn, mr_fn, mp_fn in tests:
    sync = True
    t_metcu = _time(gpu_fn, sync_gpu=sync)
    t_metrust = _time(mr_fn, sync_gpu=False)
    t_metpy = _time(mp_fn, sync_gpu=False)

    # Parity vs metpy
    try:
        ref = mp_fn()
        if t_metcu is not None:
            actual = gpu_fn()
            mr_err = _maxrel(actual, ref)
        else:
            mr_err = float("nan")
    except Exception:
        mr_err = float("nan")

    spd_mp = (t_metpy / t_metcu) if (t_metcu and t_metpy) else float("nan")
    spd_mr = (t_metrust / t_metcu) if (t_metcu and t_metrust) else float("nan")

    def _f(v, w=10):
        if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
            return f"{'--':>{w}s}"
        return f"{v:>{w}.2f}"

    print(f"{name:40s}  {_f(t_metcu)}  {_f(t_metrust,11)}  {_f(t_metpy)}  {_f(spd_mp)}x  {_f(spd_mr)}x  {_f(mr_err)}")
    results.append({
        "name": name,
        "metcu_ms": t_metcu,
        "metrust_ms": t_metrust,
        "metpy_ms": t_metpy,
        "speedup_vs_metpy": spd_mp,
        "speedup_vs_metrust": spd_mr,
        "maxrel_vs_metpy": mr_err,
    })

print()
print("Notes:")
print("  - metcu times include GPU sync. Best-of-5 after warmup.")
print("  - 'maxrel' = max relative error vs MetPy (NaN if call failed).")
print("  - cape_cin x50 cols: metcu does it in one batched kernel call; metrust/metpy loop 50x in Python.")
