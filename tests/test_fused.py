"""Parity + microbenchmark for metcu.fused kernels."""

import numpy as np
import cupy as cp
import time

from metcu.kernels import thermo as metcu
from metcu import fused


RTOL = 1e-12


def _max_relerr(a, b):
    a = cp.asnumpy(a).ravel()
    b = cp.asnumpy(b).ravel()
    denom = np.maximum(np.abs(a), np.abs(b))
    mask = denom > 0
    err = np.zeros_like(a)
    err[mask] = np.abs(a[mask] - b[mask]) / denom[mask]
    return float(err.max())


def _random_inputs(n, seed=0):
    rng = np.random.default_rng(seed)
    p = rng.uniform(200.0, 1050.0, n)
    t = rng.uniform(-60.0, 50.0, n)
    td = t - rng.uniform(0.0, 30.0, n)
    return (cp.asarray(p, dtype=cp.float64),
            cp.asarray(t, dtype=cp.float64),
            cp.asarray(td, dtype=cp.float64))


def verify_theta_bundle():
    p, t, td = _random_inputs(1_000_000, seed=1)

    theta_f, thetav_f, thetae_f = fused.theta_thetav_thetae(p, t, td)

    theta_u = metcu.potential_temperature(p, t)
    w_sat = metcu.saturation_mixing_ratio(p, td)
    thetav_u = metcu.virtual_potential_temperature(p, t, w_sat)
    thetae_u = metcu.equivalent_potential_temperature(p, t, td)

    e1 = _max_relerr(theta_f, theta_u)
    e2 = _max_relerr(thetav_f, thetav_u)
    e3 = _max_relerr(thetae_f, thetae_u)
    print(f"  theta    max relerr = {e1:.2e}")
    print(f"  theta_v  max relerr = {e2:.2e}")
    print(f"  theta_e  max relerr = {e3:.2e}")
    assert e1 <= RTOL and e2 <= RTOL and e3 <= RTOL


def verify_svp_bundle():
    p, t, td = _random_inputs(1_000_000, seed=2)

    es_f, e_f, ws_f, rh_f = fused.svp_e_mr_rh(p, t, td)

    es_u = metcu.saturation_vapor_pressure(t)
    e_u = metcu.vapor_pressure(td)
    ws_u = metcu.saturation_mixing_ratio(p, t)
    rh_u = metcu.relative_humidity_from_dewpoint(t, td)

    e1 = _max_relerr(es_f, es_u)
    e2 = _max_relerr(e_f, e_u)
    e3 = _max_relerr(ws_f, ws_u)
    e4 = _max_relerr(rh_f, rh_u)
    print(f"  es  max relerr = {e1:.2e}")
    print(f"  e   max relerr = {e2:.2e}")
    print(f"  ws  max relerr = {e3:.2e}")
    print(f"  rh  max relerr = {e4:.2e}")
    assert max(e1, e2, e3, e4) <= RTOL


def verify_full_bundle():
    p, t, td = _random_inputs(1_000_000, seed=3)

    theta_f, thetav_f, thetae_f, e_f, es_f, w_f, ws_f, rh_f = \
        fused.t_td_to_thermo_bundle(p, t, td)

    theta_u = metcu.potential_temperature(p, t)
    w_sat = metcu.saturation_mixing_ratio(p, td)
    thetav_u = metcu.virtual_potential_temperature(p, t, w_sat)
    thetae_u = metcu.equivalent_potential_temperature(p, t, td)
    e_u = metcu.vapor_pressure(td)
    es_u = metcu.saturation_vapor_pressure(t)
    w_u = metcu.mixing_ratio(e_u, p)
    ws_u = metcu.saturation_mixing_ratio(p, t)
    rh_u = metcu.relative_humidity_from_dewpoint(t, td)

    errs = {
        'theta':   _max_relerr(theta_f,   theta_u),
        'theta_v': _max_relerr(thetav_f,  thetav_u),
        'theta_e': _max_relerr(thetae_f,  thetae_u),
        'e':       _max_relerr(e_f,       e_u),
        'es':      _max_relerr(es_f,      es_u),
        'w':       _max_relerr(w_f,       w_u),
        'ws':      _max_relerr(ws_f,      ws_u),
        'rh':      _max_relerr(rh_f,      rh_u),
    }
    for k, v in errs.items():
        print(f"  {k:7s} max relerr = {v:.2e}")
    assert max(errs.values()) <= RTOL


def _bench(fn, iters=100):
    # warmup
    for _ in range(5):
        fn()
    cp.cuda.Stream.null.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    cp.cuda.Stream.null.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / iters


def benchmark():
    N = 4_000_000
    p, t, td = _random_inputs(N, seed=42)

    # --- 3-output bundle ---
    def unfused3():
        theta = metcu.potential_temperature(p, t)
        w = metcu.saturation_mixing_ratio(p, td)
        thv = metcu.virtual_potential_temperature(p, t, w)
        the = metcu.equivalent_potential_temperature(p, t, td)
        return theta, thv, the

    def fused3():
        return fused.theta_thetav_thetae(p, t, td)

    t_un3 = _bench(unfused3)
    t_fu3 = _bench(fused3)
    print(f"\n[3-output] unfused: {t_un3:.3f} ms  fused: {t_fu3:.3f} ms  "
          f"speedup: {t_un3/t_fu3:.2f}x")

    # --- 8-output bundle ---
    def unfused8():
        theta = metcu.potential_temperature(p, t)
        w_sat = metcu.saturation_mixing_ratio(p, td)
        thv = metcu.virtual_potential_temperature(p, t, w_sat)
        the = metcu.equivalent_potential_temperature(p, t, td)
        e_ = metcu.vapor_pressure(td)
        es_ = metcu.saturation_vapor_pressure(t)
        w_ = metcu.mixing_ratio(e_, p)
        ws_ = metcu.saturation_mixing_ratio(p, t)
        rh_ = metcu.relative_humidity_from_dewpoint(t, td)
        return theta, thv, the, e_, es_, w_, ws_, rh_

    def fused8():
        return fused.t_td_to_thermo_bundle(p, t, td)

    t_un8 = _bench(unfused8)
    t_fu8 = _bench(fused8)
    print(f"[8-output] unfused: {t_un8:.3f} ms  fused: {t_fu8:.3f} ms  "
          f"speedup: {t_un8/t_fu8:.2f}x")


def test_all():
    print("Parity: theta bundle")
    verify_theta_bundle()
    print("Parity: svp bundle")
    verify_svp_bundle()
    print("Parity: full 8-bundle")
    verify_full_bundle()
    benchmark()


if __name__ == "__main__":
    test_all()
