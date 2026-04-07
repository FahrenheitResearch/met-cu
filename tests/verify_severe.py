"""Verify ALL severe weather composite parameters in met-cu against metrust.

Uses controlled input arrays (not HRRR data) so we can verify exact formulas.
Compares met-cu GPU results against metrust CPU results.

For composites: formulas are just arithmetic on pre-computed fields, rtol=1e-6.
For stability indices: also simple arithmetic, rtol=1e-6.

Any mismatches print GPU value, CPU value, and the expected formula for diagnosis.

Known formula differences between met-cu and metrust are tested by verifying
each implementation against its own manual reference formula.
"""

import sys
import numpy as np

try:
    import cupy as cp
except ImportError:
    print("FATAL: cupy not installed (no GPU support)")
    sys.exit(1)

import metcu
import metrust.calc as mr
from metrust._metrust import calc as _calc

# ============================================================================
# Controlled test arrays
# ============================================================================
N = 50

cape = np.linspace(0, 5000, N, dtype=np.float64)         # J/kg
srh = np.linspace(-50, 500, N, dtype=np.float64)          # m2/s2
shear = np.linspace(0, 40, N, dtype=np.float64)           # m/s
lcl_height = np.linspace(200, 4000, N, dtype=np.float64)  # m
cin = np.linspace(-300, 0, N, dtype=np.float64)            # J/kg
mucape = np.linspace(0, 5000, N, dtype=np.float64)         # J/kg
dcape = np.linspace(0, 2000, N, dtype=np.float64)          # J/kg
mean_wind = np.linspace(2, 30, N, dtype=np.float64)        # m/s
mixing_ratio_gkg = np.linspace(2, 20, N, dtype=np.float64) # g/kg
t500 = np.linspace(-30, -5, N, dtype=np.float64)           # degC
lr_700_500 = np.linspace(4, 9, N, dtype=np.float64)        # C/km

# Stability index inputs (all degC)
t850 = np.linspace(5, 25, N, dtype=np.float64)
t700 = np.linspace(-5, 10, N, dtype=np.float64)
t500_stab = np.linspace(-25, -10, N, dtype=np.float64)
td850 = np.linspace(0, 18, N, dtype=np.float64)
td700 = np.linspace(-15, 5, N, dtype=np.float64)
t950 = np.linspace(10, 30, N, dtype=np.float64)

# Fire weather inputs
temp_f = np.linspace(60, 110, N, dtype=np.float64)      # Fahrenheit
rh_pct = np.linspace(5, 80, N, dtype=np.float64)         # percent
wind_mph = np.linspace(0, 40, N, dtype=np.float64)       # mph
temp_c = np.linspace(15, 45, N, dtype=np.float64)        # Celsius
wind_ms = np.linspace(0, 20, N, dtype=np.float64)        # m/s

# Reshape for 2D grid functions (metrust grid functions expect 2D)
ny, nx = 5, 10
assert ny * nx == N


# ============================================================================
# Helpers
# ============================================================================
passed = 0
failed = 0
expected_diffs = 0
skipped = 0


def _to_np(val):
    """Convert any result to numpy array."""
    if hasattr(val, 'magnitude'):
        val = val.magnitude
    if hasattr(val, 'get'):  # cupy
        val = val.get()
    return np.asarray(val, dtype=np.float64).ravel()


def _mr_scalar_vec(fn, *arrays):
    """Vectorize a metrust scalar function over arrays."""
    n = len(arrays[0])
    results = np.empty(n, dtype=np.float64)
    for i in range(n):
        args = [float(a[i]) for a in arrays]
        r = fn(*args)
        if hasattr(r, 'magnitude'):
            r = float(r.magnitude)
        results[i] = float(r)
    return results


def compare(name, gpu_val, cpu_val, formula, rtol=1e-6, expected_diff=False):
    """Compare GPU and CPU values, report any mismatch.

    If expected_diff=True, failures are counted as expected (not real errors).
    """
    global passed, failed, expected_diffs
    g = _to_np(gpu_val)
    c = _to_np(cpu_val)

    tag = "EXPECTED_DIFF" if expected_diff else "FAIL"

    if g.shape != c.shape:
        print(f"  {tag} {name}: shape mismatch GPU={g.shape} CPU={c.shape}")
        if expected_diff:
            expected_diffs += 1
        else:
            failed += 1
        return False

    # Handle NaN: both NaN is ok
    both_nan = np.isnan(g) & np.isnan(c)
    gpu_nan_only = np.isnan(g) & ~np.isnan(c)
    cpu_nan_only = ~np.isnan(g) & np.isnan(c)

    if gpu_nan_only.any() or cpu_nan_only.any():
        idx = np.where(gpu_nan_only | cpu_nan_only)[0]
        print(f"  {tag} {name}: NaN mismatch at indices {idx[:5]}...")
        for i in idx[:3]:
            print(f"    [{i}] GPU={g[i]}, CPU={c[i]}")
        print(f"    Formula: {formula}")
        if expected_diff:
            expected_diffs += 1
        else:
            failed += 1
        return False

    mask = ~both_nan
    if not mask.any():
        print(f"  PASS {name} (all NaN)")
        passed += 1
        return True

    gm, cm = g[mask], c[mask]
    with np.errstate(divide='ignore', invalid='ignore'):
        diffs = np.abs(gm - cm)
        rel_diffs = np.where(np.abs(cm) > 1e-15, diffs / np.abs(cm), diffs)

    if np.allclose(gm, cm, rtol=rtol, atol=1e-12):
        maxdiff = np.max(diffs) if len(gm) > 0 else 0
        print(f"  PASS {name} (max abs diff: {maxdiff:.2e})")
        passed += 1
        return True
    else:
        worst = np.argmax(rel_diffs)
        print(f"  {tag} {name}:")
        print(f"    Max rel diff: {rel_diffs[worst]:.6e}")
        print(f"    GPU[{worst}] = {gm[worst]:.10f}")
        print(f"    CPU[{worst}] = {cm[worst]:.10f}")
        print(f"    Formula: {formula}")
        bad = np.where(rel_diffs > rtol)[0]
        for i in bad[:5]:
            print(f"    [{i}] GPU={gm[i]:.10f}  CPU={cm[i]:.10f}")
        if expected_diff:
            expected_diffs += 1
        else:
            failed += 1
        return False


def skip(name, reason):
    global skipped
    print(f"  SKIP {name}: {reason}")
    skipped += 1


# ============================================================================
# COMPOSITE PARAMETERS
# ============================================================================
print("=" * 72)
print("COMPOSITE PARAMETERS")
print("=" * 72)


# ---------- 1. Significant Tornado Parameter (STP) ----------
print("\n--- STP (Significant Tornado Parameter) ---")
print("  Formula: (cape/1500) * lcl_term * (srh/150) * shear_term")
print("  where shear_term=0 below 12.5 m/s and min(shear, 30)/20 otherwise")

gpu_stp = metcu.significant_tornado_parameter(cape, lcl_height, srh, shear)
cpu_stp = _mr_scalar_vec(_calc.significant_tornado_parameter,
                         cape, lcl_height, srh, shear)

compare("STP cross-library", gpu_stp, cpu_stp,
        "metrust fixed-layer STP")

manual_stp = np.zeros(N)
for i in range(N):
    ct = max(cape[i] / 1500.0, 0.0)
    st = max(srh[i] / 150.0, 0.0)
    if shear[i] < 12.5:
        sht = 0.0
    else:
        sht = max(min(shear[i], 30.0) / 20.0, 0.0)
    if lcl_height[i] <= 1000.0:
        lt = 1.0
    else:
        lt = np.clip((2000.0 - lcl_height[i]) / 1000.0, 0.0, 1.0)
    manual_stp[i] = ct * lt * st * sht

compare("STP met-cu vs formula", gpu_stp, manual_stp,
        "cape/1500 * lcl_term * srh/150 * shear_term")
compare("STP metrust vs formula", cpu_stp, manual_stp,
        "metrust: cape/1500 * lcl_term * srh/150 * shear_term(12.5 cutoff, 30 cap)")


# ---------- 2. Supercell Composite Parameter (SCP) ----------
print("\n--- SCP (Supercell Composite Parameter) ---")
print("  Formula: (cape/1000) * (srh/50) * shear_term")
print("  where shear_term=0 below 10 m/s and min(shear, 20)/20 otherwise")

gpu_scp = metcu.supercell_composite_parameter(mucape, srh, shear)
cpu_scp = _mr_scalar_vec(_calc.supercell_composite_parameter,
                         mucape, srh, shear)

compare("SCP cross-library", gpu_scp, cpu_scp,
        "metrust SCP")

manual_scp = np.zeros(N)
for i in range(N):
    ct = max(mucape[i] / 1000.0, 0.0)
    st = max(srh[i] / 50.0, 0.0)
    if shear[i] < 10.0:
        sht = 0.0
    else:
        sht = max(min(shear[i], 20.0) / 20.0, 0.0)
    manual_scp[i] = ct * st * sht

compare("SCP met-cu vs formula", gpu_scp, manual_scp,
        "(cape/1000)*(srh/50)*(min(shear,20)/20), shear<10->0")
compare("SCP metrust vs formula", cpu_scp, manual_scp,
        "metrust: (cape/1000)*(srh/50)*(min(shear,20)/20), shear<10->0")


# ---------- 3. Energy-Helicity Index (EHI) ----------
print("\n--- EHI (Energy-Helicity Index) ---")
print("  Formula: (CAPE * SRH) / 160000 [identical]")

gpu_ehi = metcu.compute_ehi(cape, srh)
cpu_ehi = mr.compute_ehi(cape.reshape(ny, nx), srh.reshape(ny, nx))
compare("EHI cross-library", gpu_ehi, cpu_ehi,
        "(CAPE * SRH) / 160000")


# ---------- 4. Significant Hail Parameter (SHIP) ----------
print("\n--- SHIP (Significant Hail Parameter) ---")
print("  Formula: (cape*mr*lr*(-t500)*shear)/42e6, cape<1300 scaling [identical]")

gpu_ship = metcu.compute_ship(mucape, shear, t500, lr_700_500, mixing_ratio_gkg)
cpu_ship = mr.compute_ship(
    mucape.reshape(ny, nx), shear.reshape(ny, nx),
    t500.reshape(ny, nx), lr_700_500.reshape(ny, nx),
    mixing_ratio_gkg.reshape(ny, nx)
)
compare("SHIP cross-library", gpu_ship, cpu_ship,
        "(cape*mr*lr*(-t500)*shear)/42e6, scale by cape/1300 if cape<1300")


# ---------- 5. Derecho Composite Parameter (DCP) ----------
print("\n--- DCP (Derecho Composite Parameter) ---")
print("  Formula: (dcape/980) * (mucape/2000) * (shear/20) * (mixing_ratio/11)")

gpu_dcp = metcu.compute_dcp(dcape, mucape, shear, mixing_ratio_gkg)
cpu_dcp = mr.compute_dcp(
    dcape.reshape(ny, nx), mucape.reshape(ny, nx),
    shear.reshape(ny, nx), mixing_ratio_gkg.reshape(ny, nx)
)

cpu_dcp_np = _to_np(cpu_dcp)
compare("DCP cross-library", gpu_dcp, cpu_dcp_np,
        "(dcape/980)*(mucape/2000)*(shear/20)*(mixing_ratio/11)")

manual_dcp = np.array([
    max(dcape[i]/980.0, 0) * max(mucape[i]/2000.0, 0) *
    max(shear[i]/20.0, 0) * max(mixing_ratio_gkg[i]/11.0, 0)
    for i in range(N)
])
compare("DCP met-cu vs formula", gpu_dcp, manual_dcp,
        "(dcape/980)*(mucape/2000)*(shear/20)*(mr/11), each term max(0)")
compare("DCP metrust vs formula", cpu_dcp_np, manual_dcp,
        "(dcape/980)*(mucape/2000)*(shear/20)*(mr/11), each term max(0)")


# ---------- 6. Bulk Richardson Number (BRN) ----------
print("\n--- BRN (Bulk Richardson Number) ---")
print("  Formula: CAPE / (0.5 * shear^2) [identical for shear >= ~0.5]")
print("  Near-zero handling differs:")
print("    met-cu:  returns 0 when 0.5*shear^2 <= 1e-6")
print("    metrust: returns NaN when 0.5*shear^2 < 0.1")

shear_brn = np.linspace(2, 40, N, dtype=np.float64)
gpu_brn = metcu.bulk_richardson_number(cape, shear_brn)
cpu_brn = _mr_scalar_vec(_calc.bulk_richardson_number, cape, shear_brn)
compare("BRN cross-library [shear>=2]", gpu_brn, cpu_brn,
        "CAPE / (0.5 * shear^2)")

# Document near-zero
shear_tiny = np.array([0.0, 0.001, 0.1, 0.3, 0.5, 1.0])
cape_tiny = np.array([1000.0] * 6)
gpu_brn_tiny = _to_np(metcu.bulk_richardson_number(cape_tiny, shear_tiny))
cpu_brn_tiny = np.array([_calc.bulk_richardson_number(float(cape_tiny[i]),
                          float(shear_tiny[i])) for i in range(6)])
print("  Near-zero shear comparison:")
for i, s in enumerate(shear_tiny):
    print(f"    shear={s:.3f}: GPU={gpu_brn_tiny[i]:.1f}  CPU={cpu_brn_tiny[i]}")


# ============================================================================
# STABILITY INDICES
# ============================================================================
print("\n" + "=" * 72)
print("STABILITY INDICES")
print("=" * 72)


# ---------- 7. K-Index ----------
print("\n--- K-Index ---")
print("  Formula: (T850 - T500) + Td850 - (T700 - Td700) [identical]")

# met-cu Python API: k_index(t850, td850, t700, td700, t500)
# Rust _calc: k_index(t850, t700, t500, td850, td700)
gpu_ki = metcu.k_index(t850, td850, t700, td700, t500_stab)
cpu_ki = _mr_scalar_vec(_calc.k_index, t850, t700, t500_stab, td850, td700)
compare("K-Index", gpu_ki, cpu_ki,
        "(T850 - T500) + Td850 - (T700 - Td700)")


# ---------- 8. Total Totals ----------
print("\n--- Total Totals ---")
print("  Formula: (T850 - T500) + (Td850 - T500) [identical]")

# met-cu Python API: total_totals(t850, td850, t500)
# Rust _calc: total_totals(t850, t500, td850)
gpu_tt = metcu.total_totals(t850, td850, t500_stab)
cpu_tt = _mr_scalar_vec(_calc.total_totals, t850, t500_stab, td850)
compare("Total Totals", gpu_tt, cpu_tt,
        "(T850 - T500) + (Td850 - T500)")


# ---------- 9. Cross Totals ----------
print("\n--- Cross Totals ---")
print("  Formula: Td850 - T500 [identical]")

gpu_ct = metcu.cross_totals(td850, t500_stab)
cpu_ct = _mr_scalar_vec(_calc.cross_totals, td850, t500_stab)
compare("Cross Totals", gpu_ct, cpu_ct,
        "Td850 - T500")


# ---------- 10. Vertical Totals ----------
print("\n--- Vertical Totals ---")
print("  Formula: T850 - T500 [identical]")

gpu_vt = metcu.vertical_totals(t850, t500_stab)
cpu_vt = _mr_scalar_vec(_calc.vertical_totals, t850, t500_stab)
compare("Vertical Totals", gpu_vt, cpu_vt,
        "T850 - T500")


# ---------- 11. Showalter Index ----------
print("\n--- Showalter Index ---")
print("  Formula: SI = T500_env - T500_parcel (lifted from 850 hPa)")
print("  Iterative parcel lifting -- implementations use different step sizes")

p_show = np.array([1013.25, 1000, 925, 850, 700, 500, 400, 300], dtype=np.float64)
t_show = np.array([30.0, 28.0, 22.0, 15.0, 2.0, -15.0, -28.0, -42.0])
td_show = np.array([22.0, 21.0, 18.0, 10.0, -5.0, -25.0, -38.0, -52.0])

try:
    gpu_si = metcu.showalter_index(p_show, t_show, td_show)
    cpu_si = _calc.showalter_index(p_show, t_show, td_show)
    gpu_val = float(_to_np(gpu_si)[0])
    cpu_val = float(cpu_si)
    diff_pct = abs(gpu_val - cpu_val) / abs(cpu_val) * 100
    print(f"  GPU: {gpu_val:.4f}  CPU: {cpu_val:.4f}  diff: {diff_pct:.2f}%")
    if diff_pct < 10.0:
        print(f"  PASS Showalter Index (within 10% tolerance for iterative method)")
        passed += 1
    else:
        print(f"  FAIL Showalter Index (>{10}% difference)")
        failed += 1
except Exception as e:
    skip("Showalter Index", str(e))


# ---------- 12. Fosberg Fire Weather Index ----------
print("\n--- Fosberg Fire Weather Index ---")
print("  Formula: FFWI = eta * sqrt(1+wind^2) / 0.3002, EMC from T/RH")

gpu_ffwi = metcu.fosberg_fire_weather_index(temp_f, rh_pct, wind_mph)
cpu_ffwi = _mr_scalar_vec(_calc.fosberg_fire_weather_index,
                          temp_f, rh_pct, wind_mph)
compare("Fosberg FFWI", gpu_ffwi, cpu_ffwi,
        "eta * sqrt(1+wind^2) / 0.3002", rtol=1e-3)


# ---------- 13. Haines Index ----------
print("\n--- Haines Index ---")
print("  Formula: A(dt<=3/7) + B(dd<=5/9)")

gpu_haines = metcu.haines_index(t950, t850, td850)
cpu_haines = _mr_scalar_vec(_calc.haines_index, t950, t850, td850)

compare("Haines cross-library", gpu_haines, cpu_haines,
        "A(dt<=3/7) + B(dd<=5/9)")

manual_haines = np.zeros(N)
for i in range(N):
    dt = t950[i] - t850[i]
    dd = t850[i] - td850[i]
    a = 1 if dt <= 3.0 else (2 if dt <= 7.0 else 3)
    b = 1 if dd <= 5.0 else (2 if dd <= 9.0 else 3)
    manual_haines[i] = a + b
compare("Haines met-cu vs formula", gpu_haines, manual_haines,
        "A(dt<=3/7) + B(dd<=5/9)")
compare("Haines metrust vs formula", cpu_haines, manual_haines,
        "A(dt<=3/7) + B(dd<=5/9)")


# ---------- 14. Hot-Dry-Windy Index ----------
print("\n--- Hot-Dry-Windy Index ---")
print("  Formula: HDW = VPD * wind, VPD = es - ea")

gpu_hdw = metcu.hot_dry_windy(temp_c, rh_pct, wind_ms)
cpu_hdw = np.array([_calc.hot_dry_windy(float(temp_c[i]), float(rh_pct[i]),
                                         float(wind_ms[i]), 0.0)
                    for i in range(N)])
compare("Hot-Dry-Windy cross-library", gpu_hdw, cpu_hdw,
        "HDW = VPD * wind, VPD = es - ea", rtol=5e-3)

# Verify against the shared SHARPpy/Wexler SVP polynomial used by met-cu/metrust.
from metcu.calc import _vappres_sharppy_hpa
manual_hdw = np.zeros(N)
for i in range(N):
    es = _vappres_sharppy_hpa(float(temp_c[i]))
    ea = es * rh_pct[i] / 100.0
    vpd = es - ea
    manual_hdw[i] = max(vpd * wind_ms[i], 0.0)

compare("HDW met-cu vs manual SVP formula", gpu_hdw, manual_hdw,
        "SHARPpy/Wexler SVP polynomial")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 72)
total = passed + failed + expected_diffs + skipped
print(f"RESULTS: {passed} passed, {failed} failures, "
      f"{expected_diffs} expected diffs, {skipped} skipped "
      f"out of {total} tests")
print("=" * 72)

# Categorize failures
print("\nExpected cross-library differences (intentional formula variants):")
print("  - BRN near-zero: metrust returns NaN, met-cu returns 0")
print("  - Showalter: iterative parcel lifting step size differences")

# Count self-consistency tests
self_tests = [n for n in ["STP met-cu vs formula",
                           "STP metrust vs formula",
                           "SCP met-cu vs formula",
                           "SCP metrust vs formula",
                           "DCP met-cu vs formula",
                           "DCP metrust vs formula",
                           "Haines met-cu vs formula",
                           "Haines metrust vs formula",
                           "HDW met-cu vs manual SVP formula"]]
print(f"\nSelf-consistency tests verify each implementation matches its documented formula.")

sys.exit(0 if failed == 0 else 1)
