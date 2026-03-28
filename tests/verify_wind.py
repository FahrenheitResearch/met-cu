"""
Verification: met-cu (GPU/CUDA) vs metrust (CPU/Rust) wind & shear functions
on REAL HRRR data (2026-03-27 18:00Z).

Tests:
  1. Per-element wind functions at full grid (~1.9M points):
     - wind_speed, wind_direction, wind_components (round-trip)
  2. Column wind functions on 50 random sounding profiles:
     - bulk_shear (0-6 km), storm_relative_helicity (0-3 km),
       bunkers_storm_motion, mean_wind (0-6 km)
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import cupy as cp
import time
import sys

# ======================================================================
# Download real HRRR data
# ======================================================================
print("=" * 78)
print("  WIND VERIFICATION: met-cu (GPU) vs metrust (CPU) on REAL HRRR")
print("=" * 78)

from rusbie import Herbie

print("\n[1/5] Downloading HRRR data (2026-03-27 18:00Z) ...")
t0 = time.time()

H_sfc = Herbie("2026-03-27 18:00", model="hrrr", product="sfc", fxx=0, verbose=False)
H_prs = Herbie("2026-03-27 18:00", model="hrrr", product="prs", fxx=0, verbose=False)


def get_field(H, search):
    """Read a single 2-D GRIB field."""
    try:
        ds = H.xarray(search)
        if isinstance(ds, list):
            ds = ds[0]
        var = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if var:
            return ds[var[0]].values
    except Exception:
        pass
    return None


def read_prs_var(H, search):
    """Download a single pressure-level variable, returning (data, plevels)."""
    ds = H.xarray(
        search + ".*mb",
        backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}},
    )
    if isinstance(ds, list):
        ds = ds[0]
    var = list(ds.data_vars)[0]
    pres_coord = [c for c in ds.coords if "isobaric" in c.lower()][0]
    plevs = ds[pres_coord].values.copy().astype(np.float64)
    if plevs.max() > 2000:
        plevs = plevs / 100.0
    sort_idx = np.argsort(plevs)[::-1]
    return ds[var].values[sort_idx], plevs[sort_idx]


# Surface fields
u10 = get_field(H_sfc, "UGRD:10 m")
v10 = get_field(H_sfc, "VGRD:10 m")
h_sfc = get_field(H_sfc, "HGT:surface")
assert u10 is not None and v10 is not None, "Failed to download 10-m winds"
assert h_sfc is not None, "Failed to download surface height"

# Pressure-level profiles
# NOTE: Use a separate Herbie instance for HGT to avoid cfgrib caching
# issues where loading UGRD/VGRD first can corrupt the HGT level count.
print("  Reading pressure-level profiles ...")
prs_data = {}
prs_levels = {}

for search in [":UGRD:", ":VGRD:"]:
    data, levels = read_prs_var(H_prs, search)
    prs_data[search] = data
    prs_levels[search] = levels
    print(f"    {search}: {len(levels)} levels, "
          f"{levels[-1]:.0f}-{levels[0]:.0f} hPa")

# Fresh Herbie instance for HGT
H_prs2 = Herbie("2026-03-27 18:00", model="hrrr", product="prs", fxx=0, verbose=False)
data, levels = read_prs_var(H_prs2, ":HGT:")
prs_data[":HGT:"] = data
prs_levels[":HGT:"] = levels
print(f"    :HGT:: {len(levels)} levels, "
      f"{levels[-1]:.0f}-{levels[0]:.0f} hPa")

# Find common pressure levels (round to 0.5 hPa to handle float issues)
def _round_levels(p):
    return set(np.round(p * 2) / 2)  # round to nearest 0.5

all_sets = [_round_levels(prs_levels[s]) for s in [":UGRD:", ":VGRD:", ":HGT:"]]
common_set = all_sets[0].intersection(*all_sets[1:])
common_p = np.sort(np.array(list(common_set)))[::-1]
print(f"    Common levels: {len(common_p)}, "
      f"{common_p[0]:.0f}-{common_p[-1]:.0f} hPa")

# Subset to common levels
for search in [":UGRD:", ":VGRD:", ":HGT:"]:
    full_p_r = np.round(prs_levels[search] * 2) / 2
    idx = np.array([np.argmin(np.abs(full_p_r - plev)) for plev in common_p])
    prs_data[search] = prs_data[search][idx]
prs_data["p_levels"] = common_p

NY, NX = u10.shape
N = NY * NX
NLEV = len(common_p)
print(f"    Grid: {NY}x{NX} = {N:,}, {NLEV} levels")
print(f"    Downloaded in {time.time() - t0:.0f}s")

# Reshape: (nlev, ny, nx) -> (ncols, nlev)
p_levels = prs_data["p_levels"].astype(np.float64)
u_3d = prs_data[":UGRD:"].reshape(NLEV, -1).T.astype(np.float64)
v_3d = prs_data[":VGRD:"].reshape(NLEV, -1).T.astype(np.float64)
h_3d = prs_data[":HGT:"].reshape(NLEV, -1).T.astype(np.float64)

# Heights to AGL
h_sfc_flat = h_sfc.ravel().astype(np.float64)
h_3d_agl = h_3d - h_sfc_flat[:, np.newaxis]

# Surface winds flat
u10_flat = u10.ravel().astype(np.float64)
v10_flat = v10.ravel().astype(np.float64)

# Print diagnostics
print(f"    AGL height range: [{h_3d_agl.min():.0f}, {h_3d_agl.max():.0f}] m")
print(f"    AGL first level: median={np.median(h_3d_agl[:, 0]):.0f} m")
print(f"    AGL last level:  median={np.median(h_3d_agl[:, -1]):.0f} m")

passed = 0
failed = 0
total_tests = 0


def report(name, ok, detail=""):
    global passed, failed, total_tests
    total_tests += 1
    if ok:
        passed += 1
        print(f"    PASS  {name}  {detail}")
    else:
        failed += 1
        print(f"    FAIL  {name}  {detail}")


# ======================================================================
# SECTION 1: Per-element wind functions at full grid
# ======================================================================
print(f"\n[2/5] Per-element wind functions ({N:,} points) ...")

import metcu
import metrust.calc as mr

# --- wind_speed ---
gpu_ws = cp.asnumpy(metcu.wind_speed(u10_flat, v10_flat))
cpu_ws = np.asarray(mr.wind_speed(u10_flat, v10_flat).magnitude)
diff_ws = np.abs(gpu_ws - cpu_ws)
report("wind_speed", diff_ws.max() < 1e-10,
       f"max_diff={diff_ws.max():.2e}, mean={diff_ws.mean():.2e}")

# --- wind_direction ---
gpu_wd = cp.asnumpy(metcu.wind_direction(u10_flat, v10_flat))
cpu_wd = np.asarray(mr.wind_direction(u10_flat, v10_flat).magnitude)
diff_wd = np.minimum(np.abs(gpu_wd - cpu_wd), 360 - np.abs(gpu_wd - cpu_wd))
report("wind_direction", diff_wd.max() < 1e-8,
       f"max_diff={diff_wd.max():.2e}, mean={diff_wd.mean():.2e}")

# --- wind_components (round-trip) ---
gpu_u, gpu_v = metcu.wind_components(gpu_ws, gpu_wd)
gpu_u = cp.asnumpy(gpu_u)
gpu_v = cp.asnumpy(gpu_v)
cpu_u, cpu_v = mr.wind_components(cpu_ws, cpu_wd)
cpu_u = np.asarray(cpu_u.magnitude)
cpu_v = np.asarray(cpu_v.magnitude)
diff_uv = max(np.abs(gpu_u - cpu_u).max(), np.abs(gpu_v - cpu_v).max())
report("wind_components (GPU vs CPU)", diff_uv < 1e-8,
       f"max_diff={diff_uv:.2e}")

# Round-trip: u,v -> speed,dir -> u,v should recover originals
rt_diff = max(np.abs(gpu_u - u10_flat).max(), np.abs(gpu_v - v10_flat).max())
report("wind_components (u/v recovery)", rt_diff < 1e-8,
       f"max_diff_vs_original={rt_diff:.2e}")


# ======================================================================
# SECTION 2: Column wind functions on 50 random soundings
# ======================================================================
print(f"\n[3/5] Column wind functions (50 random profiles) ...")

np.random.seed(42)
# Valid columns: first AGL level near surface, last level > 8 km
valid_mask = (
    (h_3d_agl[:, 0] >= -100) &
    (h_3d_agl[:, 0] < 1000) &
    (h_3d_agl[:, -1] > 8000) &
    np.all(np.diff(h_3d_agl, axis=1) > 0, axis=1)  # monotonically increasing
)
valid_idx = np.where(valid_mask)[0]
n_test = min(50, len(valid_idx))
if n_test == 0:
    print("    ERROR: No valid columns found for column tests!")
    print(f"    AGL[:, 0] stats: min={h_3d_agl[:, 0].min():.0f}, "
          f"max={h_3d_agl[:, 0].max():.0f}, "
          f"median={np.median(h_3d_agl[:, 0]):.0f}")
    print(f"    AGL[:, -1] stats: min={h_3d_agl[:, -1].min():.0f}, "
          f"max={h_3d_agl[:, -1].max():.0f}")
    # Relax criteria
    valid_mask2 = (
        (h_3d_agl[:, 0] < 2000) &
        (h_3d_agl[:, -1] > 6000)
    )
    valid_idx = np.where(valid_mask2)[0]
    n_test = min(50, len(valid_idx))
    print(f"    Relaxed criteria: {len(valid_idx):,} valid columns")

test_cols = np.random.choice(valid_idx, size=n_test, replace=False)
print(f"    Valid columns: {len(valid_idx):,}, testing {n_test}")

# Helper to extract float from GPU/CPU results
def _f(val):
    if hasattr(val, 'get'):
        return float(cp.asnumpy(val))
    if hasattr(val, 'magnitude'):
        return float(val.magnitude)
    return float(val)

# Collect results
bs_gpu_u, bs_gpu_v, bs_cpu_u, bs_cpu_v = [], [], [], []
mw_gpu_u, mw_gpu_v, mw_cpu_u, mw_cpu_v = [], [], [], []
bk_gpu_rm, bk_cpu_rm, bk_gpu_lm, bk_cpu_lm = [], [], [], []
srh_gpu_tot, srh_cpu_tot = [], []

for col_idx in test_cols:
    u_col = u_3d[col_idx]
    v_col = v_3d[col_idx]
    h_col = h_3d_agl[col_idx]

    # --- bulk_shear (0-6 km) ---
    gbs = metcu.bulk_shear(u_col, v_col, h_col, bottom=0, top=6000)
    cbs = mr.bulk_shear(u_col, v_col, h_col, bottom=0, top=6000)
    bs_gpu_u.append(_f(gbs[0])); bs_gpu_v.append(_f(gbs[1]))
    bs_cpu_u.append(_f(cbs[0])); bs_cpu_v.append(_f(cbs[1]))

    # --- mean_wind (0-6 km) ---
    gmw = metcu.mean_wind(u_col, v_col, h_col, 0, 6000)
    cmw = mr.mean_wind(u_col, v_col, h_col, 0, 6000)
    mw_gpu_u.append(_f(gmw[0])); mw_gpu_v.append(_f(gmw[1]))
    mw_cpu_u.append(_f(cmw[0])); mw_cpu_v.append(_f(cmw[1]))

    # --- bunkers_storm_motion ---
    gbk = metcu.bunkers_storm_motion(u_col, v_col, h_col)
    cbk = mr.bunkers_storm_motion(u_col, v_col, h_col)
    (gru, grv), (glu, glv), _ = gbk
    (cru, crv), (clu, clv), _ = cbk
    gru, grv = _f(gru), _f(grv)
    glu, glv = _f(glu), _f(glv)
    cru, crv = _f(cru), _f(crv)
    clu, clv = _f(clu), _f(clv)
    bk_gpu_rm.append((gru, grv)); bk_cpu_rm.append((cru, crv))
    bk_gpu_lm.append((glu, glv)); bk_cpu_lm.append((clu, clv))

    # --- storm_relative_helicity (0-3 km) ---
    # Each impl uses its own Bunkers RM
    # met-cu 4-arg form: (height, u, v, depth) with keyword storm_u/storm_v
    gsrh = metcu.storm_relative_helicity(
        h_col, u_col, v_col, 3000.0, storm_u=gru, storm_v=grv)
    csrh = mr.storm_relative_helicity(
        h_col, u_col, v_col, depth=3000.0, storm_u=cru, storm_v=crv)
    srh_gpu_tot.append(_f(gsrh[2]))
    srh_cpu_tot.append(_f(csrh[2]))


# ======================================================================
# SECTION 3: Analyze column function results
# ======================================================================
print(f"\n[4/5] Analyzing column function results ...")

# --- bulk_shear ---
bs_gu, bs_gv = np.array(bs_gpu_u), np.array(bs_gpu_v)
bs_cu, bs_cv = np.array(bs_cpu_u), np.array(bs_cpu_v)
bs_max = max(np.abs(bs_gu - bs_cu).max(), np.abs(bs_gv - bs_cv).max())
bs_gmag = np.sqrt(bs_gu**2 + bs_gv**2)
bs_cmag = np.sqrt(bs_cu**2 + bs_cv**2)
bs_corr = np.corrcoef(bs_gmag, bs_cmag)[0, 1]
report("bulk_shear (0-6km)", bs_max < 1.0,
       f"max_diff={bs_max:.4f} m/s, corr(mag)={bs_corr:.6f}")

# --- mean_wind ---
mw_gu, mw_gv = np.array(mw_gpu_u), np.array(mw_gpu_v)
mw_cu, mw_cv = np.array(mw_cpu_u), np.array(mw_cpu_v)
mw_max = max(np.abs(mw_gu - mw_cu).max(), np.abs(mw_gv - mw_cv).max())
mw_gmag = np.sqrt(mw_gu**2 + mw_gv**2)
mw_cmag = np.sqrt(mw_cu**2 + mw_cv**2)
mw_corr = np.corrcoef(mw_gmag, mw_cmag)[0, 1]
report("mean_wind (0-6km)", mw_corr > 0.95,
       f"max_diff={mw_max:.4f} m/s, corr(mag)={mw_corr:.6f}")

# --- bunkers: RM direction ---
bk_gru = np.array([r[0] for r in bk_gpu_rm])
bk_grv = np.array([r[1] for r in bk_gpu_rm])
bk_cru = np.array([r[0] for r in bk_cpu_rm])
bk_crv = np.array([r[1] for r in bk_cpu_rm])
g_rm_dir = np.degrees(np.arctan2(-bk_gru, -bk_grv)) % 360
c_rm_dir = np.degrees(np.arctan2(-bk_cru, -bk_crv)) % 360
dir_diff = np.minimum(np.abs(g_rm_dir - c_rm_dir),
                      360 - np.abs(g_rm_dir - c_rm_dir))
g_rm_spd = np.sqrt(bk_gru**2 + bk_grv**2)
c_rm_spd = np.sqrt(bk_cru**2 + bk_crv**2)
spd_diff = np.abs(g_rm_spd - c_rm_spd)
rm_dir_corr = np.corrcoef(g_rm_dir, c_rm_dir)[0, 1]
rm_spd_corr = np.corrcoef(g_rm_spd, c_rm_spd)[0, 1]
report("bunkers_storm_motion (RM dir)", rm_dir_corr > 0.80,
       f"max_dir_diff={dir_diff.max():.1f}deg, median={np.median(dir_diff):.1f}deg, "
       f"corr={rm_dir_corr:.6f}")
report("bunkers_storm_motion (RM spd)", rm_spd_corr > 0.90,
       f"max_spd_diff={spd_diff.max():.2f} m/s, corr={rm_spd_corr:.6f}")

# --- bunkers: LM speed ---
bk_glu = np.array([r[0] for r in bk_gpu_lm])
bk_glv = np.array([r[1] for r in bk_gpu_lm])
bk_clu = np.array([r[0] for r in bk_cpu_lm])
bk_clv = np.array([r[1] for r in bk_cpu_lm])
g_lm_spd = np.sqrt(bk_glu**2 + bk_glv**2)
c_lm_spd = np.sqrt(bk_clu**2 + bk_clv**2)
lm_spd_corr = np.corrcoef(g_lm_spd, c_lm_spd)[0, 1]
report("bunkers_storm_motion (LM spd)", lm_spd_corr > 0.90,
       f"corr={lm_spd_corr:.6f}")

# --- storm_relative_helicity ---
srh_g = np.array(srh_gpu_tot)
srh_c = np.array(srh_cpu_tot)
srh_diff = np.abs(srh_g - srh_c)
# GPU uses per-column Bunkers, metrust uses its own Bunkers - compare pattern
srh_corr = np.corrcoef(srh_g, srh_c)[0, 1]
report("storm_relative_helicity (0-3km) [pattern]", srh_corr > 0.85,
       f"corr={srh_corr:.6f}, max_diff={srh_diff.max():.1f} m^2/s^2, "
       f"mean_gpu={srh_g.mean():.1f}, mean_cpu={srh_c.mean():.1f}")


# ======================================================================
# Summary
# ======================================================================
print(f"\n[5/5] Summary")
print("=" * 78)
print(f"  Passed: {passed}/{total_tests}")
print(f"  Failed: {failed}/{total_tests}")

if failed > 0:
    print("\n  ** SOME TESTS FAILED **")
    sys.exit(1)
else:
    print("\n  ALL TESTS PASSED")
    sys.exit(0)
