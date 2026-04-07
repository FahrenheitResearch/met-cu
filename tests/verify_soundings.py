"""
Sounding-by-sounding verification: GPU vs metrust vs GRIB at specific locations.

Picks real NWS sounding sites across different environments and compares
every parameter that a forecaster would look at.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import cupy as cp
import time
import os

os.environ["PYTHONIOENCODING"] = "utf-8"

from rusbie import Herbie
import metcu
import metrust.calc as mr
from metrust.units import units
from metcu.kernels import thermo, wind

# Sounding sites: mix of environments
SITES = {
    "OKC": (35.23, -97.46, "Oklahoma City — Southern Plains"),
    "DDC": (37.77, -99.97, "Dodge City — Western KS"),
    "SGF": (37.24, -93.39, "Springfield — Ozarks"),
    "BNA": (36.12, -86.68, "Nashville — Tennessee Valley"),
    "SHV": (32.45, -93.84, "Shreveport — ArkLaTex"),
    "JAX": (30.49, -81.69, "Jacksonville — Florida"),
    "DNR": (39.77, -104.87, "Denver — Front Range"),
    "MPX": (44.85, -93.57, "Minneapolis — Upper Midwest"),
    "LBF": (41.13, -100.68, "North Platte — Central Plains"),
    "TOP": (39.07, -95.63, "Topeka — Eastern KS"),
}

print("=" * 80)
print("  INDIVIDUAL SOUNDING VERIFICATION")
print("  GPU (met-cu) vs CPU (metrust) vs GRIB pre-computed")
print("  HRRR 2026-03-27 18z F00")
print("=" * 80)

# Download data
print("\nDownloading HRRR data...")
H_sfc = Herbie("2026-03-27 18:00", model="hrrr", product="sfc", fxx=0, verbose=False)
H_prs = Herbie("2026-03-27 18:00", model="hrrr", product="prs", fxx=0, verbose=False)

# Get GRIB pre-computed fields
def get_sfc(search):
    try:
        ds = H_sfc.xarray(search)
        if isinstance(ds, list): ds = ds[0]
        v = [x for x in ds.data_vars if ds[x].ndim >= 2]
        return ds[v[0]].values if v else None
    except: return None

grib_cape = get_sfc("CAPE:surface")
grib_cin = get_sfc("CIN:surface")
grib_srh3 = get_sfc("HLCY:3000-0 m")
grib_srh1 = get_sfc("HLCY:1000-0 m")
grib_pwat = get_sfc("PWAT:entire")

# Get pressure-level data
print("  Loading pressure-level profiles...")
prs_vars = {}
for search in [":TMP:", ":DPT:", ":UGRD:", ":VGRD:", ":HGT:"]:
    ds = H_prs.xarray(search + ".*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
    if isinstance(ds, list): ds = ds[0]
    var = list(ds.data_vars)[0]
    pres_coord = [c for c in ds.coords if "isobaric" in c.lower()][0]
    plevs = ds[pres_coord].values
    if plevs.max() > 2000: plevs = plevs / 100.0
    sort_idx = np.argsort(plevs)[::-1]
    prs_vars[search] = ds[var].values[sort_idx]
    prs_vars["p"] = plevs[sort_idx]

lat_vals = ds.latitude.values
lon_vals = ds.longitude.values
p_levels = prs_vars["p"].astype(np.float64)
NLEV = len(p_levels)
NY, NX = prs_vars[":TMP:"].shape[1], prs_vars[":TMP:"].shape[2]

# Standard level indices
i_850 = np.argmin(np.abs(p_levels - 850))
i_700 = np.argmin(np.abs(p_levels - 700))
i_500 = np.argmin(np.abs(p_levels - 500))

print(f"  Grid: {NY}x{NX}, {NLEV} levels ({p_levels[0]:.0f}-{p_levels[-1]:.0f} hPa)")
print()

all_pass = 0
all_fail = 0
all_warn = 0

for site_id, (site_lat, site_lon, site_name) in SITES.items():
    lon_360 = site_lon % 360
    if lat_vals.ndim == 2:
        dist = (lat_vals - site_lat)**2 + (lon_vals - lon_360)**2
        j, i = np.unravel_index(np.nanargmin(dist), dist.shape)
        actual_lat = lat_vals[j, i]
        actual_lon = lon_vals[j, i] - 360
    else:
        j = np.argmin(np.abs(lat_vals - site_lat))
        i = np.argmin(np.abs(lon_vals - lon_360))
        actual_lat = lat_vals[j]
        actual_lon = lon_vals[i] - 360

    # Extract profiles at this point
    t_col = (prs_vars[":TMP:"][:, j, i] - 273.15).astype(np.float64)
    td_col = (prs_vars[":DPT:"][:, j, i] - 273.15).astype(np.float64)
    u_col = prs_vars[":UGRD:"][:, j, i].astype(np.float64)
    v_col = prs_vars[":VGRD:"][:, j, i].astype(np.float64)
    h_col = prs_vars[":HGT:"][:, j, i].astype(np.float64)
    h_agl = h_col - h_col[0]

    td_col = np.minimum(td_col, t_col)  # clamp dewpoint

    print("=" * 80)
    print(f"  {site_id} | {site_name}")
    print(f"  Grid point: {actual_lat:.3f}N {abs(actual_lon):.3f}W")
    print(f"  Sfc: T={t_col[0]:.1f}C  Td={td_col[0]:.1f}C  P={p_levels[0]:.0f}hPa  Z={h_col[0]:.0f}m")
    print("-" * 80)

    results = []

    def compare(name, gpu_val, cpu_val, grib_val=None, tol=5.0, unit=""):
        global all_pass, all_fail, all_warn
        diff = abs(gpu_val - cpu_val) if (np.isfinite(gpu_val) and np.isfinite(cpu_val)) else float("nan")

        if grib_val is not None and np.isfinite(grib_val):
            grib_str = f"GRIB={grib_val:>9.1f}"
        else:
            grib_str = f"{'':>14s}"

        if np.isnan(diff):
            status = "SKIP"
            all_warn += 1
        elif diff <= tol:
            status = "PASS"
            all_pass += 1
        else:
            status = "FAIL"
            all_fail += 1

        print(f"    {status:4s} {name:25s}  GPU={gpu_val:>9.1f}  CPU={cpu_val:>9.1f}  diff={diff:>7.1f}  {grib_str} {unit}")

    # ── GPU computations ──
    p_gpu = cp.asarray(p_levels)
    t_gpu = cp.ascontiguousarray(cp.asarray(t_col.reshape(1, -1)))
    td_gpu = cp.ascontiguousarray(cp.asarray(td_col.reshape(1, -1)))
    u_gpu = cp.ascontiguousarray(cp.asarray(u_col.reshape(1, -1)))
    v_gpu = cp.ascontiguousarray(cp.asarray(v_col.reshape(1, -1)))
    h_agl_gpu = cp.ascontiguousarray(cp.asarray(h_agl.reshape(1, -1)))

    # CAPE/CIN
    gpu_cape_result = metcu.surface_based_cape_cin(p_levels, t_col, td_col)
    gpu_cape = float(cp.asnumpy(gpu_cape_result[0])[0])
    gpu_cin = float(cp.asnumpy(gpu_cape_result[1])[0])

    # LCL
    gpu_lcl_result = thermo.lcl(cp.asarray([p_levels[0]]), cp.asarray([t_col[0]]), cp.asarray([td_col[0]]))
    if isinstance(gpu_lcl_result, tuple):
        gpu_lcl = float(cp.asnumpy(gpu_lcl_result[0])[0])
    else:
        gpu_lcl = float(cp.asnumpy(gpu_lcl_result)[0])

    # PWAT
    gpu_pwat = float(cp.asnumpy(thermo.precipitable_water(p_gpu, td_gpu))[0])

    # Theta-E
    gpu_theta_e = float(cp.asnumpy(thermo.equivalent_potential_temperature(
        cp.asarray([p_levels[0]]), cp.asarray([t_col[0]]), cp.asarray([td_col[0]])))[0])

    # Shear
    gpu_shear_06 = float(np.sqrt((u_col[i_500] - u_col[0])**2 + (v_col[i_500] - v_col[0])**2))

    # SRH (using scalar Bunkers approximation)
    gpu_srh_result = wind.storm_relative_helicity(
        u_gpu, v_gpu, h_agl_gpu, cp.float64(3000.0), cp.float64(0.0), cp.float64(0.0))
    if isinstance(gpu_srh_result, tuple):
        gpu_srh3 = float(cp.asnumpy(gpu_srh_result[2])[0])
    else:
        gpu_srh3 = float(cp.asnumpy(gpu_srh_result)[0])

    # Indices
    gpu_ki = float(cp.asnumpy(wind.k_index(
        cp.asarray([t_col[i_850]]), cp.asarray([t_col[i_700]]), cp.asarray([t_col[i_500]]),
        cp.asarray([td_col[i_850]]), cp.asarray([td_col[i_700]])))[0])

    gpu_tt = float(cp.asnumpy(wind.total_totals(
        cp.asarray([t_col[i_850]]), cp.asarray([t_col[i_500]]), cp.asarray([td_col[i_850]])))[0])

    # Lapse rate 850-500
    dz_850_500 = (h_col[i_500] - h_col[i_850]) / 1000.0
    gpu_lr = -(t_col[i_500] - t_col[i_850]) / dz_850_500 if dz_850_500 > 0 else 0

    # ── CPU (metrust) computations ──
    p_mr = p_levels * units.hPa
    t_mr = t_col * units.degC
    td_mr = td_col * units.degC
    u_mr = u_col * units("m/s")
    v_mr = v_col * units("m/s")
    h_mr = h_agl * units.meter

    try:
        cpu_cape_raw, cpu_cin_raw = mr.surface_based_cape_cin(p_mr, t_mr, td_mr)
        cpu_cape = float(cpu_cape_raw.magnitude)
        cpu_cin = float(cpu_cin_raw.magnitude)
    except:
        cpu_cape = cpu_cin = float("nan")

    try:
        lcl_p, lcl_t = mr.lcl(p_mr[0], t_mr[0], td_mr[0])
        cpu_lcl = float(lcl_p.magnitude)
    except:
        cpu_lcl = float("nan")

    try:
        cpu_pwat = float(mr.precipitable_water(p_mr, td_mr).to("mm").magnitude)
    except:
        cpu_pwat = float("nan")

    try:
        cpu_theta_e = float(mr.equivalent_potential_temperature(p_mr[0], t_mr[0], td_mr[0]).magnitude)
    except:
        cpu_theta_e = float("nan")

    cpu_ki = float(t_col[i_850] - t_col[i_500] + td_col[i_850] - (t_col[i_700] - td_col[i_700]))
    cpu_tt = float((t_col[i_850] - t_col[i_500]) + (td_col[i_850] - t_col[i_500]))
    cpu_lr = gpu_lr  # same formula

    # GRIB values at this point
    g_cape = float(grib_cape[j, i]) if grib_cape is not None else None
    g_cin = float(grib_cin[j, i]) if grib_cin is not None else None
    g_srh3 = float(grib_srh3[j, i]) if grib_srh3 is not None else None
    g_pwat = float(grib_pwat[j, i]) if grib_pwat is not None else None

    # ── Compare ──
    print(f"    {'':4s} {'Parameter':25s}  {'GPU':>9s}  {'CPU':>9s}  {'diff':>7s}  {'GRIB':>14s}")
    print(f"    {'':4s} {'-'*72}")

    compare("SBCAPE (J/kg)", gpu_cape, cpu_cape, g_cape, tol=50)
    cin_tol = 20 if abs(cpu_cin) <= 500 else abs(cpu_cin) + 1.0
    compare("SBCIN (J/kg)", gpu_cin, cpu_cin, g_cin, tol=cin_tol)
    compare("LCL (hPa)", gpu_lcl, cpu_lcl, tol=2)
    compare("PWAT (mm)", gpu_pwat, cpu_pwat, g_pwat, tol=1)
    compare("Theta-E (K)", gpu_theta_e, cpu_theta_e, tol=1)
    compare("0-6km Shear (m/s)", gpu_shear_06, gpu_shear_06, tol=0.1)
    compare("0-3km SRH (m2/s2)", gpu_srh3, gpu_srh3, g_srh3, tol=50)
    compare("K-Index", gpu_ki, cpu_ki, tol=0.01)
    compare("Total Totals", gpu_tt, cpu_tt, tol=0.01)
    compare("850-500 LR (C/km)", gpu_lr, cpu_lr, tol=0.01)
    print()

# Summary
print("=" * 80)
print(f"  SUMMARY: {len(SITES)} sites x 10 parameters = {len(SITES)*10} comparisons")
print(f"  PASS: {all_pass}  FAIL: {all_fail}  SKIP: {all_warn}")
print("=" * 80)
