"""
REAL HRRR: Pre-computed GRIB fields vs GPU from-scratch computation.

Downloads actual HRRR data, reads the model's pre-computed CAPE/CIN/SRH,
then recomputes them from scratch on the GPU from raw T/Td/wind profiles.
Plots both side by side.
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import cupy as cp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import time
import os

os.makedirs("tests/real_plots", exist_ok=True)
out = "tests/real_plots"

from rusbie import Herbie

print("Downloading HRRR data...")
t0 = time.time()

H_sfc = Herbie("2026-03-27 18:00", model="hrrr", product="sfc", fxx=0, verbose=False)
H_prs = Herbie("2026-03-27 18:00", model="hrrr", product="prs", fxx=0, verbose=False)

def get_field(H, search):
    try:
        ds = H.xarray(search)
        if isinstance(ds, list): ds = ds[0]
        var = [v for v in ds.data_vars if ds[v].ndim >= 2]
        if var: return ds[var[0]].values
    except: pass
    return None

# Pre-computed fields from GRIB
print("  Reading pre-computed GRIB fields...")
grib_cape = get_field(H_sfc, "CAPE:surface")
grib_cin = get_field(H_sfc, "CIN:surface")
grib_srh = get_field(H_sfc, "HLCY:3000-0 m")
grib_srh01 = get_field(H_sfc, "HLCY:1000-0 m")
grib_refl = get_field(H_sfc, "REFC:entire")
grib_pwat = get_field(H_sfc, "PWAT:entire")
grib_t2m = get_field(H_sfc, "TMP:2 m")
grib_td2m = get_field(H_sfc, "DPT:2 m")
grib_u10 = get_field(H_sfc, "UGRD:10 m")
grib_v10 = get_field(H_sfc, "VGRD:10 m")
grib_vis = get_field(H_sfc, "VIS:surface")
grib_uh = get_field(H_sfc, "MXUPHL:5000-2000")

# Pressure-level profiles for from-scratch computation
print("  Reading pressure-level profiles...")
levels_to_get = [":TMP:", ":DPT:", ":UGRD:", ":VGRD:", ":HGT:"]
prs_data = {}
prs_levels = {}  # per-variable pressure levels
for search in levels_to_get:
    ds = H_prs.xarray(search + ".*mb", backend_kwargs={"filter_by_keys": {"typeOfLevel": "isobaricInhPa"}})
    if isinstance(ds, list): ds = ds[0]
    var = list(ds.data_vars)[0]
    pres_coord = [c for c in ds.coords if "isobaric" in c.lower()][0]
    plevs = ds[pres_coord].values
    if plevs.max() > 2000: plevs = plevs / 100.0
    sort_idx = np.argsort(plevs)[::-1]
    prs_data[search] = ds[var].values[sort_idx]
    prs_levels[search] = plevs[sort_idx]
    print(f"    {search}: {len(plevs)} levels, {plevs.min():.0f}-{plevs.max():.0f} hPa")

# Find common pressure levels across all variables (some may have fewer)
common_p = prs_levels[levels_to_get[0]]
for search in levels_to_get[1:]:
    common_p = np.intersect1d(common_p, prs_levels[search])
common_p = np.sort(common_p)[::-1]  # descending (surface-first)
print(f"    Common levels: {len(common_p)}, {common_p[0]:.0f}-{common_p[-1]:.0f} hPa")

# Subset each variable to the common levels
for search in levels_to_get:
    full_p = prs_levels[search]
    idx = np.array([np.where(full_p == plev)[0][0] for plev in common_p])
    prs_data[search] = prs_data[search][idx]
prs_data["p_levels"] = common_p

NY, NX = grib_t2m.shape
N = NY * NX
NLEV = len(prs_data["p_levels"])
print(f"  Grid: {NY}x{NX} = {N:,}, {NLEV} pressure levels")
print(f"  Downloaded in {time.time()-t0:.0f}s")

# Reshape 3D data to (ncols, nlevels) for GPU
p_levels = prs_data["p_levels"].astype(np.float64)
t_3d = prs_data[":TMP:"].reshape(NLEV, -1).T.astype(np.float64) - 273.15  # K -> C
td_3d = prs_data[":DPT:"].reshape(NLEV, -1).T.astype(np.float64) - 273.15
u_3d = prs_data[":UGRD:"].reshape(NLEV, -1).T.astype(np.float64)
v_3d = prs_data[":VGRD:"].reshape(NLEV, -1).T.astype(np.float64)
h_3d = prs_data[":HGT:"].reshape(NLEV, -1).T.astype(np.float64)

# Convert surface fields
t2m_c = grib_t2m - 273.15
td2m_c = grib_td2m - 273.15

# ================================================================
# GPU COMPUTATION FROM SCRATCH
# ================================================================
print("\nComputing on GPU from raw profiles...")
from metcu.kernels import thermo, wind, grid

p_gpu = cp.asarray(p_levels)
t3_gpu = cp.asarray(t_3d)
td3_gpu = cp.asarray(td_3d)
u3_gpu = cp.asarray(u_3d)
v3_gpu = cp.asarray(v_3d)
h3_gpu = cp.asarray(h_3d)
t_sfc_gpu = cp.asarray(t2m_c.ravel())
td_sfc_gpu = cp.asarray(td2m_c.ravel())
u_sfc_gpu = cp.asarray(grib_u10.ravel())
v_sfc_gpu = cp.asarray(grib_v10.ravel())
dx_gpu = cp.full((NY, NX), 3000.0)
dy_gpu = cp.full((NY, NX), 3000.0)

cp.cuda.Device().synchronize()
t_compute = time.perf_counter()

# CAPE/CIN from scratch (the big one)
print("  CAPE/CIN...")
gpu_cape_result = thermo.cape_cin(p_gpu, t3_gpu, td3_gpu)
if isinstance(gpu_cape_result, tuple):
    gpu_cape = cp.asnumpy(gpu_cape_result[0]).reshape(NY, NX)
    gpu_cin = cp.asnumpy(gpu_cape_result[1]).reshape(NY, NX)
else:
    gpu_cape = cp.asnumpy(gpu_cape_result).reshape(NY, NX)
    gpu_cin = np.zeros((NY, NX))

# PWAT from scratch
print("  PWAT...")
gpu_pwat = cp.asnumpy(thermo.precipitable_water(p_gpu, td3_gpu)).reshape(NY, NX)

# Theta-E
print("  Theta-E...")
gpu_theta_e = cp.asnumpy(thermo.equivalent_potential_temperature(
    cp.full(N, p_levels[0]), t_sfc_gpu, td_sfc_gpu)).reshape(NY, NX)

# Shear
print("  Shear...")
i_500 = np.argmin(np.abs(p_levels - 500))
i_850 = np.argmin(np.abs(p_levels - 850))
shear_u = u3_gpu[:, i_500] - u3_gpu[:, 0]
shear_v = v3_gpu[:, i_500] - v3_gpu[:, 0]
gpu_shear = cp.asnumpy(cp.sqrt(shear_u**2 + shear_v**2)).reshape(NY, NX)

# SRH — heights must be AGL, and we need real storm motion
print("  Converting heights to AGL...")
h3_agl_gpu = h3_gpu - h3_gpu[:, 0:1]  # subtract surface height from each column

print("  Bunkers storm motion...")
(rm_u, rm_v), (lm_u, lm_v), (mw_u, mw_v) = wind.bunkers_storm_motion(u3_gpu, v3_gpu, h3_agl_gpu)

print("  SRH (per-column Bunkers right-mover)...")
# SRH kernel takes scalar storm motion, but we have per-column vectors.
# Use a vectorized approach: call the raw kernel directly with per-column storm motion.
ncols, nlevels = u3_gpu.shape
srh_pos = cp.empty(ncols, dtype=cp.float64)
srh_neg = cp.empty(ncols, dtype=cp.float64)
srh_tot = cp.empty(ncols, dtype=cp.float64)

# We need a per-column SRH kernel. The existing kernel takes scalar storm_u/v.
# Compute SRH with the 0-6km mean wind as a reasonable scalar approximation,
# OR loop-free: compute using the per-column kernel directly.
# Let's patch: use the existing kernel with column-wise Bunkers via a custom kernel.
_srh_percol_code = r"""
extern "C" __global__
void srh_percol_kernel(
    const double* u,
    const double* v,
    const double* heights,
    const double* storm_u_arr,
    const double* storm_v_arr,
    double depth,
    double* srh_pos_out,
    double* srh_neg_out,
    double* srh_total_out,
    int ncols,
    int nlevels
) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    if (col >= ncols) return;

    double pos = 0.0, neg = 0.0;
    int offset = col * nlevels;
    double su = storm_u_arr[col];
    double sv = storm_v_arr[col];

    for (int k = 1; k < nlevels; k++) {
        if (heights[offset + k] > depth) break;

        double sru0 = u[offset + k-1] - su;
        double srv0 = v[offset + k-1] - sv;
        double sru1 = u[offset + k] - su;
        double srv1 = v[offset + k] - sv;

        double val = (sru1 - sru0) * (srv0 + srv1) / 2.0
                   - (srv1 - srv0) * (sru0 + sru1) / 2.0;

        if (val > 0) pos += val;
        else neg += val;
    }

    srh_pos_out[col] = pos;
    srh_neg_out[col] = neg;
    srh_total_out[col] = pos + neg;
}
"""
_srh_percol_mod = cp.RawModule(code=_srh_percol_code)
_srh_percol_kern = _srh_percol_mod.get_function("srh_percol_kernel")

_BLOCK = 256
_grid = ((ncols + _BLOCK - 1) // _BLOCK,)
_block = (min(ncols, _BLOCK),)

u3_cont = cp.ascontiguousarray(u3_gpu, dtype=cp.float64)
v3_cont = cp.ascontiguousarray(v3_gpu, dtype=cp.float64)
h3_agl_cont = cp.ascontiguousarray(h3_agl_gpu, dtype=cp.float64)

_srh_percol_kern(_grid, _block, (
    u3_cont, v3_cont, h3_agl_cont,
    rm_u, rm_v, cp.float64(3000.0),
    srh_pos, srh_neg, srh_tot,
    np.int32(ncols), np.int32(nlevels),
))
gpu_srh = cp.asnumpy(srh_tot).reshape(NY, NX)

# SRH 0-1km for STP (uses same per-column Bunkers kernel)
print("  SRH 0-1km...")
srh01_pos = cp.empty(ncols, dtype=cp.float64)
srh01_neg = cp.empty(ncols, dtype=cp.float64)
srh01_tot = cp.empty(ncols, dtype=cp.float64)
_srh_percol_kern(_grid, _block, (
    u3_cont, v3_cont, h3_agl_cont,
    rm_u, rm_v, cp.float64(1000.0),
    srh01_pos, srh01_neg, srh01_tot,
    np.int32(ncols), np.int32(nlevels),
))
gpu_srh01 = cp.asnumpy(srh01_tot).reshape(NY, NX)

# STP: significant_tornado_parameter(sbcape, lcl_height, srh_0_1km, bulk_shear_0_6km)
# LCL height via hypsometric equation: H = (Rd*Tv/g) * ln(p_sfc/p_lcl)
print("  STP...")
lcl_result = thermo.lcl(cp.full(N, p_levels[0]), t_sfc_gpu, td_sfc_gpu)
if isinstance(lcl_result, tuple):
    lcl_p_arr = lcl_result[0]  # LCL pressure in hPa (CuPy array)
else:
    lcl_p_arr = lcl_result
# Hypsometric LCL height: H = (Rd * T_mean_K / g) * ln(p_sfc / p_lcl)
t_mean_k = t_sfc_gpu + 273.15  # approximate mean layer temperature
p_sfc_arr = cp.full(N, p_levels[0], dtype=cp.float64)
lcl_hgt_gpu = (287.05 * t_mean_k / 9.80665) * cp.log(p_sfc_arr / lcl_p_arr)
lcl_hgt = cp.asnumpy(cp.clip(lcl_hgt_gpu, 0, 5000)).reshape(NY, NX)

gpu_stp = cp.asnumpy(wind.significant_tornado_parameter(
    cp.asarray(gpu_cape.ravel()),
    cp.asarray(lcl_hgt.ravel()),
    cp.asarray(gpu_srh01.ravel()),
    cp.asarray(gpu_shear.ravel()))).reshape(NY, NX)

# Vorticity
print("  Vorticity...")
gpu_vort = cp.asnumpy(grid.vorticity(
    cp.asarray(grib_u10.astype(np.float64)), cp.asarray(grib_v10.astype(np.float64)),
    dx_gpu, dy_gpu))

# Wind speed
gpu_wspd = cp.asnumpy(wind.wind_speed(u_sfc_gpu, v_sfc_gpu)).reshape(NY, NX)

# ================================================================
# NEW GPU-ONLY PRODUCTS
# ================================================================

# Extract standard level indices
i_700 = np.argmin(np.abs(p_levels - 700))
i_850 = np.argmin(np.abs(p_levels - 850))

# Standard level 2D fields (Celsius)
t850_2d = t_3d[:, i_850].reshape(NY, NX)
t700_2d = t_3d[:, i_700].reshape(NY, NX)
t500_2d = t_3d[:, i_500].reshape(NY, NX)
td850_2d = td_3d[:, i_850].reshape(NY, NX)
td700_2d = td_3d[:, i_700].reshape(NY, NX)
u850_2d = u_3d[:, i_850].reshape(NY, NX)
v850_2d = v_3d[:, i_850].reshape(NY, NX)
h850_2d = h_3d[:, i_850].reshape(NY, NX)
h500_2d = h_3d[:, i_500].reshape(NY, NX)

# SCP: supercell_composite_parameter(mucape, srh_eff, bulk_shear_eff)
print("  SCP...")
gpu_scp = cp.asnumpy(wind.supercell_composite_parameter(
    cp.asarray(gpu_cape.ravel()),
    cp.asarray(gpu_srh.ravel()),
    cp.asarray(gpu_shear.ravel()))).reshape(NY, NX)

# EHI: energy_helicity_index = CAPE * SRH / 160000
print("  EHI...")
gpu_ehi = cp.asnumpy(wind.compute_ehi(
    cp.asarray(gpu_cape.ravel()),
    cp.asarray(gpu_srh.ravel()))).reshape(NY, NX)

# K-Index: KI = (T850-T500) + Td850 - (T700-Td700), all in degC
print("  K-Index...")
gpu_ki = cp.asnumpy(wind.k_index(
    cp.asarray(t850_2d.ravel()), cp.asarray(t700_2d.ravel()),
    cp.asarray(t500_2d.ravel()), cp.asarray(td850_2d.ravel()),
    cp.asarray(td700_2d.ravel()))).reshape(NY, NX)

# Total Totals: TT = (T850-T500) + (Td850-T500)
print("  Total Totals...")
gpu_tt = cp.asnumpy(wind.total_totals(
    cp.asarray(t850_2d.ravel()), cp.asarray(t500_2d.ravel()),
    cp.asarray(td850_2d.ravel()))).reshape(NY, NX)

# 850-500mb Lapse Rate (C/km)
print("  850-500mb Lapse Rate...")
dz_km = (h500_2d - h850_2d) / 1000.0
dz_km = np.where(dz_km < 0.1, 0.1, dz_km)  # avoid division by zero
gpu_lapse = (t850_2d - t500_2d) / dz_km  # positive = temperature decreasing with height

# Frontogenesis at 850mb (needs 2D theta field, u, v)
print("  Frontogenesis (850mb)...")
theta850_2d = (t850_2d + 273.15) * (1000.0 / p_levels[i_850]) ** 0.2857
gpu_fronto = cp.asnumpy(grid.frontogenesis(
    cp.asarray(theta850_2d), cp.asarray(u850_2d), cp.asarray(v850_2d),
    dx_gpu, dy_gpu))

# Temperature Advection at 850mb
print("  Temperature Advection (850mb)...")
gpu_tadv = cp.asnumpy(grid.advection(
    cp.asarray(t850_2d), cp.asarray(u850_2d), cp.asarray(v850_2d),
    dx_gpu, dy_gpu))

cp.cuda.Device().synchronize()
compute_ms = (time.perf_counter() - t_compute) * 1000
print(f"\n  Total GPU compute: {compute_ms:.0f}ms")

# ================================================================
# PLOTTING: GRIB pre-computed vs GPU from-scratch
# ================================================================
print("\nGenerating comparison plots...")

def plot_comparison(grib_data, gpu_data, title_grib, title_gpu, cmap, fname, levels=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

    if levels is not None:
        cf1 = ax1.contourf(grib_data, levels=levels, cmap=cmap, extend="both")
        cf2 = ax2.contourf(gpu_data, levels=levels, cmap=cmap, extend="both")
    else:
        vmin = min(np.nanmin(grib_data), np.nanmin(gpu_data))
        vmax = max(np.nanmax(grib_data), np.nanmax(gpu_data))
        cf1 = ax1.contourf(grib_data, 20, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")
        cf2 = ax2.contourf(gpu_data, 20, cmap=cmap, vmin=vmin, vmax=vmax, extend="both")

    plt.colorbar(cf1, ax=ax1, shrink=0.7)
    plt.colorbar(cf2, ax=ax2, shrink=0.7)
    ax1.set_title(title_grib, fontsize=13, fontweight="bold")
    ax2.set_title(title_gpu, fontsize=13, fontweight="bold")
    for ax in (ax1, ax2):
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle("HRRR 2026-03-27 18z F00 | GRIB Pre-computed vs GPU From-Scratch",
                 fontsize=11, color="gray")
    fig.savefig(os.path.join(out, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")

def plot_single(data, title, cmap, fname, levels=None):
    fig, ax = plt.subplots(figsize=(14, 8))
    if levels is not None:
        cf = ax.contourf(data, levels=levels, cmap=cmap, extend="both")
    else:
        cf = ax.contourf(data, 20, cmap=cmap, extend="both")
    plt.colorbar(cf, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(os.path.join(out, fname), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  {fname}")

# Side-by-side comparisons
if grib_cape is not None:
    plot_comparison(grib_cape, gpu_cape,
        "GRIB SBCAPE (J/kg)", "GPU SBCAPE from scratch (J/kg)",
        "hot_r", "01_cape_comparison.png",
        levels=[0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000])

if grib_cin is not None:
    plot_comparison(grib_cin, gpu_cin,
        "GRIB SBCIN (J/kg)", "GPU SBCIN from scratch (J/kg)",
        "BuPu_r", "02_cin_comparison.png",
        levels=[-500, -300, -200, -100, -50, -25, -10, 0])

if grib_srh is not None:
    plot_comparison(grib_srh, gpu_srh,
        "GRIB 0-3km SRH (m2/s2)", "GPU 0-3km SRH from scratch (m2/s2)",
        "BuPu", "03_srh_comparison.png",
        levels=[0, 50, 100, 150, 200, 300, 400, 500])

# PWAT: both GRIB and GPU are in mm (kg/m2 = mm for water)
if grib_pwat is not None:
    plot_comparison(grib_pwat, gpu_pwat,
        "GRIB PWAT (mm)", "GPU PWAT from scratch (mm)",
        "GnBu", "04_pwat_comparison.png",
        levels=[0, 5, 10, 15, 20, 25, 30, 40, 50, 60])

if grib_srh01 is not None:
    plot_comparison(grib_srh01, gpu_srh01,
        "GRIB 0-1km SRH (m2/s2)", "GPU 0-1km SRH from scratch (m2/s2)",
        "BuPu", "04b_srh01_comparison.png",
        levels=[0, 25, 50, 100, 150, 200, 300, 400])

# GPU-only products (not in GRIB)
plot_single(gpu_stp, "GPU STP (from scratch CAPE/SRH/shear/LCL)", "RdPu",
    "05_stp_gpu.png", levels=[0, 0.5, 1, 2, 3, 4, 5, 8, 10])

plot_single(gpu_theta_e, "GPU Surface Theta-E (K)", "magma",
    "06_theta_e_gpu.png")

plot_single(gpu_shear, "GPU 0-6km Bulk Shear (m/s)", "YlOrRd",
    "07_shear_gpu.png", levels=[0, 5, 10, 15, 20, 25, 30, 35, 40])

plot_single(gpu_vort * 1e5, "GPU 10m Vorticity (x10-5 /s)", "RdBu_r",
    "08_vorticity_gpu.png", levels=np.arange(-30, 35, 5))

plot_single(gpu_wspd * 1.944, "GPU 10m Wind Speed (kts)", "YlOrRd",
    "09_wspd_gpu.png", levels=[0, 5, 10, 15, 20, 25, 30, 35, 40])

plot_single(gpu_scp, "GPU SCP (from scratch CAPE/SRH/shear)", "OrRd",
    "10_scp_gpu.png", levels=[0, 0.5, 1, 2, 4, 6, 8, 10, 15])

plot_single(gpu_ehi, "GPU EHI (CAPE*SRH/160000)", "YlOrRd",
    "11_ehi_gpu.png", levels=[0, 0.5, 1, 2, 3, 4, 5, 8])

plot_single(gpu_ki, "GPU K-Index (C)", "RdYlGn_r",
    "12_kindex_gpu.png", levels=np.arange(-10, 45, 5))

plot_single(gpu_tt, "GPU Total Totals (C)", "RdYlGn_r",
    "13_total_totals_gpu.png", levels=np.arange(30, 65, 3))

plot_single(gpu_lapse, "GPU 850-500mb Lapse Rate (C/km)", "RdYlBu_r",
    "14_lapse_rate_gpu.png", levels=np.arange(3, 10, 0.5))

plot_single(gpu_fronto * 1e9, "GPU 850mb Frontogenesis (x10-9 K/m/s)", "RdBu_r",
    "15_frontogenesis_gpu.png", levels=np.arange(-20, 22, 2))

plot_single(gpu_tadv * 3600, "GPU 850mb Temperature Advection (C/hr)", "RdBu_r",
    "16_temp_advection_gpu.png", levels=np.arange(-5, 5.5, 0.5))

# Also plot the raw GRIB fields that look nice
if grib_refl is not None:
    plot_single(grib_refl, "HRRR Composite Reflectivity (dBZ)", "turbo",
        "17_refl_grib.png", levels=np.arange(-10, 75, 5))

plot_single(t2m_c * 9/5 + 32, "HRRR 2m Temperature (F)", "RdYlBu_r",
    "18_temp_grib.png")

plot_single(td2m_c * 9/5 + 32, "HRRR 2m Dewpoint (F)", "YlGn",
    "19_dewpoint_grib.png")

if grib_vis is not None:
    plot_single(np.clip(grib_vis / 1609.34, 0, 10), "HRRR Visibility (miles)", "YlGnBu_r",
        "20_visibility_grib.png", levels=[0, 0.5, 1, 2, 3, 5, 7, 10])

print(f"\nDone. {len(os.listdir(out))} plots in {out}/")
print(f"GPU compute time: {compute_ms:.0f}ms for all from-scratch products")
