"""
Publication-quality meteorology plots with cartopy map overlays.

Each function: accepts data -> GPU compute -> GPU render -> cartopy GeoAxes figure.
Uses Lambert Conformal projection with state/country borders, coastlines, and
horizontal colorbars for a clean mesoanalysis style.

Fast path (mesoanalysis / mesoanalysis_fast):
  Renders cartopy map borders ONCE as a transparent PNG overlay, then composites
  GPU-colormapped data with PIL for ~50ms per frame after the first.
"""
import io
import os
import hashlib
import warnings
import numpy as np
import metcu.gpu_ops as cp
import matplotlib
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.axes import Axes
from matplotlib.colors import Normalize, BoundaryNorm
from PIL import Image, ImageDraw, ImageFont
from metcu.kernels import thermo, wind, grid

# HRRR Lambert grid approximate extent in lon/lat
_HRRR_EXTENT = [-134, -60, 21, 53]
# CONUS display extent
_CONUS_EXTENT = [-125, -66, 23, 50]
# HRRR native Lambert Conformal projection parameters
_HRRR_GLOBE = ccrs.Globe(semimajor_axis=6371229.0, semiminor_axis=6371229.0)
_LAMBERT = ccrs.LambertConformal(
    central_longitude=-97.5,
    central_latitude=38.5,
    standard_parallels=(38.5,),
    globe=_HRRR_GLOBE,
)
_PLATE = ccrs.PlateCarree()
_ORIGINAL_PCOLORMESH = Axes.pcolormesh
_APPROXIMATE_GRID_WARNING = (
    "Plotting without 2D latitude/longitude coordinates falls back to an "
    "approximate HRRR extent and may not align with cartopy features. Pass "
    "`lats` and `lons` for geographically correct output."
)
_warned_about_approximate_grid = False

# Cartopy features at 50m resolution
_STATES = cfeature.NaturalEarthFeature(
    'cultural', 'admin_1_states_provinces_lines', '50m',
    facecolor='none', edgecolor='black', linewidth=0.4)
_COASTLINE = cfeature.NaturalEarthFeature(
    'physical', 'coastline', '50m',
    facecolor='none', edgecolor='black', linewidth=0.7)
_BORDERS = cfeature.NaturalEarthFeature(
    'cultural', 'admin_0_boundary_lines_land', '50m',
    facecolor='none', edgecolor='black', linewidth=0.4)


def _add_map_features(ax):
    """Add state borders, coastlines, and country borders to a GeoAxes."""
    ax.add_feature(_STATES)
    ax.add_feature(_COASTLINE)
    ax.add_feature(_BORDERS)
    ax.set_extent(_CONUS_EXTENT, crs=_PLATE)


def _warn_about_approximate_grid():
    """Warn once when plotting without real geographic coordinates."""
    global _warned_about_approximate_grid
    if _warned_about_approximate_grid:
        return
    warnings.warn(_APPROXIMATE_GRID_WARNING, RuntimeWarning, stacklevel=3)
    _warned_about_approximate_grid = True


def _coerce_geo_array(name, values, shape):
    """Convert a latitude/longitude grid to float64 with an exact shape match."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.shape != shape:
        raise ValueError(
            f"{name} must have shape {shape}, got {arr.shape}"
        )
    return arr


def _normalize_longitudes(lons):
    """Normalize longitudes into [-180, 180] for cartopy PlateCarree."""
    lon_arr = np.asarray(lons, dtype=np.float64)
    return np.where(lon_arr > 180.0, lon_arr - 360.0, lon_arr)


def _extract_geo_grid(fields, shape):
    """Return 2D latitude/longitude arrays from a field dict when available."""
    grid = fields.get('grid')
    if isinstance(grid, dict):
        lat = grid.get('lats', grid.get('lat', grid.get('latitude')))
        lon = grid.get('lons', grid.get('lon', grid.get('longitude')))
    elif isinstance(grid, (tuple, list)) and len(grid) == 2:
        lat, lon = grid
    else:
        lat = lon = None

    if lat is None:
        for key in ('lats', 'lat', 'latitude'):
            if fields.get(key) is not None:
                lat = fields[key]
                break

    if lon is None:
        for key in ('lons', 'lon', 'longitude'):
            if fields.get(key) is not None:
                lon = fields[key]
                break

    if lat is None and lon is None:
        return None, None
    if lat is None or lon is None:
        raise ValueError("Both latitude and longitude grids must be provided together")

    return (
        _coerce_geo_array('lats', lat, shape),
        _normalize_longitudes(_coerce_geo_array('lons', lon, shape)),
    )


def _infer_field_shape(fields):
    """Infer the primary 2D field shape for mesoanalysis plots."""
    for key in ('t2m', 'td2m', 'cape', 'refc'):
        value = fields.get(key)
        if value is None:
            continue
        arr = np.asarray(value)
        if arr.ndim != 2:
            raise ValueError(f"Field {key!r} must be 2D, got shape {arr.shape}")
        return arr.shape
    raise ValueError("No valid 2D fields found")


def _figure_to_image(fig, dpi, save=None):
    """Render a matplotlib figure to a PIL RGBA image and optionally save it."""
    if save:
        fig.savefig(save, dpi=dpi, facecolor=fig.get_facecolor(), bbox_inches='tight')

    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, facecolor=fig.get_facecolor(),
                bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).convert('RGBA')
    image.load()
    buf.close()
    plt.close(fig)
    return image


def plot_field(ax, data, title, cmap='RdYlBu_r', vmin=None, vmax=None,
               extend='both', colorbar=True, cb_label=None,
               extent=None, discrete_levels=None, lats=None, lons=None,
               transform=None):
    """Generic field plot with cartopy map overlay.

    Parameters
    ----------
    ax : cartopy GeoAxes
        Must have a cartopy projection set.
    data : 2D array
        Field to plot.
    title : str
        Panel title.
    cmap : str or Colormap
        Matplotlib colormap.
    vmin, vmax : float
        Color limits.
    extend : str
        Colorbar extend mode.
    colorbar : bool
        Whether to add a horizontal colorbar.
    cb_label : str
        Colorbar label.
    extent : list
        [lon_min, lon_max, lat_min, lat_max] for imshow.
    discrete_levels : array-like or None
        If provided, use BoundaryNorm for discrete color bands.
    lats, lons : 2D arrays or None
        If provided, plot on the actual curvilinear grid with pcolormesh.
    transform : cartopy CRS or None
        CRS for the provided lats/lons. Defaults to PlateCarree.
    """
    data_f = np.asarray(data, dtype=np.float64)
    if vmin is None:
        vmin = float(np.nanmin(data_f))
    if vmax is None:
        vmax = float(np.nanmax(data_f))

    if isinstance(cmap, str):
        cmap_obj = plt.get_cmap(cmap)
    else:
        cmap_obj = cmap

    if extent is None:
        extent = _HRRR_EXTENT

    if discrete_levels is not None:
        norm = BoundaryNorm(discrete_levels, cmap_obj.N, clip=True)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    if lats is not None or lons is not None:
        if lats is None or lons is None:
            raise ValueError("Both lats and lons are required when plotting on a geogrid")
        lats_f = _coerce_geo_array('lats', lats, data_f.shape)
        lons_f = _normalize_longitudes(_coerce_geo_array('lons', lons, data_f.shape))
        im = _ORIGINAL_PCOLORMESH(
            ax,
            lons_f,
            lats_f,
            data_f,
            cmap=cmap_obj,
            norm=norm,
            shading='auto',
            transform=transform or _PLATE,
            rasterized=True,
        )
    else:
        _warn_about_approximate_grid()
        # HRRR Lambert grid: col 0 = east in some upstream readers, so the
        # approximate image path preserves the historic flip.
        im = ax.imshow(np.fliplr(data_f), origin='lower', cmap=cmap_obj, norm=norm,
                       extent=extent, transform=_PLATE,
                       interpolation='bilinear')

    _add_map_features(ax)
    ax.set_title(title, fontsize=10, fontweight='bold')

    if colorbar:
        cb = plt.colorbar(im, ax=ax, orientation='horizontal',
                          shrink=0.8, pad=0.05, extend=extend)
        if cb_label:
            cb.set_label(cb_label, fontsize=8)
        cb.ax.tick_params(labelsize=7)

    return im


def plot_temperature(ax, t2m, cmap='RdYlBu_r', units='F', colorbar=True, **kwargs):
    """Plot 2m temperature field with cartopy."""
    t = t2m - 273.15 if np.nanmax(t2m) > 100 else np.array(t2m, dtype=np.float64)
    if units == 'F':
        t = t * 9 / 5 + 32
        label = 'Temperature (F)'
        vmin, vmax = -20, 120
    else:
        label = 'Temperature (C)'
        vmin, vmax = -40, 50
    return plot_field(ax, t, label, cmap=cmap, vmin=vmin, vmax=vmax,
                      colorbar=colorbar, cb_label=label, **kwargs)


def plot_dewpoint(ax, td2m, cmap='YlGn', units='F', colorbar=True, **kwargs):
    """Plot 2m dewpoint field with cartopy."""
    td = td2m - 273.15 if np.nanmax(td2m) > 100 else np.array(td2m, dtype=np.float64)
    if units == 'F':
        td = td * 9 / 5 + 32
        label = 'Dewpoint (F)'
        vmin, vmax = -10, 85
    else:
        label = 'Dewpoint (C)'
        vmin, vmax = -30, 35
    return plot_field(ax, td, label, cmap=cmap, vmin=vmin, vmax=vmax,
                      colorbar=colorbar, cb_label=label, **kwargs)


def plot_cape(ax, cape, cmap='hot_r', colorbar=True, **kwargs):
    """Plot CAPE field with cartopy."""
    levels = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000])
    return plot_field(ax, cape, 'SBCAPE (J/kg)', cmap=cmap,
                      vmin=0, vmax=5000, colorbar=colorbar,
                      cb_label='CAPE (J/kg)', discrete_levels=levels, **kwargs)


def plot_reflectivity(ax, refc, cmap='turbo', colorbar=True, **kwargs):
    """Plot composite reflectivity with cartopy."""
    return plot_field(ax, refc, 'Composite Reflectivity (dBZ)', cmap=cmap,
                      vmin=-10, vmax=75, colorbar=colorbar,
                      cb_label='dBZ', **kwargs)


def plot_srh(ax, srh, cmap='BuPu', colorbar=True, **kwargs):
    """Plot storm-relative helicity with cartopy."""
    levels = np.array([0, 50, 100, 150, 200, 300, 400, 500])
    return plot_field(ax, srh, '0-3km SRH (m2/s2)', cmap=cmap,
                      vmin=0, vmax=500, colorbar=colorbar,
                      cb_label='SRH (m\u00b2/s\u00b2)', discrete_levels=levels,
                      extend='max', **kwargs)


def plot_shear(ax, shear, cmap='YlOrRd', colorbar=True, **kwargs):
    """Plot bulk wind shear with cartopy."""
    levels = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    return plot_field(ax, shear, '0-6km Bulk Shear (m/s)', cmap=cmap,
                      vmin=0, vmax=40, colorbar=colorbar,
                      cb_label='Shear (m/s)', discrete_levels=levels,
                      extend='max', **kwargs)


def plot_stp(ax, stp, cmap='RdPu', colorbar=True, **kwargs):
    """Plot Significant Tornado Parameter with cartopy."""
    levels = np.array([0, 0.5, 1, 2, 3, 4, 5, 8, 10])
    return plot_field(ax, stp, 'STP', cmap=cmap,
                      vmin=0, vmax=10, colorbar=colorbar,
                      cb_label='STP', discrete_levels=levels,
                      extend='max', **kwargs)


def plot_theta_e(ax, t2m, td2m, p_sfc=1013.0, cmap='magma', colorbar=True, **kwargs):
    """Compute and plot surface theta-e on GPU with cartopy."""
    t = t2m - 273.15 if np.nanmax(t2m) > 100 else np.array(t2m, dtype=np.float64)
    td = td2m - 273.15 if np.nanmax(td2m) > 100 else np.array(td2m, dtype=np.float64)
    n = t.size
    t_gpu = cp.asarray(t.ravel().astype(np.float64))
    td_gpu = cp.asarray(td.ravel().astype(np.float64))
    theta_e = cp.asnumpy(thermo.equivalent_potential_temperature(
        cp.full(n, p_sfc), t_gpu, td_gpu)).reshape(t.shape)
    cp.get_default_memory_pool().free_all_blocks()
    return plot_field(ax, theta_e, 'Theta-E (K)', cmap=cmap,
                      colorbar=colorbar, cb_label='Theta-E (K)', **kwargs)


def plot_vorticity(ax, u, v, dx=3000.0, dy=3000.0, cmap='RdBu_r', colorbar=True, **kwargs):
    """Compute and plot relative vorticity on GPU with cartopy."""
    vort = cp.asnumpy(grid.vorticity(
        cp.asarray(u.astype(np.float64)),
        cp.asarray(v.astype(np.float64)),
        dx, dy)) * 1e5
    cp.get_default_memory_pool().free_all_blocks()
    return plot_field(ax, vort, 'Vorticity (x10\u207b\u2075 /s)', cmap=cmap,
                      vmin=-30, vmax=30, colorbar=colorbar,
                      cb_label='Vorticity (x10\u207b\u2075 /s)', **kwargs)


def plot_frontogenesis(ax, t, u, v, dx=3000.0, dy=3000.0, cmap='RdBu_r', colorbar=True, **kwargs):
    """Compute and plot frontogenesis on GPU with cartopy."""
    t_c = t - 273.15 if np.nanmax(t) > 100 else np.array(t, dtype=np.float64)
    fronto = cp.asnumpy(grid.frontogenesis(
        cp.asarray(t_c.astype(np.float64)),
        cp.asarray(u.astype(np.float64)),
        cp.asarray(v.astype(np.float64)),
        dx, dy)) * 1e9
    cp.get_default_memory_pool().free_all_blocks()
    return plot_field(ax, fronto, 'Frontogenesis (x10\u207b\u2079 K/m/s)', cmap=cmap,
                      colorbar=colorbar,
                      cb_label='Frontogenesis (x10\u207b\u2079 K/m/s)', **kwargs)


def plot_wind_speed(ax, u, v, cmap='YlOrRd', knots=True, colorbar=True, **kwargs):
    """Plot wind speed from u/v components with cartopy."""
    wspd = np.sqrt(u**2 + v**2)
    if knots:
        wspd *= 1.944
        label = 'Wind Speed (kts)'
        levels = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    else:
        label = 'Wind Speed (m/s)'
        levels = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    return plot_field(ax, wspd, label, cmap=cmap,
                      vmin=0, vmax=40, colorbar=colorbar,
                      cb_label=label, discrete_levels=levels,
                      extend='max', **kwargs)


def plot_pwat(ax, pwat, cmap='GnBu', colorbar=True, **kwargs):
    """Plot precipitable water with cartopy."""
    return plot_field(ax, pwat, 'PWAT (mm)', cmap=cmap,
                      colorbar=colorbar, cb_label='PWAT (mm)', **kwargs)


def mesoanalysis_cartopy(fields, title='', figsize=(24, 16), dpi=120, save=None):
    """Generate a full 12-panel mesoanalysis with cartopy map overlays (slow path).

    This is the original implementation that re-renders cartopy features on every
    frame.  Kept for reference; prefer mesoanalysis() / mesoanalysis_fast() for
    animation workloads.

    fields should contain: t2m, td2m, cape, refc, srh3, shear, u10, v10, pwat
    Optional: stp, cin, srh1

    Returns the figure.
    """
    proj = _LAMBERT

    fig, axes = plt.subplots(3, 4, figsize=figsize,
                             subplot_kw={'projection': proj})
    fig.patch.set_facecolor('#1a1a2e')

    ny, nx = _infer_field_shape(fields)
    lats, lons = _extract_geo_grid(fields, (ny, nx))
    geo_kwargs = {'lats': lats, 'lons': lons} if lats is not None else {}
    n = ny * nx

    t2m = fields.get('t2m', np.zeros((ny, nx)))
    td2m = fields.get('td2m', t2m - 5)
    u10 = fields.get('u10', np.zeros((ny, nx)))
    v10 = fields.get('v10', np.zeros((ny, nx)))

    # Row 1: Temp, Dewpoint, CAPE, Reflectivity
    plot_temperature(axes[0, 0], t2m, **geo_kwargs)
    plot_dewpoint(axes[0, 1], td2m, **geo_kwargs)
    plot_cape(axes[0, 2], fields.get('cape', np.zeros((ny, nx))), **geo_kwargs)
    plot_reflectivity(axes[0, 3], fields.get('refc', np.zeros((ny, nx))), **geo_kwargs)

    # Row 2: SRH, Shear, STP, Theta-E
    plot_srh(axes[1, 0], fields.get('srh3', np.zeros((ny, nx))), **geo_kwargs)
    plot_shear(axes[1, 1], fields.get('shear', np.zeros((ny, nx))), **geo_kwargs)
    stp = fields.get('stp')
    if stp is None:
        cape = fields.get('cape', np.zeros((ny, nx)))
        srh1 = fields.get('srh1', np.zeros((ny, nx)))
        shear = fields.get('shear', np.zeros((ny, nx)))
        t_c = t2m - 273.15 if t2m.max() > 100 else t2m
        td_c = td2m - 273.15 if td2m.max() > 100 else td2m
        t_gpu = cp.asarray(t_c.ravel().astype(np.float64))
        td_gpu = cp.asarray(td_c.ravel().astype(np.float64))
        lcl_p, lcl_t = thermo.lcl(cp.full(n, 1013.0), t_gpu, td_gpu)
        lcl_h = cp.clip((1013.0 - lcl_p.ravel()) * 10, 0, 5000)
        stp = cp.asnumpy(wind.significant_tornado_parameter(
            cp.asarray(cape.ravel().astype(np.float64)), lcl_h,
            cp.asarray(srh1.ravel().astype(np.float64)),
            cp.asarray(shear.ravel().astype(np.float64)))).reshape(ny, nx)
        cp.get_default_memory_pool().free_all_blocks()
    plot_stp(axes[1, 2], stp, **geo_kwargs)
    plot_theta_e(axes[1, 3], t2m, td2m, **geo_kwargs)

    # Row 3: PWAT, Vorticity, Frontogenesis, Wind Speed
    plot_pwat(axes[2, 0], fields.get('pwat', np.zeros((ny, nx))), **geo_kwargs)
    plot_vorticity(axes[2, 1], u10, v10, **geo_kwargs)
    plot_frontogenesis(axes[2, 2], t2m, u10, v10, **geo_kwargs)
    plot_wind_speed(axes[2, 3], u10, v10, **geo_kwargs)

    for ax in axes.flat:
        ax.set_facecolor('#1a1a2e')
        ax.title.set_color('white')
        try:
            ax.spines['geo'].set_edgecolor('#444466')
        except (KeyError, AttributeError):
            pass

    fig.suptitle(title, fontsize=16, color='#00ff88', fontweight='bold', y=0.98)
    fig.subplots_adjust(hspace=0.35, wspace=0.1)

    if save:
        fig.savefig(save, dpi=dpi, facecolor=fig.get_facecolor(),
                    bbox_inches='tight')

    return fig


# ---------------------------------------------------------------------------
# Cached overlay system for fast mesoanalysis rendering
# ---------------------------------------------------------------------------
_OVERLAY_CACHE_DIR = os.path.join(os.path.dirname(__file__), '_overlay_cache')
os.makedirs(_OVERLAY_CACHE_DIR, exist_ok=True)

_overlay_cache = {}  # in-memory cache: key -> PIL Image (RGBA)


def _get_map_overlay(panel_w, panel_h, extent=(-125, -66, 23, 50)):
    """Get or create a cached transparent map overlay image.

    First call renders with cartopy (~700ms).  Subsequent calls return the
    cached PIL Image instantly.
    """
    key = f"{panel_w}x{panel_h}_{extent}"
    if key in _overlay_cache:
        return _overlay_cache[key]

    # Check disk cache
    cache_file = os.path.join(
        _OVERLAY_CACHE_DIR,
        f"overlay_{hashlib.md5(key.encode()).hexdigest()[:8]}.png",
    )
    if os.path.exists(cache_file):
        overlay = Image.open(cache_file).convert('RGBA')
        _overlay_cache[key] = overlay
        return overlay

    # Render with cartopy (slow, but only once)
    dpi = 100
    fig_w = panel_w / dpi
    fig_h = panel_h / dpi

    fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1], projection=_LAMBERT)
    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_1_states_provinces_lines', '50m',
        facecolor='none', edgecolor='black', linewidth=0.5))
    ax.add_feature(cfeature.NaturalEarthFeature(
        'physical', 'coastline', '50m',
        facecolor='none', edgecolor='black', linewidth=0.8))
    ax.add_feature(cfeature.NaturalEarthFeature(
        'cultural', 'admin_0_boundary_lines_land', '50m',
        facecolor='none', edgecolor='black', linewidth=0.5))
    ax.patch.set_alpha(0)
    fig.patch.set_alpha(0)

    fig.savefig(cache_file, dpi=dpi, transparent=True)
    plt.close(fig)

    overlay = Image.open(cache_file).convert('RGBA')
    _overlay_cache[key] = overlay
    return overlay


def _compute_all_panels(fields):
    """Compute all 12 mesoanalysis products on GPU.

    Returns list of (data_2d, title, cmap_name, vmin, vmax, cb_label).
    """
    ny, nx = _infer_field_shape(fields)
    n = ny * nx

    _z = np.zeros((ny, nx))
    t2m = fields.get('t2m') if fields.get('t2m') is not None else _z
    td2m = fields.get('td2m') if fields.get('td2m') is not None else (t2m - 5)
    t2m_c = (t2m - 273.15) if t2m.max() > 100 else np.array(t2m, dtype=np.float64)
    td2m_c = (td2m - 273.15) if td2m.max() > 100 else np.array(td2m, dtype=np.float64)
    u10 = fields.get('u10') if fields.get('u10') is not None else _z
    v10 = fields.get('v10') if fields.get('v10') is not None else _z

    # GPU compute derived products
    t_gpu = cp.asarray(t2m_c.ravel().astype(np.float64))
    td_gpu = cp.asarray(td2m_c.ravel().astype(np.float64))

    theta_e = cp.asnumpy(thermo.equivalent_potential_temperature(
        cp.full(n, 1013.0), t_gpu, td_gpu)).reshape(ny, nx)

    # LCL for STP
    lcl_r = thermo.lcl(cp.full(n, 1013.0), t_gpu, td_gpu)
    lcl_p = cp.asnumpy(lcl_r[0] if isinstance(lcl_r, tuple) else lcl_r)
    lcl_h = np.clip((1013.0 - lcl_p.ravel()) * 10, 0, 5000)

    _zeros = np.zeros((ny, nx))
    cape = fields.get('cape') if fields.get('cape') is not None else _zeros
    srh1 = fields.get('srh1') if fields.get('srh1') is not None else _zeros
    shear = fields.get('shear') if fields.get('shear') is not None else _zeros

    stp = cp.asnumpy(wind.significant_tornado_parameter(
        cp.asarray(cape.ravel().astype(np.float64)),
        cp.asarray(lcl_h.ravel().astype(np.float64)),
        cp.asarray(srh1.ravel().astype(np.float64)),
        cp.asarray(shear.ravel().astype(np.float64)))).reshape(ny, nx)

    ug = cp.asarray(u10.astype(np.float64))
    vg = cp.asarray(v10.astype(np.float64))
    tg = cp.asarray(t2m_c.astype(np.float64))
    vort = cp.asnumpy(grid.vorticity(ug, vg, 3000.0, 3000.0)) * 1e5
    fronto = cp.asnumpy(grid.frontogenesis(tg, ug, vg, 3000.0, 3000.0)) * 1e9
    wspd = np.sqrt(u10**2 + v10**2) * 1.944

    cp.get_default_memory_pool().free_all_blocks()

    refc = fields.get('refc') if fields.get('refc') is not None else _zeros
    srh3 = fields.get('srh3') if fields.get('srh3') is not None else _zeros
    pwat = fields.get('pwat') if fields.get('pwat') is not None else _zeros

    return [
        (t2m_c * 9/5 + 32,     "Temperature (F)",      "RdYlBu_r",  -20, 120, "F"),
        (td2m_c * 9/5 + 32,    "Dewpoint (F)",         "YlGn",        10,  85, "F"),
        (cape,                   "SBCAPE (J/kg)",        "hot_r",        0, 5000, "J/kg"),
        (refc,                   "Reflectivity (dBZ)",   "turbo",      -10,  75, "dBZ"),
        (srh3,                   "0-3km SRH (m2/s2)",   "BuPu",         0, 500, "m2/s2"),
        (shear,                  "0-6km Shear (m/s)",    "YlOrRd",       0,  40, "m/s"),
        (stp,                    "STP",                  "RdPu",         0,  10, ""),
        (theta_e,                "Theta-E (K)",          "magma",      280, 370, "K"),
        (pwat,                   "PWAT (mm)",            "GnBu",         0,  70, "mm"),
        (vort,                   "Vorticity (x10-5)",    "RdBu_r",     -30,  30, "1/s"),
        (fronto,                 "Frontogenesis",        "RdBu_r",      -5,   5, "K/m/s"),
        (wspd,                   "Wind Speed (kts)",     "YlOrRd",       0,  40, "kts"),
    ]


def _mesoanalysis_fast_legacy(fields, title='', save=None):
    """12-panel mesoanalysis using cached map overlays.

    First call: ~8s (renders 12 map overlays to transparent PNGs).
    Subsequent calls: ~50-100ms (GPU colormap + PIL composite).

    Parameters
    ----------
    fields : dict
        Must contain 2D arrays for t2m, td2m, cape, refc, etc.
    title : str
        Suptitle text (e.g. 'HRRR 2026-03-28 18:00 F06').
    save : str or None
        Path to save the output PNG.
    This legacy compositor is fast, but it assumes an image-like HRRR extent.
    Use the cartopy path when exact geographic alignment matters.
    """
    # Panel layout
    cols, rows = 4, 3
    panel_w, panel_h = 400, 280
    gap = 4
    header_h = 40
    cb_h = 30
    total_w = cols * panel_w + (cols + 1) * gap
    total_h = header_h + rows * (panel_h + cb_h) + (rows + 1) * gap

    # Get cached map overlay (rendered once, then free)
    overlay = _get_map_overlay(panel_w, panel_h)

    # Compute all data fields on GPU
    panels = _compute_all_panels(fields)

    # Build composite image with PIL
    composite = Image.new('RGBA', (total_w, total_h), (26, 26, 46, 255))
    draw = ImageDraw.Draw(composite)

    # Fonts
    try:
        font_title = ImageFont.truetype("consola.ttf", 18)
    except OSError:
        font_title = ImageFont.load_default()
    try:
        font_label = ImageFont.truetype("consola.ttf", 11)
    except OSError:
        font_label = ImageFont.load_default()

    # Title bar
    draw.text((gap + 5, 8), title, fill=(0, 255, 136), font=font_title)

    for idx, (data, panel_title, cmap_name, vmin, vmax, cb_label) in enumerate(panels):
        row, col = divmod(idx, cols)
        x = gap + col * (panel_w + gap)
        y = header_h + gap + row * (panel_h + cb_h + gap)

        # GPU colormap -> RGBA
        # HRRR Lambert grid: col 0 = east, needs horizontal flip
        cmap = plt.get_cmap(cmap_name)
        data_f = np.fliplr(np.asarray(data, dtype=np.float64))
        norm = np.clip((data_f - vmin) / (vmax - vmin + 1e-12), 0, 1)
        rgba = (cmap(norm) * 255).astype(np.uint8)

        # Create PIL image, resize to panel size
        data_img = Image.fromarray(rgba, 'RGBA').resize(
            (panel_w, panel_h), Image.BILINEAR)

        # Composite: data + map overlay
        panel_img = Image.alpha_composite(data_img, overlay)

        # Paste into composite
        composite.paste(panel_img, (x, y))

        # Panel title
        draw.text((x + 3, y + 2), panel_title, fill=(255, 255, 255), font=font_label)

        # Simple colorbar (horizontal gradient strip)
        cb_y = y + panel_h + 2
        for cx in range(panel_w):
            t = cx / panel_w
            r, g, b, a = cmap(t)
            draw.line([(x + cx, cb_y), (x + cx, cb_y + 12)],
                      fill=(int(r * 255), int(g * 255), int(b * 255)))
        # Colorbar tick labels
        for i, val in enumerate(np.linspace(vmin, vmax, 5)):
            lx = x + int(i / 4 * (panel_w - 20))
            draw.text((lx, cb_y + 13), f"{val:.0f}",
                      fill=(200, 200, 200), font=font_label)

    if save:
        composite.save(save, 'PNG')

    return composite


def mesoanalysis_fast(fields, title='', save=None, dpi=150, figsize=(22, 13)):
    """12-panel mesoanalysis with accuracy-first behavior on a geogrid.

    If `fields` includes 2D latitude/longitude arrays, this function uses the
    cartopy path so the Lambert grid is rendered correctly.  Otherwise it falls
    back to the legacy cached-overlay compositor for speed.
    """
    ny, nx = _infer_field_shape(fields)
    lats, lons = _extract_geo_grid(fields, (ny, nx))
    if lats is None or lons is None:
        return _mesoanalysis_fast_legacy(fields, title=title, save=save)

    fig = mesoanalysis_cartopy(fields, title=title, figsize=figsize, dpi=dpi, save=None)
    return _figure_to_image(fig, dpi=dpi, save=save)


# Default mesoanalysis points to the fast path
mesoanalysis = mesoanalysis_fast


def single_plot(data, title, cmap='RdYlBu_r', vmin=None, vmax=None,
                cb_label=None, save=None, dpi=200, figsize=(14, 9),
                discrete_levels=None, extend='both', lats=None, lons=None):
    """Single high-resolution map with cartopy.

    Parameters
    ----------
    data : 2D array
        Field to plot.
    title : str
        Figure title (model info + valid time).
    cmap : str
        Colormap name.
    vmin, vmax : float
        Color limits.
    cb_label : str
        Colorbar label.
    save : str or None
        Path to save the figure.
    dpi : int
        Output resolution.
    figsize : tuple
        Figure size in inches.
    discrete_levels : array-like or None
        If provided, use discrete color bands.
    extend : str
        Colorbar extend mode.

    Returns
    -------
    fig : matplotlib Figure
    """
    fig = plt.figure(figsize=figsize, facecolor='white')
    ax = fig.add_subplot(1, 1, 1, projection=_LAMBERT)

    plot_field(ax, data, title, cmap=cmap, vmin=vmin, vmax=vmax,
               colorbar=True, cb_label=cb_label,
               discrete_levels=discrete_levels, extend=extend,
               lats=lats, lons=lons)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig
