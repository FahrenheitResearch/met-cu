"""
Publication-quality meteorology plots with cartopy map overlays.

Each function: accepts data -> GPU compute -> GPU render -> cartopy GeoAxes figure.
Uses Lambert Conformal projection with state/country borders, coastlines, and
horizontal colorbars for a clean mesoanalysis style.
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable
from metcu.kernels import thermo, wind, grid

# HRRR Lambert grid approximate extent in lon/lat
_HRRR_EXTENT = [-134, -60, 21, 53]
# CONUS display extent
_CONUS_EXTENT = [-125, -66, 23, 50]
# Default projection
_LAMBERT = ccrs.LambertConformal(central_longitude=-96, standard_parallels=(33, 45))
_PLATE = ccrs.PlateCarree()

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


def plot_field(ax, data, title, cmap='RdYlBu_r', vmin=None, vmax=None,
               extend='both', colorbar=True, cb_label=None,
               extent=None, discrete_levels=None):
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

    im = ax.imshow(data_f, origin='lower', cmap=cmap_obj, norm=norm,
                   extent=extent, transform=_PLATE,
                   interpolation='bilinear')

    _add_map_features(ax)
    ax.set_title(title, fontsize=10, fontweight='bold')

    sm = ScalarMappable(norm=norm, cmap=cmap_obj)
    sm.set_array(data_f)

    if colorbar:
        cb = plt.colorbar(sm, ax=ax, orientation='horizontal',
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
                      colorbar=colorbar, cb_label=label)


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
                      colorbar=colorbar, cb_label=label)


def plot_cape(ax, cape, cmap='hot_r', colorbar=True, **kwargs):
    """Plot CAPE field with cartopy."""
    levels = np.array([0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000])
    return plot_field(ax, cape, 'SBCAPE (J/kg)', cmap=cmap,
                      vmin=0, vmax=5000, colorbar=colorbar,
                      cb_label='CAPE (J/kg)', discrete_levels=levels)


def plot_reflectivity(ax, refc, cmap='turbo', colorbar=True, **kwargs):
    """Plot composite reflectivity with cartopy."""
    return plot_field(ax, refc, 'Composite Reflectivity (dBZ)', cmap=cmap,
                      vmin=-10, vmax=75, colorbar=colorbar,
                      cb_label='dBZ')


def plot_srh(ax, srh, cmap='BuPu', colorbar=True, **kwargs):
    """Plot storm-relative helicity with cartopy."""
    levels = np.array([0, 50, 100, 150, 200, 300, 400, 500])
    return plot_field(ax, srh, '0-3km SRH (m2/s2)', cmap=cmap,
                      vmin=0, vmax=500, colorbar=colorbar,
                      cb_label='SRH (m\u00b2/s\u00b2)', discrete_levels=levels,
                      extend='max')


def plot_shear(ax, shear, cmap='YlOrRd', colorbar=True, **kwargs):
    """Plot bulk wind shear with cartopy."""
    levels = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40])
    return plot_field(ax, shear, '0-6km Bulk Shear (m/s)', cmap=cmap,
                      vmin=0, vmax=40, colorbar=colorbar,
                      cb_label='Shear (m/s)', discrete_levels=levels,
                      extend='max')


def plot_stp(ax, stp, cmap='RdPu', colorbar=True, **kwargs):
    """Plot Significant Tornado Parameter with cartopy."""
    levels = np.array([0, 0.5, 1, 2, 3, 4, 5, 8, 10])
    return plot_field(ax, stp, 'STP', cmap=cmap,
                      vmin=0, vmax=10, colorbar=colorbar,
                      cb_label='STP', discrete_levels=levels,
                      extend='max')


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
                      colorbar=colorbar, cb_label='Theta-E (K)')


def plot_vorticity(ax, u, v, dx=3000.0, dy=3000.0, cmap='RdBu_r', colorbar=True, **kwargs):
    """Compute and plot relative vorticity on GPU with cartopy."""
    vort = cp.asnumpy(grid.vorticity(
        cp.asarray(u.astype(np.float64)),
        cp.asarray(v.astype(np.float64)),
        dx, dy)) * 1e5
    cp.get_default_memory_pool().free_all_blocks()
    return plot_field(ax, vort, 'Vorticity (x10\u207b\u2075 /s)', cmap=cmap,
                      vmin=-30, vmax=30, colorbar=colorbar,
                      cb_label='Vorticity (x10\u207b\u2075 /s)')


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
                      cb_label='Frontogenesis (x10\u207b\u2079 K/m/s)')


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
                      extend='max')


def plot_pwat(ax, pwat, cmap='GnBu', colorbar=True, **kwargs):
    """Plot precipitable water with cartopy."""
    return plot_field(ax, pwat, 'PWAT (mm)', cmap=cmap,
                      colorbar=colorbar, cb_label='PWAT (mm)')


def mesoanalysis(fields, title='', figsize=(24, 16), dpi=120, save=None):
    """Generate a full 12-panel mesoanalysis with cartopy map overlays.

    fields should contain: t2m, td2m, cape, refc, srh3, shear, u10, v10, pwat
    Optional: stp, cin, srh1

    Returns the figure.
    """
    proj = _LAMBERT

    fig, axes = plt.subplots(3, 4, figsize=figsize,
                             subplot_kw={'projection': proj})
    fig.patch.set_facecolor('#1a1a2e')

    ny = nx = None
    for key in ('t2m', 'td2m', 'cape', 'refc'):
        if key in fields and fields[key] is not None:
            ny, nx = fields[key].shape
            break
    if ny is None:
        raise ValueError("No valid 2D fields found")
    n = ny * nx

    t2m = fields.get('t2m', np.zeros((ny, nx)))
    td2m = fields.get('td2m', t2m - 5)
    u10 = fields.get('u10', np.zeros((ny, nx)))
    v10 = fields.get('v10', np.zeros((ny, nx)))

    # Row 1: Temp, Dewpoint, CAPE, Reflectivity
    plot_temperature(axes[0, 0], t2m)
    plot_dewpoint(axes[0, 1], td2m)
    plot_cape(axes[0, 2], fields.get('cape', np.zeros((ny, nx))))
    plot_reflectivity(axes[0, 3], fields.get('refc', np.zeros((ny, nx))))

    # Row 2: SRH, Shear, STP, Theta-E
    plot_srh(axes[1, 0], fields.get('srh3', np.zeros((ny, nx))))
    plot_shear(axes[1, 1], fields.get('shear', np.zeros((ny, nx))))
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
    plot_stp(axes[1, 2], stp)
    plot_theta_e(axes[1, 3], t2m, td2m)

    # Row 3: PWAT, Vorticity, Frontogenesis, Wind Speed
    plot_pwat(axes[2, 0], fields.get('pwat', np.zeros((ny, nx))))
    plot_vorticity(axes[2, 1], u10, v10)
    plot_frontogenesis(axes[2, 2], t2m, u10, v10)
    plot_wind_speed(axes[2, 3], u10, v10)

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


def single_plot(data, title, cmap='RdYlBu_r', vmin=None, vmax=None,
                cb_label=None, save=None, dpi=200, figsize=(14, 9),
                discrete_levels=None, extend='both'):
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
               discrete_levels=discrete_levels, extend=extend)

    fig.tight_layout()

    if save:
        fig.savefig(save, dpi=dpi, bbox_inches='tight', facecolor='white')

    return fig
