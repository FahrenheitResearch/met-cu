"""
GPU-accelerated meteorology plots.

Each function: downloads/accepts data -> GPU compute -> GPU render -> matplotlib figure.
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from metcu.mpl.gpu_render import gpu_contourf, gpu_pcolormesh
from metcu.kernels import thermo, wind, grid


def plot_temperature(ax, t2m, cmap='RdYlBu_r', units='F', **kwargs):
    """Plot 2m temperature field.

    t2m: 2D array in Kelvin or Celsius (auto-detected)
    """
    t = t2m - 273.15 if np.nanmax(t2m) > 100 else t2m
    if units == 'F':
        t = t * 9 / 5 + 32
        levels = kwargs.pop('levels', np.arange(-20, 125, 5))
        label = 'Temperature (F)'
    else:
        levels = kwargs.pop('levels', np.arange(-40, 50, 2.5))
        label = 'Temperature (C)'
    sm = gpu_contourf(ax, t, levels=levels, cmap=cmap, extend='both', **kwargs)
    ax.set_title(label)
    return sm


def plot_dewpoint(ax, td2m, cmap='YlGn', units='F', **kwargs):
    """Plot 2m dewpoint field."""
    td = td2m - 273.15 if np.nanmax(td2m) > 100 else td2m
    if units == 'F':
        td = td * 9 / 5 + 32
        levels = kwargs.pop('levels', np.arange(-10, 85, 5))
        label = 'Dewpoint (F)'
    else:
        levels = kwargs.pop('levels', np.arange(-30, 35, 2.5))
        label = 'Dewpoint (C)'
    sm = gpu_contourf(ax, td, levels=levels, cmap=cmap, extend='both', **kwargs)
    ax.set_title(label)
    return sm


def plot_cape(ax, cape, cmap='hot_r', **kwargs):
    """Plot CAPE field."""
    levels = kwargs.pop('levels', [0, 250, 500, 1000, 1500, 2000, 3000, 4000, 5000])
    sm = gpu_contourf(ax, cape, levels=levels, cmap=cmap, extend='max', **kwargs)
    ax.set_title('SBCAPE (J/kg)')
    return sm


def plot_reflectivity(ax, refc, cmap='turbo', **kwargs):
    """Plot composite reflectivity."""
    levels = kwargs.pop('levels', np.arange(-10, 75, 5))
    sm = gpu_contourf(ax, refc, levels=levels, cmap=cmap, extend='both', **kwargs)
    ax.set_title('Composite Reflectivity (dBZ)')
    return sm


def plot_srh(ax, srh, cmap='BuPu', **kwargs):
    """Plot storm-relative helicity."""
    levels = kwargs.pop('levels', [0, 50, 100, 150, 200, 300, 400, 500])
    sm = gpu_contourf(ax, srh, levels=levels, cmap=cmap, extend='max', **kwargs)
    ax.set_title('0-3km SRH (m2/s2)')
    return sm


def plot_shear(ax, shear, cmap='YlOrRd', **kwargs):
    """Plot bulk wind shear."""
    levels = kwargs.pop('levels', [0, 5, 10, 15, 20, 25, 30, 35, 40])
    sm = gpu_contourf(ax, shear, levels=levels, cmap=cmap, extend='max', **kwargs)
    ax.set_title('0-6km Bulk Shear (m/s)')
    return sm


def plot_stp(ax, stp, cmap='RdPu', **kwargs):
    """Plot Significant Tornado Parameter."""
    levels = kwargs.pop('levels', [0, 0.5, 1, 2, 3, 4, 5, 8, 10])
    sm = gpu_contourf(ax, stp, levels=levels, cmap=cmap, extend='max', **kwargs)
    ax.set_title('STP')
    return sm


def plot_theta_e(ax, t2m, td2m, p_sfc=1013.0, cmap='magma', **kwargs):
    """Compute and plot surface theta-e on GPU."""
    t = t2m - 273.15 if np.nanmax(t2m) > 100 else t2m
    td = td2m - 273.15 if np.nanmax(td2m) > 100 else td2m
    n = t.size
    t_gpu = cp.asarray(t.ravel().astype(np.float64))
    td_gpu = cp.asarray(td.ravel().astype(np.float64))
    theta_e = cp.asnumpy(thermo.equivalent_potential_temperature(
        cp.full(n, p_sfc), t_gpu, td_gpu)).reshape(t.shape)
    cp.get_default_memory_pool().free_all_blocks()
    levels = kwargs.pop('levels', None)
    sm = gpu_contourf(ax, theta_e, levels=levels or 20, cmap=cmap, extend='both', **kwargs)
    ax.set_title('Theta-E (K)')
    return sm


def plot_vorticity(ax, u, v, dx=3000.0, dy=3000.0, cmap='RdBu_r', **kwargs):
    """Compute and plot relative vorticity on GPU."""
    vort = cp.asnumpy(grid.vorticity(
        cp.asarray(u.astype(np.float64)),
        cp.asarray(v.astype(np.float64)),
        dx, dy)) * 1e5
    cp.get_default_memory_pool().free_all_blocks()
    levels = kwargs.pop('levels', np.arange(-30, 35, 5))
    sm = gpu_contourf(ax, vort, levels=levels, cmap=cmap, extend='both', **kwargs)
    ax.set_title('Vorticity (x10-5 /s)')
    return sm


def plot_frontogenesis(ax, t, u, v, dx=3000.0, dy=3000.0, cmap='RdBu_r', **kwargs):
    """Compute and plot frontogenesis on GPU."""
    t_c = t - 273.15 if np.nanmax(t) > 100 else t
    fronto = cp.asnumpy(grid.frontogenesis(
        cp.asarray(t_c.astype(np.float64)),
        cp.asarray(u.astype(np.float64)),
        cp.asarray(v.astype(np.float64)),
        dx, dy)) * 1e9
    cp.get_default_memory_pool().free_all_blocks()
    levels = kwargs.pop('levels', None)
    sm = gpu_contourf(ax, fronto, levels=levels or 20, cmap=cmap, extend='both', **kwargs)
    ax.set_title('Frontogenesis (x10-9 K/m/s)')
    return sm


def plot_wind_speed(ax, u, v, cmap='YlOrRd', knots=True, **kwargs):
    """Plot wind speed from u/v components."""
    wspd = np.sqrt(u**2 + v**2)
    if knots:
        wspd *= 1.944
        label = 'Wind Speed (kts)'
    else:
        label = 'Wind Speed (m/s)'
    levels = kwargs.pop('levels', [0, 5, 10, 15, 20, 25, 30, 35, 40])
    sm = gpu_contourf(ax, wspd, levels=levels, cmap=cmap, extend='max', **kwargs)
    ax.set_title(label)
    return sm


def plot_pwat(ax, pwat, cmap='GnBu', **kwargs):
    """Plot precipitable water."""
    levels = kwargs.pop('levels', None)
    sm = gpu_contourf(ax, pwat, levels=levels or 20, cmap=cmap, extend='both', **kwargs)
    ax.set_title('PWAT (mm)')
    return sm


def mesoanalysis(fields, title='', figsize=(22, 13), dpi=120, save=None):
    """Generate a full 12-panel mesoanalysis from a dict of fields.

    fields should contain: t2m, td2m, cape, refc, srh3, shear, u10, v10, pwat
    Optional: stp, cin, srh1

    All GPU computation and rendering happens here.
    Returns the figure.
    """
    fig, axes = plt.subplots(3, 4, figsize=figsize)
    fig.patch.set_facecolor('black')

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
        # Compute STP on GPU
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
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('black')
        ax.title.set_color('white')

    fig.suptitle(title, fontsize=14, color='lime', fontweight='bold', y=0.98)

    if save:
        fig.savefig(save, dpi=dpi, facecolor='black', bbox_inches='tight')

    return fig
