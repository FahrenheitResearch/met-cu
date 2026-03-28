"""
GPU-accelerated matplotlib for meteorology.

Usage:
    import metcu.mpl  # auto-patches matplotlib
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.contourf(data, levels=20, cmap='RdYlBu_r')  # GPU accelerated!
    plt.colorbar()
    plt.savefig('output.png')

Or explicit:
    from metcu.mpl import gpu_contourf, gpu_pcolormesh, gpu_barbs
    gpu_contourf(ax, data, levels=20, cmap='RdYlBu_r')
"""

from matplotlib.axes import Axes

from . import gpu_render
from .gpu_render import gpu_contourf, gpu_pcolormesh, gpu_barbs
from .compat import GPUContourSet, GPUQuadMesh
from .met_plots import (
    plot_temperature, plot_dewpoint, plot_cape, plot_reflectivity,
    plot_srh, plot_shear, plot_stp, plot_theta_e, plot_vorticity,
    plot_frontogenesis, plot_wind_speed, plot_pwat,
    mesoanalysis, mesoanalysis_fast, mesoanalysis_cartopy,
    single_plot, plot_field,
)

__all__ = [
    'gpu_contourf', 'gpu_pcolormesh', 'gpu_barbs',
    'GPUContourSet', 'GPUQuadMesh',
    'enable', 'disable', 'is_enabled',
    'plot_temperature', 'plot_dewpoint', 'plot_cape', 'plot_reflectivity',
    'plot_srh', 'plot_shear', 'plot_stp', 'plot_theta_e', 'plot_vorticity',
    'plot_frontogenesis', 'plot_wind_speed', 'plot_pwat',
    'mesoanalysis', 'mesoanalysis_fast', 'mesoanalysis_cartopy',
    'single_plot', 'plot_field',
]

# ---- Store originals ----
_originals = {
    'contourf': Axes.contourf,
    'pcolormesh': Axes.pcolormesh,
    'imshow': Axes.imshow,
    'barbs': Axes.barbs,
}

_patched = False


# ---- Wrapper functions that match Axes method signatures ----

def _patched_contourf(self, *args, **kwargs):
    """Drop-in replacement for Axes.contourf using GPU."""
    return gpu_contourf(self, *args, **kwargs)


def _patched_pcolormesh(self, *args, **kwargs):
    """Drop-in replacement for Axes.pcolormesh using GPU.

    Falls back to original for small arrays (e.g., colorbar internals)
    to maintain full API compatibility.
    """
    import numpy as np
    # Find the data array from args
    data_arr = args[0] if len(args) == 1 else (args[2] if len(args) >= 3 else None)
    if data_arr is not None:
        data_arr = np.asarray(data_arr)
        if data_arr.size < 10000:
            return _originals['pcolormesh'](self, *args, **kwargs)
    return gpu_pcolormesh(self, *args, **kwargs)


def _patched_barbs(self, *args, **kwargs):
    """Drop-in replacement for Axes.barbs using GPU."""
    return gpu_barbs(self, *args, **kwargs)


# ---- Enable / disable ----

def enable():
    """Monkey-patch matplotlib Axes to use GPU-accelerated rendering."""
    global _patched

    # Wire up originals into gpu_render so it can call imshow/barbs
    gpu_render._original_contourf = _originals['contourf']
    gpu_render._original_pcolormesh = _originals['pcolormesh']
    gpu_render._original_imshow = _originals['imshow']
    gpu_render._original_barbs = _originals['barbs']

    Axes.contourf = _patched_contourf
    Axes.pcolormesh = _patched_pcolormesh
    Axes.barbs = _patched_barbs
    # NOTE: we do NOT patch imshow — our GPU funcs *use* imshow for output

    _patched = True


def disable():
    """Restore original matplotlib Axes methods."""
    global _patched

    Axes.contourf = _originals['contourf']
    Axes.pcolormesh = _originals['pcolormesh']
    Axes.imshow = _originals['imshow']
    Axes.barbs = _originals['barbs']

    _patched = False


def is_enabled():
    """Return True if GPU patches are currently active."""
    return _patched


# ---- Auto-enable on import ----
enable()
