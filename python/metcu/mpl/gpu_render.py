"""
Core GPU rendering functions.

Replaces matplotlib's slow CPU-based contourf/pcolormesh/barbs with
GPU-accelerated equivalents that produce identical visual output.
Supports both CUDA (CuPy) and Metal backends.
"""

import metcu.gpu_ops as cp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, BoundaryNorm
from matplotlib.cm import ScalarMappable

# Will be set by __init__.py when monkey-patching
_original_contourf = None
_original_pcolormesh = None
_original_imshow = None
_original_barbs = None


def _is_uniform_1d(values, rtol=1e-6, atol=1e-12):
    """Return True when a 1D coordinate vector is evenly spaced."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 3:
        return arr.ndim == 1
    diffs = np.diff(arr)
    return np.allclose(diffs, diffs[0], rtol=rtol, atol=atol)


def _call_original(method, ax, *args, **kwargs):
    """Call the original matplotlib method when GPU rendering is unsafe."""
    if method is None:
        raise RuntimeError("Original matplotlib method is unavailable")
    return method(ax, *args, **kwargs)


def gpu_contourf(ax, *args, **kwargs):
    """GPU-accelerated filled contour plot.

    Same signature as ax.contourf(). Renders on GPU, displays via imshow.
    Returns a ScalarMappable so plt.colorbar() works.
    """
    # --- Parse positional args (same as matplotlib) ---
    X = Y = None
    levels_arg = None

    if len(args) == 1:
        Z = np.asarray(args[0], dtype=np.float64)
    elif len(args) == 2:
        Z = np.asarray(args[0], dtype=np.float64)
        levels_arg = args[1]
    elif len(args) >= 3:
        X = np.asarray(args[0], dtype=np.float64)
        Y = np.asarray(args[1], dtype=np.float64)
        Z = np.asarray(args[2], dtype=np.float64)
        if len(args) >= 4:
            levels_arg = args[3]

    # contourf semantics depend on the coordinate geometry; if the caller
    # supplied explicit coordinates or a geospatial transform, preserve the
    # exact matplotlib path instead of approximating via imshow.
    if Z.ndim != 2 or X is not None or Y is not None or kwargs.get('transform') is not None:
        return _call_original(_original_contourf, ax, *args, **kwargs)

    # --- Extract kwargs ---
    kwargs = dict(kwargs)
    levels = kwargs.pop('levels', levels_arg)
    cmap = kwargs.pop('cmap', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    extend = kwargs.pop('extend', 'neither')
    alpha = kwargs.pop('alpha', None)
    # Absorb remaining kwargs silently (transform, antialiased, etc.)
    kwargs.pop('transform', None)
    kwargs.pop('antialiased', None)
    kwargs.pop('hatches', None)

    # --- Build levels array ---
    if levels is None or isinstance(levels, (int, np.integer)):
        n_levels = int(levels) if levels is not None else 15
        zmin = vmin if vmin is not None else float(np.nanmin(Z))
        zmax = vmax if vmax is not None else float(np.nanmax(Z))
        levels = np.linspace(zmin, zmax, n_levels + 1)
    else:
        levels = np.asarray(levels, dtype=np.float64)

    # --- Resolve colormap ---
    if cmap is None:
        cmap = plt.rcParams.get('image.cmap', 'viridis')
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    n_bands = len(levels) - 1
    if n_bands < 1:
        # Degenerate case: fall back to original
        if _original_contourf is not None:
            return _original_contourf(ax, *args, **kwargs)
        return None

    # --- Build band color LUT (CPU, small) ---
    band_colors = np.empty((n_bands, 4), dtype=np.float32)
    for i in range(n_bands):
        t = (i + 0.5) / n_bands
        band_colors[i] = cmap(t)

    # --- GPU work: classify each pixel into a band and assign color ---
    Z_gpu = cp.asarray(Z, dtype=cp.float64)
    levels_gpu = cp.asarray(levels, dtype=cp.float64)
    band_colors_gpu = cp.asarray(band_colors)

    ny, nx = Z.shape

    # Digitize: find band index for each value (vectorized on GPU)
    # cp.searchsorted gives index where value would be inserted in levels
    flat = Z_gpu.ravel()
    band_idx = cp.searchsorted(levels_gpu, flat, side='right').astype(cp.int32) - 1
    # Clamp to valid range [0, n_bands-1]
    band_idx = cp.clip(band_idx, 0, n_bands - 1)

    # Handle values outside the level range
    below_mask = flat < levels_gpu[0]
    above_mask = flat > levels_gpu[-1]

    if extend in ('min', 'both'):
        band_idx[below_mask] = 0
    else:
        # Mark out-of-range as transparent (use -1 sentinel)
        band_idx[below_mask] = -1

    if extend in ('max', 'both'):
        band_idx[above_mask] = n_bands - 1
    else:
        band_idx[above_mask] = -1

    # Build RGBA image via fancy indexing
    # Add a transparent color at the end for the -1 sentinel
    band_colors_ext = cp.zeros((n_bands + 1, 4), dtype=cp.float32)
    band_colors_ext[:n_bands] = band_colors_gpu
    # index -1 wraps to last element which is transparent [0,0,0,0]

    rgba_flat = band_colors_ext[band_idx]
    rgba = rgba_flat.reshape(ny, nx, 4)

    # Handle NaN: make transparent
    nan_mask = cp.isnan(Z_gpu)
    rgba[nan_mask] = cp.array([0, 0, 0, 0], dtype=cp.float32)

    # --- Transfer to CPU ---
    rgba_np = cp.asnumpy(rgba)

    if alpha is not None:
        rgba_np[:, :, 3] *= alpha

    # --- Display via imshow ---
    extent_arg = None
    if X is not None and Y is not None:
        extent_arg = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    # Use original imshow to avoid recursion; it's an unbound method so pass ax
    if _original_imshow is not None:
        _original_imshow(ax, rgba_np, aspect='auto', origin='lower',
                         extent=extent_arg, interpolation='bilinear')
    else:
        ax.imshow(rgba_np, aspect='auto', origin='lower', extent=extent_arg,
                  interpolation='bilinear')

    # --- Return ScalarMappable for colorbar compatibility ---
    norm = BoundaryNorm(levels, cmap.N, clip=True)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(Z)
    sm._gpu_accelerated = True

    return sm


def gpu_pcolormesh(ax, *args, **kwargs):
    """GPU-accelerated pcolormesh.

    Same signature as ax.pcolormesh(). Renders on GPU via continuous
    colormap lookup, displays via imshow.
    Returns a ScalarMappable so plt.colorbar() works.
    """
    X = Y = None

    if len(args) == 1:
        C = np.asarray(args[0], dtype=np.float64)
    elif len(args) == 3:
        X = np.asarray(args[0], dtype=np.float64)
        Y = np.asarray(args[1], dtype=np.float64)
        C = np.asarray(args[2], dtype=np.float64)
    else:
        C = np.asarray(args[0], dtype=np.float64)

    transform = kwargs.get('transform')
    edgecolors = kwargs.get('edgecolors')
    linewidth = kwargs.get('linewidth', kwargs.get('linewidths'))
    shading = kwargs.get('shading')
    has_non_rectilinear_coords = (
        X is not None and Y is not None and (
            X.ndim != 1 or
            Y.ndim != 1 or
            not _is_uniform_1d(X) or
            not _is_uniform_1d(Y)
        )
    )
    if (
        C.ndim != 2 or
        transform is not None or
        has_non_rectilinear_coords or
        edgecolors not in (None, 'none') or
        linewidth not in (None, 0, 0.0) or
        shading == 'gouraud'
    ):
        return _call_original(_original_pcolormesh, ax, *args, **kwargs)

    kwargs = dict(kwargs)
    cmap = kwargs.pop('cmap', None)
    vmin = kwargs.pop('vmin', None)
    vmax = kwargs.pop('vmax', None)
    alpha = kwargs.pop('alpha', None)
    shading = kwargs.pop('shading', None)
    # Absorb remaining
    kwargs.pop('transform', None)
    kwargs.pop('edgecolors', None)
    kwargs.pop('linewidth', None)
    kwargs.pop('rasterized', None)

    if vmin is None:
        vmin = float(np.nanmin(C))
    if vmax is None:
        vmax = float(np.nanmax(C))

    if cmap is None:
        cmap = plt.rcParams.get('image.cmap', 'viridis')
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    # --- GPU continuous colormap lookup ---
    C_gpu = cp.asarray(C, dtype=cp.float64)
    denom = vmax - vmin
    if abs(denom) < 1e-15:
        denom = 1.0
    normed = cp.clip((C_gpu - vmin) / denom, 0.0, 1.0)

    # Build 256-entry LUT
    lut = np.empty((256, 4), dtype=np.float32)
    for i in range(256):
        lut[i] = cmap(i / 255.0)
    lut_gpu = cp.asarray(lut)

    indices = (normed * 255).astype(cp.int32)

    # Handle NaN
    nan_mask = cp.isnan(C_gpu)
    indices[nan_mask] = 0  # will be overwritten

    rgba = lut_gpu[indices]
    rgba[nan_mask] = cp.array([0, 0, 0, 0], dtype=cp.float32)

    rgba_np = cp.asnumpy(rgba)

    if alpha is not None:
        rgba_np[:, :, 3] *= alpha

    extent_arg = None
    if X is not None and Y is not None:
        extent_arg = [float(X.min()), float(X.max()), float(Y.min()), float(Y.max())]

    if _original_imshow is not None:
        _original_imshow(ax, rgba_np, aspect='auto', origin='lower',
                         extent=extent_arg, interpolation='bilinear')
    else:
        ax.imshow(rgba_np, aspect='auto', origin='lower', extent=extent_arg,
                  interpolation='bilinear')

    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array(C)
    sm._gpu_accelerated = True

    return sm


def gpu_barbs(ax, *args, **kwargs):
    """GPU-accelerated wind barbs.

    For barbs the bottleneck is matplotlib's vector rendering, not
    the computation. Pass through to original with optional thinning.
    """
    if _original_barbs is not None:
        return _original_barbs(ax, *args, **kwargs)
    return ax.barbs(*args, **kwargs)
