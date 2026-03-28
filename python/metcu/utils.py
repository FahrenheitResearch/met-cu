"""met-cu utility functions for GPU array management and CUDA tuning."""

import cupy as cp
import numpy as np


def to_gpu(arr):
    """Convert any array-like to cupy GPU array.

    Handles numpy arrays, Python lists/scalars, and pint Quantities.

    Parameters
    ----------
    arr : array-like, cupy.ndarray, or pint.Quantity
        Input data.

    Returns
    -------
    cupy.ndarray
        Float64 array on the current GPU device.
    """
    if isinstance(arr, cp.ndarray):
        return arr
    if hasattr(arr, 'magnitude'):  # pint Quantity
        arr = arr.magnitude
    return cp.asarray(arr, dtype=cp.float64)


def to_cpu(arr):
    """Convert cupy array to numpy.

    Parameters
    ----------
    arr : cupy.ndarray or numpy.ndarray
        Input array.

    Returns
    -------
    numpy.ndarray
    """
    if isinstance(arr, cp.ndarray):
        return cp.asnumpy(arr)
    return np.asarray(arr)


def strip_units(x):
    """Strip pint units, return raw magnitude.

    Parameters
    ----------
    x : pint.Quantity or array-like
        Input that may carry pint units.

    Returns
    -------
    numpy.ndarray or scalar
        Raw numeric value without unit metadata.
    """
    if hasattr(x, 'magnitude'):
        return x.magnitude
    return x


def get_gpu_info():
    """Print and return GPU device properties.

    Returns
    -------
    dict
        Device properties from CUDA runtime.
    """
    dev = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(dev.id)
    print(f"GPU: {props['name'].decode()}")
    print(f"Compute capability: {props['major']}.{props['minor']}")
    print(f"Memory: {props['totalGlobalMem'] / 1e9:.1f} GB")
    print(f"SM count: {props['multiProcessorCount']}")
    return props


def optimal_block_size(n):
    """Choose optimal CUDA block size for N elements.

    Parameters
    ----------
    n : int
        Number of elements.

    Returns
    -------
    int
        Recommended block size (threads per block).
    """
    if n < 256:
        return 64
    if n < 1024:
        return 128
    return 256


def optimal_grid_2d(ny, nx, block=(16, 16)):
    """Compute grid dimensions for 2D kernel.

    Parameters
    ----------
    ny : int
        Number of rows.
    nx : int
        Number of columns.
    block : tuple of (int, int)
        Block dimensions (default (16, 16)).

    Returns
    -------
    tuple of (int, int)
        Grid dimensions (grid_x, grid_y).
    """
    return ((nx + block[0] - 1) // block[0],
            (ny + block[1] - 1) // block[1])
