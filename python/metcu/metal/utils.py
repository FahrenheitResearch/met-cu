"""met-cu utility functions for Metal GPU array management."""

import numpy as np
from .runtime import MetalArray, MetalDevice, metal_device, to_metal, to_numpy


def to_gpu(arr):
    """Convert any array-like to MetalArray on the GPU.

    Handles numpy arrays, Python lists/scalars, and pint Quantities.
    """
    return to_metal(arr)


def to_cpu(arr):
    """Convert MetalArray to numpy."""
    return to_numpy(arr)


def strip_units(x):
    """Strip pint units, return raw magnitude."""
    if hasattr(x, 'magnitude'):
        return x.magnitude
    return x


def get_gpu_info():
    """Print and return Metal GPU device info."""
    dev = metal_device()
    name = dev.name
    print(f"GPU: {name}")
    print(f"Backend: Metal (Apple Silicon)")
    return {"name": name, "backend": "metal"}


def optimal_block_size(n):
    """Choose optimal threadgroup size for N elements."""
    if n < 256:
        return 64
    if n < 1024:
        return 128
    return 256


def optimal_grid_2d(ny, nx, block=(16, 16)):
    """Compute grid dimensions for 2D kernel."""
    return ((nx + block[0] - 1) // block[0],
            (ny + block[1] - 1) // block[1])
