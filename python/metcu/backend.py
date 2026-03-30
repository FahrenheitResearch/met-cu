"""
Backend selection for met-cu.

Automatically picks the best available GPU backend:
  1. CUDA (CuPy) — NVIDIA GPUs
  2. Metal — Apple Silicon (M1/M2/M3/M4)
  3. CPU (numpy) — fallback

Usage
-----
    from metcu.backend import gpu, Backend

    arr = gpu.array([1.0, 2.0, 3.0])   # uses best backend
    result = arr.get()                   # numpy array
"""

from __future__ import annotations

import platform
import sys
from enum import Enum


class Backend(Enum):
    CUDA = "cuda"
    METAL = "metal"
    CPU = "cpu"


_active_backend: Backend | None = None


def detect_backend() -> Backend:
    """Detect the best available backend."""
    global _active_backend
    if _active_backend is not None:
        return _active_backend

    # Check environment variable override
    import os
    env_backend = os.environ.get('METCU_BACKEND', '').lower()
    if env_backend in ('metal', 'cuda', 'cpu'):
        _active_backend = Backend(env_backend)
        return _active_backend

    # Try CUDA first
    try:
        import cupy as cp  # noqa: F401
        cp.cuda.Device(0)  # will raise if no CUDA device
        _active_backend = Backend.CUDA
        return _active_backend
    except Exception:
        pass

    # Try Metal (macOS only)
    if platform.system() == "Darwin":
        try:
            from metcu.metal.runtime import is_available
            if is_available():
                _active_backend = Backend.METAL
                return _active_backend
        except Exception:
            pass

    # CPU fallback
    _active_backend = Backend.CPU
    return _active_backend


def get_backend() -> Backend:
    """Return the current backend (auto-detect on first call)."""
    return detect_backend()


def set_backend(backend: Backend | str):
    """Force a specific backend."""
    global _active_backend
    if isinstance(backend, str):
        backend = Backend(backend.lower())
    _active_backend = backend


def get_kernels():
    """Import and return the appropriate kernel module for the active backend.

    Returns a module-like namespace with thermo, wind, grid sub-modules.
    """
    b = get_backend()
    if b == Backend.CUDA:
        from metcu import kernels
        return kernels
    elif b == Backend.METAL:
        from metcu import metal as metal_kernels
        return metal_kernels
    else:
        # CPU fallback — use numpy implementations
        raise ImportError(
            "No GPU backend available. Install cupy (NVIDIA) or "
            "pyobjc-framework-Metal (macOS) for GPU acceleration."
        )
