"""
Backend-agnostic GPU array operations.

Provides a unified namespace (`gpu`) that wraps either CuPy (CUDA) or
MetalArray + numpy (Metal) so that calc.py can use `gpu.asarray(...)`,
`gpu.asnumpy(...)`, etc. without backend-specific code.
"""

from __future__ import annotations

import numpy as np
from metcu.backend import get_backend, Backend

_backend = get_backend()

if _backend == Backend.METAL:
    from metcu.metal.runtime import MetalArray, to_metal, to_numpy

    def asarray(data, dtype=None):
        """Convert to GPU array (MetalArray on Metal backend)."""
        if isinstance(data, MetalArray):
            return data
        return to_metal(data)

    def asnumpy(arr):
        """Convert GPU array to numpy."""
        if isinstance(arr, MetalArray):
            return arr.numpy()
        return np.asarray(arr, dtype=np.float64)

    def ascontiguousarray(arr, dtype=None):
        if isinstance(arr, MetalArray):
            return arr
        return to_metal(arr)

    def zeros(shape, dtype=None):
        return MetalArray(shape=shape if isinstance(shape, tuple) else (shape,))

    def zeros_like(arr):
        if isinstance(arr, MetalArray):
            return MetalArray(shape=arr.shape)
        return MetalArray(data=np.zeros_like(np.asarray(arr)))

    def ones_like(arr):
        if isinstance(arr, MetalArray):
            return MetalArray(data=np.ones(arr.shape, dtype=np.float32))
        return MetalArray(data=np.ones_like(np.asarray(arr), dtype=np.float32))

    def empty(n, dtype=None):
        return MetalArray(shape=(n,) if isinstance(n, int) else n)

    def empty_like(arr):
        if isinstance(arr, MetalArray):
            return MetalArray(shape=arr.shape)
        return MetalArray(data=np.empty_like(np.asarray(arr)))

    def full(shape, val, dtype=None):
        if isinstance(shape, int):
            shape = (shape,)
        return MetalArray(data=np.full(shape, val, dtype=np.float32))

    def full_like(arr, val):
        if isinstance(arr, MetalArray):
            return MetalArray(data=np.full(arr.shape, val, dtype=np.float32))
        return MetalArray(data=np.full_like(np.asarray(arr), val, dtype=np.float32))

    def flip(arr, axis=0):
        a = asnumpy(arr)
        return to_metal(np.flip(a, axis=axis).copy())

    # Math operations (fall back to numpy, return MetalArray)
    def sqrt(x):
        return to_metal(np.sqrt(asnumpy(x)))

    def abs(x):
        return to_metal(np.abs(asnumpy(x)))

    def sin(x):
        return to_metal(np.sin(asnumpy(x)))

    def cos(x):
        return to_metal(np.cos(asnumpy(x)))

    def exp(x):
        return to_metal(np.exp(asnumpy(x)))

    def log(x):
        return to_metal(np.log(asnumpy(x)))

    def deg2rad(x):
        return to_metal(np.deg2rad(asnumpy(x)))

    def rad2deg(x):
        return to_metal(np.rad2deg(asnumpy(x)))

    def arccos(x):
        return to_metal(np.arccos(asnumpy(x)))

    def clip(x, a, b):
        return to_metal(np.clip(asnumpy(x), a, b))

    def where(cond, x, y):
        c = asnumpy(cond) if isinstance(cond, MetalArray) else np.asarray(cond)
        xn = asnumpy(x) if isinstance(x, MetalArray) else x
        yn = asnumpy(y) if isinstance(y, MetalArray) else y
        return to_metal(np.where(c, xn, yn))

    def maximum(x, y):
        xn = asnumpy(x) if isinstance(x, MetalArray) else x
        yn = asnumpy(y) if isinstance(y, MetalArray) else y
        return to_metal(np.maximum(xn, yn))

    def minimum(x, y):
        xn = asnumpy(x) if isinstance(x, MetalArray) else x
        yn = asnumpy(y) if isinstance(y, MetalArray) else y
        return to_metal(np.minimum(xn, yn))

    def mean(x, axis=None):
        return to_metal(np.mean(asnumpy(x), axis=axis))

    def sum(x, axis=None):
        result = np.sum(asnumpy(x), axis=axis)
        if np.ndim(result) == 0:
            return float(result)
        return to_metal(result)

    def argmax(x, axis=None):
        return int(np.argmax(asnumpy(x), axis=axis))

    def trapz(y, x=None, axis=-1):
        yn = asnumpy(y)
        xn = asnumpy(x) if x is not None else None
        return float(np.trapz(yn, xn, axis=axis))

    def gradient(arr, *args, **kwargs):
        a = asnumpy(arr)
        np_args = [asnumpy(x) if isinstance(x, MetalArray) else x for x in args]
        result = np.gradient(a, *np_args, **kwargs)
        if isinstance(result, list):
            return [to_metal(r) for r in result]
        return to_metal(result)

    def nan_to_num(x, nan=0.0):
        return to_metal(np.nan_to_num(asnumpy(x), nan=nan))

    def isnan(x):
        return to_metal(np.isnan(asnumpy(x)).astype(np.float32))

    def searchsorted(a, v, side='left'):
        an = asnumpy(a)
        vn = asnumpy(v) if isinstance(v, MetalArray) else np.asarray(v)
        return to_metal(np.searchsorted(an, vn, side=side).astype(np.float32))

    def array(data, dtype=None):
        return to_metal(np.asarray(data, dtype=np.float32 if dtype is None else dtype))

    class _NoOpPool:
        def free_all_blocks(self): pass

    def get_default_memory_pool():
        return _NoOpPool()

    nan = np.nan
    int32 = np.int32
    int64 = np.int64
    float32 = np.float32
    float64 = np.float64

    ndarray = MetalArray

else:
    # CUDA backend — just re-export cupy
    import cupy as _cp

    asarray = _cp.asarray
    asnumpy = _cp.asnumpy
    ascontiguousarray = _cp.ascontiguousarray
    zeros = _cp.zeros
    zeros_like = _cp.zeros_like
    ones_like = _cp.ones_like
    empty = _cp.empty
    empty_like = _cp.empty_like
    full = _cp.full
    full_like = _cp.full_like
    flip = _cp.flip
    sqrt = _cp.sqrt
    abs = _cp.abs
    sin = _cp.sin
    cos = _cp.cos
    exp = _cp.exp
    log = _cp.log
    deg2rad = _cp.deg2rad
    rad2deg = _cp.rad2deg
    arccos = _cp.arccos
    clip = _cp.clip
    where = _cp.where
    maximum = _cp.maximum
    minimum = _cp.minimum
    mean = _cp.mean
    sum = _cp.sum
    argmax = _cp.argmax
    trapz = _cp.trapz
    gradient = _cp.gradient
    nan_to_num = _cp.nan_to_num
    isnan = _cp.isnan
    searchsorted = _cp.searchsorted
    array = _cp.array
    get_default_memory_pool = _cp.get_default_memory_pool
    nan = _cp.nan
    int32 = _cp.int32
    int64 = _cp.int64
    float32 = _cp.float32
    float64 = _cp.float64
    ndarray = _cp.ndarray
