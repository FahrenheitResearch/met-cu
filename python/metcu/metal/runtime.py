"""
Metal compute runtime for met-cu.

Provides GPU array management and kernel dispatch on Apple Silicon via PyObjC.
Metal compute shaders use float32 (Metal on M1 does not support float64 in
compute kernels), so all arrays are stored as float32 on the GPU.  The Python
API accepts float64 and converts transparently.

Key classes
-----------
MetalDevice   - singleton wrapping MTLDevice / command queue
MetalArray    - GPU buffer + shape/strides, numpy-interoperable
MetalKernel   - compiled compute pipeline from MSL source
"""

from __future__ import annotations

import ctypes
import platform
import struct
import numpy as np
from typing import Optional, Tuple, Union

_METAL_AVAILABLE = False
_device_singleton: Optional["MetalDevice"] = None

if platform.system() == "Darwin":
    try:
        import Metal  # pyobjc-framework-Metal
        import MetalKit  # noqa: F401
        _METAL_AVAILABLE = True
    except ImportError:
        try:
            import objc as _objc  # noqa: F401
            Metal = _objc.loadBundle(
                "Metal", {},
                bundle_path="/System/Library/Frameworks/Metal.framework",
            )
            _METAL_AVAILABLE = True
        except Exception:
            pass


def is_available() -> bool:
    """Return True if Metal GPU compute is available on this machine."""
    return _METAL_AVAILABLE


# --------------------------------------------------------------------------
# MetalDevice
# --------------------------------------------------------------------------
class MetalDevice:
    """Singleton wrapper around an MTLDevice and its command queue."""

    def __init__(self):
        if not _METAL_AVAILABLE:
            raise RuntimeError("Metal is not available on this platform")
        self.device = Metal.MTLCreateSystemDefaultDevice()
        if self.device is None:
            raise RuntimeError("No Metal GPU device found")
        self.queue = self.device.newCommandQueue()
        self._pipeline_cache: dict[str, "MetalKernel"] = {}

    @property
    def name(self) -> str:
        return self.device.name()

    def compile(self, source: str, function_name: str) -> "MetalKernel":
        """Compile MSL source and return a MetalKernel for *function_name*."""
        cache_key = f"{function_name}:{hash(source)}"
        if cache_key in self._pipeline_cache:
            return self._pipeline_cache[cache_key]
        kernel = MetalKernel(self, source, function_name)
        self._pipeline_cache[cache_key] = kernel
        return kernel

    def alloc(self, nbytes: int) -> "Metal.MTLBuffer":
        """Allocate a shared-memory Metal buffer."""
        buf = self.device.newBufferWithLength_options_(
            max(nbytes, 4), Metal.MTLResourceStorageModeShared
        )
        if buf is None:
            raise MemoryError(f"Metal: could not allocate {nbytes} bytes")
        return buf


def metal_device() -> MetalDevice:
    """Return the global MetalDevice singleton."""
    global _device_singleton
    if _device_singleton is None:
        _device_singleton = MetalDevice()
    return _device_singleton


# --------------------------------------------------------------------------
# MetalArray  — thin GPU buffer wrapper
# --------------------------------------------------------------------------
class MetalArray:
    """A GPU-resident array backed by a Metal buffer (float32).

    Provides basic numpy interop (shape, dtype, .numpy(), arithmetic).
    """

    def __init__(
        self,
        data: Union[np.ndarray, "MetalArray", list, float, int, None] = None,
        shape: Optional[Tuple[int, ...]] = None,
        dtype=np.float32,
        _buffer=None,
        _device: Optional[MetalDevice] = None,
    ):
        self._dev = _device or metal_device()
        self.dtype = np.dtype(np.float32)  # always float32 on Metal

        if _buffer is not None:
            # Wrap an existing Metal buffer
            self._buf = _buffer
            self.shape = shape if shape is not None else (
                _buffer.length() // 4,)
        elif data is not None:
            arr = np.asarray(data, dtype=np.float32).copy()
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            self.shape = arr.shape
            nbytes = arr.nbytes
            self._buf = self._dev.alloc(nbytes)
            # Copy data into Metal buffer
            buf_ptr = self._buf.contents()
            ctypes.memmove(buf_ptr, arr.ctypes.data, nbytes)
        else:
            # Zero-filled
            if shape is None:
                shape = (0,)
            self.shape = shape
            nbytes = int(np.prod(shape)) * 4
            self._buf = self._dev.alloc(max(nbytes, 4))
            # Zero it
            ctypes.memset(self._buf.contents(), 0, max(nbytes, 4))

    # ---- properties ----
    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape))

    @property
    def nbytes(self) -> int:
        return self.size * 4

    @property
    def buffer(self):
        return self._buf

    # ---- conversions ----
    def numpy(self) -> np.ndarray:
        """Copy GPU data back to a numpy float64 array."""
        n = self.size
        out = np.empty(n, dtype=np.float32)
        ctypes.memmove(out.ctypes.data, self._buf.contents(), n * 4)
        return out.reshape(self.shape).astype(np.float64)

    def get(self) -> np.ndarray:
        """Alias for .numpy() — matches CuPy convention."""
        return self.numpy()

    def __array__(self, dtype=None):
        arr = self.numpy()
        if dtype is not None:
            arr = arr.astype(dtype)
        return arr

    def __float__(self):
        return float(self.numpy().flat[0])

    def __repr__(self):
        return f"MetalArray(shape={self.shape}, dtype=float32)"

    def __len__(self):
        return self.shape[0] if self.shape else 0

    def reshape(self, *new_shape):
        if len(new_shape) == 1 and isinstance(new_shape[0], (tuple, list)):
            new_shape = tuple(new_shape[0])
        return MetalArray(
            _buffer=self._buf,
            shape=new_shape,
            _device=self._dev,
        )

    def copy(self) -> "MetalArray":
        new_buf = self._dev.alloc(self.nbytes)
        ctypes.memmove(new_buf.contents(), self._buf.contents(), self.nbytes)
        return MetalArray(_buffer=new_buf, shape=self.shape, _device=self._dev)

    def __getitem__(self, key):
        # Fall back to numpy for indexing, then push back
        arr = self.numpy()
        result = arr[key]
        if isinstance(result, np.ndarray):
            return MetalArray(result, _device=self._dev)
        return result  # scalar

    def __setitem__(self, key, value):
        arr = self.numpy()
        arr[key] = value if not isinstance(
            value, MetalArray) else value.numpy()
        new = MetalArray(arr, _device=self._dev)
        self._buf = new._buf
        self.shape = new.shape


# --------------------------------------------------------------------------
# MetalKernel — compiled compute pipeline
# --------------------------------------------------------------------------
class MetalKernel:
    """A compiled Metal compute pipeline state, ready for dispatch."""

    def __init__(self, device: MetalDevice, source: str, function_name: str):
        self._dev = device
        self.function_name = function_name
        opts = Metal.MTLCompileOptions.alloc().init()
        opts.setFastMathEnabled_(True)
        err = None
        library, err = device.device.newLibraryWithSource_options_error_(
            source, opts, None
        )
        if err is not None:
            raise RuntimeError(
                f"Metal shader compile error for '{function_name}': {err}"
            )
        func = library.newFunctionWithName_(function_name)
        if func is None:
            raise RuntimeError(
                f"Metal function '{function_name}' not found in compiled library"
            )
        self._pipeline, err = device.device.newComputePipelineStateWithFunction_error_(
            func, None
        )
        if err is not None:
            raise RuntimeError(f"Metal pipeline error: {err}")
        self._max_threads = self._pipeline.maxTotalThreadsPerThreadgroup()

    def dispatch(
        self,
        buffers: list,
        grid_size: Tuple[int, ...],
        threadgroup_size: Optional[Tuple[int, ...]] = None,
    ):
        """Encode and submit a compute command.

        Parameters
        ----------
        buffers : list of MetalArray or MTLBuffer
            Kernel arguments in order.  MetalArrays are unwrapped.
            bytes objects are uploaded as small constant buffers.
            int/float scalars are packed into 4-byte buffers.
        grid_size : (x,) or (x, y) or (x, y, z)
            Total threads to dispatch.
        threadgroup_size : optional (x,) or (x, y) or (x, y, z)
            Threads per threadgroup (auto if None).
        """
        cmd = self._dev.queue.commandBuffer()
        enc = cmd.computeCommandEncoder()
        enc.setComputePipelineState_(self._pipeline)

        for idx, buf in enumerate(buffers):
            if isinstance(buf, MetalArray):
                enc.setBuffer_offset_atIndex_(buf.buffer, 0, idx)
            elif isinstance(buf, bytes):
                # Small constant data
                tmp = self._dev.alloc(len(buf))
                ctypes.memmove(tmp.contents(), buf, len(buf))
                enc.setBuffer_offset_atIndex_(tmp, 0, idx)
            elif isinstance(buf, (int, np.integer)):
                data = struct.pack("i", int(buf))
                tmp = self._dev.alloc(4)
                ctypes.memmove(tmp.contents(), data, 4)
                enc.setBuffer_offset_atIndex_(tmp, 0, idx)
            elif isinstance(buf, (float, np.floating)):
                data = struct.pack("f", float(buf))
                tmp = self._dev.alloc(4)
                ctypes.memmove(tmp.contents(), data, 4)
                enc.setBuffer_offset_atIndex_(tmp, 0, idx)
            else:
                # Assume raw MTLBuffer
                enc.setBuffer_offset_atIndex_(buf, 0, idx)

        # Normalise grid dims to 3-tuple
        gs = _pad3(grid_size)
        if threadgroup_size is None:
            tw = min(self._max_threads, gs[0])
            ts = (tw, 1, 1)
        else:
            ts = _pad3(threadgroup_size)

        enc.dispatchThreads_threadsPerThreadgroup_(
            Metal.MTLSizeMake(*gs),
            Metal.MTLSizeMake(*ts),
        )
        enc.endEncoding()
        cmd.commit()
        cmd.waitUntilCompleted()


def _pad3(t: Tuple[int, ...]) -> Tuple[int, int, int]:
    if len(t) == 1:
        return (t[0], 1, 1)
    if len(t) == 2:
        return (t[0], t[1], 1)
    return (t[0], t[1], t[2])


# --------------------------------------------------------------------------
# Convenience: to_metal / to_numpy
# --------------------------------------------------------------------------
def to_metal(arr, device: Optional[MetalDevice] = None) -> MetalArray:
    """Convert any array-like to a MetalArray on the GPU."""
    if isinstance(arr, MetalArray):
        return arr
    if hasattr(arr, "magnitude"):  # pint
        arr = arr.magnitude
    return MetalArray(data=arr, _device=device)


def to_numpy(arr) -> np.ndarray:
    """Convert MetalArray or numpy array to numpy float64."""
    if isinstance(arr, MetalArray):
        return arr.numpy()
    return np.asarray(arr, dtype=np.float64)
