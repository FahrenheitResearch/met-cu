"""
PTX runtime for met-cu.

Replaces CuPy's runtime CUDA-C compilation with on-disk PTX files. Every kernel
is materialized once via NVRTC into ``python/metcu/ptx/<name>.ptx`` and
thereafter loaded directly via the CUDA driver. Hand-tuned PTX files in that
directory are loaded as-is — recompilation only happens when a .ptx file is
missing.

Public surface:
    PtxKernel(name)            -- callable wrapping a loaded PTX function
    PtxRawModule(name_list)    -- RawModule-shaped object with .get_function(n)
    install_kernel_patches()   -- monkey-patch cp.RawKernel/cp.RawModule so the
                                  existing thermo/wind/grid sources route here
                                  without modification
    compile_source_to_ptx(src) -- NVRTC helper used by the build step

The on-disk layout::

    python/metcu/ptx/
        <kernel_name>.ptx          # one file per kernel function
        _sources/<kernel_name>.cu  # original CUDA C source (for diffing/audit)
"""

from __future__ import annotations

import os
import sys
import threading
from pathlib import Path
from typing import Iterable

import cupy as cp
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).resolve().parent
PTX_DIR = _THIS_DIR / "ptx"
PTX_SRC_DIR = PTX_DIR / "_sources"

PTX_DIR.mkdir(parents=True, exist_ok=True)
PTX_SRC_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# NVRTC compile (build-time only — runtime path uses cached .ptx)
# ---------------------------------------------------------------------------

_NVRTC_OPTIONS_DEFAULT = (
    "--std=c++14",
    "--device-as-default-execution-space",
    # No --use_fast_math: met-cu must match metrust-py to ~1e-10, and
    # fast-math approximations of pow/exp/log/div break that tolerance.
)


def _detect_arch_flag() -> str:
    """Pick an --gpu-architecture flag matching the active device, clamped
    to the highest arch NVRTC actually supports."""
    try:
        from cupy.cuda import nvrtc
        supported = sorted(nvrtc.getSupportedArchs())
    except Exception:
        supported = [70, 75, 80, 86, 89, 90]
    try:
        dev = cp.cuda.Device()
        cc_raw = dev.compute_capability  # CuPy returns "120", "89", etc.
        cc = int(cc_raw)
    except Exception:
        cc = 70
    # Pick the highest supported arch <= device cc; fall back to highest
    # supported if device is newer than NVRTC knows about.
    eligible = [a for a in supported if a <= cc]
    chosen = eligible[-1] if eligible else supported[-1]
    return f"--gpu-architecture=compute_{chosen}"


def compile_source_to_ptx(source: str, name_hint: str = "kernel") -> str:
    """Compile a CUDA C source string to PTX text via NVRTC.

    NVRTC links the necessary libdevice intrinsics (exp/log/pow/...) for us
    so the resulting PTX is standalone and loadable via cuModuleLoadData.
    """
    from cupy.cuda import nvrtc

    prog = nvrtc.createProgram(source, f"{name_hint}.cu", [], [])
    try:
        opts = list(_NVRTC_OPTIONS_DEFAULT) + [_detect_arch_flag()]
        try:
            nvrtc.compileProgram(prog, opts)
        except nvrtc.NVRTCError:
            log = nvrtc.getProgramLog(prog)
            raise RuntimeError(f"NVRTC compilation failed for {name_hint}:\n{log}")
        ptx_bytes = nvrtc.getPTX(prog)
    finally:
        nvrtc.destroyProgram(prog)

    if isinstance(ptx_bytes, bytes):
        return ptx_bytes.decode("utf-8", errors="replace")
    return ptx_bytes


# ---------------------------------------------------------------------------
# PTX module cache + materialisation
# ---------------------------------------------------------------------------

_module_cache: dict[str, "cp.RawModule"] = {}
_kernel_cache: dict[str, "cp.RawKernel"] = {}
_lock = threading.Lock()

# Pristine references captured at import time so the runtime can keep using
# the real CuPy classes after install_kernel_patches() rebinds them.
_REAL_RawModule = cp.RawModule
_REAL_RawKernel = cp.RawKernel

try:
    from cupy.cuda.driver import CUDADriverError as _CUDADriverError
except Exception:
    _CUDADriverError = RuntimeError


def ptx_path(name: str) -> Path:
    return PTX_DIR / f"{name}.ptx"


def src_path(name: str) -> Path:
    return PTX_SRC_DIR / f"{name}.cu"


def materialize(name: str, source: str) -> Path:
    """Ensure ``<name>.ptx`` exists on disk and matches the current source.

    The CUDA C source is also archived under ``ptx/_sources`` so the swarm
    has a reference when hand-tuning.
    """
    pp = ptx_path(name)
    sp = src_path(name)
    source_changed = False

    # Keep the archived source in sync and treat a source diff as an invalidation
    # event for the generated PTX. Otherwise runtime behavior can drift away from
    # the checked-in Python/CUDA source after exploratory edits.
    try:
        archived = sp.read_text(encoding="utf-8") if sp.exists() else None
        source_changed = archived != source
        if source_changed:
            sp.write_text(source, encoding="utf-8")
    except OSError:
        pass

    ptx_fresh = False
    try:
        if pp.exists() and pp.stat().st_size > 0:
            if sp.exists():
                ptx_fresh = pp.stat().st_mtime >= sp.stat().st_mtime
            else:
                ptx_fresh = True
    except OSError:
        ptx_fresh = False

    if ptx_fresh and not source_changed:
        return pp

    ptx_text = compile_source_to_ptx(source, name)
    pp.write_text(ptx_text, encoding="utf-8")
    return pp


def load_kernel(name: str, source: str | None = None) -> "cp.RawKernel":
    """Load a kernel function by name from its .ptx file.

    If the .ptx file doesn't exist and ``source`` is provided, compile it
    first. Cached by name.
    """
    with _lock:
        cached = _kernel_cache.get(name)
        if cached is not None:
            return cached

        pp = ptx_path(name)
        if not pp.exists():
            if source is None:
                raise FileNotFoundError(
                    f"PTX file missing for kernel '{name}' and no source provided: {pp}"
                )
            materialize(name, source)

        try:
            mod = _REAL_RawModule(path=str(pp))
            kern = mod.get_function(name)
        except Exception as exc:
            msg = str(exc).lower()
            invalid_ptx = isinstance(exc, _CUDADriverError) and (
                "cuda_error_invalid_ptx" in msg
                or "invalid ptx" in msg
                or "cuda_error_invalid_image" in msg
                or "kernel image is invalid" in msg
            )
            if not invalid_ptx or source is None:
                raise

            # Cached PTX can be architecture-specific. If a checked-in PTX file
            # is invalid on the current GPU, drop it and regenerate from the
            # original CUDA source so the runtime stays portable.
            try:
                pp.unlink()
            except OSError:
                pass
            materialize(name, source)
            mod = _REAL_RawModule(path=str(pp))
            kern = mod.get_function(name)

        _module_cache[name] = mod
        _kernel_cache[name] = kern
        return kern


# ---------------------------------------------------------------------------
# Shim objects mimicking cp.RawKernel / cp.RawModule
# ---------------------------------------------------------------------------


class PtxKernel:
    """Drop-in replacement for cp.RawKernel backed by an on-disk .ptx file."""

    def __init__(self, code: str, name: str, *args, **kwargs):
        self._code = code
        self._name = name
        self._kern: cp.RawKernel | None = None
        # Eagerly materialise the .ptx file so the on-disk artifact matches
        # the source at import time. The actual function load (cuModuleLoad)
        # is still deferred to first call.
        try:
            materialize(name, code)
        except Exception:
            # Defer hard failures to first __call__ so import doesn't crash.
            pass

    def _ensure(self):
        if self._kern is None:
            self._kern = load_kernel(self._name, self._code)
        return self._kern

    def __call__(self, grid, block, args, **kwargs):
        return self._ensure()(grid, block, args, **kwargs)

    def __getattr__(self, item):
        # Forward attribute access (e.g. .attributes) to the real kernel.
        return getattr(self._ensure(), item)


_DTYPE_MAP = {
    "float64": ("double", np.float64),
    "float32": ("float", np.float32),
    "int32":   ("int", np.int32),
    "int64":   ("long long", np.int64),
    "bool":    ("bool", np.bool_),
}


def _parse_params(params: str):
    """Parse a CuPy ElementwiseKernel param string into [(ctype, np_dtype, name), ...].

    Accepts entries like ``"float64 pressure"`` or ``"raw float64 buf"``.
    """
    out = []
    if not params:
        return out
    for chunk in params.split(","):
        toks = chunk.strip().split()
        if not toks:
            continue
        is_raw = False
        if toks[0] == "raw":
            is_raw = True
            toks = toks[1:]
        if len(toks) != 2:
            raise ValueError(f"Cannot parse ElementwiseKernel param: {chunk!r}")
        cupy_dtype, name = toks
        if cupy_dtype not in _DTYPE_MAP:
            raise NotImplementedError(
                f"PtxElementwiseKernel unsupported dtype {cupy_dtype!r} in {params!r}"
            )
        ctype, np_dtype = _DTYPE_MAP[cupy_dtype]
        out.append((ctype, np_dtype, name, is_raw))
    return out


class PtxElementwiseKernel:
    """Drop-in replacement for cp.ElementwiseKernel.

    Generates an explicit RawKernel-style CUDA C source from the
    elementwise template, materialises it as a .ptx file via the same
    pipeline used for hand-written kernels, and handles broadcasting +
    output allocation at call time.
    """

    def __init__(self, in_params, out_params, operation, name,
                 preamble="", options=(), reduce_dims=True,
                 no_return=False, return_tuple=False, loop_prep="",
                 after_loop="", **kwargs):
        self._name = name
        self._in_specs = _parse_params(in_params)
        self._out_specs = _parse_params(out_params)
        self._operation = operation
        self._preamble = preamble
        self._no_return = no_return
        self._return_tuple = return_tuple
        self._code = self._generate_cuda()
        try:
            materialize(name, self._code)
        except Exception:
            pass
        self._kern: cp.RawKernel | None = None

    # --- code generation ----------------------------------------------------
    def _generate_cuda(self) -> str:
        in_args = ", ".join(
            f"const {ct}* __restrict__ {nm}_arr" for ct, _, nm, _ in self._in_specs
        )
        out_args = ", ".join(
            f"{ct}* __restrict__ {nm}_arr" for ct, _, nm, _ in self._out_specs
        )
        in_unpack = "\n    ".join(
            f"{ct} {nm} = {nm}_arr[i];" for ct, _, nm, _ in self._in_specs
        )
        out_decl = "\n    ".join(
            f"{ct} {nm};" for ct, _, nm, _ in self._out_specs
        )
        out_store = "\n    ".join(
            f"{nm}_arr[i] = {nm};" for ct, _, nm, _ in self._out_specs
        )
        signature = ", ".join(
            x for x in (in_args, out_args, "long long n_total") if x
        )
        # CuPy's ElementwiseKernel does not require trailing ';' on the
        # operation; append one if missing so the wrapper compiles cleanly.
        op = self._operation.rstrip()
        if op and not op.endswith((";", "}")):
            op = op + ";"
        return f"""// Auto-generated PTX-elementwise wrapper for {self._name}
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_PI_2
#define M_PI_2 1.57079632679489661923
#endif
{self._preamble}

extern "C" __global__
void {self._name}({signature}) {{
    long long i = (long long)blockDim.x * (long long)blockIdx.x + (long long)threadIdx.x;
    if (i >= n_total) return;
    {in_unpack}
    {out_decl}
    {{
{op}
    }}
    {out_store}
}}
"""

    # --- launch -------------------------------------------------------------
    def _ensure_kern(self):
        if self._kern is None:
            self._kern = load_kernel(self._name, self._code)
        return self._kern

    def __call__(self, *args, **kwargs):
        n_in = len(self._in_specs)
        n_out = len(self._out_specs)

        in_arrays = list(args[:n_in])
        provided_outs = list(args[n_in:n_in + n_out])

        # Coerce inputs to cupy arrays of correct dtype.
        coerced_in = []
        for (ct, np_dt, nm, _), val in zip(self._in_specs, in_arrays):
            coerced_in.append(cp.asarray(val, dtype=np_dt))

        # Broadcast inputs to a common shape.
        if coerced_in:
            shapes = [a.shape for a in coerced_in]
            try:
                broadcast_shape = np.broadcast_shapes(*shapes)
            except ValueError as e:
                raise ValueError(
                    f"PtxElementwiseKernel {self._name}: shape mismatch {shapes}"
                ) from e
            coerced_in = [cp.broadcast_to(a, broadcast_shape) for a in coerced_in]
            # broadcast_to may return non-contiguous views — make contiguous
            coerced_in = [cp.ascontiguousarray(a) for a in coerced_in]
        else:
            broadcast_shape = ()

        n_total = int(np.prod(broadcast_shape)) if broadcast_shape else 1

        # Allocate / validate outputs.
        if not provided_outs:
            provided_outs = [
                cp.empty(broadcast_shape, dtype=np_dt)
                for _, np_dt, _, _ in self._out_specs
            ]
        else:
            provided_outs = [
                cp.ascontiguousarray(o, dtype=np_dt)
                for o, (_, np_dt, _, _) in zip(provided_outs, self._out_specs)
            ]

        # Launch.
        kern = self._ensure_kern()
        threads = 256
        blocks = (n_total + threads - 1) // threads if n_total > 0 else 1
        kern_args = []
        for a in coerced_in:
            kern_args.append(a.ravel())
        for a in provided_outs:
            kern_args.append(a.ravel())
        kern_args.append(np.int64(n_total))
        if n_total > 0:
            kern((blocks,), (threads,), tuple(kern_args))

        if self._no_return:
            return None
        if n_out == 1:
            return provided_outs[0]
        if self._return_tuple:
            return tuple(provided_outs)
        return tuple(provided_outs)


class PtxRawModule:
    """Drop-in replacement for cp.RawModule.

    The CuPy original compiles a multi-function CUDA C blob and exposes
    ``get_function(name)``. We split that into per-function PTX files: each
    ``get_function`` call materialises one PTX file containing the requested
    function.
    """

    def __init__(self, code=None, *, path=None, options=(), backend="nvrtc",
                 translate_cucomplex=False, enable_cooperative_groups=False,
                 name_expressions=None, jitify=False):
        # Pass-through path mode (already PTX/cubin) — just use cupy directly.
        if code is None and path is not None:
            self._real = _REAL_RawModule(
                path=path, options=options, backend=backend,
                translate_cucomplex=translate_cucomplex,
                enable_cooperative_groups=enable_cooperative_groups,
                name_expressions=name_expressions, jitify=jitify,
            )
            self._code = None
        else:
            self._real = None
            self._code = code

    def get_function(self, name: str):
        if self._real is not None:
            return self._real.get_function(name)
        return load_kernel(name, self._code)


# ---------------------------------------------------------------------------
# Monkey-patch installation
# ---------------------------------------------------------------------------

_PATCHED = False
_orig_RawKernel = None
_orig_RawModule = None
_orig_ElementwiseKernel = None


def install_kernel_patches() -> None:
    """Replace cp.RawKernel / cp.RawModule / cp.ElementwiseKernel with
    PTX-backed shims. Must be called BEFORE thermo/wind/grid are imported.
    """
    global _PATCHED, _orig_RawKernel, _orig_RawModule, _orig_ElementwiseKernel
    if _PATCHED:
        return
    _orig_RawKernel = cp.RawKernel
    _orig_RawModule = cp.RawModule
    _orig_ElementwiseKernel = cp.ElementwiseKernel
    cp.RawKernel = PtxKernel  # type: ignore[assignment]
    cp.RawModule = PtxRawModule  # type: ignore[assignment]
    cp.ElementwiseKernel = PtxElementwiseKernel  # type: ignore[assignment]
    _PATCHED = True


def uninstall_kernel_patches() -> None:
    global _PATCHED
    if not _PATCHED:
        return
    cp.RawKernel = _orig_RawKernel  # type: ignore[assignment]
    cp.RawModule = _orig_RawModule  # type: ignore[assignment]
    cp.ElementwiseKernel = _orig_ElementwiseKernel  # type: ignore[assignment]
    _PATCHED = False


# ---------------------------------------------------------------------------
# Build helpers (used by tools/build_ptx.py)
# ---------------------------------------------------------------------------


def list_ptx_files() -> list[Path]:
    return sorted(PTX_DIR.glob("*.ptx"))


def manifest() -> dict[str, str]:
    """Map kernel name -> sha256 of its .ptx (for change detection)."""
    import hashlib
    out: dict[str, str] = {}
    for p in list_ptx_files():
        out[p.stem] = hashlib.sha256(p.read_bytes()).hexdigest()
    return out
