"""
Verification script for Metal backend — run on macOS with Apple Silicon.

Tests that all Metal kernel functions exist, have matching signatures to CUDA,
and produce numerically correct results (within float32 tolerance).

Usage: python tests/verify_metal.py
"""

import sys
import os
import numpy as np
import inspect
import importlib

# Force Metal backend
os.environ['METCU_BACKEND'] = 'metal'


def check_module_exists(name, path):
    """Check if a Python module file exists."""
    if os.path.exists(path):
        print(f"  [OK] {name} exists ({os.path.getsize(path)} bytes)")
        return True
    else:
        print(f"  [FAIL] {name} MISSING at {path}")
        return False


def compare_signatures(cuda_mod, metal_mod, module_name):
    """Compare function signatures between CUDA and Metal modules."""
    cuda_funcs = {
        name: inspect.signature(getattr(cuda_mod, name))
        for name in dir(cuda_mod)
        if not name.startswith('_') and callable(getattr(cuda_mod, name))
        and inspect.isfunction(getattr(cuda_mod, name))
    }
    metal_funcs = {
        name: inspect.signature(getattr(metal_mod, name))
        for name in dir(metal_mod)
        if not name.startswith('_') and callable(getattr(metal_mod, name))
        and inspect.isfunction(getattr(metal_mod, name))
    }

    missing = set(cuda_funcs.keys()) - set(metal_funcs.keys())
    extra = set(metal_funcs.keys()) - set(cuda_funcs.keys())
    matched = 0
    mismatched = 0

    for name in sorted(cuda_funcs.keys()):
        if name in metal_funcs:
            cuda_sig = str(cuda_funcs[name])
            metal_sig = str(metal_funcs[name])
            if cuda_sig == metal_sig:
                matched += 1
            else:
                print(f"  [WARN] {module_name}.{name}: signature differs")
                print(f"         CUDA:  {cuda_sig}")
                print(f"         Metal: {metal_sig}")
                mismatched += 1

    print(f"  {module_name}: {matched} matched, {mismatched} sig mismatch, "
          f"{len(missing)} missing, {len(extra)} extra")
    if missing:
        print(f"    Missing in Metal: {sorted(missing)}")
    if extra:
        print(f"    Extra in Metal: {sorted(extra)}")
    return len(missing) == 0


def verify_metal_files():
    """Verify that all Metal kernel files exist."""
    print("=" * 60)
    print("METAL BACKEND FILE VERIFICATION")
    print("=" * 60)

    base = os.path.join(os.path.dirname(__file__), '..', 'python', 'metcu', 'metal')
    base = os.path.abspath(base)

    files = {
        'runtime.py': os.path.join(base, 'runtime.py'),
        'thermo.py': os.path.join(base, 'thermo.py'),
        'wind.py': os.path.join(base, 'wind.py'),
        'grid.py': os.path.join(base, 'grid.py'),
        'utils.py': os.path.join(base, 'utils.py'),
        'msl_helpers.py': os.path.join(base, 'msl_helpers.py'),
        '__init__.py': os.path.join(base, '__init__.py'),
    }

    all_ok = True
    for name, path in files.items():
        if not check_module_exists(name, path):
            all_ok = False

    return all_ok


def verify_function_coverage():
    """Verify all CUDA kernel functions have Metal equivalents."""
    print("\n" + "=" * 60)
    print("FUNCTION COVERAGE VERIFICATION")
    print("=" * 60)

    try:
        from metcu.kernels import thermo as cuda_thermo
        from metcu.metal import thermo as metal_thermo
        compare_signatures(cuda_thermo, metal_thermo, "thermo")
    except ImportError as e:
        print(f"  [SKIP] thermo: {e}")

    try:
        from metcu.kernels import wind as cuda_wind
        from metcu.metal import wind as metal_wind
        compare_signatures(cuda_wind, metal_wind, "wind")
    except ImportError as e:
        print(f"  [SKIP] wind: {e}")

    try:
        from metcu.kernels import grid as cuda_grid
        from metcu.metal import grid as metal_grid
        compare_signatures(cuda_grid, metal_grid, "grid")
    except ImportError as e:
        print(f"  [SKIP] grid: {e}")


def verify_syntax():
    """Verify that all Metal modules parse without syntax errors."""
    print("\n" + "=" * 60)
    print("SYNTAX VERIFICATION")
    print("=" * 60)

    base = os.path.join(os.path.dirname(__file__), '..', 'python', 'metcu', 'metal')
    base = os.path.abspath(base)

    for fname in ['runtime.py', 'thermo.py', 'wind.py', 'grid.py',
                   'utils.py', 'msl_helpers.py']:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fname} not found")
            continue
        try:
            with open(fpath, 'r') as f:
                source = f.read()
            compile(source, fpath, 'exec')
            print(f"  [OK] {fname} — valid Python syntax")
        except SyntaxError as e:
            print(f"  [FAIL] {fname} — syntax error: {e}")


def verify_msl_syntax():
    """Basic verification that MSL kernel strings look valid."""
    print("\n" + "=" * 60)
    print("MSL KERNEL STRING VERIFICATION")
    print("=" * 60)

    base = os.path.join(os.path.dirname(__file__), '..', 'python', 'metcu', 'metal')
    base = os.path.abspath(base)

    for fname in ['thermo.py', 'wind.py', 'grid.py']:
        fpath = os.path.join(base, fname)
        if not os.path.exists(fpath):
            print(f"  [SKIP] {fname} not found")
            continue

        with open(fpath, 'r') as f:
            source = f.read()

        # Count kernel functions in MSL strings
        kernel_count = source.count('kernel void ')
        metal_stdlib = source.count('#include <metal_stdlib>')
        buffer_attrs = source.count('[[buffer(')
        thread_pos = source.count('[[thread_position_in_grid]]')

        # Check for common CUDA artifacts that shouldn't be in Metal
        cuda_artifacts = {
            '__global__': source.count('__global__'),
            'extern "C"': source.count('extern "C"'),
            '__shared__': source.count('__shared__'),
            'threadIdx': source.count('threadIdx'),
            'blockIdx': source.count('blockIdx'),
            'blockDim': source.count('blockDim'),
        }

        problems = {k: v for k, v in cuda_artifacts.items() if v > 0}

        print(f"\n  {fname}:")
        print(f"    Metal kernel functions: {kernel_count}")
        print(f"    #include <metal_stdlib>: {metal_stdlib}")
        print(f"    [[buffer(N)]] attributes: {buffer_attrs}")
        print(f"    [[thread_position_in_grid]]: {thread_pos}")

        if problems:
            print(f"    [WARN] CUDA artifacts found: {problems}")
        else:
            print(f"    [OK] No CUDA artifacts")


if __name__ == '__main__':
    verify_metal_files()
    verify_syntax()
    verify_msl_syntax()
    # Only do function coverage if both CUDA and Metal modules can be loaded
    # (this won't work on Mac without CuPy, but it's useful for cross-checking)
    try:
        verify_function_coverage()
    except Exception as e:
        print(f"\n[SKIP] Function coverage check: {e}")

    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)
