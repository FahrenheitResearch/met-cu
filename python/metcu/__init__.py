"""met-cu — Custom CUDA kernels for every meteorological calculation."""

__version__ = "0.2.1"


def __getattr__(name):
    """Lazy import: expose everything from metcu.calc and metcu.constants."""
    import metcu.calc as _calc  # noqa: F811
    import metcu.constants as _constants  # noqa: F811

    # Populate this module's namespace so future lookups are direct
    _all = {}
    for mod in (_calc, _constants):
        for attr in dir(mod):
            if not attr.startswith('_'):
                _all[attr] = getattr(mod, attr)
    globals().update(_all)

    if name in _all:
        return _all[name]
    raise AttributeError(f"module 'metcu' has no attribute {name!r}")


def __dir__():
    """List all public attributes including lazy-loaded ones."""
    import metcu.calc as _calc
    import metcu.constants as _constants
    base = list(globals().keys())
    for mod in (_calc, _constants):
        base.extend(attr for attr in dir(mod) if not attr.startswith('_'))
    return sorted(set(base))
