# Metal compute backend for met-cu (Apple Silicon / macOS)
from .runtime import (  # noqa: F401
    MetalDevice,
    MetalArray,
    metal_device,
    to_metal,
    to_numpy,
    is_available,
)
