"""
Compatibility layer for GPU-rendered plots.

Provides wrapper objects that satisfy matplotlib's colorbar and layout APIs
so that plt.colorbar(), fig.tight_layout(), etc. work seamlessly with
GPU-accelerated renders.
"""

import numpy as np
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.cm import ScalarMappable


class GPUContourSet(ScalarMappable):
    """Mimics matplotlib.contour.QuadContourSet for colorbar compatibility.

    matplotlib.colorbar.Colorbar checks for .levels, .cmap, .norm, and
    calls .changed().  This provides all of those.
    """

    def __init__(self, ax, levels, cmap, norm=None, extend='neither'):
        self.ax = ax
        self.levels = np.asarray(levels)
        self.extend = extend
        self._gpu_accelerated = True

        if norm is None:
            norm = BoundaryNorm(self.levels, cmap.N, clip=True)

        super().__init__(norm=norm, cmap=cmap)

        # Attributes that colorbar inspects
        self.collections = []
        self._contour_generator = None

    # --- colorbar API ---

    @property
    def vmin(self):
        return float(self.levels[0])

    @property
    def vmax(self):
        return float(self.levels[-1])

    def _get_allsegs(self):
        """Return empty segments (colorbar doesn't actually need real paths)."""
        return [[] for _ in range(len(self.levels) - 1)]

    def _get_allkinds(self):
        return [[] for _ in range(len(self.levels) - 1)]

    @property
    def allsegs(self):
        return self._get_allsegs()

    @property
    def allkinds(self):
        return self._get_allkinds()


class GPUQuadMesh(ScalarMappable):
    """Mimics matplotlib.collections.QuadMesh for colorbar compatibility."""

    def __init__(self, ax, cmap, norm=None, vmin=None, vmax=None):
        self.ax = ax
        self._gpu_accelerated = True

        if norm is None:
            norm = Normalize(vmin=vmin, vmax=vmax)

        super().__init__(norm=norm, cmap=cmap)
