"""CUDA Graph capture helpers for metcu.

Batch many small stencil kernel launches into a single graph submission to
amortize launch overhead on sm_120 where grid stencils are launch-bound.

Usage
-----
    import cupy as cp
    import metcu
    from metcu.graph import graph_capture

    u = cp.random.rand(2048, 2048, dtype=cp.float32)
    v = cp.random.rand(2048, 2048, dtype=cp.float32)
    dx = dy = cp.float32(1000.0)

    # Warm-up so any lazy allocations / JIT happen before capture.
    _ = metcu.vorticity(u, v, dx, dy)

    with graph_capture() as g:
        vort = metcu.vorticity(u, v, dx, dy)
        div  = metcu.divergence(u, v, dx, dy)

    for _ in range(1000):
        g.launch()
    cp.cuda.get_current_stream().synchronize()

Limitations
-----------
* Capture must occur on a non-blocking stream. The context manager creates one
  for you if the current stream is the default (legacy) stream.
* Every op inside the ``with`` block must be pure GPU work. Anything that
  triggers a device->host sync (``.get()``, ``float(x)``, shape checks that
  materialize) will abort the capture.
* Output buffers allocated *inside* the captured region are reused on every
  replay -- CuPy's caching allocator returns the same pointers to the graph.
  This is fine as long as you treat the handles returned inside the ``with``
  block as "latest replay result" slots, not as independent tensors.
* For maximum safety, pre-allocate outputs outside the block and call the
  low-level raw kernels in ``metcu.kernels.grid`` directly.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Optional

import cupy as cp


@contextmanager
def graph_capture(stream: Optional[cp.cuda.Stream] = None):
    """Capture all CuPy kernel launches inside the ``with`` block into a graph.

    Parameters
    ----------
    stream : cupy.cuda.Stream, optional
        Stream to capture on. If ``None`` a fresh non-blocking stream is
        created (required by CUDA -- you cannot capture on the legacy/default
        stream).

    Yields
    ------
    handle : _GraphHandle
        Object with a ``.launch(stream=None)`` method that re-submits the
        captured work, and a ``.graph`` attribute exposing the raw CuPy
        ``Graph``.
    """
    owns_stream = False
    if stream is None:
        stream = cp.cuda.Stream(non_blocking=True)
        owns_stream = True

    handle = _GraphHandle(stream)

    with stream:
        stream.begin_capture()
        try:
            yield handle
        except Exception:
            # Best-effort: end capture so the stream isn't left in capture mode.
            try:
                stream.end_capture()
            except Exception:
                pass
            raise
        graph = stream.end_capture()

    graph.upload(stream=stream)
    handle._graph = graph
    handle._owns_stream = owns_stream


class _GraphHandle:
    """Lightweight handle returned by :func:`graph_capture`."""

    def __init__(self, capture_stream: cp.cuda.Stream):
        self._capture_stream = capture_stream
        self._graph: Optional[cp.cuda.graph.Graph] = None
        self._owns_stream = False

    @property
    def graph(self) -> cp.cuda.graph.Graph:
        if self._graph is None:
            raise RuntimeError("graph_capture block has not exited yet")
        return self._graph

    def launch(self, stream: Optional[cp.cuda.Stream] = None) -> None:
        """Replay the captured graph on ``stream`` (default: capture stream)."""
        g = self.graph
        if stream is None:
            stream = self._capture_stream
        with stream:
            g.launch(stream=stream)

    def synchronize(self) -> None:
        self._capture_stream.synchronize()
