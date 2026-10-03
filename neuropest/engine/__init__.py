from .lif_numba import LIFEngine
from .lif_ref import ReferenceEngine
from .network import Network
from .params import LIFParams

__all__ = ["LIFEngine", "ReferenceEngine", "Network", "LIFParams", "create_engine"]


def create_engine(net: Network, dt: float = 0.5, seed: int = 0, backend: str = "cpu", adapter: int | None = None):
    """CPU (event-driven numba) or GPU (WebGPU, `pip install neuropest[gpu]`) engine; same interface."""
    if backend == "gpu":
        from .lif_wgpu import WGPUEngine

        return WGPUEngine(net, dt=dt, seed=seed, adapter=adapter)
    return LIFEngine(net, dt=dt, seed=seed)
