"""
Abelian Sandpile Model - High-fidelity GPU-accelerated simulation.

Core components:
- SandpileEngine: Main simulation engine (CuPy GPU / NumPy CPU)
- StatisticsTracker: Avalanche analysis
- StateManager: Persistence (save/load)
- SandpileGUI: Interactive PyGame visualization
- DensityColorMap: Color mapping for density visualization
"""

from .core.engine import SandpileEngine
from .core.grid import Grid, HAS_CUPY, GPU_AVAILABLE
from .toppling import parallel_topple_step, stabilize
from .deposition import PatternFactory
from .statistics import StatisticsTracker
from .persistence import StateManager
from .visualization.gui import SandpileGUI
from .visualization.colormap import DensityColorMap
from .visualization.exporter import VideoExporter, FrameExporter

__version__ = "1.0.0"
__author__ = "Developer"
__all__ = [
    "SandpileEngine",
    "Grid",
    "StatisticsTracker",
    "StateManager",
    "SandpileGUI",
    "DensityColorMap",
    "PatternFactory",
    "parallel_topple_step",
    "stabilize",
    "VideoExporter",
    "FrameExporter",
    "HAS_CUPY",
    "GPU_AVAILABLE",
]

# Quick check for GPU
if HAS_CUPY:
    try:
        import cupy as cp
        _gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name']
        print(f"[OK] CuPy initialized - GPU: {_gpu_name.decode() if isinstance(_gpu_name, bytes) else _gpu_name}")
    except Exception as e:
        print(f"[WARN] CuPy available but GPU error: {e}")
        GPU_AVAILABLE = False
else:
    print("[WARN] CuPy not installed - falling back to CPU (NumPy). Install cupy-cuda12x for GPU acceleration.")
