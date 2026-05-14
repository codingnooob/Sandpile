"""
Abelian Sandpile Model - Core Engine

GPU-accelerated parallel toppling using CuPy with fallback to NumPy+Numba.
Implements the standard Bak-Tang-Wiesenfeld sandpile dynamics with open boundaries.
"""

from .engine import SandpileEngine
from .grid import Grid

__all__ = ["SandpileEngine", "Grid"]
