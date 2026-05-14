"""
SandpileEngine: Core simulation engine.

Manages grid state, toppling, deposition, and statistics.
Supports both GPU (CuPy) and CPU (NumPy/Numba) backends.
"""

import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from pathlib import Path

from .grid import Grid, get_array_module, HAS_CUPY, GPU_AVAILABLE, ensure_gpu
from sandpile.toppling import parallel_topple_step, stabilize, detect_avalanche_boundary

try:
    import cupy as cp
except ImportError:
    cp = None


class SandpileEngine:
    """
    Core Abelian Sandpile Model simulation engine.

    Features:
    - GPU-accelerated parallel toppling via CuPy (or CPU via NumPy/Numba)
    - Open boundary conditions (grains fall off edges)
    - Configurable threshold (default 4)
    - In-place state modification for efficiency
    - Statistics tracking (avalanche size, duration, etc.)
    """

    def __init__(
        self,
        width: int = 500,
        height: int = 500,
        threshold: int = 4,
        use_gpu: bool = True,
        boundary: str = 'open',
        max_iterations: int = 10000,
    ):
        """
        Initialize sandpile simulation engine.

        Args:
            width: Grid width in cells
            height: Grid height in cells
            threshold: Toppling threshold (grains needed to topple)
            use_gpu: Try to use GPU acceleration if available
            boundary: Boundary condition ('open' or 'periodic')
            max_iterations: Safety limit for stabilization loops
        """
        self.width = width
        self.height = height
        self.threshold = threshold
        self.boundary = boundary
        self.max_iterations = max_iterations

        # Determine backend
        self.use_gpu = use_gpu and HAS_CUPY and GPU_AVAILABLE
        self.backend = 'cupy' if self.use_gpu else 'numpy'

        # Initialize grid (all zeros)
        if self.use_gpu:
            self._grid = cp.zeros((height, width), dtype=cp.int32)
        else:
            self._grid = np.zeros((height, width), dtype=np.int32)

        # Statistics tracking
        self._total_grains = 0
        self._stabilization_iterations = 0
        self._total_toppled = 0

        # Avalanche monitoring
        self._avalanche_active = False
        self._current_avalanche_size = 0
        self._current_avalanche_duration = 0
        self._avalanche_history: List[Dict[str, Any]] = []

    @property
    def grid(self) -> Grid:
        """Get grid as Grid wrapper (provides backend-agnostic access)"""
        return Grid(self._grid, backend=self.backend)

    def get_grid(self) -> Grid:
        """Convenience method to get Grid object"""
        return self.grid

    def get_grid_cpu(self) -> np.ndarray:
        """Get grid as NumPy array on CPU (always)"""
        if self.use_gpu:
            return cp.asnumpy(self._grid)
        return self._grid.copy()

    def set_grid(self, array: np.ndarray):
        """Set grid from NumPy array (copies to internal storage)"""
        if array.shape != (self.height, self.width):
            raise ValueError(f"Array shape {array.shape} does not match grid size")

        if self.use_gpu:
            self._grid = cp.asarray(array, dtype=cp.int32)
        else:
            self._grid = array.astype(np.int32).copy()

    def deposit(self, x: int, y: int, grains: int = 1):
        """
        Add grains to a specific cell.

        Args:
            x: X coordinate (0 to width-1)
            y: Y coordinate (0 to height-1)
            grains: Number of grains to add
        """
        if not (0 <= x < self.width and 0 <= y < self.height):
            raise ValueError(f"Coordinates ({x},{y}) out of bounds")

        self._grid[y, x] += grains
        self._total_grains += grains
        self._avalanche_active = True

    def deposit_random(self, grains: int = 1, count: int = 1):
        """
        Add grains to random cells (uniform distribution).

        Args:
            grains: Grains per deposition
            count: Number of random deposition events
        """
        xp = cp if self.use_gpu else np

        for _ in range(count):
            x = int(xp.random.randint(0, self.width))
            y = int(xp.random.randint(0, self.height))
            self.deposit(x, y, grains)

    def deposit_pattern(
        self,
        pattern: str,
        grains: int = 1,
        **kwargs
    ):
        """
        Deposit grains according to a preset pattern.

        Args:
            pattern: Pattern name ('point', 'line_h', 'line_v', 'circle',
                     'square', 'cross', 'diamond', 'random')
            grains: Grains per affected cell
            **kwargs: Pattern-specific parameters
                - center: (x, y) tuple, default center of grid
                - radius: for circle/diamond patterns
                - length: for line patterns
                - size: for square (side length)
        """
        from ..deposition import PatternFactory

        # Extract 'center' with default, remove from kwargs to avoid duplicate
        center = kwargs.pop('center', (self.width // 2, self.height // 2))
        pattern = PatternFactory.get(pattern, center=center, **kwargs)
        coords = pattern.get_coordinates(self.width, self.height)

        for x, y in coords:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.deposit(x, y, grains)

    def deposit_uniform(self, grains_per_cell: int = 1):
        """Add same number of grains to every cell."""
        xp = cp if self.use_gpu else np
        self._grid += grains_per_cell
        self._total_grains += grains_per_cell * self.width * self.height
        self._avalanche_active = True

    def stabilize(self, callback=None, max_iterations: int = None) -> Tuple[int, int]:
        """
        Stabilize the grid by toppling until all cells are below threshold.

        Args:
            callback: Optional callable(iteration, changed) for progress updates
            max_iterations: Override engine's default max_iterations if provided

        Returns:
            tuple: (iterations, total_toppled)
        """
        if not self._avalanche_active and self._grid.max() < self.threshold:
            return 0, 0

        # Record state before stabilization for avalanche metrics
        state_before = self.get_grid_cpu().copy()

        # Use provided max_iterations or fall back to engine default
        max_iters = max_iterations if max_iterations is not None else self.max_iterations

        # Perform stabilization
        iterations, total_toppled = stabilize(
            self._grid,
            threshold=self.threshold,
            max_iterations=max_iters,
            progress_callback=callback
        )

        self._stabilization_iterations += iterations
        self._total_toppled += total_toppled

        # Record avalanche if any toppling occurred
        if total_toppled > 0:
            state_after = self.get_grid_cpu()
            self._record_avalanche(state_before, state_after, iterations, total_toppled)

        return iterations, total_toppled

    def step(self) -> bool:
        """
        Perform a single toppling step (one iteration).

        Returns:
            True if grid changed (unstable cells toppled), False otherwise
        """
        changed, _ = parallel_topple_step(self._grid, self.threshold)
        return changed

    def is_stable(self) -> bool:
        """Check if grid is stable (no cells at or above threshold)"""
        xp = cp if self.use_gpu else np
        return bool(xp.all(self._grid < self.threshold))

    def get_total_grains(self) -> int:
        """Get total grains in the system"""
        if self.use_gpu:
            return int(cp.sum(self._grid))
        return int(np.sum(self._grid))

    def get_statistics(self) -> Dict[str, Any]:
        """Get current simulation statistics"""
        return {
            'width': self.width,
            'height': self.height,
            'total_grains': self.get_total_grains(),
            'threshold': self.threshold,
            'backend': self.backend,
            'gpu_available': GPU_AVAILABLE,
            'stabilization_iterations': self._stabilization_iterations,
            'total_toppled_cells': self._total_toppled,
            'avalanche_count': len(self._avalanche_history),
        }

    def get_avalanche_history(self) -> List[Dict[str, Any]]:
        """Get list of all recorded avalanches"""
        return self._avalanche_history.copy()

    def clear_avalanche_history(self):
        """Clear the avalanche history"""
        self._avalanche_history.clear()

    def get_last_avalanche(self) -> Optional[Dict[str, Any]]:
        """Get the most recent avalanche data"""
        if self._avalanche_history:
            return self._avalanche_history[-1]
        return None

    def _record_avalanche(
        self,
        state_before: np.ndarray,
        state_after: np.ndarray,
        iterations: int,
        toppled_count: int
    ):
        """Internal: record avalanche metrics"""
        boundary = detect_avalanche_boundary(state_before, state_after, self.threshold)
        avalanche_size = int(np.sum(boundary))
        changed_cells = np.argwhere(boundary)

        if len(changed_cells) > 0:
            ymax, xmax = changed_cells.max(axis=0)
            ymin, xmin = changed_cells.min(axis=0)
            radius = max(ymax - ymin, xmax - xmin) // 2
        else:
            radius = 0

        avalanche = {
            'timestamp': len(self._avalanche_history),
            'size': avalanche_size,
            'duration': iterations,
            'toppled_cells': toppled_count,
            'radius': radius,
            'grains_added': self._total_grains,
        }
        self._avalanche_history.append(avalanche)

    # Persistence methods
    def save(self, filepath: str):
        """
        Save engine state to file.

        Args:
            filepath: Path to .npz file
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'grid': self.get_grid_cpu(),
            'width': self.width,
            'height': self.height,
            'threshold': self.threshold,
            'total_grains': self._total_grains,
            'stabilization_iterations': self._stabilization_iterations,
            'total_toppled': self._total_toppled,
            'avalanche_history': self._avalanche_history,
        }

        np.savez_compressed(filepath, **save_dict)
        print(f"Saved state to {filepath}")

    def load(self, filepath: str):
        """
        Load engine state from file.

        Args:
            filepath: Path to .npz file
        """
        data = np.load(filepath, allow_pickle=True)

        self.width = int(data['width'])
        self.height = int(data['height'])
        self.threshold = int(data['threshold'])
        self._total_grains = int(data['total_grains'])
        self._stabilization_iterations = int(data['stabilization_iterations'])
        self._total_toppled = int(data['total_toppled'])
        self._avalanche_history = list(data['avalanche_history'])

        self.set_grid(data['grid'])

        print(f"Loaded state from {filepath}")

    def clear(self):
        """Reset grid to all zeros"""
        xp = cp if self.use_gpu else np
        self._grid.fill(0)
        self._total_grains = 0
        self._avalanche_history.clear()

    def __repr__(self) -> str:
        return (
            f"SandpileEngine(size=({self.width}x{self.height}), "
            f"backend={self.backend}, grains={self.get_total_grains()})"
        )
