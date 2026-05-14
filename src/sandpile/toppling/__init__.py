"""
Parallel vectorized toppling algorithm.

Implements the efficient parallel toppling rule:
- All cells with >= threshold grains topple simultaneously
- Each toppling cell loses threshold grains (default 4)
- Each neighbor receives 1 grain (cardinal directions only)
- Open boundaries: grains at edges fall off into the void
"""

from ..core.grid import get_array_module, ensure_gpu, HAS_CUPY, GPU_AVAILABLE
from ..core.grid import jit, prange, HAS_NUMBA  # noqa: F401 (re-export for numba decorate)
import numpy as np

if HAS_CUPY:
    import cupy as cp


def parallel_topple_step(grid, threshold=4):
    """
    Perform one parallel toppling step on the grid.

    Uses vectorized array operations for maximum efficiency.
    All unstable cells (>= threshold) topple simultaneously.

    Args:
        grid: CuPy/NumPy 2D array (modified in-place)
        threshold: toppling threshold (default 4)

    Returns:
        tuple: (changed: bool, toppled_count: int)
            changed: True if any cell toppled
            toppled_count: number of cells that toppled
    """
    xp = get_array_module('cupy' if HAS_CUPY else 'numpy')

    # Create boolean mask of cells exceeding threshold
    mask = grid >= threshold

    # Count toppling cells
    toppled_count = int(xp.sum(mask))

    if toppled_count == 0:
        return False, 0

    # Each toppling cell loses threshold grains
    grid[mask] -= threshold

    # Distribute to neighbors using array slicing (open boundaries)
    # Down neighbor (cell above gives to cell below)
    grid[:-1, :] += mask[1:, :]
    # Up neighbor (cell below gives to cell above)
    grid[1:, :] += mask[:-1, :]
    # Right neighbor (cell left gives to cell right)
    grid[:, :-1] += mask[:, 1:]
    # Left neighbor (cell right gives to cell left)
    grid[:, 1:] += mask[:, :-1]

    return True, toppled_count


def stabilize(grid, threshold=4, max_iterations=10000, progress_callback=None):
    """
    Stabilize the grid by repeatedly applying parallel toppling until stable.

    Args:
        grid: CuPy/NumPy 2D array (modified in-place)
        threshold: toppling threshold
        max_iterations: safety limit to prevent infinite loops
        progress_callback: optional callable(iteration, changed) called each step

    Returns:
        tuple: (iterations: int, total_toppled: int)
            iterations: number of toppling steps performed
            total_toppled: cumulative cells that toppled
    """
    iterations = 0
    total_toppled = 0

    while iterations < max_iterations:
        changed, toppled = parallel_topple_step(grid, threshold)

        if progress_callback:
            progress_callback(iterations, changed)

        if not changed:
            break

        iterations += 1
        total_toppled += toppled

    if iterations >= max_iterations:
        import warnings
        warnings.warn(
            f"Stabilization reached max iterations ({max_iterations}). "
            "Grid may not be fully stable."
        )

    return iterations, total_toppled


# For CPU fallback with Numba
if HAS_NUMBA:
    @jit(nopython=True, parallel=True, fastmath=True)
    def parallel_topple_step_numba(grid, threshold=4):
        """
        Numba-accelerated parallel toppling for CPU fallback.
        Note: Numba doesn't support boolean masking as efficiently,
              so we use explicit loops with parallel=True.
        """
        h, w = grid.shape
        toppled_count = 0

        # Create a copy of which cells need to topple
        topple_mask = np.zeros((h, w), dtype=np.bool_)

        for i in prange(h):
            for j in range(w):
                if grid[i, j] >= threshold:
                    topple_mask[i, j] = True
                    toppled_count += 1

        if toppled_count == 0:
            return False, 0

        # Apply toppling
        for i in prange(h):
            for j in range(w):
                if topple_mask[i, j]:
                    grid[i, j] -= threshold

        # Distribute to neighbors (vectorized where possible)
        # Down
        for i in range(h - 1):
            for j in range(w):
                if topple_mask[i, j]:
                    grid[i + 1, j] += 1
        # Up
        for i in range(1, h):
            for j in range(w):
                if topple_mask[i, j]:
                    grid[i - 1, j] += 1
        # Right
        for i in range(h):
            for j in range(w - 1):
                if topple_mask[i, j]:
                    grid[i, j + 1] += 1
        # Left
        for i in range(h):
            for j in range(1, w):
                if topple_mask[i, j]:
                    grid[i, j - 1] += 1

        return True, toppled_count

    def stabilize_numba(grid, threshold=4, max_iterations=10000):
        """Numba-accelerated stabilization loop."""
        iterations = 0
        total_toppled = 0

        while iterations < max_iterations:
            changed, toppled = parallel_topple_step_numba(grid, threshold)
            if not changed:
                break
            iterations += 1
            total_toppled += toppled

        return iterations, total_toppled


def stabilize_cpu(grid, threshold=4, max_iterations=10000):
    """CPU stabilization using either Numba or pure NumPy."""
    if HAS_NUMBA:
        return stabilize_numba(grid, threshold, max_iterations)
    else:
        return stabilize(grid, threshold, max_iterations)


def detect_avalanche_boundary(grid_before, grid_after, threshold=4):
    """
    Detect cells that toppled during stabilization (avalanche boundary).

    Args:
        grid_before: grid state before stabilization (NumPy array)
        grid_after: grid state after stabilization (NumPy array)

    Returns:
        Binary mask (NumPy bool array) where True = cell toppled at least once
    """
    # Cells that changed significantly are those involved in the avalanche
    diff = np.abs(grid_after - grid_before)
    boundary = diff > 0
    return boundary


def count_avalanche_radius(grid_before, grid_after, threshold=4):
    """
    Calculate the maximum Manhattan distance from the drop point
    to any cell that changed (approximate avalanche radius).
    """
    changed = np.argwhere(grid_after != grid_before)
    if len(changed) == 0:
        return 0

    # Find max Manhattan distance from any changed cell to any other
    # (more precise: find bounding box radius)
    ymax, xmax = changed.max(axis=0)
    ymin, xmin = changed.min(axis=0)
    radius = max(ymax - ymin, xmax - xmin) // 2
    return int(radius)
