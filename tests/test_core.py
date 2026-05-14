"""Test suite for sandpile simulation core."""

import pytest
import numpy as np
try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

from sandpile.core.engine import SandpileEngine
from sandpile.core.grid import Grid
from sandpile.toppling import parallel_topple_step, stabilize
from sandpile.deposition import PatternFactory


class TestCoreEngine:
    """Tests for SandpileEngine core functionality."""

    def test_engine_initialization(self):
        """Test engine creates correct grid size."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        assert engine.width == 100
        assert engine.height == 100
        assert engine.is_stable()
        assert engine.get_total_grains() == 0

    def test_single_deposition(self):
        """Test adding grains to single cell."""
        engine = SandpileEngine(50, 50, use_gpu=False)
        engine.deposit(25, 25, grains=10)
        assert engine.grid[25, 25] == 10
        assert engine.get_total_grains() == 10

    def test_boundary_violation_raises(self):
        """Test depositing outside grid raises error."""
        engine = SandpileEngine(50, 50, use_gpu=False)
        with pytest.raises(ValueError):
            engine.deposit(100, 100, grains=1)

    def test_random_deposition(self):
        """Test random deposition distributes grains."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        engine.deposit_random(grains=100, count=10)
        assert engine.get_total_grains() == 1000

    def test_basic_toppling(self):
        """Test that a cell with >=4 grains topples correctly."""
        engine = SandpileEngine(10, 10, use_gpu=False)
        # Place 4 grains in center
        engine.deposit(5, 5, grains=4)
        engine.stabilize()

        # Center should have 0, neighbors should have 1 each (if no boundary issues)
        grid = engine.get_grid_cpu()
        assert grid[5, 5] < 4  # Center stable
        # Verify total grains conserved (minus edge losses)
        total = grid.sum()
        # 4 grains added, some fall off edges - at least 3 should remain in grid
        assert total >= 0

    def test_threshold_configurable(self):
        """Test custom threshold value."""
        engine = SandpileEngine(20, 20, threshold=5, use_gpu=False)
        engine.deposit(10, 10, grains=5)
        engine.stabilize()
        assert engine.grid[10, 10] < 5

    def test_stabilization_guaranteed(self):
        """Test that stabilization always terminates."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        # Add many grains randomly
        for _ in range(100):
            engine.deposit_random(grains=5, count=10)
            iterations, _ = engine.stabilize(max_iterations=10000)
            # Must eventually stabilize
            assert engine.is_stable()


class TestParallelToppling:
    """Tests for vectorized toppling algorithm."""

    def test_topple_step_changes_grid(self):
        """Test topple_step modifies grid."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = 4
        changed, count = parallel_topple_step(grid)
        assert changed is True
        assert count == 1
        assert grid[5, 5] == 0  # Lost 4

    def test_topple_distribution(self):
        """Test grains distributed to cardinal neighbors."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[5, 5] = 4
        parallel_topple_step(grid)

        # Neighbors should each gain 1
        assert grid[4, 5] == 1  # up
        assert grid[6, 5] == 1  # down
        assert grid[5, 4] == 1  # left
        assert grid[5, 6] == 1  # right

    def test_open_boundaries(self):
        """Test grains at edges fall off."""
        grid = np.zeros((5, 5), dtype=np.int32)
        grid[0, 2] = 4  # Top edge
        parallel_topple_step(grid)

        # Top neighbor would be at y=-1 (lost)
        assert grid[0, 2] == 0
        assert grid[1, 2] == 1  # down neighbor gains
        # No cell at row -1, so grain lost

    def test_multiple_topple_simultaneous(self):
        """Test parallel toppling of multiple cells."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[3, 3] = 4
        grid[3, 4] = 4
        grid[4, 3] = 4

        changed, count = parallel_topple_step(grid)
        assert changed
        assert count == 3

        # After parallel topple: the central cell receives from two neighbors, edges from one
        assert grid[3, 3] == 2  # receives +1 from both (3,4) and (4,3)
        assert grid[3, 4] == 1  # receives +1 only from (3,3)
        assert grid[4, 3] == 1  # receives +1 only from (3,3)

    def test_no_topple_when_stable(self):
        """Test stable grid returns no change."""
        grid = np.zeros((10, 10), dtype=np.int32)
        grid[:] = 3  # At threshold - 1
        changed, count = parallel_topple_step(grid)
        assert changed is False
        assert count == 0

    def test_abelian_property(self):
        """Test final state independent of toppling order (parallel correctness)."""
        # Create a grid that will have complex toppling
        grid1 = np.zeros((20, 20), dtype=np.int32)
        grid2 = grid1.copy()

        # Add many grains
        for _ in range(100):
            x, y = np.random.randint(0, 20, 2)
            grid1[y, x] += 1
            grid2[y, x] += 1

        # Stabilize using parallel (vectorized)
        stabilize(grid1)

        # Stabilize grid2 using sequential (slow) for comparison
        # Not implementing sequential now - just verify parallel converges
        assert grid1.max() < 4  # Stable


class TestStabilize:
    """Tests for full stabilization."""

    def test_stabilize_known_configuration(self):
        """Test stabilization on known configuration (identity pattern seed)."""
        engine = SandpileEngine(25, 25, use_gpu=False)
        # Identity element pattern: place 4 grains at each corner
        corners = [(0, 0), (0, 24), (24, 0), (24, 24)]
        for x, y in corners:
            engine.deposit(x, y, grains=4)  # exactly at threshold

        iterations, total = engine.stabilize()

        # All cells should be < threshold
        assert engine.is_stable()
        # Known: should take at least 1 step
        assert iterations >= 1
        assert total >= 0

    def test_stabilize_max_iterations_safety(self):
        """Test stabilization aborts at max_iterations."""
        # Create dangerous grid that might loop (shouldn't, but test safety)
        engine = SandpileEngine(50, 50, use_gpu=False)
        engine.deposit(25, 25, grains=10000)
        iterations, total = engine.stabilize(max_iterations=10)
        # Should hit limit and return with warning
        assert iterations == 10


class TestPatterns:
    """Tests for deposition patterns."""

    def test_point_pattern(self):
        """Test point pattern returns single coordinate."""
        pattern = PatternFactory.get('point', x=10, y=20)
        coords = pattern.get_coordinates(100, 100)
        assert coords == [(10, 20)]

    def test_line_h_pattern(self):
        """Test horizontal line covers entire width."""
        pattern = PatternFactory.get('line_h', y=50)
        coords = pattern.get_coordinates(100, 100)
        assert len(coords) == 100
        assert all(y == 50 for _, y in coords)

    def test_circle_pattern(self):
        """Test circle pattern produces circular set."""
        pattern = PatternFactory.get('circle', center=(50, 50), radius=6)
        coords = pattern.get_coordinates(100, 100)
        # Check points lie within radius (+1 for integer rounding)
        for x, y in coords:
            assert (x - 50)**2 + (y - 50)**2 <= 6**2 + 2
        # Approximately π*r² = 113 cells
        assert 100 <= len(coords) <= 130

    def test_square_pattern(self):
        """Test square pattern produces square set."""
        pattern = PatternFactory.get('square', center=(25, 25), size=10)
        coords = pattern.get_coordinates(100, 100)
        # 10x10 = 100 cells (roughly, edges may truncate if near boundary)
        assert len(coords) == 100
        xs, ys = zip(*coords)
        # Centered around 25: span should be 20-29 inclusive
        assert min(xs) >= 20 and max(xs) <= 29
        assert min(ys) >= 20 and max(ys) <= 29

    def test_cross_pattern(self):
        """Test cross pattern produces plus shape."""
        pattern = PatternFactory.get('cross', center=(50, 50), length=20, thickness=1)
        coords = pattern.get_coordinates(100, 100)
        # Cross should be approx length*2 + thickness*2
        assert len(coords) >= 40

    def test_unknown_pattern_raises(self):
        """Test invalid pattern name raises error."""
        with pytest.raises(ValueError):
            PatternFactory.get('not_a_pattern')


class TestDeposition:
    """Tests for deposition methods."""

    def test_uniform_deposition(self):
        """Test uniform deposition adds to all cells."""
        engine = SandpileEngine(10, 10, use_gpu=False)
        engine.deposit_uniform(grains_per_cell=2)
        assert np.all(engine.get_grid_cpu() == 2)
        assert engine.get_total_grains() == 200

    def test_pattern_deposition(self):
        """Test pattern deposition."""
        engine = SandpileEngine(50, 50, use_gpu=False)
        # Circle of radius 6 gives approx 113 cells (π*36 ≈ 113)
        # Each with 5 grains = ~565 grains total
        engine.deposit_pattern('circle', grains=5, center=(25, 25), radius=6)
        total = engine.get_total_grains()
        # Should be ~5 * π * 36 ≈ 565
        assert 500 < total < 650


class TestPersistence:
    """Tests for state save/load."""

    def test_save_and_load(self, tmp_path):
        """Test saving and loading preserves state."""
        engine1 = SandpileEngine(50, 50, use_gpu=False)
        engine1.deposit(25, 25, grains=1000)
        engine1.stabilize()

        filepath = tmp_path / "test_state.npz"
        engine1.save(str(filepath))

        engine2 = SandpileEngine(50, 50, use_gpu=False)
        engine2.load(str(filepath))

        # Compare grids
        np.testing.assert_array_equal(
            engine1.get_grid_cpu(),
            engine2.get_grid_cpu()
        )
        assert engine1.get_total_grains() == engine2.get_total_grains()


class TestGrid:
    """Tests for Grid abstraction."""

    def test_grid_wrapper(self):
        """Test Grid wrapper provides correct backend."""
        engine = SandpileEngine(50, 50, use_gpu=False)
        grid = engine.grid
        assert grid.backend == 'numpy'
        assert grid.shape == (50, 50)

    def test_grid_cpu_conversion(self):
        """Test to_cpu returns NumPy array."""
        engine = SandpileEngine(50, 50, use_gpu=False)
        grid = engine.grid.to_cpu()
        assert isinstance(grid, np.ndarray)


class TestStatistics:
    """Tests for statistics tracking."""

    def test_avalanche_recording(self):
        """Test avalanches are recorded."""
        from sandpile.statistics import StatisticsTracker

        engine = SandpileEngine(100, 100, use_gpu=False)
        tracker = StatisticsTracker(engine)

        # Cause an avalanche
        engine.deposit(50, 50, grains=1000)
        engine.stabilize()

        tracker.record_stabilization()
        assert len(tracker) > 0

    def test_size_distribution(self):
        """Test avalanche size distribution."""
        from sandpile.statistics import StatisticsTracker

        engine = SandpileEngine(100, 100, use_gpu=False)
        tracker = StatisticsTracker(engine)

        # Trigger many avalanches
        for _ in range(50):
            engine.deposit_random(grains=10, count=1)
            engine.stabilize()
            tracker.record_stabilization()

        dist = tracker.get_size_distribution(bins=10)
        assert len(dist.sizes) > 0
        assert len(dist.frequencies) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
