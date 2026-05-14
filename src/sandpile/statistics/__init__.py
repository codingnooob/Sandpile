"""
Statistics tracking for avalanche analysis.

Measures: avalanche size, duration, radius, grain movements.
Collects historical data for power-law distribution analysis.
"""

import numpy as np
from typing import List, Dict, Any, Optional
from collections import deque
from dataclasses import dataclass, field


@dataclass
class AvalancheMetrics:
    """Metrics for a single avalanche event."""
    size: int              # Number of unique cells involved
    duration: int          # Number of toppling steps
    toppled_cells: int     # Total cells that toppled (sum per step)
    radius: int            # Approximate geometric radius
    grains_moved: int      # Total grains redistributed
    timestamp: int         # Sequential ID


@dataclass
class DistributionData:
    """Avalanche size distribution for power-law analysis."""
    sizes: np.ndarray      # Unique avalanche sizes
    frequencies: np.ndarray  # Count of each size
    bins: np.ndarray       # Bin edges if binned


class StatisticsTracker:
    """
    Tracks sandpile statistics during simulation.

    Automatically detects avalanche events by monitoring
    toppling activity between stable states.
    """

    def __init__(self, engine, max_history: int = 10000):
        """
        Initialize tracker.

        Args:
            engine: SandpileEngine instance to monitor
            max_history: Maximum avalanches to keep in memory
        """
        self.engine = engine
        self.max_history = max_history

        self._avalanche_history: List[AvalancheMetrics] = []
        self._size_distribution_cache: Optional[DistributionData] = None
        self._dirty = True  # Flag to recompute distribution

    def record_stabilization(self) -> Optional[AvalancheMetrics]:
        """
        Record a completed stabilization (avalanche).

        Called after engine.stabilize() to capture metrics.

        Returns:
            AvalancheMetrics if an avalanche occurred, else None
        """
        last = self.engine.get_last_avalanche()
        if last:
            metrics = AvalancheMetrics(
                size=last['size'],
                duration=last['duration'],
                toppled_cells=last['toppled_cells'],
                radius=last['radius'],
                grains_moved=last['toppled_cells'] * self.engine.threshold,
                timestamp=last['timestamp'],
            )
            self._avalanche_history.append(metrics)
            self._dirty = True

            # Trim history if needed
            if len(self._avalanche_history) > self.max_history:
                self._avalanche_history = self._avalanche_history[-self.max_history:]

            return metrics
        return None

    def get_avalanche_sizes(self) -> np.ndarray:
        """Get array of all recorded avalanche sizes."""
        if not self._avalanche_history:
            return np.array([], dtype=int)
        return np.array([a.size for a in self._avalanche_history], dtype=int)

    def get_avalanche_durations(self) -> np.ndarray:
        """Get array of all recorded avalanche durations (steps)."""
        if not self._avalanche_history:
            return np.array([], dtype=int)
        return np.array([a.duration for a in self._avalanche_history], dtype=int)

    def get_avalanche_radii(self) -> np.ndarray:
        """Get array of all recorded avalanche radii."""
        if not self._avalanche_history:
            return np.array([], dtype=int)
        return np.array([a.radius for a in self._avalanche_history], dtype=int)

    def get_size_distribution(self, bins: int = 50) -> DistributionData:
        """
        Compute avalanche size distribution.

        Args:
            bins: Number of histogram bins (or 'auto' for automatic)

        Returns:
            DistributionData object
        """
        sizes = self.get_avalanche_sizes()
        if len(sizes) == 0:
            return DistributionData(
                sizes=np.array([]),
                frequencies=np.array([]),
                bins=np.array([])
            )

        if self._dirty or self._size_distribution_cache is None:
            if bins == 'auto':
                # Use Sturges or Freedman-Diaconis
                unique_sizes = np.unique(sizes)
                if len(unique_sizes) < 10:
                    bin_edges = np.concatenate([
                        unique_sizes - 0.5,
                        [unique_sizes[-1] + 0.5]
                    ])
                else:
                    # Freedman-Diaconis
                    q75, q25 = np.percentile(sizes, [75, 25])
                    iqr = q75 - q25
                    if iqr > 0:
                        bin_width = 2 * iqr / (len(sizes) ** (1/3))
                        bin_edges = np.arange(
                            sizes.min(), sizes.max() + bin_width, bin_width
                        )
                    else:
                        bin_edges = 50
            else:
                bin_edges = bins

            hist, bin_edges = np.histogram(sizes, bins=bin_edges, density=False)
            centers = (bin_edges[:-1] + bin_edges[1:]) / 2

            self._size_distribution_cache = DistributionData(
                sizes=centers,
                frequencies=hist,
                bins=bin_edges
            )
            self._dirty = False

        return self._size_distribution_cache

    def get_summary_statistics(self) -> Dict[str, float]:
        """
        Compute summary statistics over all avalanches.

        Returns:
            Dict with mean, median, max, std for size/duration/radius
        """
        sizes = self.get_avalanche_sizes()
        durations = self.get_avalanche_durations()
        radii = self.get_avalanche_radii()

        def safe_stats(arr):
            if len(arr) == 0:
                return {'mean': 0, 'median': 0, 'max': 0, 'std': 0, 'count': 0}
            return {
                'mean': float(np.mean(arr)),
                'median': float(np.median(arr)),
                'max': float(np.max(arr)),
                'std': float(np.std(arr)),
                'count': len(arr),
            }

        return {
            'size': safe_stats(sizes),
            'duration': safe_stats(durations),
            'radius': safe_stats(radii),
            'total_avalanches': len(self._avalanche_history),
        }

    def estimate_power_law_exponent(
        self,
        min_size: Optional[int] = None,
        method: str = 'mle'
    ) -> float:
        """
        Estimate power-law exponent α from avalanche size distribution.

        The sandpile model is expected to follow P(s) ∝ s^(-α)
        with α ≈ 1.0 for the 2D Abelian case.

        Args:
            min_size: Minimum avalanche size to include (estimates cutoff)
            method: 'mle' for maximum likelihood or 'lsq' for least-squares

        Returns:
            Estimated exponent α (positive number)
        """
        sizes = self.get_avalanche_sizes()
        if len(sizes) < 10:
            return np.nan

        if min_size is None:
            # Use minimum size where distribution follows power law
            min_size = int(np.percentile(sizes, 10))

        # Filter sizes >= min_size
        sample = sizes[sizes >= min_size]

        if len(sample) < 5:
            return np.nan

        if method == 'mle':
            # Maximum likelihood estimator for power-law exponent
            # α = 1 + n / (sum(log(x_i / x_min)))
            xmin = min_size
            n = len(sample)
            alpha = 1 + n / np.sum(np.log(sample / xmin))
            return alpha
        else:
            # Least-squares fit on log-log plot
            unique, counts = np.unique(sample, return_counts=True)
            log_s = np.log(unique)
            log_p = np.log(counts / len(sample))
            # Linear regression
            coeffs = np.polyfit(log_s, log_p, 1)
            return -coeffs[0]  # slope is -α

    def get_activity_heatmap(self) -> np.ndarray:
        """
        Get heatmap of toppling activity (cells that have toppled most).

        Returns:
            2D NumPy array (same shape as grid) with activity counts
        """
        # This would require engine to expose topple counts per cell
        # For now, return zeros (placeholder for future enhancement)
        return np.zeros((self.engine.height, self.engine.width), dtype=int)

    def clear(self):
        """Clear all statistics."""
        self._avalanche_history.clear()
        self._size_distribution_cache = None
        self._dirty = True

    def __len__(self) -> int:
        return len(self._avalanche_history)

    def __repr__(self) -> str:
        return f"StatisticsTracker(avalanches={len(self._avalanche_history)})"
