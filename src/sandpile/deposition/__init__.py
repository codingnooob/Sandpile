"""
Deposition strategies and patterns.

Provides various ways to deposit grains: single point, random,
and geometric patterns (lines, circles, squares, crosses, diamonds).
"""

import numpy as np
from typing import List, Tuple, Callable
from abc import ABC, abstractmethod


class DepositionStrategy(ABC):
    """Base class for deposition strategies."""

    @abstractmethod
    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        """Return list of (x, y) coordinates to deposit."""
        pass


class PointDeposition(DepositionStrategy):
    """Single point deposition (default: center)."""

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        return [(self.x, self.y)]


class RandomDeposition(DepositionStrategy):
    """Uniform random deposition."""

    def __init__(self, count: int = 10, rng=None):
        self.count = count
        self.rng = rng or np.random.default_rng()

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        xs = self.rng.integers(0, width, size=self.count)
        ys = self.rng.integers(0, height, size=self.count)
        return list(zip(xs, ys))


class LineHorizontalDeposition(DepositionStrategy):
    """Horizontal line across the grid."""

    def __init__(self, y: int = None):
        self.y = y

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        y = self.y if self.y is not None else height // 2
        return [(x, y) for x in range(width)]


class LineVerticalDeposition(DepositionStrategy):
    """Vertical line across the grid."""

    def __init__(self, x: int = None):
        self.x = x

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        x = self.x if self.x is not None else width // 2
        return [(x, y) for y in range(height)]


class CircleDeposition(DepositionStrategy):
    """Filled circle pattern."""

    def __init__(self, center: Tuple[int, int] = None, radius: int = 10):
        self.center = center
        self.radius = radius

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        cx, cy = self.center if self.center else (width // 2, height // 2)
        coords = []
        for y in range(max(0, cy - self.radius), min(height, cy + self.radius + 1)):
            for x in range(max(0, cx - self.radius), min(width, cx + self.radius + 1)):
                if (x - cx) ** 2 + (y - cy) ** 2 <= self.radius ** 2:
                    coords.append((x, y))
        return coords


class SquareDeposition(DepositionStrategy):
    """Filled square pattern."""

    def __init__(self, center: Tuple[int, int] = None, size: int = 10):
        """
        Args:
            size: Side length in cells. Even sizes produce axis-aligned squares
                  perfectly centered between cells; odd sizes include the center cell.
        """
        self.center = center
        self.size = size

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        cx, cy = self.center if self.center else (width // 2, height // 2)
        half = self.size // 2

        if self.size % 2 == 0:
            # Even size: exactly size cells, centered as close as possible
            x0 = max(0, cx - half)
            x1 = min(width, cx + half)  # exclusive
            y0 = max(0, cy - half)
            y1 = min(height, cy + half)
        else:
            # Odd size: inclusive range
            x0 = max(0, cx - half)
            x1 = min(width, cx + half + 1)
            y0 = max(0, cy - half)
            y1 = min(height, cy + half + 1)

        coords = []
        for y in range(y0, y1):
            for x in range(x0, x1):
                coords.append((x, y))
        return coords


class CrossDeposition(DepositionStrategy):
    """Plus-shaped cross pattern."""

    def __init__(self, center: Tuple[int, int] = None, length: int = 20, thickness: int = 1):
        self.center = center
        self.length = length
        self.thickness = thickness

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        cx, cy = self.center if self.center else (width // 2, height // 2)
        half = self.length // 2
        coords = []

        # Horizontal line
        for x in range(max(0, cx - half), min(width, cx + half + 1)):
            for dy in range(-self.thickness // 2, self.thickness // 2 + 1):
                y = cy + dy
                if 0 <= y < height:
                    coords.append((x, y))

        # Vertical line
        for y in range(max(0, cy - half), min(height, cy + half + 1)):
            for dx in range(-self.thickness // 2, self.thickness // 2 + 1):
                x = cx + dx
                if 0 <= x < width:
                    coords.append((x, y))

        # Remove duplicates (intersection)
        return list(set(coords))


class DiamondDeposition(DepositionStrategy):
    """Filled diamond (rotated square) pattern."""

    def __init__(self, center: Tuple[int, int] = None, radius: int = 10):
        self.center = center
        self.radius = radius

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        cx, cy = self.center if self.center else (width // 2, height // 2)
        coords = []
        for y in range(max(0, cy - self.radius), min(height, cy + self.radius + 1)):
            for x in range(max(0, cx - self.radius), min(width, cx + self.radius + 1)):
                # Manhattan distance
                if abs(x - cx) + abs(y - cy) <= self.radius:
                    coords.append((x, y))
        return coords


class UniformDeposition(DepositionStrategy):
    """Add same number of grains to every cell."""

    def get_coordinates(self, width: int, height: int) -> List[Tuple[int, int]]:
        # Return sentinel value - engine handles uniform deposit differently
        return []


class PatternFactory:
    """Factory for creating deposition pattern strategies."""

    _PATTERNS = {
        'point': PointDeposition,
        'random': RandomDeposition,
        'line_h': LineHorizontalDeposition,
        'line_v': LineVerticalDeposition,
        'circle': CircleDeposition,
        'square': SquareDeposition,
        'cross': CrossDeposition,
        'diamond': DiamondDeposition,
        'uniform': UniformDeposition,
    }

    @classmethod
    def get(
        cls,
        pattern_name: str,
        **kwargs
    ) -> DepositionStrategy:
        """
        Get a deposition strategy by name.

        Args:
            pattern_name: Name of pattern (see _PATTERNS keys)
            **kwargs: Arguments to pass to pattern constructor.
                     Generic 'center' (x,y) is automatically mapped to
                     pattern-specific coordinates (e.g., 'y' for line_h).

        Returns:
            DepositionStrategy instance

        Raises:
            ValueError: Unknown pattern name
        """
        pattern_name = pattern_name.lower().strip()
        if pattern_name not in cls._PATTERNS:
            raise ValueError(
                f"Unknown pattern '{pattern_name}'. "
                f"Available: {list(cls._PATTERNS.keys())}"
            )

        # Normalize generic 'center' argument to pattern-specific params
        if 'center' in kwargs:
            center = kwargs.pop('center')
            cx, cy = center
            if pattern_name == 'point':
                kwargs['x'], kwargs['y'] = cx, cy
            elif pattern_name == 'line_h':
                kwargs['y'] = cy
            elif pattern_name == 'line_v':
                kwargs['x'] = cx
            else:
                # circle, square, cross, diamond accept center as-is
                kwargs['center'] = center

        return cls._PATTERNS[pattern_name](**kwargs)

    @classmethod
    def available_patterns(cls) -> List[str]:
        """List all available pattern names."""
        return list(cls._PATTERNS.keys())


# Convenience function
def create_deposition_strategy(
    pattern: str,
    **kwargs
) -> DepositionStrategy:
    """Convenience wrapper for PatternFactory.get()"""
    return PatternFactory.get(pattern, **kwargs)
