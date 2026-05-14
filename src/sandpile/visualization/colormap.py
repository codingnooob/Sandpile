"""
Color mapping for sandpile density visualization.

Provides linear and logarithmic color maps.
Supports custom color palettes and HSV interpolation.
"""

import numpy as np
from typing import Tuple, List
import pygame


class DensityColorMap:
    """
    Maps sandpile density levels (0, 1, 2, 3) to RGB colors.

    Supports:
    - Linear scaling (0=dark, 3=bright)
    - Logarithmic scaling for high dynamic range
    - Custom color palettes
    - Matplotlib colormap integration
    """

    # Default scientific colormaps (dark→bright)
    DEFAULT_COLORS = {
        0: (40, 40, 40),      # Dark gray (empty)
        1: (255, 200, 100),   # Light tan
        2: (210, 105, 30),    # Orange (chocolate)
        3: (180, 0, 0),       # Dark red (critical)
    }

    # High-contrast palettes
    PALETTES = {
        'default': DEFAULT_COLORS,
        'heat': {
            0: (0, 0, 0),
            1: (255, 0, 0),
            2: (255, 165, 0),
            3: (255, 255, 0),
        },
        'ocean': {
            0: (0, 10, 30),
            1: (0, 100, 200),
            2: (0, 200, 255),
            3: (200, 255, 255),
        },
        'viridis_like': {
            0: (68, 1, 84),
            1: (59, 82, 139),
            2: (33, 154, 143),
            3: (253, 231, 37),
        },
        'grayscale': {
            0: (0, 0, 0),
            1: (85, 85, 85),
            2: (170, 170, 170),
            3: (255, 255, 255),
        },
    }

    def __init__(
        self,
        palette: str = 'default',
        mode: str = 'linear',
        log_scale: bool = False,
        log_base: float = 2.0,
        over_color: Tuple[int, int, int] = None,
    ):
        """
        Initialize color map.

        Args:
            palette: 'default', 'heat', 'ocean', 'viridis_like', 'grayscale'
            mode: 'linear' or 'logarithmic'
            log_scale: Enable logarithmic scaling for values > 3
            log_base: Base for logarithm (default 2)
            over_color: Color for values exceeding normal range (> 3)
        """
        self.palette_name = palette
        self.mode = mode
        self.log_scale = log_scale
        self.log_base = log_base
        self.over_color = over_color or (255, 255, 255)  # White for overflow

        # Get base colors
        if palette not in self.PALETTES:
            raise ValueError(f"Unknown palette '{palette}'. Available: {list(self.PALETTES.keys())}")
        self.base_colors = self.PALETTES[palette].copy()

        # Precompute lookup table for speed (0-255 intensity levels)
        self._lut = self._build_lut()

    def _build_lut(self) -> List[Tuple[int, int, int]]:
        """
        Build 256-entry lookup table for fast color mapping.

        Uses linear interpolation between defined color stops.
        """
        lut = []

        # Extract stops
        levels = sorted(self.base_colors.keys())
        colors = [self.base_colors[l] for l in levels]

        for i in range(256):
            # Map i (0-255) to level (0-3)
            if self.log_scale:
                # Logarithmic: compress higher values
                # i' = base^(i/scale) - 1
                max_level = max(levels)
                level = self.log_base ** (i / 64) - 1
                level = min(level, max_level)
            else:
                # Linear: map 0-255 → 0-3
                level = (i / 255) * max(levels)

            # Interpolate color
            color = self._interpolate_color(levels, colors, level)
            lut.append(color)

        return lut

    def _interpolate_color(
        self,
        levels: List[int],
        colors: List[Tuple[int, int, int]],
        level: float
    ) -> Tuple[int, int, int]:
        """Interpolate RGB color for a given level between known stops."""
        if level <= levels[0]:
            return colors[0]
        if level >= levels[-1]:
            # Check if we have over_color for overflow
            if level > levels[-1] and self.over_color:
                return self.over_color
            return colors[-1]

        # Find bracketing levels
        for i in range(len(levels) - 1):
            l0, l1 = levels[i], levels[i + 1]
            c0, c1 = colors[i], colors[i + 1]

            if l0 <= level <= l1:
                # Linear interpolation factor
                t = (level - l0) / (l1 - l0)
                r = int(c0[0] * (1 - t) + c1[0] * t)
                g = int(c0[1] * (1 - t) + c1[1] * t)
                b = int(c0[2] * (1 - t) + c1[2] * t)
                return (r, g, b)

        return colors[-1]

    def get_color(self, value: int) -> Tuple[int, int, int]:
        """
        Get RGB color for a density value.

        Args:
            value: Density level (0, 1, 2, 3, or higher)

        Returns:
            (R, G, B) tuple each in [0, 255]
        """
        if value < 0:
            return (0, 0, 0)
        if value > 3:
            return self.over_color

        # Map level to 0-255 index
        idx = int(value * 85)  # 0→0, 1→85, 2→170, 3→255
        idx = max(0, min(255, idx))
        return self._lut[idx]

    def get_pygame_color(self, value: int) -> pygame.Color:
        """Get color as pygame.Color object."""
        rgb = self.get_color(value)
        return pygame.Color(*rgb)

    def map_grid_to_surface(self, grid: np.ndarray, zoom: int = 1) -> pygame.Surface:
        """
        Convert entire grid to a PyGame surface.

        Args:
            grid: 2D NumPy array (density levels 0-3+)
            zoom: Pixel multiplier (1 = 1 cell = 1 pixel)

        Returns:
            pygame.Surface with mapped colors
        """
        h, w = grid.shape
        surface = pygame.Surface((w * zoom, h * zoom))

        for y in range(h):
            for x in range(w):
                color = self.get_pygame_color(int(grid[y, x]))
                rect = pygame.Rect(x * zoom, y * zoom, zoom, zoom)
                surface.fill(color, rect)

        return surface

    def get_palette_names(self) -> List[str]:
        """List available palette names."""
        return list(self.PALETTES.keys())

    def set_palette(self, palette: str):
        """Change the active palette and rebuild LUT."""
        if palette not in self.PALETTES:
            raise ValueError(f"Unknown palette: {palette}")
        self.palette_name = palette
        self.base_colors = self.PALETTES[palette].copy()
        self._lut = self._build_lut()
