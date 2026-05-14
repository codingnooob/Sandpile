"""Tests for visualization components."""

import pytest
import numpy as np
import pygame

from sandpile.core.engine import SandpileEngine
from sandpile.visualization.colormap import DensityColorMap
from sandpile.visualization.renderer import Renderer


class TestColorMap:
    """Tests for DensityColorMap."""

    def test_default_colors(self):
        """Test default palette has correct stops."""
        cmap = DensityColorMap(palette='default')
        assert cmap.get_color(0) == (40, 40, 40)
        assert cmap.get_color(1) == (255, 200, 100)
        assert cmap.get_color(2) == (210, 105, 30)
        assert cmap.get_color(3) == (180, 0, 0)

    def test_out_of_range_values(self):
        """Test values < 0 and > 3 handled."""
        cmap = DensityColorMap(palette='default')
        assert cmap.get_color(-1) == (0, 0, 0)
        assert cmap.get_color(10) == cmap.over_color

    def test_palette_switching(self):
        """Test changing palette updates colors."""
        cmap = DensityColorMap(palette='heat')
        heat1 = cmap.get_color(1)
        cmap.set_palette('default')
        default1 = cmap.get_color(1)
        assert heat1 != default1

    def test_available_palettes(self):
        """Test listing palettes."""
        cmap = DensityColorMap()
        names = cmap.get_palette_names()
        assert 'default' in names
        assert 'heat' in names
        assert 'ocean' in names


class TestRenderer:
    """Tests for PyGame renderer."""

    def test_renderer_initialization(self):
        """Test renderer creates window."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        cmap = DensityColorMap()
        renderer = Renderer(engine, cmap, width=400, height=400)

        assert renderer.screen is not None
        assert renderer.cell_size >= 1

    def test_screen_to_grid_conversion(self):
        """Test coordinate mapping."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        cmap = DensityColorMap()
        renderer = Renderer(engine, cmap, width=400, height=400)

        # Convert screen center to grid
        gx, gy = renderer.screen_to_grid(200, 200)
        assert 0 <= gx < 100
        assert 0 <= gy < 100

    def test_zoom_changes_cell_size(self):
        """Test zoom affects cell rendering."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        cmap = DensityColorMap()
        renderer = Renderer(engine, cmap, width=800, height=800)

        initial_cell = renderer.cell_size
        renderer.set_zoom(2)
        assert renderer.get_zoom() == 2
        # Cell size in pixels should increase
        assert renderer.cell_size >= initial_cell


class TestIntegration:
    """Integration tests: engine + renderer."""

    def test_full_simulation_render(self):
        """Test complete simulation renders without error."""
        engine = SandpileEngine(100, 100, use_gpu=False)
        cmap = DensityColorMap()
        renderer = Renderer(engine, cmap, width=400, height=400, fps=60)

        # Run simulation
        engine.deposit(50, 50, grains=1000)
        engine.stabilize()

        # Render
        renderer.render()
        # No exception = success
        renderer.quit()
