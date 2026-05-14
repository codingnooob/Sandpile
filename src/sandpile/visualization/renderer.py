"""
PyGame-based renderer for sandpile visualization.

Handles drawing the grid, color mapping, overlays, and UI elements.
"""

import pygame
import numpy as np
from typing import Optional, Tuple, List


class Renderer:
    """
    Main rendering engine for sandpile visualization.

    Manages a PyGame window and renders the grid with configurable
    color maps, zoom, overlays, and UI controls.
    """

    def __init__(
        self,
        engine,
        colormap,
        width: int = 800,
        height: int = 800,
        fps: int = 60,
    ):
        """
        Initialize renderer.

        Args:
            engine: SandpileEngine instance
            colormap: DensityColorMap instance
            width: Window width in pixels
            height: Window height in pixels
            fps: Target frames per second
        """
        self.engine = engine
        self.colormap = colormap
        self.window_width = width
        self.window_height = height
        self.fps = fps

        # Derived: cell size and offsets
        self._calculate_layout()

        # PyGame init
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Abelian Sandpile Model - GPU Accelerated")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Consolas', 14)
        self.font_large = pygame.font.SysFont('Consolas', 18, bold=True)

        # UI state
        self.running = False
        self.paused = False
        self.show_grid = False
        self.show_stats = True
        self.display_mode = 'density'  # 'density' or 'activity'
        self._zoom = 1

        # Cached surface for grid
        self._grid_surface: Optional[pygame.Surface] = None
        self._surface_needs_update = True

    def _calculate_layout(self):
        """Calculate zoom and offsets to fit grid in window."""
        self.cell_size = min(
            self.window_width // self.engine.width,
            self.window_height // self.engine.height
        )
        if self.cell_size < 1:
            self.cell_size = 1  # Allow scrolling for huge grids?

        self.grid_width = self.engine.width * self.cell_size
        self.grid_height = self.engine.height * self.cell_size

        self.offset_x = (self.window_width - self.grid_width) // 2
        self.offset_y = (self.window_height - self.grid_height) // 2

    def set_zoom(self, zoom: int):
        """Set zoom level (1 = pixel-perfect, >1 = scaled up)."""
        self._zoom = max(1, zoom)
        self._surface_needs_update = True

    def get_zoom(self) -> int:
        return self._zoom

    def screen_to_grid(self, screen_x: int, screen_y: int) -> Tuple[int, int]:
        """Convert screen coordinates to grid cell coordinates."""
        gx = (screen_x - self.offset_x) // self.cell_size
        gy = (screen_y - self.offset_y) // self.cell_size
        return gx, gy

    def grid_to_screen(self, gx: int, gy: int) -> Tuple[int, int]:
        """Convert grid cell coordinates to screen center pixel."""
        sx = self.offset_x + gx * self.cell_size
        sy = self.offset_y + gy * self.cell_size
        return sx, sy

    def render(self):
        """Main render call - draws everything."""
        self.screen.fill((30, 30, 30))  # Dark background

        # Render grid
        self._render_grid()

        # Render UI overlays
        if self.show_stats:
            self._render_stats()

        if self.show_grid:
            self._render_grid_lines()

        pygame.display.flip()

    def _render_grid(self):
        """Render the sandpile grid."""
        grid = self.engine.get_grid_cpu()
        h, w = grid.shape
        zoom = self._zoom

        # Create/lazy-update surface cache
        if self._grid_surface is None or self._surface_needs_update:
            size = (w * zoom, h * zoom)
            self._grid_surface = pygame.Surface(size)

            # Draw all cells
            for y in range(h):
                for x in range(w):
                    color = self.colormap.get_pygame_color(int(grid[y, x]))
                    rect = pygame.Rect(x * zoom, y * zoom, zoom, zoom)
                    self._grid_surface.fill(color, rect)

            self._surface_needs_update = False

        # Blit to screen with offsets
        self.screen.blit(self._grid_surface, (self.offset_x, self.offset_y))

        # Draw border around grid
        border_rect = pygame.Rect(
            self.offset_x - 1,
            self.offset_y - 1,
            self.grid_width + 2,
            self.grid_height + 2
        )
        pygame.draw.rect(self.screen, (100, 100, 100), border_rect, 1)

    def _render_grid_lines(self):
        """Draw grid lines showing cell boundaries."""
        for x in range(self.engine.width + 1):
            sx = self.offset_x + x * self.cell_size
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (sx, self.offset_y),
                (sx, self.offset_y + self.grid_height),
                1
            )
        for y in range(self.engine.height + 1):
            sy = self.offset_y + y * self.cell_size
            pygame.draw.line(
                self.screen,
                (60, 60, 60),
                (self.offset_x, sy),
                (self.offset_x + self.grid_width, sy),
                1
            )

    def _render_stats(self):
        """Render simulation statistics overlay."""
        stats = self.engine.get_statistics()

        lines = [
            f"Grid: {self.engine.width}x{self.engine.height}",
            f"Backend: {stats['backend']} {'(GPU)' if stats['gpu_available'] and self.engine.use_gpu else '(CPU)'}",
            f"Total Grains: {stats['total_grains']:,}",
            f"Threshold: {self.engine.threshold}",
            "",
            "Controls:",
            "Left Click: Drop at cursor",
            "Right Click: Toggle rain",
            "Space: Pause/Resume",
            "C: Clear, S: Save, L: Load",
            "1-7: Patterns, E: Export",
            "M: Stats, G: Grid lines",
        ]

        # Draw semi-transparent panel
        panel_width = 280
        panel_height = len(lines) * 18 + 20
        panel = pygame.Surface((panel_width, panel_height), pygame.SRCALPHA)
        panel.fill((0, 0, 0, 180))
        self.screen.blit(panel, (10, 10))

        # Render text
        y = 20
        for line in lines:
            if line:
                text = self.font.render(line, True, (220, 220, 220))
                self.screen.blit(text, (20, y))
            y += 18

        # FPS counter
        fps = self.clock.get_fps()
        fps_text = self.font.render(f"FPS: {fps:.1f}", True, (0, 255, 0))
        self.screen.blit(fps_text, (self.window_width - 100, 20))

        # Avalanche info if last avalanche exists
        last = self.engine.get_last_avalanche()
        if last:
            avy_text = self.font.render(
                f"Last: size={last['size']}, dur={last['duration']}",
                True, (255, 200, 0)
            )
            self.screen.blit(avy_text, (self.window_width - 250, 50))

    def handle_events(self) -> bool:
        """
        Process PyGame events. Returns False if window should close.

        Returns:
            True to continue, False to exit
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

            elif event.type == pygame.KEYDOWN:
                return self._handle_keydown(event)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                self._handle_mousedown(event)

        return True

    def _handle_keydown(self, event) -> bool:
        """Handle keyboard input."""
        key = event.key

        if key == pygame.K_ESCAPE:
            return False

        elif key == pygame.K_SPACE:
            self.paused = not self.paused
            print(f"{'Paused' if self.paused else 'Resumed'}")

        elif key == pygame.K_c:
            self.engine.clear()
            self._surface_needs_update = True

        elif key == pygame.K_s:
            from ..persistence import StateManager
            sm = StateManager(self.engine)
            sm.save()

        elif key == pygame.K_l:
            from ..persistence import StateManager
            sm = StateManager(self.engine)
            sm.load_last() or sm.list_saved()  # TODO: better loading

        elif key == pygame.K_r:
            self.engine.deposit_random(grains=1, count=1)
            self._surface_needs_update = True

        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            # Increase drop size (stored externally in GUI controller)
            pass  # Handled by controller

        elif key == pygame.K_MINUS:
            # Decrease drop size
            pass

        elif key == pygame.K_g:
            self.show_grid = not self.show_grid

        elif key == pygame.K_m:
            self.show_stats = not self.show_stats

        elif key == pygame.K_e:
            self.export_frame()

        elif pygame.K_1 <= key <= pygame.K_7:
            # Pattern selection
            patterns = ['point', 'line_h', 'line_v', 'circle',
                       'square', 'cross', 'diamond']
            idx = key - pygame.K_1
            if idx < len(patterns):
                pattern = patterns[idx]
                print(f"Pattern: {pattern}")
                # Handled by controller

        return True

    def _handle_mousedown(self, event):
        """Handle mouse clicks."""
        x, y = pygame.mouse.get_pos()

        # Check if click is within grid bounds
        grid_rect = pygame.Rect(
            self.offset_x, self.offset_y,
            self.grid_width, self.grid_height
        )
        if not grid_rect.collidepoint(x, y):
            return

        gx, gy = self.screen_to_grid(x, y)

        if event.button == 1:  # Left click - single drop
            self.engine.deposit(gx, gy, grains=1)
            self._surface_needs_update = True

        elif event.button == 3:  # Right click - toggle rain
            # Handled by controller
            pass

    def export_frame(self, filename: str = None):
        """
        Export current frame as PNG.

        Args:
            filename: Output path, auto-generated if None
        """
        from datetime import datetime
        if filename is None:
            filename = f"sandpile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

        # Render without UI overlays for clean image
        self._render_grid()
        pygame.image.save(self.screen, filename)
        print(f"[OK] Exported {filename}")

    def tick(self):
        """Maintain consistent frame rate."""
        self.clock.tick(self.fps)

    def quit(self):
        """Shutdown renderer."""
        pygame.quit()
