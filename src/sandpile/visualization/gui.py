"""
Main GUI controller combining engine, renderer, and user interaction.

Provides the main application loop and connects all components.
"""

import pygame
import numpy as np
from typing import Optional, Tuple
import time

from ..core.engine import SandpileEngine
from .renderer import Renderer
from .colormap import DensityColorMap
from ..deposition import PatternFactory
from ..persistence import StateManager


class SandpileGUI:
    """
    Complete GUI application for sandpile simulation.

    Manages:
    - Engine lifecycle
    - Rendering
    - User input (keyboard, mouse)
    - Continuous rain mode
    - Export operations
    """

    # Pattern name to key mapping
    PATTERNS = {
        pygame.K_1: 'point',
        pygame.K_2: 'line_h',
        pygame.K_3: 'line_v',
        pygame.K_4: 'circle',
        pygame.K_5: 'square',
        pygame.K_6: 'cross',
        pygame.K_7: 'diamond',
    }

    def __init__(
        self,
        engine: Optional[SandpileEngine] = None,
        width: int = 500,
        height: int = 500,
        use_gpu: bool = True,
        window_width: int = 800,
        window_height: int = 800,
        fps: int = 60,
    ):
        """
        Initialize sandpile GUI.

        Args:
            engine: Existing engine or None to create new
            width: Grid width (if creating new engine)
            height: Grid height
            use_gpu: Use GPU acceleration
            window_width: PyGame window width
            window_height: PyGame window height
            fps: Target frame rate
        """
        # Create engine if not provided
        if engine is None:
            self.engine = SandpileEngine(width, height, use_gpu=use_gpu)
        else:
            self.engine = engine

        # Create color map
        self.colormap = DensityColorMap(palette='default', mode='linear')

        # Create renderer
        self.renderer = Renderer(
            engine=self.engine,
            colormap=self.colormap,
            width=window_width,
            height=window_height,
            fps=fps
        )

        # GUI state
        self.continuous_rain = False
        self.rain_interval = 0.1  # seconds between drops
        self._last_rain_time = 0
        self.drop_grains = 1
        self.selected_pattern = 'point'

        # State manager
        self.state_manager = StateManager(self.engine)

    def run(self):
        """Main application loop."""
        self.renderer.running = True

        print("\n" + "="*60)
        print("  ABELIAN SANDPILE MODEL - Interactive Simulation")
        print("="*60)
        print("\nControls:")
        print("  Left Click     - Drop sand at cursor")
        print("  Right Click    - Toggle continuous random rain")
        print("  Space          - Pause/Resume")
        print("  C              - Clear grid")
        print("  S / L          - Save / Load state")
        print("  R              - Random drop")
        print("  +/-            - Increase/decrease drop size")
        print("  1-7            - Select pattern (point, lines, circle, square, cross, diamond)")
        print("  E              - Export frame as PNG")
        print("  M              - Toggle stats panel")
        print("  G              - Toggle grid lines")
        print("  ESC            - Quit")
        print("="*60 + "\n")

        while self.renderer.running:
            # Handle events
            if not self.renderer.handle_events():
                break

            # Update simulation if not paused
            if not self.renderer.paused:
                self._update()

            # Render
            self.renderer.render()

            # Maintain FPS
            self.renderer.tick()

        self._cleanup()

    def _update(self):
        """Update simulation state each frame."""
        current_time = time.time()

        # Handle continuous rain
        if self.continuous_rain and (current_time - self._last_rain_time) >= self.rain_interval:
            self.engine.deposit_random(grains=self.drop_grains, count=1)
            self.engine.stabilize()
            self._last_rain_time = current_time
            self.renderer._surface_needs_update = True

    def handle_keydown(self, event):
        """Handle individual keyboard events."""
        key = event.key

        # Exit
        if key == pygame.K_ESCAPE:
            self.renderer.running = False
            return

        # Toggle pause
        elif key == pygame.K_SPACE:
            self.renderer.paused = not self.renderer.paused
            print(f"{'Paused' if self.renderer.paused else 'Running'}")

        # Clear
        elif key == pygame.K_c:
            self.engine.clear()
            self.renderer._surface_needs_update = True
            print("Grid cleared")

        # Save
        elif key == pygame.K_s:
            self.state_manager.save()

        # Load - show list and let user pick? For now load most recent
        elif key == pygame.K_l:
            states = self.state_manager.list_saved()
            if states:
                self.state_manager.load(states[0]['filename'])
                self.renderer._surface_needs_update = True
            else:
                print("No saved states found")

        # Random drop
        elif key == pygame.K_r:
            self.engine.deposit_random(grains=self.drop_grains, count=1)
            self.engine.stabilize()
            self.renderer._surface_needs_update = True

        # Increase drop size
        elif key == pygame.K_EQUALS or key == pygame.K_PLUS:
            self.drop_grains = min(100, self.drop_grains + 1)
            print(f"Drop size: {self.drop_grains}")

        # Decrease drop size
        elif key == pygame.K_MINUS:
            self.drop_grains = max(1, self.drop_grains - 1)
            print(f"Drop size: {self.drop_grains}")

        # Toggle rain mode
        elif key == pygame.K_x:  # Extra: X toggles rain
            self.continuous_rain = not self.continuous_rain
            print(f"Rain mode: {'ON' if self.continuous_rain else 'OFF'}")

        # Toggle stats display
        elif key == pygame.K_m:
            self.renderer.show_stats = not self.renderer.show_stats

        # Toggle grid lines
        elif key == pygame.K_g:
            self.renderer.show_grid = not self.renderer.show_grid

        # Export frame
        elif key == pygame.K_e:
            self.renderer.export_frame()

        # Pattern selection
        elif key in self.PATTERNS:
            self.selected_pattern = self.PATTERNS[key]
            print(f"Pattern: {self.selected_pattern} (multiple grains for full pattern)")

    def handle_mouse_down(self, button: int, pos: Tuple[int, int]):
        """Handle mouse button clicks."""
        x, y = pos

        # Check bounds
        rect = pygame.Rect(
            self.renderer.offset_x, self.renderer.offset_y,
            self.renderer.grid_width, self.renderer.grid_height
        )
        if not rect.collidepoint(x, y):
            return

        gx, gy = self.renderer.screen_to_grid(x, y)

        if button == 1:  # Left click - point deposition
            self.engine.deposit(gx, gy, grains=self.drop_grains)
            self.engine.stabilize()
            self.renderer._surface_needs_update = True

        elif button == 3:  # Right click - toggle rain
            self.continuous_rain = not self.continuous_rain
            print(f"Rain mode: {'ON' if self.continuous_rain else 'OFF'}")

    def deposit_current_pattern(self):
        """Deposit grains according to currently selected pattern."""
        try:
            self.engine.deposit_pattern(
                self.selected_pattern,
                grains=self.drop_grains
            )
            self.engine.stabilize()
            self.renderer._surface_needs_update = True
            print(f"Dropped pattern: {self.selected_pattern}")
        except Exception as e:
            print(f"Error: {e}")

    def _cleanup(self):
        """Clean up resources."""
        pygame.quit()

    def export_animation(
        self,
        output_path: str = "sandpile_animation.mp4",
        frames: int = 300,
        fps: int = 30
    ):
        """
        Export MP4 video of continuous random rain.

        Args:
            output_path: Output video file
            frames: Number of frames to capture
            fps: Frames per second in output
        """
        try:
            import cv2
        except ImportError:
            print("OpenCV (cv2) required for video export. Install: pip install opencv-python")
            return

        print(f"Exporting {frames} frames to {output_path}...")

        # Setup video writer
        frame_size = (self.renderer.window_width, self.renderer.window_height)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Render loop without display
        for i in range(frames):
            # Deposit and stabilize
            self.engine.deposit_random(grains=5, count=1)
            self.engine.stabilize()

            # Render to offscreen surface
            self.renderer.render()

            # Convert PyGame surface to OpenCV image
            array = pygame.surfarray.array3d(self.renderer.screen)
            array = array.swapaxes(0, 1)  # PyGame uses (x,y) order
            array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

            out.write(array)

            if i % 30 == 0:
                print(f"  Frame {i}/{frames}")

        out.release()
        print(f"[OK] Video saved to {output_path}")
