"""
Export sandpile simulations to images and videos.

Supports PNG frame export and MP4 video via OpenCV or Matplotlib.
"""

import numpy as np
from typing import Optional, List
from pathlib import Path
from datetime import datetime
import warnings


class FrameExporter:
    """Export individual frames as PNG images."""

    def __init__(self, engine, output_dir: str = "./sandpile_exports"):
        self.engine = engine
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.frame_count = 0

    def capture(
        self,
        grid: Optional[np.ndarray] = None,
        filename: Optional[str] = None
    ) -> str:
        """
        Capture and save a single frame.

        Args:
            grid: Grid array (uses engine grid if None)
            filename: Custom filename or auto-generated

        Returns:
            Path to saved image
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        if grid is None:
            grid = self.engine.get_grid_cpu()

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=3, interpolation='nearest')
        ax.axis('off')
        plt.tight_layout(pad=0)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"frame_{self.frame_count:06d}_{timestamp}.png"

        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=150, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        self.frame_count += 1
        return str(filepath)

    def capture_sequence(
        self,
        steps: int,
        deposit_func,
        filename_pattern: str = "frame_{:06d}.png"
    ) -> List[str]:
        """
        Capture a sequence of frames while running simulation.

        Args:
            steps: Number of frames to capture
            deposit_func: Callable() called each step before capture
            filename_pattern: Python format string for filenames

        Returns:
            List of saved file paths
        """
        paths = []
        for i in range(steps):
            deposit_func()
            self.engine.stabilize()
            filename = filename_pattern.format(i)
            path = self.capture(filename=filename)
            paths.append(path)
            if i % 10 == 0:
                print(f"  Captured {i}/{steps}")
        return paths


class VideoExporter:
    """Export video (MP4) using OpenCV or Matplotlib animation."""

    def __init__(
        self,
        engine,
        output_path: str = "sandpile.mp4",
        fps: int = 30,
        codec: str = 'mp4v',
        width: Optional[int] = None,
        height: Optional[int] = None,
    ):
        """
        Initialize video exporter.

        Args:
            engine: SandpileEngine instance
            output_path: Output video file path
            fps: Frames per second
            codec: FourCC video codec ('mp4v', 'XVID', 'avc1')
            width: Video width (uses grid shape if None)
            height: Video height
        """
        self.engine = engine
        self.output_path = output_path
        self.fps = fps
        self.codec = codec

        grid = engine.get_grid_cpu()
        self.width = width or grid.shape[1] * 4  # 4x scale for visibility
        self.height = height or grid.shape[0] * 4

        self.writer = None
        self._frame_buffer: List[np.ndarray] = []
        self._initialized = False

    def _init_writer(self):
        """Lazy-initialize video writer."""
        try:
            import cv2
        except ImportError:
            raise ImportError(
                "OpenCV (cv2) required for video export. "
                "Install: pip install opencv-python"
            )

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        self.writer = cv2.VideoWriter(
            self.output_path,
            fourcc,
            self.fps,
            (self.width, self.height)
        )
        self._initialized = True

    def capture_frame(self, grid: Optional[np.ndarray] = None):
        """
        Capture a frame for later video assembly.

        Args:
            grid: Grid to render (uses engine grid if None)
        """
        if grid is None:
            grid = self.engine.get_grid_cpu()

        # Render with matplotlib (Agg backend for headless)
        import matplotlib
        matplotlib.use('Agg')  # Ensure non-interactive backend
        import matplotlib.pyplot as plt
        import io

        fig, ax = plt.subplots(figsize=(self.width/100, self.height/100), dpi=100)
        ax.imshow(grid, cmap='YlOrRd', vmin=0, vmax=3, interpolation='nearest')
        ax.axis('off')
        plt.tight_layout(pad=0)

        # Save to in-memory PNG buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

        # Read PNG and convert to RGB array
        buf.seek(0)
        from PIL import Image
        img = Image.open(buf).convert('RGB')
        image = np.array(img)

        # Resize if needed
        if image.shape[:2] != (self.height, self.width):
            import cv2
            image = cv2.resize(image, (self.width, self.height),
                              interpolation=cv2.INTER_NEAREST)

        self._frame_buffer.append(image)

    def finalize(self) -> str:
        """
        Write buffered frames to video file.

        Returns:
            Output file path
        """
        if not self._frame_buffer:
            warnings.warn("No frames captured")
            return self.output_path

        self._init_writer()

        for i, frame in enumerate(self._frame_buffer):
            # OpenCV uses BGR
            frame_bgr = frame[..., ::-1]
            self.writer.write(frame_bgr)

            if i % 30 == 0:
                print(f"  Wrote {i}/{len(self._frame_buffer)} frames")

        self.writer.release()
        print(f"[OK] Video saved to {self.output_path}")

        # Clear buffer
        self._frame_buffer.clear()

        return self.output_path

    def capture_and_write_realtime(
        self,
        steps: int,
        deposit_func,
    ) -> str:
        """
        Capture and write video in real-time as simulation runs.

        Args:
            steps: Number of frames
            deposit_func: Callable(i) to run simulation step i

        Returns:
            Output file path
        """
        self._init_writer()

        for i in range(steps):
            deposit_func(i)
            self.engine.stabilize()
            self.capture_frame()
            self.writer.write(self._frame_buffer[-1][..., ::-1])
            self._frame_buffer.pop()  # avoid double storage

            if i % 10 == 0:
                print(f"  Frame {i}/{steps}")

        self.writer.release()
        print(f"[OK] Video saved to {self.output_path}")
        return self.output_path


def export_grid_as_image(
    grid: np.ndarray,
    filepath: str,
    colormap: str = 'YlOrRd',
    dpi: int = 150,
):
    """
    Simple function to export a grid as a standalone image.

    Args:
        grid: 2D NumPy array
        filepath: Output path (.png, .jpg, .pdf, .svg)
        colormap: Matplotlib colormap name
        dpi: Resolution for raster formats
    """
    import matplotlib.pyplot as plt
    from matplotlib.cm import get_cmap

    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(grid, cmap=colormap, vmin=0, vmax=3, interpolation='nearest')
    ax.set_axis_off()
    plt.tight_layout(pad=0)

    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(filepath, dpi=dpi, bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    print(f"[OK] Saved image to {filepath}")
