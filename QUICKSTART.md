# Quick Start Guide

## Installation

```bash
# Install Python 3.9+ and pip

# Install core dependencies
pip install -r requirements.txt

# Install CuPy for GPU acceleration (choose one):
# CUDA 12.x (RTX 30/40, most common):
pip install cupy-cuda12x

# OR CUDA 11.x (older GPUs):
# pip install cupy-cuda11x

# Install package (development mode)
pip install -e .

# Optional: dev dependencies
pip install -e ".[dev]"
```

## Run the Simulation

```bash
# Launch interactive GUI
sandpile

# With custom grid size
sandpile --size 1000x800

# Run performance benchmark
sandpile --benchmark

# Export 300-frame video
sandpile --export-video output.mp4 --frames 300

# Use CPU only (no GPU)
sandpile --no-gpu
```

## Python API

```python
from sandpile import SandpileEngine

# Create engine
engine = SandpileEngine(500, 500)

# Add grains
engine.deposit(250, 250, grains=1000)   # precise location
engine.deposit_random(grains=10)         # random uniform
engine.deposit_pattern('circle', grains=5, radius=10)  # preset

# Stabilize
engine.stabilize()

# Access results
grid = engine.get_grid_cpu()  # NumPy array
print(f"Total grains: {engine.get_total_grains()}")
```

## Controls (GUI)

| Key | Action |
|-----|--------|
| Left click | Drop sand |
| Right click | Toggle continuous rain |
| Space | Pause/resume |
| C | Clear grid |
| S / L | Save / Load |
| 1-7 | Select pattern |
| E | Export PNG |
| +/- | Change drop size |
| M | Toggle stats |
| G | Toggle grid lines |

## Examples

### Generate a fractal pattern

```python
from sandpile import SandpileEngine
from sandpile.visualization.exporter import export_grid_as_image

engine = SandpileEngine(500, 500)
engine.deposit(250, 250, grains=10000)
engine.stabilize()
export_grid_as_image(engine.get_grid_cpu(), "fractal.png")
```

### Capture continuous rain

```python
from sandpile import SandpileEngine, SandpileGUI

engine = SandpileEngine(500, 500, use_gpu=True)
gui = SandpileGUI(engine)

# Right-click in GUI to toggle rain mode
# Press E to export frames
# Press M to see avalanche stats
gui.run()
```

### Analyze avalanche statistics

```python
from sandpile import SandpileEngine, StatisticsTracker

engine = SandpileEngine(300, 300)
tracker = StatisticsTracker(engine)

for _ in range(100):
    engine.deposit_random(grains=5)
    engine.stabilize()
    tracker.record_stabilization()

print(f"Total avalanches: {len(tracker)}")
print(f"Mean size: {tracker.get_summary_statistics()['size']['mean']:.1f}")
```

## Troubleshooting

**"No module named 'cupy'"** → CuPy not installed. Install with `pip install cupy-cuda12x` (CUDA 12) or `cupy-cuda11x` (CUDA 11). For CPU-only mode, use `sandpile --no-gpu`.

**PyGame window closes immediately** → Press `Space` to pause, or run from terminal: `python -m sandpile`.

**Low FPS on large grids** → Reduce grid size with `--size` or enable GPU (`cupy-cuda12x`).

**OpenCV video export fails** → Ensure `opencv-python` installed: `pip install opencv-python`.

## Expected Behavior

- **Single drop**: Creates intricate fractal-like patterns (10,000+ grains)
- **Continuous rain**: Produces steady-state SOC with ongoing avalanches
- **Pattern deposits**: Generate symmetrical dendritic structures
- **Statistics**: Avalanche sizes follow power-law distribution P(s) ∝ s^(-α) with α ≈ 1.0
