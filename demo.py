#!/usr/bin/env python3
"""Quick demo: generate a sandpile fractal and export it."""
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
from sandpile import SandpileEngine

engine = SandpileEngine(300, 300)
engine.deposit(150, 150, grains=5000)
print("Stabilizing...")
iters, toppled = engine.stabilize()
print(f"Stabilized in {iters} steps, {toppled} toppled cells")
grid = engine.get_grid_cpu()

# Save via matplotlib
plt.figure(figsize=(6,6))
plt.imshow(grid, cmap='YlOrRd', vmin=0, vmax=3, interpolation='nearest')
plt.axis('off')
plt.tight_layout(pad=0)
plt.savefig("demo_fractal.png", dpi=150, bbox_inches='tight', pad_inches=0)
plt.close()
print("Saved demo_fractal.png")
print(f"Total grains: {engine.get_total_grains()}")
print(f"Max cell: {grid.max()} (stable < 4)")
