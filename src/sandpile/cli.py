"""
Command-line interface (CLI) for sandpile simulation.

Usage:
    sandpile                  # Launch interactive GUI with default 500x500 grid
    sandpile --size 1000x1000 # Custom grid size
    sandpile --no-gpu         # Force CPU mode
    sandpile --help           # Show usage help
"""

import argparse
import sys
from pathlib import Path

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from sandpile import SandpileEngine, SandpileGUI, GPU_AVAILABLE
from sandpile.visualization.exporter import VideoExporter


def main():
    parser = argparse.ArgumentParser(
        prog="sandpile",
        description="Abelian Sandpile Model - GPU-accelerated simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                     Start GUI with 500x500 grid
  %(prog)s --size 1000x1000   Start with 1000x1000 grid
  %(prog)s --no-gpu           Force CPU mode
  %(prog)s --benchmark        Run performance benchmark
        """
    )

    parser.add_argument(
        '--size', '-s',
        type=str,
        default='500x500',
        help='Grid size as WIDTHxHEIGHT (e.g., 800x600)'
    )

    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Disable GPU acceleration (use CPU only)'
    )

    parser.add_argument(
        '--benchmark', '-b',
        action='store_true',
        help='Run performance benchmark and exit'
    )

    parser.add_argument(
        '--export-video', '-e',
        type=str,
        metavar='FILE',
        help='Export video to FILE (runs simulation, saves, exits)'
    )

    parser.add_argument(
        '--frames', '-f',
        type=int,
        default=300,
        help='Number of frames for video export (default: 300)'
    )

    parser.add_argument(
        '--version', '-v',
        action='store_true',
        help='Show version and exit'
    )

    args = parser.parse_args()

    if args.version:
        from sandpile import __version__
        print(f"Sandpile {__version__}")
        return

    if args.benchmark:
        run_benchmark()
        return

    if args.export_video:
        run_video_export(
            output=args.export_video,
            frames=args.frames,
            size=args.size,
            use_gpu=not args.no_gpu
        )
        return

    # Normal GUI mode
    try:
        w, h = map(int, args.size.lower().split('x'))
    except ValueError:
        print(f"Error: Invalid size format '{args.size}'. Use WIDTHxHEIGHT (e.g., 500x500)")
        sys.exit(1)

    use_gpu = not args.no_gpu and GPU_AVAILABLE
    if args.no_gpu:
        print("GPU acceleration disabled by user flag")
    elif GPU_AVAILABLE:
        print("GPU acceleration enabled")
    else:
        print("GPU not available - using CPU mode")

    try:
        engine = SandpileEngine(width=w, height=h, use_gpu=use_gpu)
        gui = SandpileGUI(engine)
        gui.run()
    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(0)
    except Exception as e:
        print(f"Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_benchmark():
    """Run performance benchmark and print results."""
    print("\n" + "="*60)
    print("  ABELIAN SANDPILE - Performance Benchmark")
    print("="*60 + "\n")

    import time
    import numpy as np
    from sandpile import SandpileEngine, HAS_CUPY

    sizes = [200, 500, 1000]
    backends = ['cpu', 'gpu'] if HAS_CUPY else ['cpu']

    results = []

    for size in sizes:
        for backend in backends:
            if backend == 'gpu' and not HAS_CUPY:
                continue

            print(f"\nGrid {size}x{size} | Backend: {backend.upper()}")

            engine = SandpileEngine(size, size, use_gpu=(backend == 'gpu'))

            # Benchmark: single drop stabilization
            times = []
            for _ in range(5):
                engine.clear()
                engine.deposit(size//2, size//2, grains=1000)
                start = time.time()
                engine.stabilize()
                elapsed = time.time() - start
                times.append(elapsed)

            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"  Stabilization: {mean_time*1000:.2f} ± {std_time*1000:.2f} ms")

            # Benchmark: continuous rain (10 drops)
            engine.clear()
            times = []
            for _ in range(10):
                engine.deposit_random(grains=1, count=1)
                start = time.time()
                engine.stabilize()
                elapsed = time.time() - start
                times.append(elapsed)

            mean_time = np.mean(times)
            print(f"  Avg per drop: {mean_time*1000:.2f} ms (est. {1000/mean_time:.0f} FPS)")

            results.append((size, backend, mean_time))

    print("\n" + "="*60)
    print("Benchmark complete")
    print("="*60 + "\n")

    # Print summary table
    print(f"{'Size':>8} | {'Backend':>8} | {'Avg Time (ms)':>14}")
    print("-" * 36)
    for size, backend, t in results:
        print(f"{size:>8}x | {backend:>8} | {t*1000:>14.2f}")


def run_video_export(output: str, frames: int, size: str, use_gpu: bool):
    """Run simulation and export as video."""
    try:
        w, h = map(int, size.lower().split('x'))
    except ValueError:
        print(f"Error: Invalid size format '{size}'")
        sys.exit(1)

    print(f"\nExporting video: {frames} frames, {w}x{h} grid\n")

    engine = SandpileEngine(w, h, use_gpu=use_gpu)
    exporter = VideoExporter(engine, output_path=output, fps=30)

    def deposit_step(i):
        # Continuous random rain
        engine.deposit_random(grains=5, count=1)

    exporter.capture_and_write_realtime(frames, deposit_step)
    print(f"\n✓ Video export complete: {output}")


if __name__ == '__main__':
    main()
