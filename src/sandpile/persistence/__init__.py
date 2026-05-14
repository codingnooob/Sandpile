"""
State persistence: save/load sandpile engine states.

Uses NumPy's .npz compressed format for portability.
Supports grid state, metadata, and avalanche history.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime


class StateManager:
    """Manages saving and loading of sandpile engine states."""

    def __init__(self, engine, storage_dir: str = "./sandpile_states"):
        """
        Initialize state manager.

        Args:
            engine: SandpileEngine instance
            storage_dir: Directory for saved states
        """
        self.engine = engine
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        filename: Optional[str] = None,
        compressed: bool = True,
        include_history: bool = True
    ) -> str:
        """
        Save current engine state to file.

        Args:
            filename: Custom filename or None for auto-generated
            compressed: Use npz_compressed for smaller files
            include_history: Include avalanche history

        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"sandpile_{self.engine.width}x{self.engine.height}_{timestamp}.npz"

        filepath = self.storage_dir / filename

        # Build save dict
        save_dict = {
            'grid': self.engine.get_grid_cpu(),
            'width': self.engine.width,
            'height': self.engine.height,
            'threshold': self.engine.threshold,
            'total_grains': self.engine._total_grains,
            'stabilization_iterations': self.engine._stabilization_iterations,
            'total_toppled': self.engine._total_toppled,
        }

        if include_history:
            save_dict['avalanche_history'] = self.engine._avalanche_history

        # Save metadata as JSON string for readability
        metadata = {
            'saved_at': datetime.now().isoformat(),
            'backend': self.engine.backend,
            'boundary': self.engine.boundary,
            'avalanche_count': len(self.engine._avalanche_history),
        }
        save_dict['metadata_json'] = json.dumps(metadata)

        if compressed:
            np.savez_compressed(filepath, **save_dict)
        else:
            np.savez(filepath, **save_dict)

        print(f"[OK] Saved state to {filepath} ({filepath.stat().st_size / 1024:.1f} KB)")
        return str(filepath)

    def load(self, filename: str) -> bool:
        """
        Load engine state from file.

        Args:
            filename: Filename in storage_dir (or full path)

        Returns:
            True if load succeeded
        """
        filepath = Path(filename)
        if not filepath.is_absolute():
            filepath = self.storage_dir / filename

        if not filepath.exists():
            print(f"[FAIL] File not found: {filepath}")
            return False

        try:
            data = np.load(filepath, allow_pickle=True)

            # Load core parameters
            self.engine.width = int(data['width'])
            self.engine.height = int(data['height'])
            self.engine.threshold = int(data['threshold'])
            self.engine._total_grains = int(data['total_grains'])
            self.engine._stabilization_iterations = int(data['stabilization_iterations'])
            self.engine._total_toppled = int(data['total_toppled'])

            # Load grid
            self.engine.set_grid(data['grid'])

            # Load avalanche history if present
            if 'avalanche_history' in data:
                self.engine._avalanche_history = list(data['avalanche_history'])
            else:
                self.engine._avalanche_history = []

            # Print metadata if available
            if 'metadata_json' in data:
                meta = json.loads(str(data['metadata_json']))
                print(f"[OK] Loaded state from {filepath.name}")
                print(f"  Saved: {meta.get('saved_at', 'unknown')}")
                print(f"  Avalanches: {meta.get('avalanche_count', 'N/A')}")

            return True

        except Exception as e:
            print(f"[FAIL] Failed to load {filepath}: {e}")
            return False

    def list_saved(self) -> list:
        """
        List all saved states in storage_dir.

        Returns:
            List of filenames with metadata
        """
        states = []
        for f in sorted(self.storage_dir.glob("*.npz")):
            try:
                data = np.load(f, allow_pickle=True)
                meta = {}
                if 'metadata_json' in data:
                    meta = json.loads(str(data['metadata_json']))
                grid_shape = data['grid'].shape
                states.append({
                    'filename': f.name,
                    'size': f.stat().st_size,
                    'shape': grid_shape,
                    'saved_at': meta.get('saved_at', 'unknown'),
                    'avalanches': meta.get('avalanche_count', 0),
                })
            except Exception:
                continue
        return states

    def load_last(self) -> bool:
        """
        Load the most recently saved state file.

        Returns:
            True if load succeeded, False if no saved states exist
        """
        states = self.list_saved()
        if not states:
            print("No saved states found")
            return False

        # Sort by modification time (most recent first)
        states_sorted = sorted(
            states,
            key=lambda s: (self.storage_dir / s['filename']).stat().st_mtime,
            reverse=True
        )
        last = states_sorted[0]
        return self.load(last['filename'])

    def delete(self, filename: str) -> bool:
        """
        Delete a saved state file.

        Args:
            filename: Filename to delete

        Returns:
            True if deleted
        """
        filepath = self.storage_dir / filename
        if filepath.exists():
            filepath.unlink()
            print(f"[OK] Deleted {filename}")
            return True
        return False

    def cleanup_old(self, keep_count: int = 10):
        """
        Keep only the most recent N states.

        Args:
            keep_count: Number of recent files to keep
        """
        files = sorted(
            self.storage_dir.glob("*.npz"),
            key=lambda f: f.stat().st_mtime,
            reverse=True
        )
        for f in files[keep_count:]:
            f.unlink()
        print(f"[OK] Cleaned up, kept {min(keep_count, len(files))} most recent states")
