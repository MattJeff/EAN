"""Advanced monitoring utilities: unified logger + spike/energy visualisation.

Dependencies: matplotlib and imageio (optional).  If missing, methods that rely
on them will raise an informative ImportError.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import csv

from statistics import mean

# -----------------------------------------------------------------------------
# Unified CSV + JSON logger ----------------------------------------------------
# -----------------------------------------------------------------------------
class UnifiedLogger:  # pylint: disable=too-few-public-methods
    """Log network-level metrics to *both* CSV and JSON-Lines.

    Parameters
    ----------
    csv_path : str | Path
        Destination CSV file.  Header is written on creation.
    json_path : str | Path | None, default None
        Destination JSON-Lines file (one JSON per row).  If *None*, JSON output
        is disabled.
    """

    def __init__(self, csv_path: str | Path, *, json_path: str | Path | None = None):  # noqa: D401
        self.csv_path = Path(csv_path)
        self.json_path = Path(json_path) if json_path is not None else None

        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                csv.writer(f).writerow([
                    "tick",
                    "n_assemblies",
                    "mean_energy_global",
                    "reward",
                ])

    # ------------------------------------------------------------------
    def log(self, network: "Any", *, reward: float | None = None) -> None:  # noqa: D401
        """Append a new entry for *network* state."""
        energies: List[float] = [n.energy for asm in network.assemblies.values() for n in asm.neurons]
        payload: Dict[str, Any] = {
            "tick": network.tick,
            "n_assemblies": len(network.assemblies),
            "mean_energy_global": mean(energies) if energies else 0.0,
            "reward": reward,
        }

        # CSV ----------------------------------------------------------------
        with self.csv_path.open("a", newline="") as f:
            csv.writer(f).writerow([
                payload["tick"],
                payload["n_assemblies"],
                payload["mean_energy_global"],
                payload["reward"] if reward is not None else "",
            ])

        # JSON ----------------------------------------------------------------
        if self.json_path is not None:
            with self.json_path.open("a") as f:
                json.dump(payload, f)
                f.write("\n")


# -----------------------------------------------------------------------------
# Spike / energy visualisation -------------------------------------------------
# -----------------------------------------------------------------------------
class SpikeEnergyVisualizer:  # pylint: disable=too-few-public-methods
    """Collect 2-D energy & spike grids and export an animation.

    The visualiser stores frames in memory (as RGB arrays) then writes a GIF or
    MP4 using *imageio*.
    """

    def __init__(self, *, cmap: str = "viridis", spike_color: str = "red") -> None:  # noqa: D401
        self._frames: List["Any"] = []
        self.cmap = cmap
        self.spike_color = spike_color

    # ------------------------------------------------------------------
    def add_frame(self, energy: "Any", spikes: "Any") -> None:  # noqa: D401
        """Register a new frame.

        Parameters
        ----------
        energy : 2-D array-like (H×W)
            Energy per cell.
        spikes : 2-D bool array-like (H×W)
            *True* for cells that spiked on this tick.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        energy_np = np.asarray(energy, dtype=float)
        spikes_np = np.asarray(spikes, dtype=bool)

        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(energy_np, cmap=self.cmap, vmin=0.0, vmax=float(energy_np.max() or 1.0))
        # Overlay spikes
        ys, xs = np.nonzero(spikes_np)
        ax.scatter(xs, ys, c=self.spike_color, s=10)
        ax.axis("off")
        fig.tight_layout(pad=0)

        # Convert to RGB array --------------------------------------------
        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self._frames.append(frame)
        plt.close(fig)

    # ------------------------------------------------------------------
    def save(self, out_path: str | Path, *, fps: int = 5) -> None:  # noqa: D401
        """Write collected frames to *out_path* (GIF/MP4 depending on ext)."""
        if not self._frames:
            raise RuntimeError("No frames recorded; call add_frame() first")

        import imageio.v3 as iio

        ext = Path(out_path).suffix.lower()
        if ext in {".gif", ".mp4"}:
            iio.imwrite(out_path, self._frames, fps=fps)
        else:
            raise ValueError("Unsupported output format; use .gif or .mp4")
