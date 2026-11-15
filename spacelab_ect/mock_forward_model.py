"""Mock ECT forward model: known fluid distribution -> synthetic capacitances.

*** SYNTHETIC DATA ONLY ***
This is an APPROXIMATE synthetic model for validating the reconstruction
pipeline. It is NOT a physically validated ECT solver and its outputs must
never be represented as actual hardware (FDC1004) measurements.

Documented approximations:
  1. LINEARITY: delta-capacitance is linear in the voxel fluid fraction,
     via the heuristic sensitivity matrix (see sensitivity.py). Real ECT
     is nonlinear (soft-field effect).
  2. BASELINE: the empty-capsule capacitance of each pair is a simple
     parallel-plate-inspired estimate scaled by distance — a plausible
     magnitude, not a field solution.
  3. GAIN: the fluid response gain maps the heuristic sensitivity response
     to a capacitance change of a plausible order of magnitude (fF).
  4. NOISE: additive Gaussian noise plus optional drift; real sensor noise
     has structure (quantization, temperature drift, parasitics).

The ForwardModel interface (``measure``) is designed so an FEM-based
implementation (e.g. Gmsh + FEniCSx) can replace this class later.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from .config import ECTConfig
from .geometry import CapsuleGrid, build_electrodes, build_measurement_pairs
from .sensitivity import SensitivityModel

EPS0 = 8.8541878128e-12  # vacuum permittivity [F/m]


class ForwardModel(Protocol):
    """Interface for any forward model (mock now, FEM later)."""

    def measure(self, fluid_volume: np.ndarray | None) -> np.ndarray:
        """Return one noiseless capacitance vector [fF] for the given fluid
        distribution (None = empty capsule)."""
        ...


class MockForwardModel:
    """Approximate synthetic forward model. See module docstring for caveats."""

    def __init__(self, config: ECTConfig):
        self.config = config
        self.grid = CapsuleGrid(config.capsule, config.grid)
        self.electrodes = build_electrodes(config.capsule, config.electrodes)
        self.pairs = build_measurement_pairs(self.electrodes, config.electrodes)
        self.sensitivity = SensitivityModel(self.grid, self.electrodes, self.pairs)
        self._baseline = self._compute_baseline_fF()
        self._gain = self._compute_gain_fF()

    # ------------------------------------------------------------------ #
    def _compute_baseline_fF(self) -> np.ndarray:
        """Empty-capsule baseline per pair: parallel-plate-inspired estimate.

        C ~ eps0 * eps_empty * A / d, scaled to fF. Plausible magnitude only.
        """
        cap = self.config.capsule
        ec = self.config.electrodes
        arc_len = 2 * np.pi * cap.radius / ec.electrodes_per_ring * ec.electrode_arc_fraction
        area = arc_len * ec.electrode_height
        baselines = []
        for i, j in self.pairs:
            d = np.linalg.norm(self.electrodes[i].center - self.electrodes[j].center)
            c = EPS0 * cap.eps_empty * area / max(d, 1e-4)
            baselines.append(c * 1e15)  # F -> fF
        return np.asarray(baselines)

    def _compute_gain_fF(self) -> float:
        """Gain mapping normalized sensitivity response to fF.

        Chosen so a completely full capsule shifts a same-ring adjacent pair
        by roughly 20% of its baseline times the permittivity contrast factor.
        Approximate scaling, not physics.
        """
        cap = self.config.capsule
        contrast = (cap.eps_fluid - cap.eps_empty) / (cap.eps_fluid + cap.eps_empty)
        return float(0.2 * np.median(self._baseline) * contrast)

    # ------------------------------------------------------------------ #
    def measure(self, fluid_volume: np.ndarray | None) -> np.ndarray:
        """Noiseless synthetic capacitances [fF] for a fluid distribution.

        fluid_volume: float array in [0,1] on the capsule grid, or None for
        the empty capsule.
        """
        if fluid_volume is None:
            return self._baseline.copy()
        if fluid_volume.shape != self.grid.shape:
            raise ValueError(
                f"fluid volume shape {fluid_volume.shape} != grid {self.grid.shape}"
            )
        f = np.clip(fluid_volume, 0.0, 1.0)[self.grid.inside]
        response = self.sensitivity.matrix @ f  # normalized rows -> mean fraction weighted
        return self._baseline + self._gain * response

    def measure_with_noise(
        self, fluid_volume: np.ndarray | None, n_repeats: int = 1
    ) -> np.ndarray:
        """Repeated noisy measurements, shape (n_repeats, n_pairs) [fF]."""
        nz = self.config.noise
        rng = np.random.default_rng(nz.seed)
        clean = self.measure(fluid_volume)
        out = np.empty((n_repeats, clean.size))
        for r in range(n_repeats):
            noise = rng.normal(0.0, nz.noise_std_fF, size=clean.size)
            out[r] = clean + noise + nz.drift_fF * r
        return out

    # ------------------------------------------------------------------ #
    def to_dataframe(
        self, measurements: np.ndarray, label: str = "synthetic"
    ) -> pd.DataFrame:
        """Tidy DataFrame of measurements, one row per (repeat, pair)."""
        measurements = np.atleast_2d(measurements)
        rows = []
        for rep in range(measurements.shape[0]):
            for p, (i, j) in enumerate(self.pairs):
                ei, ej = self.electrodes[i], self.electrodes[j]
                rows.append(
                    {
                        "repeat": rep,
                        "pair_index": p,
                        "electrode_a": i,
                        "electrode_b": j,
                        "ring_a": ei.ring,
                        "ring_b": ej.ring,
                        "same_ring": ei.ring == ej.ring,
                        "capacitance_fF": measurements[rep, p],
                        "data_type": f"SYNTHETIC_{label}",
                    }
                )
        return pd.DataFrame(rows)

    def export_csv(
        self, measurements: np.ndarray, path: str | Path, label: str = "synthetic"
    ) -> Path:
        """Export measurements to CSV, clearly labeled as synthetic."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df = self.to_dataframe(measurements, label)
        with open(path, "w") as fh:
            fh.write(
                "# SYNTHETIC ECT DATA — generated by MockForwardModel. "
                "NOT actual FDC1004 hardware measurements.\n"
            )
            df.to_csv(fh, index=False)
        return path
