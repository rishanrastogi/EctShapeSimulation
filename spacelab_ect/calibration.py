"""Baseline calibration and measurement normalization.

ECT reconstruction works on normalized measurements:

    g = (C_measured - C_empty) / (C_full - C_empty)

so g is ~0 for an empty capsule and ~1 for a full one, per pair. The full
reference here comes from the same synthetic model (a capsule entirely
full of fluid); with real hardware it would come from a calibration fill.
"""

from __future__ import annotations

import numpy as np


class Calibration:
    """Holds empty and full reference vectors and normalizes measurements."""

    def __init__(self, c_empty_fF: np.ndarray, c_full_fF: np.ndarray):
        c_empty_fF = np.asarray(c_empty_fF, dtype=float)
        c_full_fF = np.asarray(c_full_fF, dtype=float)
        if c_empty_fF.shape != c_full_fF.shape:
            raise ValueError("empty and full references must have the same shape")
        span = c_full_fF - c_empty_fF
        if np.any(np.abs(span) < 1e-12):
            raise ValueError("full and empty references are identical for some pair")
        self.c_empty = c_empty_fF
        self.c_full = c_full_fF
        self._span = span

    @classmethod
    def from_model(cls, model) -> "Calibration":
        """Build references from a forward model: empty and fully-filled capsule."""
        c_empty = model.measure(None)
        full = np.ones(model.grid.shape)
        c_full = model.measure(full)
        return cls(c_empty, c_full)

    def normalize(self, c_measured_fF: np.ndarray) -> np.ndarray:
        """Normalized measurement vector(s); accepts (n,) or (repeats, n)."""
        c = np.atleast_2d(np.asarray(c_measured_fF, dtype=float))
        g = (c - self.c_empty) / self._span
        return g[0] if np.asarray(c_measured_fF).ndim == 1 else g

    def average_repeats(self, c_measured_fF: np.ndarray) -> np.ndarray:
        """Mean over repeated measurements, shape (n_pairs,)."""
        return np.atleast_2d(c_measured_fF).mean(axis=0)
