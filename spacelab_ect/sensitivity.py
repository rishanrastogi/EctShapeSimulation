"""Approximate sensitivity matrix for the ECT electrode array.

APPROXIMATION NOTICE
--------------------
This module builds a LINEAR sensitivity model: the change in capacitance
of each electrode pair is assumed proportional to a weighted sum of the
permittivity change in each voxel. The per-voxel weight is a geometric
heuristic (product of distance-weighted "visibility" factors from the two
electrodes), which mimics the qualitative behavior of real ECT dipole
sensitivity maps — strong near the electrodes, weak in the center, and
localized between the pair — but it is NOT derived from Maxwell's
equations. Real ECT sensitivity is nonlinear in the permittivity
distribution ("soft-field" effect) and must ultimately come from an FEM
solver (e.g. FEniCSx). The interface below (`SensitivityModel.matrix`)
is the replacement point for such a solver.
"""

from __future__ import annotations

import numpy as np

from .geometry import CapsuleGrid, Electrode


def _electrode_weight(
    grid: CapsuleGrid, electrode: Electrode, softness: float = 1e-3
) -> np.ndarray:
    """Distance- and aperture-weighted visibility of each voxel from an electrode.

    Heuristic: weight = cos(angular offset within aperture)+ / (d^2 + softness^2),
    with an axial window that falls off outside the electrode's z-extent.
    """
    ex, ey, ez = electrode.center
    dx = grid.X - ex
    dy = grid.Y - ey
    dz = grid.Z - ez
    d2 = dx**2 + dy**2 + dz**2

    # Azimuthal aperture: voxels angularly close to the electrode center count more.
    voxel_theta = np.arctan2(grid.Y, grid.X)
    dtheta = np.angle(np.exp(1j * (voxel_theta - electrode.theta)))
    ang = np.clip(1.0 - np.abs(dtheta) / (np.pi / 2 + electrode.arc_half_angle), 0.0, 1.0)

    # Axial window: smooth falloff away from the electrode strip.
    half_h = (electrode.z_max - electrode.z_min) / 2
    axial = np.exp(-0.5 * (np.maximum(np.abs(grid.Z - ez) - half_h, 0.0) / (2 * half_h + 1e-12)) ** 2)

    w = ang * axial / (d2 + softness**2)
    w[~grid.inside] = 0.0
    return w


class SensitivityModel:
    """Builds and caches the approximate linear sensitivity matrix.

    matrix: shape (n_pairs, n_inside_voxels). Each row is normalized to
    unit sum so every measurement contributes on a comparable scale.
    """

    def __init__(
        self,
        grid: CapsuleGrid,
        electrodes: list[Electrode],
        pairs: list[tuple[int, int]],
    ):
        self.grid = grid
        self.electrodes = electrodes
        self.pairs = pairs
        self._matrix: np.ndarray | None = None

    @property
    def matrix(self) -> np.ndarray:
        if self._matrix is None:
            self._matrix = self._build()
        return self._matrix

    def _build(self) -> np.ndarray:
        weights = [_electrode_weight(self.grid, e) for e in self.electrodes]
        rows = []
        for i, j in self.pairs:
            s = weights[i] * weights[j]
            s_in = s[self.grid.inside]
            total = s_in.sum()
            if total > 0:
                s_in = s_in / total
            rows.append(s_in)
        return np.asarray(rows)

    def save(self, path: str) -> None:
        np.save(path, self.matrix)

    @classmethod
    def load_matrix(cls, path: str) -> np.ndarray:
        """Load a precomputed sensitivity matrix (e.g. from a future FEM solver)."""
        return np.load(path)
