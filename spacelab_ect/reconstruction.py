"""Inverse reconstruction: normalized capacitances -> 3D voxel volume.

Method: regularized linear least squares

    minimize ||A x - g||^2 + alpha^2 ||x||^2 + beta^2 ||L x||^2

where A is the (pairs x voxels) sensitivity matrix, g the normalized
measurement vector, L a discrete 3D Laplacian over the in-capsule voxels
(spatial smoothness regularization), subject to box bounds [0, 1] on x
(physically meaningful fluid fractions). Solved by projected gradient
(Landweber) iteration, which enforces the bounds at every step.

This module is independent of the mock forward model: it only consumes a
sensitivity matrix (from any source, including a future FEM solver) and a
normalized measurement vector.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp

from .geometry import CapsuleGrid


def build_laplacian(grid: CapsuleGrid) -> sp.csr_matrix:
    """Discrete graph Laplacian over voxels inside the capsule (6-neighborhood)."""
    idx_map = -np.ones(grid.shape, dtype=np.int64)
    coords = np.argwhere(grid.inside)
    idx_map[tuple(coords.T)] = np.arange(len(coords))

    rows, cols, vals = [], [], []
    offsets = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    nx, ny, nz = grid.shape
    for di, dj, dk in offsets:
        a = coords
        b = coords + np.array([di, dj, dk])
        ok = (b[:, 0] < nx) & (b[:, 1] < ny) & (b[:, 2] < nz)
        a, b = a[ok], b[ok]
        ib = idx_map[b[:, 0], b[:, 1], b[:, 2]]
        valid = ib >= 0
        ia = idx_map[a[valid, 0], a[valid, 1], a[valid, 2]]
        ib = ib[valid]
        # Edge contributes +1 on diagonal of both ends, -1 off-diagonal.
        rows += [ia, ib, ia, ib]
        cols += [ib, ia, ia, ib]
        vals += [-np.ones(len(ia)), -np.ones(len(ia)), np.ones(len(ia)), np.ones(len(ia))]
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    vals = np.concatenate(vals)
    n = len(coords)
    return sp.csr_matrix((vals, (rows, cols)), shape=(n, n))


@dataclass
class ReconstructionResult:
    volume: np.ndarray          # full 3D volume, values in [lower, upper]
    x: np.ndarray               # raw solution over inside voxels
    residual_norm: float        # ||A x - g|| after projection
    n_iterations: int


class Reconstructor:
    """Regularized least-squares ECT reconstructor with smoothness prior."""

    def __init__(
        self,
        grid: CapsuleGrid,
        sensitivity_matrix: np.ndarray,
        alpha: float = 0.02,
        beta: float = 0.05,
        lower: float = 0.0,
        upper: float = 1.0,
        projected_iterations: int = 500,
    ):
        """
        alpha: Tikhonov (amplitude) regularization weight.
        beta: spatial smoothness (Laplacian) regularization weight.
        lower/upper: physical bounds for the voxel fluid fraction.
        projected_iterations: projected-gradient (Landweber) iterations.
        """
        self.grid = grid
        self.A = np.asarray(sensitivity_matrix, dtype=float)
        if self.A.shape[1] != grid.n_inside:
            raise ValueError(
                f"sensitivity matrix has {self.A.shape[1]} columns, expected "
                f"{grid.n_inside} in-capsule voxels"
            )
        self.alpha = alpha
        self.beta = beta
        self.lower = lower
        self.upper = upper
        self.projected_iterations = projected_iterations
        self.L = build_laplacian(grid)
        self._sigma: float | None = None

    def _spectral_norm(self, n_iter: int = 30) -> float:
        """Largest singular value of A, by power iteration (cached)."""
        if self._sigma is None:
            rng = np.random.default_rng(0)
            v = rng.normal(size=self.A.shape[1])
            v /= np.linalg.norm(v)
            for _ in range(n_iter):
                w = self.A.T @ (self.A @ v)
                nw = np.linalg.norm(w)
                if nw == 0:
                    break
                v = w / nw
            self._sigma = float(np.sqrt(np.linalg.norm(self.A.T @ (self.A @ v))))
        return self._sigma

    def reconstruct(self, g: np.ndarray) -> ReconstructionResult:
        """Reconstruct from a normalized measurement vector g (n_pairs,)."""
        g = np.asarray(g, dtype=float).ravel()
        if g.size != self.A.shape[0]:
            raise ValueError(
                f"measurement vector has {g.size} entries, expected {self.A.shape[0]}"
            )
        n = self.A.shape[1]

        # Projected-gradient (Landweber) iteration from zero, matrix-free.
        # Starting at zero avoids rectifying measurement noise into phantom
        # fluid when clipping an unconstrained least-squares solution.
        sigma = self._spectral_norm()
        lap_norm = float(abs(self.L).sum(axis=1).max())  # Gershgorin bound
        step = 1.0 / (sigma**2 + self.alpha**2 + self.beta**2 * lap_norm + 1e-12)

        x = np.zeros(n)
        it = 0
        for it in range(1, self.projected_iterations + 1):
            grad = (
                self.A.T @ (self.A @ x - g)
                + self.alpha**2 * x
                + self.beta**2 * (self.L @ x)
            )
            x = np.clip(x - step * grad, self.lower, self.upper)

        residual = float(np.linalg.norm(self.A @ x - g))
        volume = self.grid.unflatten_inside(x)
        return ReconstructionResult(volume=volume, x=x, residual_norm=residual,
                                    n_iterations=it)


def threshold_volume(volume: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """Binarize a reconstructed volume.

    Default: volume-preserving threshold — keep the N highest-valued voxels,
    where N is the total fluid mass of the continuous solution (sum of the
    voxel fluid fractions). This counters the spatial smearing of the
    regularized linear reconstruction, which spreads mass over many voxels.
    """
    if threshold is None:
        target = int(round(float(volume.sum())))
        if target <= 0:
            return np.zeros_like(volume)
        flat = np.sort(volume.ravel())[::-1]
        target = min(target, flat.size)
        threshold = max(float(flat[target - 1]), 1e-6)
    return (volume >= threshold).astype(float)
