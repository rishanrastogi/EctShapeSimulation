"""Quantitative accuracy analysis: IoU, volume error, surface distances."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import CapsuleGrid


@dataclass
class AccuracyReport:
    iou: float
    volume_true_m3: float
    volume_recon_m3: float
    volume_error_pct: float
    mean_surface_distance_m: float | None
    hausdorff_distance_m: float | None

    def summary(self) -> str:
        lines = [
            f"IoU:                   {self.iou:.3f}",
            f"Ground-truth volume:   {self.volume_true_m3 * 1e6:.3f} mL",
            f"Reconstructed volume:  {self.volume_recon_m3 * 1e6:.3f} mL",
            f"Volume error:          {self.volume_error_pct:+.1f} %",
        ]
        if self.mean_surface_distance_m is not None:
            lines.append(
                f"Mean surface distance: {self.mean_surface_distance_m * 1e3:.2f} mm"
            )
        if self.hausdorff_distance_m is not None:
            lines.append(
                f"Hausdorff distance:    {self.hausdorff_distance_m * 1e3:.2f} mm"
            )
        return "\n".join(lines)


def iou(truth: np.ndarray, recon: np.ndarray) -> float:
    """Intersection over Union of two binary volumes."""
    t = truth > 0.5
    r = recon > 0.5
    union = np.logical_or(t, r).sum()
    if union == 0:
        return 1.0  # both empty
    return float(np.logical_and(t, r).sum() / union)


def volume_m3(grid: CapsuleGrid, binary: np.ndarray) -> float:
    return float((binary > 0.5).sum()) * grid.voxel_volume


def surface_distances(
    grid: CapsuleGrid, truth: np.ndarray, recon: np.ndarray
) -> tuple[float, float] | tuple[None, None]:
    """Mean symmetric surface distance and Hausdorff distance [m].

    Uses marching-cubes surface vertices; returns (None, None) when either
    volume has no surface (empty or full grid).
    """
    try:
        from skimage import measure
    except ImportError:  # pragma: no cover
        return None, None

    spacing = (grid.dx, grid.dy, grid.dz)

    def verts(vol: np.ndarray) -> np.ndarray | None:
        v = (vol > 0.5).astype(float)
        if v.min() == v.max():
            return None
        out = measure.marching_cubes(v, level=0.5, spacing=spacing)
        return out[0]

    vt = verts(truth)
    vr = verts(recon)
    if vt is None or vr is None:
        return None, None

    from scipy.spatial import cKDTree

    dt = cKDTree(vt).query(vr)[0]
    dr = cKDTree(vr).query(vt)[0]
    mean_sym = float((dt.mean() + dr.mean()) / 2)
    hausdorff = float(max(dt.max(), dr.max()))
    return mean_sym, hausdorff


def evaluate(
    grid: CapsuleGrid, truth: np.ndarray, recon_binary: np.ndarray
) -> AccuracyReport:
    """Full accuracy report for a binarized reconstruction."""
    v_true = volume_m3(grid, truth)
    v_rec = volume_m3(grid, recon_binary)
    err_pct = 100.0 * (v_rec - v_true) / v_true if v_true > 0 else float("inf")
    msd, hd = surface_distances(grid, truth, recon_binary)
    return AccuracyReport(
        iou=iou(truth, recon_binary),
        volume_true_m3=v_true,
        volume_recon_m3=v_rec,
        volume_error_pct=err_pct,
        mean_surface_distance_m=msd,
        hausdorff_distance_m=hd,
    )
