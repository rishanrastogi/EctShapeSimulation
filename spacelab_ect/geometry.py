"""Capsule and electrode geometry.

Builds the voxel grid inside the cylindrical capsule and the electrode
positions for every ring. The capsule axis is the z-axis with z = 0 at the
bottom interior face; x and y span [-radius, +radius].
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import CapsuleConfig, ElectrodeConfig, GridConfig


@dataclass
class Electrode:
    """One electrode on the capsule wall."""

    index: int          # global electrode index
    ring: int           # ring number (0 = bottom)
    pos_in_ring: int    # position within the ring
    center: np.ndarray  # (x, y, z) of the electrode center [m]
    normal: np.ndarray  # inward-pointing unit normal
    z_min: float        # axial extent
    z_max: float
    theta: float        # azimuthal angle of center [rad]
    arc_half_angle: float  # half the azimuthal arc covered [rad]


class CapsuleGrid:
    """Voxel grid clipped to the cylindrical capsule interior."""

    def __init__(self, capsule: CapsuleConfig, grid: GridConfig):
        self.capsule = capsule
        self.grid = grid
        r, h = capsule.radius, capsule.height

        # Voxel centers.
        self.x = np.linspace(-r, r, grid.nx, endpoint=True)
        self.y = np.linspace(-r, r, grid.ny, endpoint=True)
        self.z = np.linspace(0.0, h, grid.nz, endpoint=True)
        self.dx = self.x[1] - self.x[0] if grid.nx > 1 else 2 * r
        self.dy = self.y[1] - self.y[0] if grid.ny > 1 else 2 * r
        self.dz = self.z[1] - self.z[0] if grid.nz > 1 else h
        self.voxel_volume = self.dx * self.dy * self.dz

        self.X, self.Y, self.Z = np.meshgrid(self.x, self.y, self.z, indexing="ij")
        # Mask of voxels inside the cylinder.
        self.inside = (self.X**2 + self.Y**2) <= r**2

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.grid.nx, self.grid.ny, self.grid.nz)

    @property
    def n_inside(self) -> int:
        return int(self.inside.sum())

    def flatten_inside(self, volume: np.ndarray) -> np.ndarray:
        """Extract the values of voxels inside the cylinder as a 1-D vector."""
        return volume[self.inside]

    def unflatten_inside(self, vec: np.ndarray, fill: float = 0.0) -> np.ndarray:
        """Place a 1-D inside-voxel vector back into a full 3-D volume."""
        vol = np.full(self.shape, fill, dtype=float)
        vol[self.inside] = vec
        return vol


def build_electrodes(capsule: CapsuleConfig, cfg: ElectrodeConfig) -> list[Electrode]:
    """Create the electrode list from the configuration."""
    if cfg.ring_heights is not None:
        if len(cfg.ring_heights) != cfg.n_rings:
            raise ValueError(
                f"ring_heights has {len(cfg.ring_heights)} entries but n_rings={cfg.n_rings}"
            )
        heights = np.asarray(cfg.ring_heights, dtype=float)
    else:
        # Evenly spaced, keeping the electrode body inside the capsule.
        margin = cfg.electrode_height / 2
        heights = np.linspace(margin, capsule.height - margin, cfg.n_rings)

    arc_half = np.pi / cfg.electrodes_per_ring * cfg.electrode_arc_fraction
    electrodes: list[Electrode] = []
    idx = 0
    for ring, zc in enumerate(heights):
        for k in range(cfg.electrodes_per_ring):
            theta = 2 * np.pi * k / cfg.electrodes_per_ring
            cx = capsule.radius * np.cos(theta)
            cy = capsule.radius * np.sin(theta)
            electrodes.append(
                Electrode(
                    index=idx,
                    ring=ring,
                    pos_in_ring=k,
                    center=np.array([cx, cy, zc]),
                    normal=-np.array([np.cos(theta), np.sin(theta), 0.0]),
                    z_min=zc - cfg.electrode_height / 2,
                    z_max=zc + cfg.electrode_height / 2,
                    theta=theta,
                    arc_half_angle=arc_half,
                )
            )
            idx += 1
    return electrodes


def build_measurement_pairs(
    electrodes: list[Electrode], cfg: ElectrodeConfig
) -> list[tuple[int, int]]:
    """Unique (source, detector) electrode pairs to measure.

    Includes all pairs within a ring, and optionally pairs across rings
    (limited to max_ring_separation rings apart when set).
    """
    pairs: list[tuple[int, int]] = []
    n = len(electrodes)
    for i in range(n):
        for j in range(i + 1, n):
            dr = abs(electrodes[i].ring - electrodes[j].ring)
            if dr == 0:
                pairs.append((i, j))
            elif cfg.include_inter_ring:
                if cfg.max_ring_separation is None or dr <= cfg.max_ring_separation:
                    pairs.append((i, j))
    return pairs
