"""Ground-truth voxelized 3D shape generation inside the capsule.

Every generator returns a float volume in [0, 1] on the capsule grid:
1 = fluid, 0 = empty. Shapes are clipped to the cylinder interior.

Supported shapes: sphere, prolate/oblate ellipsoid, flattened disk,
concave bowl (contact-lens-like) and an asymmetric custom geometry.
Dimensions, position, orientation and grid resolution are all arguments.
"""

from __future__ import annotations

import numpy as np

from .geometry import CapsuleGrid


def _rotation_matrix(euler_xyz: tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix from intrinsic x-y-z Euler angles (radians)."""
    ax, ay, az = euler_xyz
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _local_coords(
    grid: CapsuleGrid,
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Grid coordinates translated to `center` and rotated by `orientation`."""
    X = grid.X - center[0]
    Y = grid.Y - center[1]
    Z = grid.Z - center[2]
    R = _rotation_matrix(orientation)
    # Rotate coordinates into the shape's local frame (inverse rotation).
    Ri = R.T
    Xl = Ri[0, 0] * X + Ri[0, 1] * Y + Ri[0, 2] * Z
    Yl = Ri[1, 0] * X + Ri[1, 1] * Y + Ri[1, 2] * Z
    Zl = Ri[2, 0] * X + Ri[2, 1] * Y + Ri[2, 2] * Z
    return Xl, Yl, Zl


def _finalize(grid: CapsuleGrid, mask: np.ndarray) -> np.ndarray:
    """Clip to the capsule interior and convert to float."""
    vol = np.where(grid.inside & mask, 1.0, 0.0)
    return vol


def sphere(
    grid: CapsuleGrid,
    radius: float,
    center: tuple[float, float, float],
) -> np.ndarray:
    X, Y, Z = _local_coords(grid, center)
    return _finalize(grid, X**2 + Y**2 + Z**2 <= radius**2)


def ellipsoid(
    grid: CapsuleGrid,
    semi_axes: tuple[float, float, float],
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """General ellipsoid. Use a=b<c for prolate, a=b>c for oblate."""
    a, b, c = semi_axes
    X, Y, Z = _local_coords(grid, center, orientation)
    return _finalize(grid, (X / a) ** 2 + (Y / b) ** 2 + (Z / c) ** 2 <= 1.0)


def prolate_ellipsoid(
    grid: CapsuleGrid,
    equatorial_radius: float,
    polar_radius: float,
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    if polar_radius <= equatorial_radius:
        raise ValueError("prolate requires polar_radius > equatorial_radius")
    return ellipsoid(grid, (equatorial_radius, equatorial_radius, polar_radius),
                     center, orientation)


def oblate_ellipsoid(
    grid: CapsuleGrid,
    equatorial_radius: float,
    polar_radius: float,
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    if polar_radius >= equatorial_radius:
        raise ValueError("oblate requires polar_radius < equatorial_radius")
    return ellipsoid(grid, (equatorial_radius, equatorial_radius, polar_radius),
                     center, orientation)


def flattened_disk(
    grid: CapsuleGrid,
    radius: float,
    thickness: float,
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> np.ndarray:
    """Flat cylinder (disk) of given radius and thickness."""
    X, Y, Z = _local_coords(grid, center, orientation)
    mask = (X**2 + Y**2 <= radius**2) & (np.abs(Z) <= thickness / 2)
    return _finalize(grid, mask)


def concave_bowl(
    grid: CapsuleGrid,
    outer_radius: float,
    inner_radius: float,
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    cavity_offset: float | None = None,
) -> np.ndarray:
    """Concave bowl / contact-lens shape.

    A sphere of `outer_radius` with a spherical cavity of `inner_radius`
    carved from above, its center shifted upward by `cavity_offset`
    (default: half the outer radius). Only the lower half of the outer
    sphere is kept, giving a bowl-like meniscus.
    """
    if inner_radius >= outer_radius:
        raise ValueError("inner_radius must be smaller than outer_radius")
    if cavity_offset is None:
        cavity_offset = outer_radius / 2
    X, Y, Z = _local_coords(grid, center, orientation)
    outer = X**2 + Y**2 + Z**2 <= outer_radius**2
    cavity = X**2 + Y**2 + (Z - cavity_offset) ** 2 <= inner_radius**2
    mask = outer & ~cavity & (Z <= outer_radius * 0.6)
    return _finalize(grid, mask)


def asymmetric_blob(
    grid: CapsuleGrid,
    base_radius: float,
    center: tuple[float, float, float],
    orientation: tuple[float, float, float] = (0.0, 0.0, 0.0),
    lobe_scale: float = 0.6,
    seed: int = 0,
) -> np.ndarray:
    """Asymmetric experimental geometry: a sphere plus offset lobes.

    Deterministic for a given seed so ground truth is reproducible.
    """
    rng = np.random.default_rng(seed)
    X, Y, Z = _local_coords(grid, center, orientation)
    mask = X**2 + Y**2 + Z**2 <= base_radius**2
    for _ in range(3):
        offset = rng.uniform(-base_radius, base_radius, size=3)
        r = base_radius * lobe_scale * rng.uniform(0.6, 1.0)
        mask |= (X - offset[0]) ** 2 + (Y - offset[1]) ** 2 + (Z - offset[2]) ** 2 <= r**2
    return _finalize(grid, mask)


def shape_volume(grid: CapsuleGrid, volume: np.ndarray) -> float:
    """Physical volume of the fluid [m^3]."""
    return float(volume.sum()) * grid.voxel_volume


SHAPE_REGISTRY = {
    "sphere": sphere,
    "ellipsoid": ellipsoid,
    "prolate_ellipsoid": prolate_ellipsoid,
    "oblate_ellipsoid": oblate_ellipsoid,
    "flattened_disk": flattened_disk,
    "concave_bowl": concave_bowl,
    "asymmetric_blob": asymmetric_blob,
}
