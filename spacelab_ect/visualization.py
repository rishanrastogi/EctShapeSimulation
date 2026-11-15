"""3D visualization with PyVista + scikit-image marching cubes.

Physical coordinates are preserved: meshes live in the same meter-scale
frame as the capsule (z up, origin at the capsule bottom center).

Set show=False (or run with off_screen) to export screenshots and meshes
without opening interactive windows — useful for CI and headless runs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .geometry import CapsuleGrid


def volume_to_mesh(grid: CapsuleGrid, volume: np.ndarray, level: float = 0.5):
    """Marching-cubes surface of a voxel volume as a PyVista PolyData mesh.

    Returns None when the volume has no surface at the given level.
    """
    import pyvista as pv
    from skimage import measure

    v = np.asarray(volume, dtype=float)
    if v.min() >= level or v.max() <= level:
        return None
    # Pad so shapes touching the boundary still close.
    vp = np.pad(v, 1, mode="constant", constant_values=0.0)
    verts, faces, _, _ = measure.marching_cubes(
        vp, level=level, spacing=(grid.dx, grid.dy, grid.dz)
    )
    # Undo padding offset and move into physical coordinates.
    verts -= np.array([grid.dx, grid.dy, grid.dz])
    verts += np.array([grid.x[0], grid.y[0], grid.z[0]])
    faces_pv = np.hstack(
        [np.full((faces.shape[0], 1), 3, dtype=np.int64), faces.astype(np.int64)]
    ).ravel()
    return pv.PolyData(verts, faces_pv)


def capsule_outline(grid: CapsuleGrid):
    """Translucent cylinder for context."""
    import pyvista as pv

    cap = grid.capsule
    return pv.Cylinder(
        center=(0, 0, cap.height / 2),
        direction=(0, 0, 1),
        radius=cap.radius,
        height=cap.height,
        resolution=60,
    )


def export_mesh(mesh, path: str | Path) -> Path:
    """Save a mesh as STL or PLY (chosen by extension)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.save(str(path))
    return path


def side_by_side(
    grid: CapsuleGrid,
    truth: np.ndarray,
    recon: np.ndarray,
    title: str = "Ground truth vs reconstruction",
    metrics_text: str | None = None,
    screenshot: str | Path | None = None,
    show: bool = True,
):
    """Interactive side-by-side comparison of truth and reconstruction.

    Left: ground truth. Right: reconstruction. Both with capsule outline.
    Pass screenshot=path and show=False for headless export.
    """
    import pyvista as pv

    off_screen = not show
    pl = pv.Plotter(shape=(1, 2), off_screen=off_screen, window_size=(1400, 700))

    for col, (vol, name, color) in enumerate(
        [(truth, "Ground truth", "#2a76c6"), (recon, "Reconstruction", "#d9822b")]
    ):
        pl.subplot(0, col)
        pl.add_text(f"{title}\n{name}", font_size=10)
        pl.add_mesh(capsule_outline(grid), opacity=0.12, color="white")
        mesh = volume_to_mesh(grid, vol)
        if mesh is not None:
            pl.add_mesh(mesh, color=color, smooth_shading=True)
        else:
            pl.add_text("(no surface)", position="lower_left", font_size=9)
        pl.show_axes()
    if metrics_text:
        pl.subplot(0, 1)
        pl.add_text(metrics_text, position="lower_right", font_size=9)
    pl.link_views()

    if screenshot is not None:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        pl.show(screenshot=str(screenshot), auto_close=True)
    else:
        pl.show()
    return pl


def voxel_view(
    grid: CapsuleGrid,
    volume: np.ndarray,
    title: str = "Voxel view",
    screenshot: str | Path | None = None,
    show: bool = True,
):
    """Voxel (cell) visualization of a volume, colored by fluid fraction."""
    import pyvista as pv

    off_screen = not show
    img = pv.ImageData(
        dimensions=np.array(volume.shape) + 1,
        spacing=(grid.dx, grid.dy, grid.dz),
        origin=(grid.x[0] - grid.dx / 2, grid.y[0] - grid.dy / 2, grid.z[0] - grid.dz / 2),
    )
    img.cell_data["fluid"] = volume.ravel(order="F")
    thresh = img.threshold(0.25, scalars="fluid")

    pl = pv.Plotter(off_screen=off_screen)
    pl.add_text(title, font_size=11)
    pl.add_mesh(capsule_outline(grid), opacity=0.12, color="white")
    if thresh.n_cells > 0:
        pl.add_mesh(thresh, scalars="fluid", show_edges=True, clim=[0, 1], cmap="viridis")
    pl.show_axes()
    if screenshot is not None:
        Path(screenshot).parent.mkdir(parents=True, exist_ok=True)
        pl.show(screenshot=str(screenshot), auto_close=True)
    else:
        pl.show()
    return pl
