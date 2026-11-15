"""Automated tests: shape generation, measurement generation, reconstruction,
mesh extraction and analysis."""

from __future__ import annotations

import numpy as np
import pytest

from spacelab_ect import analysis, shape_generator, visualization
from spacelab_ect.calibration import Calibration
from spacelab_ect.config import ECTConfig, GridConfig, NoiseConfig
from spacelab_ect.geometry import CapsuleGrid, build_electrodes, build_measurement_pairs
from spacelab_ect.mock_forward_model import MockForwardModel
from spacelab_ect.reconstruction import Reconstructor, build_laplacian, threshold_volume


@pytest.fixture(scope="module")
def config() -> ECTConfig:
    cfg = ECTConfig()
    cfg.grid = GridConfig(nx=16, ny=16, nz=20)  # small & fast for tests
    cfg.noise = NoiseConfig(noise_std_fF=0.3, seed=42)
    return cfg


@pytest.fixture(scope="module")
def model(config) -> MockForwardModel:
    return MockForwardModel(config)


@pytest.fixture(scope="module")
def grid(model) -> CapsuleGrid:
    return model.grid


# --------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------- #
class TestGeometry:
    def test_grid_clipped_to_cylinder(self, grid):
        assert 0 < grid.n_inside < np.prod(grid.shape)
        # Corners of the bounding box are outside the cylinder.
        assert not grid.inside[0, 0, 0]

    def test_electrode_count(self, config):
        els = build_electrodes(config.capsule, config.electrodes)
        assert len(els) == config.electrodes.n_rings * config.electrodes.electrodes_per_ring

    def test_electrodes_on_wall(self, config):
        for e in build_electrodes(config.capsule, config.electrodes):
            assert np.hypot(e.center[0], e.center[1]) == pytest.approx(config.capsule.radius)
            assert 0 <= e.center[2] <= config.capsule.height

    def test_pairs_unique_and_complete(self, config):
        els = build_electrodes(config.capsule, config.electrodes)
        pairs = build_measurement_pairs(els, config.electrodes)
        assert len(set(pairs)) == len(pairs)
        n = len(els)
        assert len(pairs) == n * (n - 1) // 2  # inter-ring enabled, no separation cap

    def test_inter_ring_disabled(self, config):
        from dataclasses import replace

        ec = replace(config.electrodes, include_inter_ring=False)
        els = build_electrodes(config.capsule, ec)
        pairs = build_measurement_pairs(els, ec)
        m = ec.electrodes_per_ring
        assert len(pairs) == ec.n_rings * m * (m - 1) // 2


# --------------------------------------------------------------------- #
# Shape generation
# --------------------------------------------------------------------- #
class TestShapes:
    def test_sphere_volume(self, grid):
        r = 0.5 * grid.capsule.radius
        vol = shape_generator.sphere(grid, r, (0, 0, grid.capsule.height / 2))
        expected = 4 / 3 * np.pi * r**3
        got = shape_generator.shape_volume(grid, vol)
        assert got == pytest.approx(expected, rel=0.35)  # coarse voxel grid

    def test_sphere_inside_capsule(self, grid):
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        assert np.all(vol[~grid.inside] == 0)

    def test_prolate_oblate_validation(self, grid):
        c = (0, 0, grid.capsule.height / 2)
        with pytest.raises(ValueError):
            shape_generator.prolate_ellipsoid(grid, 0.01, 0.005, c)
        with pytest.raises(ValueError):
            shape_generator.oblate_ellipsoid(grid, 0.005, 0.01, c)

    def test_oblate_flatter_than_sphere(self, grid):
        c = (0, 0, grid.capsule.height / 2)
        obl = shape_generator.oblate_ellipsoid(grid, 0.7 * grid.capsule.radius,
                                               0.3 * grid.capsule.radius, c)
        z_extent = np.ptp(np.argwhere(obl > 0)[:, 2])
        xy_extent = np.ptp(np.argwhere(obl > 0)[:, 0])
        assert z_extent < xy_extent

    def test_bowl_concave(self, grid):
        c = (0, 0, grid.capsule.height / 2)
        bowl = shape_generator.concave_bowl(grid, 0.65 * grid.capsule.radius,
                                            0.5 * grid.capsule.radius, c)
        assert bowl.sum() > 0
        # Center column above the bowl midpoint should be hollow near the top of the shape.
        ci, cj = grid.grid.nx // 2, grid.grid.ny // 2
        col = bowl[ci, cj, :]
        filled = np.where(col > 0)[0]
        assert filled.size > 0
        sphere_full = shape_generator.sphere(grid, 0.65 * grid.capsule.radius, c)
        assert bowl.sum() < sphere_full.sum()  # material removed

    def test_asymmetric_reproducible(self, grid):
        c = (0, 0, grid.capsule.height / 2)
        a = shape_generator.asymmetric_blob(grid, 0.4 * grid.capsule.radius, c, seed=7)
        b = shape_generator.asymmetric_blob(grid, 0.4 * grid.capsule.radius, c, seed=7)
        assert np.array_equal(a, b)

    def test_orientation_rotates_shape(self, grid):
        c = (0, 0, grid.capsule.height / 2)
        up = shape_generator.ellipsoid(grid, (0.004, 0.004, 0.012), c)
        rot = shape_generator.ellipsoid(grid, (0.004, 0.004, 0.012), c,
                                        orientation=(np.pi / 2, 0, 0))
        assert not np.array_equal(up, rot)


# --------------------------------------------------------------------- #
# Forward model
# --------------------------------------------------------------------- #
class TestForwardModel:
    def test_baseline_positive(self, model):
        base = model.measure(None)
        assert base.shape == (len(model.pairs),)
        assert np.all(base > 0)

    def test_fluid_increases_capacitance(self, model, grid):
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        assert np.all(model.measure(vol) >= model.measure(None))
        assert model.measure(vol).sum() > model.measure(None).sum()

    def test_spatial_response(self, model, grid):
        """Measurements respond to WHERE the fluid is, not just how much."""
        h = grid.capsule.height
        r = 0.35 * grid.capsule.radius
        low = shape_generator.sphere(grid, r, (0, 0, 0.25 * h))
        high = shape_generator.sphere(grid, r, (0, 0, 0.75 * h))
        d_low = model.measure(low) - model.measure(None)
        d_high = model.measure(high) - model.measure(None)
        # Bottom-ring pairs must respond more to the low sphere than the high one.
        bottom_pairs = [
            p for p, (i, j) in enumerate(model.pairs)
            if model.electrodes[i].ring == 0 and model.electrodes[j].ring == 0
        ]
        assert d_low[bottom_pairs].sum() > d_high[bottom_pairs].sum()

    def test_noise_statistics(self, model, grid):
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        reps = model.measure_with_noise(vol, n_repeats=50)
        assert reps.shape == (50, len(model.pairs))
        resid = reps - model.measure(vol)
        assert abs(resid.std() - model.config.noise.noise_std_fF) < 0.1

    def test_csv_export(self, model, grid, tmp_path):
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        reps = model.measure_with_noise(vol, n_repeats=2)
        path = model.export_csv(reps, tmp_path / "m.csv", label="test")
        text = path.read_text()
        assert "SYNTHETIC" in text
        import pandas as pd

        df = pd.read_csv(path, comment="#")
        assert len(df) == 2 * len(model.pairs)
        assert {"capacitance_fF", "electrode_a", "electrode_b"} <= set(df.columns)


# --------------------------------------------------------------------- #
# Calibration
# --------------------------------------------------------------------- #
class TestCalibration:
    def test_normalization_range(self, model):
        calib = Calibration.from_model(model)
        g_empty = calib.normalize(model.measure(None))
        g_full = calib.normalize(model.measure(np.ones(model.grid.shape)))
        assert np.allclose(g_empty, 0, atol=1e-9)
        assert np.allclose(g_full, 1, atol=1e-9)


# --------------------------------------------------------------------- #
# Reconstruction
# --------------------------------------------------------------------- #
class TestReconstruction:
    def test_laplacian_rowsum_zero(self, grid):
        L = build_laplacian(grid)
        assert np.allclose(np.asarray(L.sum(axis=1)).ravel(), 0)

    def test_sphere_reconstruction_quality(self, model, grid):
        calib = Calibration.from_model(model)
        truth = shape_generator.sphere(grid, 0.55 * grid.capsule.radius,
                                       (0, 0, grid.capsule.height / 2))
        noisy = model.measure_with_noise(truth, n_repeats=5)
        g = calib.normalize(calib.average_repeats(noisy))
        recon = Reconstructor(grid, model.sensitivity.matrix)
        result = recon.reconstruct(g)
        assert np.all(result.volume >= 0) and np.all(result.volume <= 1)
        binary = threshold_volume(result.volume)
        report = analysis.evaluate(grid, truth, binary)
        # Coarse linear reconstruction: expect meaningful but imperfect overlap.
        assert report.iou > 0.25
        assert abs(report.volume_error_pct) < 100

    def test_empty_measurement_reconstructs_empty(self, model, grid):
        calib = Calibration.from_model(model)
        g = calib.normalize(model.measure(None))
        recon = Reconstructor(grid, model.sensitivity.matrix)
        result = recon.reconstruct(g)
        assert result.volume.max() < 0.2

    def test_bad_measurement_size_raises(self, model, grid):
        recon = Reconstructor(grid, model.sensitivity.matrix)
        with pytest.raises(ValueError):
            recon.reconstruct(np.zeros(3))


# --------------------------------------------------------------------- #
# Mesh extraction & analysis
# --------------------------------------------------------------------- #
class TestMeshAndAnalysis:
    def test_mesh_extraction_and_export(self, grid, tmp_path):
        pytest.importorskip("pyvista")
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        mesh = visualization.volume_to_mesh(grid, vol)
        assert mesh is not None and mesh.n_points > 0
        # Mesh bounds in physical coordinates, inside the capsule bounding box.
        xmin, xmax, ymin, ymax, zmin, zmax = mesh.bounds
        r, h = grid.capsule.radius, grid.capsule.height
        assert xmin >= -r - grid.dx and xmax <= r + grid.dx
        assert zmin >= -grid.dz and zmax <= h + grid.dz
        out = visualization.export_mesh(mesh, tmp_path / "s.stl")
        assert out.exists() and out.stat().st_size > 0

    def test_empty_volume_no_mesh(self, grid):
        pytest.importorskip("pyvista")
        assert visualization.volume_to_mesh(grid, np.zeros(grid.shape)) is None

    def test_iou_identity(self, grid):
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        assert analysis.iou(vol, vol) == 1.0
        assert analysis.iou(vol, np.zeros_like(vol)) == 0.0

    def test_evaluate_report(self, grid):
        vol = shape_generator.sphere(grid, 0.5 * grid.capsule.radius,
                                     (0, 0, grid.capsule.height / 2))
        report = analysis.evaluate(grid, vol, vol)
        assert report.iou == 1.0
        assert report.volume_error_pct == pytest.approx(0.0)
        assert report.mean_surface_distance_m == pytest.approx(0.0, abs=1e-9)
        assert "IoU" in report.summary()
