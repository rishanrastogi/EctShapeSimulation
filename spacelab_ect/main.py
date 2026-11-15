"""End-to-end SpaceLab ECT demonstration.

For each demo shape (sphere, oblate ellipsoid, concave bowl):
  1. Generate the ground-truth voxel shape.
  2. Generate SYNTHETIC capacitance measurements with the mock forward model.
  3. Add configurable measurement noise (repeats averaged).
  4. Reconstruct the shape with regularized least squares.
  5. Extract a 3D mesh (marching cubes) and export STL.
  6. Compare reconstruction vs ground truth (IoU, volume error, surface distance).
  7. Export the synthetic measurements to CSV.

Run:  python -m spacelab_ect.main [--show]
  --show  opens interactive PyVista windows (default: headless screenshots)

All measurements produced here are SYNTHETIC — see README for the list of
modeling approximations.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import analysis, shape_generator, visualization
from .calibration import Calibration
from .config import DEFAULT_CONFIG, ECTConfig
from .mock_forward_model import MockForwardModel
from .reconstruction import Reconstructor, threshold_volume

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"


def demo_shapes(model: MockForwardModel) -> dict[str, np.ndarray]:
    """The three demonstration ground-truth shapes."""
    g = model.grid
    cap = model.config.capsule
    cz = cap.height / 2
    r = cap.radius
    return {
        "sphere": shape_generator.sphere(g, radius=0.55 * r, center=(0, 0, cz)),
        "oblate_ellipsoid": shape_generator.oblate_ellipsoid(
            g, equatorial_radius=0.7 * r, polar_radius=0.35 * r, center=(0, 0, cz)
        ),
        "concave_bowl": shape_generator.concave_bowl(
            g, outer_radius=0.65 * r, inner_radius=0.5 * r, center=(0, 0, cz)
        ),
    }


def run_case(
    name: str,
    truth: np.ndarray,
    model: MockForwardModel,
    calib: Calibration,
    recon: Reconstructor,
    n_repeats: int = 5,
    show: bool = False,
) -> analysis.AccuracyReport:
    print(f"\n=== {name} ===")

    # Synthetic measurements with noise, repeated, then averaged.
    noisy = model.measure_with_noise(truth, n_repeats=n_repeats)
    model.export_csv(noisy, DATA_DIR / f"{name}_synthetic_measurements.csv", label=name)
    c_avg = calib.average_repeats(noisy)
    g_norm = calib.normalize(c_avg)

    # Reconstruct and binarize.
    result = recon.reconstruct(g_norm)
    binary = threshold_volume(result.volume)

    # Accuracy.
    report = analysis.evaluate(model.grid, truth, binary)
    print(report.summary())

    # Meshes.
    for vol, tag in [(truth, "truth"), (binary, "recon")]:
        mesh = visualization.volume_to_mesh(model.grid, vol)
        if mesh is not None:
            visualization.export_mesh(mesh, MODELS_DIR / f"{name}_{tag}.stl")

    # Side-by-side visualization (interactive with --show, else screenshot).
    visualization.side_by_side(
        model.grid,
        truth,
        binary,
        title=f"{name} (synthetic demo)",
        metrics_text=report.summary(),
        screenshot=None if show else MODELS_DIR / f"{name}_comparison.png",
        show=show,
    )
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--show", action="store_true", help="interactive 3D windows")
    parser.add_argument("--repeats", type=int, default=5, help="noisy repeats per shape")
    args = parser.parse_args(argv)

    config: ECTConfig = DEFAULT_CONFIG
    print("SpaceLab ECT synthetic pipeline demo")
    print("NOTE: all measurements below are SYNTHETIC (mock forward model).")
    print(
        f"Capsule r={config.capsule.radius*1e3:.0f} mm, h={config.capsule.height*1e3:.0f} mm | "
        f"{config.electrodes.n_rings} rings x {config.electrodes.electrodes_per_ring} electrodes | "
        f"grid {config.grid.nx}x{config.grid.ny}x{config.grid.nz} | "
        f"noise sigma={config.noise.noise_std_fF} fF"
    )

    model = MockForwardModel(config)
    print(f"{len(model.pairs)} electrode pairs, {model.grid.n_inside} voxels in capsule")

    calib = Calibration.from_model(model)
    model.export_csv(model.measure(None), DATA_DIR / "empty_baseline_synthetic.csv",
                     label="empty_baseline")

    recon = Reconstructor(model.grid, model.sensitivity.matrix)

    reports = {}
    for name, truth in demo_shapes(model).items():
        reports[name] = run_case(name, truth, model, calib, recon,
                                 n_repeats=args.repeats, show=args.show)

    print("\n=== Summary ===")
    for name, rep in reports.items():
        print(f"{name:20s} IoU={rep.iou:.3f}  volume error={rep.volume_error_pct:+.1f}%")
    print(f"\nCSV data in {DATA_DIR}\nMeshes/screenshots in {MODELS_DIR}")


if __name__ == "__main__":
    main()
