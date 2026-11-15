# SpaceLab 3D ECT Simulation & Reconstruction

Synthetic Electrical Capacitance Tomography (ECT) pipeline for the SpaceLab
acoustic fluid-shaping experiment: generate synthetic capacitance
measurements from known 3D fluid shapes, reconstruct the shapes from those
measurements, and quantify reconstruction accuracy — so the reconstruction
pipeline is validated **before** real experimental data arrives.

> ## ⚠️ Scientific integrity notice
>
> **Every measurement this software produces is SYNTHETIC.** The forward
> model is an *approximate, heuristic* model built for pipeline validation.
> It is **not** a physically validated ECT solver and its outputs must never
> be presented as actual FDC1004 hardware measurements. See
> [Approximations](#documented-approximations) below.

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Tested with Python 3.14; any recent Python 3.11+ should work.

## Quick start

Run the full demonstration (sphere, oblate ellipsoid, concave bowl):

```bash
python -m spacelab_ect.main          # headless: writes screenshots to models/
python -m spacelab_ect.main --show   # interactive PyVista 3D windows
```

For each shape this will:

1. generate the ground-truth voxel shape,
2. generate synthetic capacitance measurements (mock forward model),
3. add configurable sensor noise over repeated measurements,
4. reconstruct the fluid distribution (regularized least squares),
5. extract a marching-cubes mesh and export STL files to `models/`,
6. render a side-by-side ground-truth vs reconstruction comparison,
7. print IoU, volumes, volume error and surface-distance metrics,
8. export the synthetic measurements as CSV to `data/`.

Typical output with the default configuration:

```
sphere               IoU=0.76  volume error=-3.0%
oblate_ellipsoid     IoU=0.88  volume error=+2.1%
concave_bowl         IoU=0.59  volume error=+0.9%
```

## HTML viewer

An interactive browser viewer shows the 3D model (ground truth vs
reconstruction, rotatable) on the left and the synthetic capacitance data
for every electrode pair on the right.

```bash
python tools/export_viewer_data.py                    # regenerate viewer/data.js
python -m http.server 8742 --directory viewer         # then open http://localhost:8742
```

## Tests

```bash
python -m pytest tests/
```

Covers shape generation, electrode geometry, measurement generation and
noise statistics, calibration, reconstruction quality, mesh extraction and
accuracy metrics (26 tests).

## Configuration

Everything hardware-related is configurable in
[`spacelab_ect/config.py`](spacelab_ect/config.py) — **no final capsule
dimensions or electrode counts are assumed**:

| Dataclass | What it controls |
|---|---|
| `CapsuleConfig` | capsule radius/height, wall thickness, fluid & empty permittivity |
| `ElectrodeConfig` | number of rings, electrodes per ring, ring heights, electrode size, inter-ring measurements on/off |
| `GridConfig` | voxel grid resolution (default 24×24×32, laptop-friendly) |
| `NoiseConfig` | Gaussian noise σ (fF), drift per repeat, RNG seed |

Example:

```python
from spacelab_ect.config import ECTConfig, CapsuleConfig, ElectrodeConfig
from spacelab_ect.mock_forward_model import MockForwardModel

cfg = ECTConfig()
cfg.capsule = CapsuleConfig(radius=0.030, height=0.100, eps_fluid=47.0)
cfg.electrodes = ElectrodeConfig(n_rings=6, electrodes_per_ring=12)
model = MockForwardModel(cfg)
```

## Project structure

```
spacelab_ect/
  config.py             # all configurable hardware/sim parameters
  geometry.py           # capsule voxel grid, electrode layout, measurement pairs
  shape_generator.py    # voxelized ground-truth shapes
  mock_forward_model.py # synthetic capacitance generation (+ CSV export)
  calibration.py        # empty/full baseline normalization
  sensitivity.py        # approximate linear sensitivity matrix
  reconstruction.py     # regularized least-squares inverse solver
  visualization.py      # PyVista 3D views, marching cubes, STL/PLY export
  analysis.py           # IoU, volume error, surface distance metrics
  main.py               # end-to-end demonstration
data/                   # synthetic CSV measurement files
models/                 # STL meshes and comparison screenshots
tests/                  # automated test suite
```

## Shapes

`shape_generator` provides: `sphere`, `ellipsoid`, `prolate_ellipsoid`,
`oblate_ellipsoid`, `flattened_disk`, `concave_bowl` (contact-lens-like) and
`asymmetric_blob`. All take dimensions, position `center=(x, y, z)`,
`orientation=(rx, ry, rz)` Euler angles, and operate on any grid resolution.
Each returns a voxel volume (1 = fluid) usable as ground truth for accuracy
comparison via `analysis.evaluate`.

## How the pipeline fits together

```
shape_generator ──► ground-truth voxel volume ──────────────┐
                                                            │
MockForwardModel.measure_with_noise ──► synthetic C [fF]    │
                 │                                          │
Calibration.normalize (vs empty/full baseline) ──► g        │
                 │                                          │
Reconstructor.reconstruct(g) ──► voxel volume ──► threshold │
                 │                                          ▼
visualization (meshes, side-by-side)  ◄──  analysis.evaluate (IoU, volume, surfaces)
```

The reconstruction engine is **fully decoupled** from the mock forward
model: it consumes only a sensitivity matrix and a normalized measurement
vector. `SensitivityModel.load_matrix()` can load a matrix produced by any
other source.

## Documented approximations

The mock forward model (`mock_forward_model.py`) and sensitivity model
(`sensitivity.py`) make these explicit approximations:

1. **Linearity.** Capacitance change is modeled as linear in the voxel
   fluid fraction. Real ECT is nonlinear in the permittivity distribution
   (the "soft-field" effect). A simple linear sensitivity model is **not
   physically exact**.
2. **Heuristic sensitivity maps.** Per-voxel weights are a geometric
   heuristic (distance/aperture-weighted visibility products), qualitatively
   similar to real ECT sensitivity (strong near electrodes, weak in the
   center) but **not derived from Maxwell's equations**.
3. **Baseline estimate.** Empty-capsule capacitances use a
   parallel-plate-inspired scaling — plausible magnitudes (tens of fF,
   FDC1004 range), not field solutions. Acrylic wall effects, fringing
   fields, shields and parasitics are not modeled.
4. **Noise model.** Additive Gaussian noise + optional linear drift. Real
   FDC1004 noise includes quantization, temperature drift and parasitics.
5. **Permittivity.** `eps_fluid = 47` for 80/20 glycerol/water is a
   literature-based estimate, not a measured value.

Every exported CSV carries a `SYNTHETIC` label in both a header comment and
a `data_type` column.

## Path to real physics and real data

- **FEM forward model:** the `ForwardModel` protocol
  (`mock_forward_model.py`) and the `SensitivityModel.matrix` interface are
  the two replacement points for a Gmsh + FEniCSx implementation. Nothing
  in `reconstruction.py`, `analysis.py` or `visualization.py` depends on
  the mock model.
- **Real measurements:** once FDC1004 data exists, build a `Calibration`
  from measured empty/full references and feed normalized vectors straight
  into `Reconstructor.reconstruct`.

## Reconstruction method

Regularized least squares, `min ‖Ax − g‖² + α²‖x‖² + β²‖Lx‖²` subject to
`0 ≤ x ≤ 1`, where `L` is a 3D graph Laplacian over in-capsule voxels
(spatial smoothness). Solved by projected gradient (Landweber) iteration,
which keeps every iterate within physical bounds. Binarization uses a
volume-preserving threshold (keeps the N strongest voxels, N = total
reconstructed fluid mass) to counter the spatial smearing inherent to
linear, regularized ECT reconstruction.
