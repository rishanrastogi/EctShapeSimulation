"""Configuration for the SpaceLab ECT simulation.

All hardware geometry is configurable — final capsule dimensions and
electrode counts are NOT assumed. Change values here or construct the
dataclasses with your own parameters.

Units: meters unless noted otherwise. Capacitance is reported in
femtofarads (fF) to match the FDC1004 measurement range, but the values
come from an approximate synthetic model, not real hardware.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CapsuleConfig:
    """Cylindrical acrylic capsule geometry.

    The capsule axis is the z-axis; z = 0 is the bottom interior face.
    """

    radius: float = 0.025          # interior radius [m] (placeholder, configurable)
    height: float = 0.080          # interior height [m] (placeholder, configurable)
    wall_thickness: float = 0.003  # acrylic wall [m] (used only descriptively)

    # Relative permittivities (dimensionless). Approximate literature values;
    # the 80/20 glycerol/water mixture value is an estimate, not a measurement.
    eps_fluid: float = 47.0        # ~80% glycerol / 20% water at room temp (approx.)
    eps_empty: float = 1.0         # air / vacuum background


@dataclass
class ElectrodeConfig:
    """Electrode array layout: multiple rings at different heights.

    All counts and positions are configurable; none of these defaults are
    final hardware values.
    """

    n_rings: int = 4                       # number of electrode rings
    electrodes_per_ring: int = 8           # electrodes in each ring
    ring_heights: tuple[float, ...] | None = None  # z of ring centers [m]; None = evenly spaced
    electrode_height: float = 0.010        # axial extent of each electrode [m]
    electrode_arc_fraction: float = 0.8    # fraction of the ring arc covered by metal
    include_inter_ring: bool = True        # measure between electrodes on different rings
    max_ring_separation: int | None = None # limit inter-ring pairs (None = all)


@dataclass
class GridConfig:
    """Voxel grid for shapes, sensitivity maps and reconstructions.

    Low resolution by default so everything runs comfortably on a laptop.
    """

    nx: int = 24
    ny: int = 24
    nz: int = 32


@dataclass
class NoiseConfig:
    """Synthetic sensor noise model (approximate, configurable).

    The FDC1004 datasheet quotes noise on the order of a few fF at typical
    output rates; the default here is a rough stand-in, not a device model.
    """

    noise_std_fF: float = 0.25     # additive Gaussian noise std dev [fF]
    drift_fF: float = 0.0          # optional constant drift per repeat [fF]
    seed: int | None = 1234        # RNG seed; None for non-reproducible noise


@dataclass
class ECTConfig:
    """Top-level configuration bundle."""

    capsule: CapsuleConfig = field(default_factory=CapsuleConfig)
    electrodes: ElectrodeConfig = field(default_factory=ElectrodeConfig)
    grid: GridConfig = field(default_factory=GridConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)


DEFAULT_CONFIG = ECTConfig()
