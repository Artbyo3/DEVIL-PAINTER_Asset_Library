"""Blanket whip-stitch edge seam alpha trim."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    stitches = [8, 12, 16, 22][(variant + int(rng.integers(0, 4))) % 4]
    sv = (v * stitches) % 1.0 - 0.5
    r_th = 0.062

    # Continuous vertical purl spine along x = -0.15
    d_spine = np.abs(x + 0.15) / r_th
    spine = np.sqrt(np.clip(1.0 - d_spine * d_spine, 0.0, 1.0)) * 0.92

    # Smooth capsule SDF for perpendicular blanket whip-stitch bar from (-0.15, 0) to (+0.25, 0)
    tx = np.clip((x - (-0.15)) / 0.40, 0.0, 1.0)
    dx_bar = x - (-0.15 + tx * 0.40)
    dy_bar = sv / float(stitches) * 2.0  # physical scale matching u-width
    # Normalized distance in (u, v) cell units
    r_cell_v = 0.12
    d_bar = np.sqrt((dx_bar / r_th) ** 2 + (sv / r_cell_v) ** 2)
    cross_bar = np.sqrt(np.clip(1.0 - d_bar * d_bar, 0.0, 1.0)) * (0.88 + 0.12 * np.sin(tx * math.pi))

    seam_ridge = np.exp(-((x + 0.15) ** 2) / 0.025) * 0.34
    _h = np.clip(np.maximum(spine, cross_bar) + seam_ridge, 0.0, 1.0).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="blanket-stitch-seam",
    kind="alpha",
    tileable=False,
    tile_axes="v",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=5)",
    default_palette="charcoal_flannel",
    description="Blanket / whip-stitch edge seam trim alpha with vertical spine and perpendicular loops (tiles vertically).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native blanket-stitch thread spine, cross-bar, and seam ridge relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
