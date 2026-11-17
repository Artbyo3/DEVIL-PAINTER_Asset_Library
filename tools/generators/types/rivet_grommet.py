"""Metal eyelet grommet and leather-pucker rivet alpha stamp."""

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
    _ = (palette, variant)
    u, v = common.make_uv_grid(size)
    x = (u - 0.5) * 2.0
    y = (v - 0.5) * 2.0
    r = np.sqrt(x * x + y * y)
    theta = np.arctan2(y, x)

    outer_r = float(rng.uniform(0.42, 0.74))
    inner_r = outer_r * float(rng.uniform(0.32, 0.56))
    ring = np.clip((outer_r - r) * 26.0, 0.0, 1.0) * np.clip((r - inner_r) * 26.0, 0.0, 1.0)
    bevel = np.sin(np.clip((r - inner_r) / max(outer_r - inner_r, 0.05), 0.0, 1.0) * math.pi)
    prongs = (
        0.25
        * np.clip(np.cos(theta * [6, 8, 12][int(rng.integers(0, 3))]) - 0.5, 0.0, 1.0)
        * np.exp(-((r - inner_r * 1.15) ** 2) / 0.004)
    )
    puckers = (
        0.22
        * np.power(0.5 + 0.5 * np.sin(theta * [12, 16, 20][int(rng.integers(0, 3))]), 2.0)
        * np.exp(-((r - outer_r * 1.18) ** 2) / 0.015)
    )
    _h = (ring * bevel + prongs + puckers).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="rivet-grommet",
    kind="alpha",
    tileable=False,
    tile_axes="none",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=1)",
    default_palette="charcoal_flannel",
    description="Radial metallic grommet eyelet with beveled washer, inner prongs, and outer fabric puckers.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native beveled grommet ring, radial prongs, and fabric pucker relief.",
    normal_strength=6.0,
    height_fn=generate_height,
)
