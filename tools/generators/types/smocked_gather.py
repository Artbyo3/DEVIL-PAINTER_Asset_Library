"""Smocked fabric gather waistband/bodice trim alpha mask."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    _ = (palette, variant)
    u, v = common.make_uv_grid(size)
    y = (v - 0.5) * 2.0
    cols = [5, 7, 9, 12][int(rng.integers(0, 4))]
    rows = [5, 7, 9, 12][int(rng.integers(0, 4))]
    p1 = np.sin(u * cols * 2.0 * math.pi + np.sin(v * rows * 2.0 * math.pi) * 1.2)
    p2 = np.sin(v * rows * 2.0 * math.pi)
    smock = np.power(0.5 + 0.5 * p1 * p2, 1.3)
    mask = np.clip((0.88 - np.abs(y)) * 14.0, 0.0, 1.0)
    return common.alpha_image(smock * mask)


SPEC = GeneratorSpec(
    name="smocked-gather",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=3)",
    default_palette="charcoal_flannel",
    description="Gathered smocked fabric cuff/waistband relief alpha band (tiles horizontally).",
    render_fn=generate,
)
