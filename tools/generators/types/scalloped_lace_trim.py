"""Scalloped eyelet lace edge trim alpha generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    _ = palette
    u, v = common.make_uv_grid(size)
    y = (v - 0.5) * 2.0
    waves = [3, 4, 6, 8][(variant + int(rng.integers(0, 4))) % 4]
    scallop_y = 0.32 + 0.18 * np.abs(np.sin(u * waves * math.pi))
    base = np.clip((scallop_y - np.abs(y)) * 22.0, 0.0, 1.0)
    eu = (u * waves) % 1.0 - 0.5
    er = np.sqrt(eu * eu + y * y)
    eyelet_hole = np.clip((er - 0.12) * 30.0, 0.0, 1.0)
    rim = np.exp(-((er - 0.14) ** 2) / 0.002) * 0.4
    return common.alpha_image(base * eyelet_hole + rim * base)


SPEC = GeneratorSpec(
    name="scalloped-lace-trim",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=2)",
    default_palette="bridal_ivory_lace",
    description="Scalloped broderie anglaise eyelet lace edge trim alpha (tiles horizontally).",
    render_fn=generate,
)
