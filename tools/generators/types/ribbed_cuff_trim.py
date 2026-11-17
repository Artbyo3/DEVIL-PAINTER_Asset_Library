"""Elastic ribbed sweater cuff and collar trim alpha band."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    _ = (np.random.default_rng(seed), palette)
    u, v = common.make_uv_grid(size)
    ribs = 20 + (variant % 5) * 6
    rib_wave = np.power(0.5 + 0.5 * np.sin(u * ribs * 2.0 * math.pi), 0.6)
    knit_v = 0.85 + 0.15 * np.sin(v * ribs * 3.0 * math.pi)
    band_mask = np.clip((0.82 - np.abs(v - 0.5) * 2.0) * 16.0, 0.0, 1.0)
    _h = (rib_wave * knit_v * band_mask).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="ribbed-cuff-trim",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (Suite 4 style=4)",
    default_palette="charcoal_flannel",
    description="Elastic ribbed sweater cuff and collar knit trim band alpha (tiles horizontally).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native vertical rib-knit wave and stitch relief band.",
    normal_strength=6.0,
    height_fn=generate_height,
)
