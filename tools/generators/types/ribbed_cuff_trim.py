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
    ribs = 12 + (variant % 5) * 4
    courses = int(round(ribs * 1.4))
    gx, gy = u * ribs, v * courses
    fx = gx - np.floor(gx)
    ix = np.floor(gx).astype(np.int32)
    iy_base = np.floor(gy).astype(np.int32)
    is_knit = (ix % 2) == 0

    r_yarn = 0.22
    knit_h = np.zeros_like(u, dtype=np.float32)
    for dy in (-1, 0, 1):
        fy = gy - (iy_base + dy)
        t = np.clip((fy + 0.18) / 1.30, 0.0, 1.0)
        x_l = 0.18 + 0.145 * np.sin(t * math.pi * 0.85) + 0.04 * (1.0 - np.cos(t * math.pi))
        x_r = 1.0 - x_l
        z_leg = 0.55 + 0.45 * np.sin(t * math.pi)
        in_span = ((fy >= -0.18) & (fy <= 1.12)).astype(np.float32)
        pl = np.sqrt(np.clip(1.0 - ((fx - x_l) / r_yarn) ** 2, 0.0, 1.0)) * in_span
        pr = np.sqrt(np.clip(1.0 - ((fx - x_r) / r_yarn) ** 2, 0.0, 1.0)) * in_span
        knit_h = np.maximum(knit_h, np.maximum(pl, pr) * z_leg)

    fy_p = gy - iy_base
    purl_h = np.sqrt(np.clip(1.0 - ((fy_p - 0.5 - 0.16 * np.cos((fx - 0.5) * math.pi)) * 2.5) ** 2, 0.0, 1.0)) * 0.42
    cuff_relief = np.where(is_knit, 0.35 + 0.65 * knit_h, 0.18 + purl_h)
    band_mask = np.clip((0.82 - np.abs(v - 0.5) * 2.0) * 16.0, 0.0, 1.0)
    _h = (cuff_relief * band_mask).astype(np.float32)
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
