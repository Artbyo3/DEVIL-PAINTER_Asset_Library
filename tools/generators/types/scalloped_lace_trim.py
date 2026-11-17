"""Scalloped eyelet lace edge trim alpha generator."""

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
    y = (v - 0.5) * 2.0
    waves = [3, 4, 6, 8][(variant + int(rng.integers(0, 4))) % 4]
    scallop_y = 0.32 + 0.18 * np.abs(np.sin(u * waves * math.pi))
    base = np.clip((scallop_y - np.abs(y)) * 22.0, 0.0, 1.0)
    eu = (u * waves) % 1.0 - 0.5
    er = np.sqrt(eu * eu + y * y)
    etheta = np.arctan2(y, eu)
    eyelet_hole = np.clip((er - 0.12) * 30.0, 0.0, 1.0)
    # Broderie anglaise radial satin-stitch rim around each eyelet plus diamond bobbinet tulle
    satin_rib = 0.78 + 0.22 * np.cos(etheta * 18.0)
    rim = np.exp(-((er - 0.145) ** 2) / 0.002) * 0.45 * satin_rib
    d1 = np.abs(((u + v) * waves * 4.0) % 1.0 - 0.5) * 2.0
    d2 = np.abs(((u - v) * waves * 4.0) % 1.0 - 0.5) * 2.0
    tulle = np.clip(0.35 + 0.45 * np.maximum(np.exp(-(d1 * d1) / 0.06), np.exp(-(d2 * d2) / 0.06)), 0.0, 1.0)
    _h = (base * eyelet_hole * tulle + rim * base).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="scalloped-lace-trim",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=2)",
    default_palette="bridal_ivory_lace",
    description="Scalloped broderie anglaise eyelet lace edge trim alpha (tiles horizontally).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native scalloped edge profile, eyelet cutouts, and raised embroidery rim.",
    normal_strength=6.0,
    height_fn=generate_height,
)
