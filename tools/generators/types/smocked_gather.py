"""Smocked fabric gather waistband/bodice trim alpha mask."""

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
    y = (v - 0.5) * 2.0
    cols = [6, 8, 10, 12][int(rng.integers(0, 4))]
    rows = [6, 8, 10, 12][int(rng.integers(0, 4))]
    # Staggered English honeycomb diamond smocking lattice: two offset sub-lattices
    u1 = (u * cols) % 1.0 - 0.5
    v1 = (v * rows) % 1.0 - 0.5
    u2 = (u * cols + 0.5) % 1.0 - 0.5
    v2 = (v * rows + 0.5) % 1.0 - 0.5
    d_diamond = np.minimum(np.abs(u1) + np.abs(v1), np.abs(u2) + np.abs(v2))
    # Convex pinched lozenge cushion + fine pleat gathers radiating into stitch tacks
    cushion = np.power(np.clip(np.sin(np.clip(d_diamond * 2.0, 0.0, 1.0) * math.pi * 0.5), 0.0, 1.0), 0.75)
    pleat_folds = 0.82 + 0.18 * np.cos(u * cols * 4.0 * math.pi)
    tack_dimple = np.exp(-((np.minimum(u1 * u1 + v1 * v1, u2 * u2 + v2 * v2)) / 0.015)) * 0.32
    smock = np.clip(cushion * pleat_folds - tack_dimple, 0.0, 1.0)
    mask = np.clip((0.88 - np.abs(y)) * 14.0, 0.0, 1.0)
    _h = (smock * mask).astype(np.float32)
    return common.alpha_image(_h), _h


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="smocked-gather",
    kind="alpha",
    tileable=False,
    tile_axes="u",
    origin="p1_work/scripts/custom_generators.py (gen_cloth_alpha_custom kind=3)",
    default_palette="charcoal_flannel",
    description="Gathered smocked fabric cuff/waistband relief alpha band (tiles horizontally).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native pleated smocked fabric gather relief field.",
    normal_strength=6.0,
    height_fn=generate_height,
)
