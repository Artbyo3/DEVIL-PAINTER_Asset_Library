"""Interlocking V-loop jersey knitwear texture generator."""

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
    u, v = common.make_uv_grid(size)
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    cols = 16 + (variant % 5) * 4
    rows = (cols * 3) // 2  # multiple of 4 for seamless row stripes
    gx, gy = u * cols, v * rows
    fx = gx - np.floor(gx)
    v_shift = 0.55 * (1.0 - np.abs(2.0 * fx - 1.0))
    gy_shifted = (gy + v_shift) % rows
    fy = gy_shifted - np.floor(gy_shifted)

    dist_edge = np.minimum(fx, 1.0 - fx)
    loop_tube = np.sin(np.clip(dist_edge * 2.2, 0.0, 1.0) * math.pi) * np.sin(fy * math.pi)
    ply_twist = 0.85 + 0.15 * np.sin((fx * 4.0 + fy * 3.0) * 2.0 * math.pi)

    h = common.normalize_field(loop_tube * ply_twist + 0.10 * fuzz)
    row_stripe = ((np.floor(gy).astype(np.int32) // 2) % 2) * 0.25
    c = common.normalize_field(0.65 * h + row_stripe + 0.10 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.14)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.8, sheen_amt=16.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="jersey-knit",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=1)",
    default_palette="heather_grey_jersey",
    description="Interlocking V-loop jersey knitwear for t-shirts, hoodies, and sweaters.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=5.8,
    height_fn=generate_height,
)
