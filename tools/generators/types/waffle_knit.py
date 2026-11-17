"""Waffle thermal knit and pique polo mesh texture generator."""

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
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    cells = 10 + (variant % 5) * 4
    gu = (u * cells) % 1.0
    gv = (v * cells) % 1.0
    pocket = np.maximum(np.abs(gu - 0.5), np.abs(gv - 0.5)) * 2.0
    ridge = np.power(pocket, 1.4)
    knit_micro = 0.5 + 0.5 * np.sin(u * cells * 6.0 * math.pi) * np.sin(v * cells * 6.0 * math.pi)

    h = common.normalize_field(0.78 * ridge + 0.22 * knit_micro)
    c = common.normalize_field(0.82 * h + 0.18 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, slub, stops, contrast=1.15)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.0, sheen_amt=14.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="waffle-knit",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=8)",
    default_palette="sage_linen",
    description="Honeycomb waffle thermal knit and pique polo mesh weave.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
