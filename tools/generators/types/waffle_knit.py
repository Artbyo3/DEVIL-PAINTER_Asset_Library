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

    cells = 8 + (variant % 5) * 4
    gu = (u * cells) % 1.0
    gv = (v * cells) % 1.0
    # Honeycomb geometry: sunken pocket center transitions steeply up curved warp/weft float walls
    dx = np.abs(gu - 0.5) * 2.0
    dy = np.abs(gv - 0.5) * 2.0
    wall_u = 1.0 - np.cos(np.clip((dx - 0.15) / 0.85, 0.0, 1.0) * math.pi * 0.5) ** 2
    wall_v = 1.0 - np.cos(np.clip((dy - 0.15) / 0.85, 0.0, 1.0) * math.pi * 0.5) ** 2
    # Corner float overlap lifts the perimeter rim around each recessed hollow
    ridge = np.clip(0.55 * np.maximum(wall_u, wall_v) + 0.45 * (wall_u + wall_v) * 0.5, 0.0, 1.0)

    # Underlying 1/1 plain-weave tabby grid visible inside the honeycomb cells & ridges
    plain_m = np.array([[1, 0], [0, 1]], dtype=np.int32)
    weave_h, _, _, _ = common.woven_fabric_field(u, v, cells * 6, cells * 6, plain_m, float_arch=0.35, ply_twist=0.10)

    h = common.normalize_field(0.72 * ridge + 0.28 * weave_h)
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
