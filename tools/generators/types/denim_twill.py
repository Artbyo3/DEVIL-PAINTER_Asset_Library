"""3x1 selvedge denim twill weave texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    fuzz = common.periodic_fbm(u, v, 8, 3, 0.5, rng)
    slub = common.periodic_fbm(u, v, 4, 3, 0.5, rng, freqy=32)

    threads = 24 + (variant % 5) * 4
    tu, tv = u * threads, v * threads
    iu, iv = np.floor(tu).astype(np.int32), np.floor(tv).astype(np.int32)
    fu, fv = tu - iu, tv - iv

    warp_over = ((iu + iv * (1 if variant % 2 == 0 else -1)) % 4) < 3
    ply_u = 0.85 + 0.15 * np.sin((fu * 2.0 + fv * 3.0) * 2.0 * math.pi)
    ply_v = 0.85 + 0.15 * np.sin((fv * 2.0 - fu * 3.0) * 2.0 * math.pi)
    yarn_u = np.sin(fu * math.pi) * ply_u * (0.9 + 0.1 * np.cos(fv * math.pi))
    yarn_v = np.sin(fv * math.pi) * ply_v * (0.85 + 0.15 * np.cos(fu * math.pi))

    h = common.normalize_field(np.where(warp_over, yarn_u, yarn_v * 0.78) + 0.08 * fuzz)
    c = common.normalize_field(np.where(warp_over, 0.25 + 0.35 * h + 0.12 * slub, 0.78 + 0.20 * h))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    micro = common.periodic_fbm(u, v, 12, 3, 0.5, rng)
    albedo = common.apply_palette_5stop(c, micro, stops, contrast=1.15)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.6, sheen_amt=18.0)


SPEC = GeneratorSpec(
    name="denim-twill",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=0)",
    default_palette="selvedge_indigo_denim",
    description="3x1 selvedge denim twill with diagonal wale, 2-ply yarn twist, and weft show-through.",
    render_fn=generate,
)
