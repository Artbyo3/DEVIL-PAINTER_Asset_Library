"""Plush corduroy wales and ribbed velveteen texture generator."""

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

    wales = 14 + (variant % 6) * 4
    wu = (u * wales) % 1.0
    rib = np.power(np.sin(wu * math.pi), 0.55)
    tuft = common.periodic_gradient_perlin(u, v, wales * 2, wales * 4, rng)

    h = common.normalize_field(rib * (0.80 + 0.20 * tuft) + 0.08 * fuzz)
    c = common.normalize_field(0.75 * h + 0.25 * slub)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, tuft, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.0, sheen_amt=22.0)


SPEC = GeneratorSpec(
    name="corduroy",
    kind="texture",
    tileable=True,
    origin="cloth_apparel_master_600_pack/procedural_texture_generator.py (arch=6)",
    default_palette="burgundy_velour",
    description="Plush corduroy wales and ribbed velveteen with plush pile tuft variation.",
    render_fn=generate,
)
