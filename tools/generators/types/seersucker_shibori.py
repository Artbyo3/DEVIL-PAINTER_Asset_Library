"""Puckered seersucker stripe and shibori tie-dye linen texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    if variant % 2 == 0:
        stripes = [6, 8, 10, 14][int(rng.integers(0, 4))]
        su = (u * stripes) % 1.0
        pucker_band = (su < 0.5).astype(np.float32)
        crinkle = common.periodic_fbm(u, v, stripes * 2, 4, 0.55, rng, freqy=stripes * 5)
        weave = 0.5 + 0.5 * np.sin(u * stripes * 16.0 * math.pi) * np.sin(v * stripes * 16.0 * math.pi)
        h = common.normalize_field(0.45 * weave + 0.55 * crinkle * (0.25 + 0.75 * pucker_band))
        c = common.normalize_field(0.55 * pucker_band + 0.30 * crinkle + 0.15 * weave)
    else:
        warp = common.periodic_fbm(u, v, 3, 4, 0.55, rng)
        rings = [2, 3, 4][int(rng.integers(0, 3))]
        du = ((u + 0.12 * (warp - 0.5)) * rings) % 1.0 - 0.5
        dv = ((v + 0.12 * (warp - 0.5)) * rings) % 1.0 - 0.5
        r = np.sqrt(du * du + dv * dv) * 2.0
        dye = np.clip(0.5 + 0.5 * np.cos(r * math.pi * float(rng.uniform(3.0, 6.0)) + warp * 4.0), 0.0, 1.0)
        linen = 0.5 + 0.5 * np.sin(u * 64.0 * math.pi) * np.sin(v * 64.0 * math.pi)
        h = common.normalize_field(0.70 * linen + 0.30 * warp)
        c = common.normalize_field(0.80 * dye + 0.20 * linen)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, None, stops, contrast=1.10)
    return common.shade_cloth_surface(albedo, h, normal_strength=4.8, sheen_amt=12.0)


SPEC = GeneratorSpec(
    name="seersucker-shibori",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_seersucker_shibori)",
    default_palette="indigo_chambray",
    description="Puckered seersucker striped cotton (even variants) and indigo shibori tie-dye linen (odd variants).",
    render_fn=generate,
)
