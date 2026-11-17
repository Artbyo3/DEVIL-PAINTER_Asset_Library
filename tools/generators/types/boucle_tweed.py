"""Heavy boucle tweed with irregular slub yarns and wool flecks."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    threads = [18, 24, 32, 40][variant % 4]
    slub_u = common.periodic_fbm(u, v, 4, 3, 0.55, rng, freqy=threads // 2)
    slub_v = common.periodic_fbm(u, v, threads // 2, 3, 0.55, rng, freqy=4)
    tu = u * threads + (slub_u - 0.5) * 0.35
    tv = v * threads + (slub_v - 0.5) * 0.35
    iu, iv = np.floor(tu).astype(np.int32), np.floor(tv).astype(np.int32)
    fu, fv = tu - iu, tv - iv
    warp_over = ((iu + iv) % 2) == 0
    thickness = np.where(warp_over, 0.65 + 0.5 * slub_u, 0.65 + 0.5 * slub_v)
    weave = np.where(warp_over, np.sin(fu * math.pi), np.sin(fv * math.pi)) * thickness
    fleck = (common.periodic_gradient_perlin(u, v, threads, threads, rng) > 0.78).astype(np.float32)

    h = common.normalize_field(weave + 0.25 * fleck)
    c = common.normalize_field(0.60 * h + 0.25 * slub_u + 0.30 * fleck)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, slub_v, stops, contrast=1.18)
    return common.shade_cloth_surface(albedo, h, normal_strength=5.8, sheen_amt=14.0)


SPEC = GeneratorSpec(
    name="boucle-tweed",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_boucle_slub_tweed)",
    default_palette="oatmeal_boucle",
    description="Heavy boucle tweed weave with slub warp/weft yarn thickness variation and wool flecks.",
    render_fn=generate,
)
