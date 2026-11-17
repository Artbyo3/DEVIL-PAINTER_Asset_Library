"""Ribbed jersey knitwear (1x1, 2x2, 3x1 wales) texture generator."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    rib_type = int(rng.choice([1, 2, 3]))
    period = rib_type + 1
    wales = int(rng.choice([3, 4, 5, 6])) * period
    courses = int(round(wales * float(rng.uniform(1.1, 1.5))))
    wu = (u * wales) % 1.0
    wv = (v * courses) % 1.0
    rib_mask = (np.floor(u * wales).astype(np.int32) % period) < rib_type
    dx = np.abs(wu - 0.5) * 2.0
    v_shift = dx * 0.45
    loop = np.sin(((wv + v_shift) % 1.0) * math.pi) * np.sin(wu * math.pi)
    yarn_twist = 0.85 + 0.15 * np.sin((wu * 8.0 + wv * 4.0) * 2.0 * math.pi)
    fuzz = common.periodic_fbm(u, v, wales, 3, 0.55, rng)

    h = common.normalize_field(np.where(rib_mask, loop * yarn_twist, loop * 0.55) + 0.15 * fuzz)
    c = common.normalize_field(0.75 * h + 0.25 * rib_mask.astype(np.float32))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.12)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.0, sheen_amt=14.0)


SPEC = GeneratorSpec(
    name="rib-jersey-knit",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_rib_jersey_knit)",
    default_palette="indigo_chambray",
    description="1x1, 2x2, and 3x1 ribbed jersey knitwear with interlocking V-loop columns.",
    render_fn=generate,
)
