"""Ribbed jersey knitwear (1x1, 2x2, 3x1 wales) texture generator."""

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

    rib_type = int(rng.choice([1, 2, 3]))
    period = rib_type + 1
    wales = int(rng.choice([3, 4, 5, 6])) * period
    courses = int(round(wales * float(rng.uniform(1.15, 1.45))))
    gx, gy = u * wales, v * courses
    fx = gx - np.floor(gx)
    ix = np.floor(gx).astype(np.int32)
    iy_base = np.floor(gy).astype(np.int32)
    rib_mask = (ix % period) < rib_type

    # Knit columns: Chamberlain interlocking V-loops
    r_yarn = 0.215
    knit_h = np.zeros_like(u, dtype=np.float32)
    for dy in (-1, 0, 1):
        fy = gy - (iy_base + dy)
        t = np.clip((fy + 0.18) / 1.30, 0.0, 1.0)
        x_l = 0.18 + 0.14 * np.sin(t * math.pi * 0.85) + 0.04 * (1.0 - np.cos(t * math.pi))
        x_r = 1.0 - x_l
        z_leg = 0.55 + 0.45 * np.sin(t * math.pi)
        in_span = ((fy >= -0.18) & (fy <= 1.12)).astype(np.float32)
        pl = np.sqrt(np.clip(1.0 - ((fx - x_l) / r_yarn) ** 2, 0.0, 1.0)) * in_span
        pr = np.sqrt(np.clip(1.0 - ((fx - x_r) / r_yarn) ** 2, 0.0, 1.0)) * in_span
        twist = 0.86 + 0.14 * np.sin((t * 5.0 + fx * 4.0) * 2.0 * math.pi)
        knit_h = np.maximum(knit_h, np.maximum(pl, pr) * twist * z_leg)

    # Purl columns: recessed horizontal semicircular purl bump arches
    fy_p = gy - iy_base
    purl_arc = np.sqrt(np.clip(1.0 - ((fy_p - 0.5 - 0.16 * np.cos((fx - 0.5) * math.pi)) * 2.5) ** 2, 0.0, 1.0))
    purl_seg = np.sqrt(np.clip(1.0 - ((fx - 0.5) * 1.95) ** 2, 0.0, 1.0))
    purl_h = purl_arc * purl_seg * 0.54

    fuzz = common.periodic_fbm(u, v, min(128, wales), 3, 0.55, rng)
    h = common.normalize_field(np.where(rib_mask, knit_h, purl_h) + 0.14 * fuzz)
    c = common.normalize_field(0.75 * h + 0.25 * rib_mask.astype(np.float32))
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, fuzz, stops, contrast=1.12)
    return common.shade_cloth_surface(albedo, h, normal_strength=6.0, sheen_amt=14.0), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="rib-jersey-knit",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_cloth_rib_jersey_knit)",
    default_palette="indigo_chambray",
    description="1x1, 2x2, and 3x1 ribbed jersey knitwear with interlocking V-loop columns.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_cloth_surface.",
    normal_strength=6.0,
    height_fn=generate_height,
)
