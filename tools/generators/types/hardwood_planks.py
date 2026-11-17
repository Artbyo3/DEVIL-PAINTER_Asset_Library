"""Hardwood planks with growth knots, deflected annual rings, and fine wood pores."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    planks = 1 if (variant % 3 == 0) else (2 + (variant % 4) * 2)
    plank_id = np.floor(v * planks)
    plank_shift = (plank_id * 0.41) % 1.0

    up = (u + plank_shift) % 1.0
    vp = (v * planks) % 1.0

    knot_f1, _, knot_id, _, _ = common.periodic_voronoi(up, v, 2 + (variant % 3), 0.75, rng)
    knot_influence = np.exp(-knot_f1 * 6.5) * (knot_id > 0.35).astype(np.float32)

    low_warp = common.periodic_fbm(up, v, 3, 4, 0.5, rng)
    ring_freq = 7 + (variant % 6) * 2
    ring_phase = (up + 0.16 * (low_warp - 0.5) + 0.28 * knot_influence) * ring_freq * 2.0 * math.pi
    rings = np.power(0.5 + 0.5 * np.sin(ring_phase), 0.62)

    fibers = common.periodic_gradient_perlin((u * 2.0) % 1.0, (v * 16.0) % 1.0, 16, 16, rng)
    grain = common.normalize_field(0.68 * rings + 0.20 * fibers + 0.12 * (1.0 - knot_influence))

    if planks > 1:
        bevel_u = np.clip(np.minimum(up, 1.0 - up) * 45.0, 0.0, 1.0)
        bevel_v = np.clip(np.minimum(vp, 1.0 - vp) * 22.0, 0.0, 1.0)
        bevel = bevel_u * bevel_v
        plank_tone = ((plank_id * 7) % 5) / 5.0
        height = common.normalize_field(grain * bevel)
        color_field = common.normalize_field((0.82 * grain + 0.18 * plank_tone) * bevel)
    else:
        height = grain
        color_field = grain

    roughness = np.clip(0.28 + 0.35 * (1.0 - grain), 0.15, 0.80).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, fibers, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=5.4, spec_intensity=0.22)


SPEC = GeneratorSpec(
    name="hardwood-planks",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_hardwood_knots_and_burl; merges V1 wood_grain & SP arch=2)",
    default_palette="burled_black_walnut",
    description="Hardwood planks & burl timber with annual rings, deflected growth knots, and wood pores.",
    render_fn=generate,
)
