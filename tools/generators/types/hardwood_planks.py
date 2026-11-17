"""Hardwood planks with growth knots, deflected annual rings, and fine wood pores."""

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
    planks = 1 if (variant % 3 == 0) else (2 + (variant % 4) * 2)
    plank_id = np.floor(v * planks)
    plank_shift = (plank_id * 0.41) % 1.0

    up = (u + plank_shift) % 1.0
    vp = (v * planks) % 1.0

    knot_f1, _, knot_id, rx, ry = common.periodic_voronoi(up, v, 2 + (variant % 3), 0.75, rng)
    # Elliptical knot mask stretched longitudinally along plank grain
    ellip_r = np.sqrt(rx * rx * 2.2 + ry * ry * 0.55) * (2 + (variant % 3))
    is_knot = (knot_id > 0.35).astype(np.float32)
    knot_influence = np.exp(-ellip_r * 3.8) * is_knot

    low_warp = common.periodic_fbm(up, v, 3, 4, 0.5, rng)
    ring_freq = 7 + (variant % 6) * 2
    # Potential-like deflection bends annual rings cleanly around elliptical knots
    ring_phase = (up + 0.16 * (low_warp - 0.5) + 0.35 * knot_influence) * ring_freq
    # Liu, Marschner & Dye (2015) / "From Words to Wood": asymmetric gradual earlywood -> abrupt latewood boundary
    ring_frac = ring_phase % 1.0
    late_cut = 0.76
    t_rise = np.clip(ring_frac / late_cut, 0.0, 1.0)
    t_drop = np.clip((1.0 - ring_frac) / (1.0 - late_cut), 0.0, 1.0)
    ew = t_rise * t_rise * (3.0 - 2.0 * t_rise)
    lw_drop = t_drop * t_drop * (3.0 - 2.0 * t_drop)
    rings = np.where(ring_frac < late_cut, 1.0 - 0.75 * ew, 0.25 + 0.75 * (1.0 - lw_drop)).astype(np.float32)

    fibers = common.periodic_gradient_perlin((u * 2.0) % 1.0, (v * 16.0) % 1.0, 16, 16, rng)
    # Ring-porous hardwood vessel pits concentrated inside softer earlywood band (rings > 0.55)
    pore_field = common.periodic_fbm((u * 3.0) % 1.0, (v * 20.0) % 1.0, 10, 3, 0.55, rng)
    vessels = np.clip((pore_field - 0.66) * 6.5, 0.0, 1.0) * np.clip((rings - 0.45) * 2.2, 0.0, 1.0)
    grain = common.normalize_field(0.64 * rings + 0.18 * fibers + 0.10 * (1.0 - knot_influence) - 0.18 * vessels)

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

    roughness = np.clip(0.24 + 0.34 * rings + 0.18 * vessels, 0.14, 0.85).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, fibers, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=5.4, spec_intensity=0.22), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="hardwood-planks",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_hardwood_knots_and_burl; merges V1 wood_grain & SP arch=2)",
    default_palette="burled_black_walnut",
    description="Hardwood planks & burl timber with annual rings, deflected growth knots, and wood pores.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=5.4,
    height_fn=generate_height,
)
