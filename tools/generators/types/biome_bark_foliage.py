"""Deeply fissured tree bark, wind-rippled sand, and mossy forest floor."""

from __future__ import annotations

import math
import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    style = variant % 3
    micro = common.periodic_fbm(u, v, 8, 4, 0.5, rng)

    if style == 0:
        warp = common.periodic_fbm(u, v, 3, 4, 0.5, rng)
        bark_u = (u * (5 + variant % 4) + 0.12 * (warp - 0.5)) % 1.0
        _, edge, cid, _, _ = common.periodic_voronoi(bark_u, (v * 2.0) % 1.0, 6, 0.85, rng)
        ridges = np.clip(edge * 3.5, 0.0, 1.0)
        height = common.normalize_field(0.70 * ridges + 0.30 * micro)
        color_field = common.normalize_field(0.60 * height + 0.25 * cid + 0.15 * warp)
    elif style == 1:
        warp = common.periodic_fbm(u, v, 3, 4, 0.5, rng)
        ripples = np.power(0.5 + 0.5 * np.sin((v * (8 + variant % 5) + 0.35 * warp) * 2.0 * math.pi), 0.7)
        height = common.normalize_field(0.78 * ripples + 0.22 * micro)
        color_field = common.normalize_field(0.72 * height + 0.28 * warp)
    else:
        f1, edge, cid, _, _ = common.periodic_voronoi(u, v, 7 + (variant % 4), 0.9, rng)
        swiss = common.periodic_fbm(u, v, 4, 5, 0.55, rng, mode="swiss")
        height = common.normalize_field(0.55 * swiss + 0.30 * (1.0 - f1) + 0.15 * micro)
        color_field = common.normalize_field(0.50 * height + 0.35 * edge + 0.15 * cid)

    roughness = np.clip(0.60 + 0.30 * (1.0 - height), 0.4, 0.95).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.15)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.2, spec_intensity=0.14)


SPEC = GeneratorSpec(
    name="biome-bark-foliage",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_biome_ground_foliage_and_bark; merges SP arch=6)",
    default_palette="ancient_redwood_bark",
    description="Biome surfaces: anisotropic tree bark (v0), rippled ground sand (v1), and mossy forest floor (v2).",
    render_fn=generate,
)
