"""Stylized cel-shaded grass blades, dual water caustics, and painterly arcane stone slabs."""

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
    style = variant % 3
    micro = common.periodic_fbm(u, v, 6, 4, 0.5, rng)

    if style == 0:
        # Tapered curved cel grass tufts arranged on warped periodic Voronoi blade clusters
        w1 = common.periodic_fbm(u, v, 4, 4, 0.5, rng)
        bu = (u * 12.0 + 0.55 * (w1 - 0.5)) % 1.0
        bv = (v * 12.0 + 0.35 * (micro - 0.5)) % 1.0
        curve_x = bu - (0.5 + 0.18 * np.sin(bv * math.pi))
        half_w = 0.36 * np.power(np.clip(1.0 - bv, 0.0, 1.0), 0.72) + 0.04
        blades = np.clip(1.0 - np.abs(curve_x) / half_w, 0.0, 1.0) * (0.4 + 0.6 * (1.0 - bv))
        quant = np.floor(common.normalize_field(0.62 * w1 + 0.38 * blades) * 5.0) / 4.0
        height = common.normalize_field(0.60 * blades + 0.40 * w1)
        color_field = common.normalize_field(0.72 * quant + 0.28 * height)
    elif style == 1:
        # Snell refraction caustic cusps along dual warped Voronoi ridge networks (sharp optical concentration)
        wu = (u + 0.06 * (micro - 0.5)) % 1.0
        wv = (v + 0.06 * (micro - 0.5)) % 1.0
        _, edge1, _, _, _ = common.periodic_voronoi(wu, wv, 6 + (variant % 4), 0.90, rng)
        _, edge2, _, _, _ = common.periodic_voronoi((wu + 0.37) % 1.0, (wv + 0.61) % 1.0, 8, 0.90, rng)
        c1 = np.exp(-edge1 * 9.5)
        c2 = np.exp(-edge2 * 10.5)
        caustic = common.normalize_field(c1 + 0.68 * c2 + 0.14 * micro)
        height = caustic
        color_field = caustic
    else:
        # Lloyd-relaxed arcane stone slabs engraved with concentric geometric ring-and-spoke runes
        f1, edge, cid, rx, ry = common.periodic_lloyd_voronoi(u, v, 5 + (variant % 4), 0.78, rng, lloyd_iters=1)
        t_g = np.clip((edge - 0.025) / 0.035, 0.0, 1.0)
        grout = t_g * t_g * (3.0 - 2.0 * t_g)
        ang = np.arctan2(ry, rx)
        rune_ring = np.exp(-((f1 - 0.065) ** 2) / 0.00018)
        rune_spokes = np.exp(-((np.sin(ang * 3.0 + cid * 6.2831853)) ** 2) / 0.04) * (f1 < 0.075)
        engraving = np.clip(rune_ring + 0.7 * rune_spokes, 0.0, 1.0) * (edge > 0.05)
        height = common.normalize_field(grout * (0.78 + 0.14 * micro - 0.28 * engraving))
        color_field = common.normalize_field(grout * (0.50 * cid + 0.25 * micro + 0.25 * engraving))

    roughness = np.clip(0.30 + 0.40 * (1.0 - height), 0.15, 0.85).astype(np.float32)
    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=5.0, spec_intensity=0.22), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="stylized-arcane-cel",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_stylized_genshin_and_arcane; merges V1 abstract_stylized_anime)",
    default_palette="ghibli_cel_meadow",
    description="Stylized cel-shaded grass blades (v0), water caustics (v1), and painterly arcane slabs (v2).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=5.0,
    height_fn=generate_height,
)
