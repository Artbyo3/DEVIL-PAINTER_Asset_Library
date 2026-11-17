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
        w1 = common.periodic_fbm(u, v, 4, 4, 0.5, rng)
        blade_u = (u * 14.0 + 0.6 * w1) % 1.0
        blade_v = (v * 14.0 + 0.4 * micro) % 1.0
        blades = np.sin(blade_u * math.pi) * np.power(1.0 - blade_v, 0.6)
        quant = np.floor(common.normalize_field(0.65 * w1 + 0.35 * blades) * 5.0) / 4.0
        height = common.normalize_field(0.60 * blades + 0.40 * w1)
        color_field = common.normalize_field(0.70 * quant + 0.30 * height)
    elif style == 1:
        _, edge1, _, _, _ = common.periodic_voronoi(u, v, 6 + (variant % 4), 0.9, rng)
        _, edge2, _, _, _ = common.periodic_voronoi((u + 0.37) % 1.0, (v + 0.61) % 1.0, 8, 0.9, rng)
        c1 = np.power(1.0 - edge1, 2.6)
        c2 = np.power(1.0 - edge2, 2.8)
        caustic = common.normalize_field(c1 + 0.65 * c2 + 0.20 * micro)
        height = caustic
        color_field = caustic
    else:
        f1, edge, cid, _, _ = common.periodic_voronoi(u, v, 5 + (variant % 4), 0.75, rng)
        grout = np.clip(edge * 5.0, 0.0, 1.0)
        swirl = 0.5 + 0.5 * np.sin((f1 * 6.0 + micro * 2.0) * 2.0 * math.pi)
        height = common.normalize_field(grout * (0.75 + 0.25 * swirl))
        color_field = common.normalize_field(grout * (0.55 * cid + 0.45 * swirl))

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
