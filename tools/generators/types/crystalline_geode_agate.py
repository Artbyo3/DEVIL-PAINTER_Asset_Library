"""Concentric banded agate slices and faceted amethyst/quartz druzy crystal clusters."""

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
    style = variant % 2
    micro = common.periodic_fbm(u, v, 8, 4, 0.5, rng)

    if style == 0:
        f1, _, _, _, _ = common.periodic_voronoi(u, v, 3 + (variant % 3), 0.8, rng)
        warp = common.periodic_fbm(u, v, 4, 4, 0.5, rng)
        rings = 0.5 + 0.5 * np.sin((f1 * (10 + variant % 6) + 0.35 * warp) * 2.0 * math.pi)
        color_field = common.normalize_field(0.72 * rings + 0.18 * (1.0 - f1) + 0.10 * micro)
        height = common.normalize_field(0.65 * (1.0 - f1) + 0.25 * rings + 0.10 * micro)
        roughness = np.full_like(height, 0.14, dtype=np.float32)
    else:
        cells = 8 + (variant % 6) * 2
        f1, edge, cid, rx, ry = common.periodic_voronoi(u, v, cells, 0.92, rng)
        angle = cid * 2.0 * math.pi
        facet_plane = 0.5 + 0.45 * (rx * np.cos(angle) + ry * np.sin(angle))
        pyramid = np.clip(1.0 - f1 * 0.85, 0.0, 1.0) * facet_plane
        height = common.normalize_field(0.85 * pyramid + 0.15 * edge)
        color_field = common.normalize_field(0.55 * cid + 0.35 * height + 0.10 * micro)
        roughness = np.full_like(height, 0.10, dtype=np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(color_field, micro, stops, contrast=1.22)
    return common.shade_pbr_surface(albedo, height, roughness=roughness, normal_strength=6.5, spec_intensity=0.45), np.asarray(height, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="crystalline-geode-agate",
    kind="texture",
    tileable=True,
    origin="procedural_textures_v2_300_pack (gen_crystalline_geode_and_agate)",
    default_palette="amethyst_druzy_geode",
    description="Concentric banded agate mineral slices (v0) and faceted druzy crystal clusters (v1).",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (height) used by shade_pbr_surface.",
    normal_strength=6.5,
    height_fn=generate_height,
)
