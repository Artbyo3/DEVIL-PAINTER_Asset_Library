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
        # Fortification agate: concentric Liesegang chalcedony precipitation bands around Voronoi silica vug centers
        f1, edge, _, _, _ = common.periodic_lloyd_voronoi(u, v, 3 + (variant % 3), 0.85, rng, lloyd_iters=1)
        warp = common.periodic_fbm(u, v, 4, 4, 0.5, rng)
        # Edge-anchored fortification contours mirror angular host-rock cavity walls (unlike circular blobs)
        contour = 0.55 * edge + 0.45 * (1.0 - f1)
        phase = (contour * (9 + variant % 5) + 0.28 * warp) * 2.0 * math.pi
        rings = 0.5 + 0.38 * np.sin(phase) + 0.12 * np.sin(3.0 * phase)
        color_field = common.normalize_field(0.68 * rings + 0.20 * edge + 0.12 * micro)
        height = common.normalize_field(0.62 * (1.0 - f1) + 0.28 * rings + 0.10 * micro)
        roughness = np.clip(0.12 + 0.18 * (1.0 - rings), 0.08, 0.40).astype(np.float32)
    else:
        # Hexagonal/trigonal quartz druzy cluster: 3-plane rhombohedral crystal termination facets per Lloyd seed
        cells = 8 + (variant % 6) * 2
        f1, edge, cid, rx, ry = common.periodic_lloyd_voronoi(u, v, cells, 0.92, rng, lloyd_iters=1)
        angle0 = cid * 2.0 * math.pi
        r_proj = np.maximum.reduce([
            rx * np.cos(angle0) + ry * np.sin(angle0),
            rx * np.cos(angle0 + 2.0943951) + ry * np.sin(angle0 + 2.0943951),
            rx * np.cos(angle0 + 4.1887902) + ry * np.sin(angle0 + 4.1887902),
        ])
        apex = np.clip(1.0 - np.abs(r_proj) * cells * 1.45, 0.0, 1.0)
        facet_tilt = 0.5 + 0.40 * ((rx * np.cos(angle0 + 0.5) + ry * np.sin(angle0 + 0.5)) * cells)
        height = common.normalize_field(0.72 * apex + 0.18 * np.clip(facet_tilt, 0.0, 1.0) + 0.10 * edge)
        color_field = common.normalize_field(0.50 * cid + 0.38 * height + 0.12 * micro)
        roughness = np.full_like(height, 0.09, dtype=np.float32)

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
