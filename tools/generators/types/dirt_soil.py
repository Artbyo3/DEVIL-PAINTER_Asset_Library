"""Compacted brown loam soil ground with organic clods, grit grain, and scattered small stones."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def _compute(
    seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0
) -> tuple[Image.Image, np.ndarray]:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)

    clod_freq = [4, 6, 8, 5][variant % 4]
    clods = common.periodic_fbm(u, v, clod_freq, 4, 0.55, rng)
    crumb = common.periodic_fbm(u, v, 24, 4, 0.55, rng)
    grit = common.periodic_fbm((u + 0.23) % 1.0, (v + 0.71) % 1.0, 48, 2, 0.5, rng)

    # Sparse embedded small pebbles/stones
    stone_cells = [16, 20, 24, 18][variant % 4]
    f1_s, edge_s, cid_s, _, _ = common.periodic_voronoi(u, v, stone_cells, 0.92, rng)
    stone_mask = np.clip(((1.0 - f1_s) * (cid_s > 0.68).astype(np.float32) - 0.68) * 14.0, 0.0, 1.0)
    stone_dome = stone_mask * np.clip(edge_s * 5.0, 0.0, 1.0)

    h = common.normalize_field(0.45 * clods + 0.30 * crumb + 0.10 * grit + 0.35 * stone_dome)
    c = common.normalize_field(0.42 * clods + 0.28 * crumb + 0.10 * grit + 0.32 * stone_mask * (0.5 + 0.5 * cid_s))
    rough = np.clip(0.86 - 0.20 * stone_mask + 0.08 * grit, 0.60, 0.96).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, grit, stops, contrast=1.16)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=5.8, spec_intensity=0.07), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="dirt-soil",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/dirt_soil.py",
    default_palette="loam_dirt_soil",
    description="Natural brown loam earth soil with organic clods, granular crumb texture, and small embedded stones.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=5.8,
    height_fn=generate_height,
)
