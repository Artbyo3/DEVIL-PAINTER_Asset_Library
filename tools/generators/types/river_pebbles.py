"""Tightly packed smooth rounded river pebbles nestled in damp streambed grit."""

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

    cells = [8, 10, 12, 14][variant % 4]
    f1, edge, cid, _, _ = common.periodic_lloyd_voronoi(u, v, cells, 0.88, rng, lloyd_iters=2)
    _, edge_s, cid_s, _, _ = common.periodic_lloyd_voronoi((u + 0.37) % 1.0, (v + 0.59) % 1.0, cells * 2, 0.88, rng, lloyd_iters=1)
    smooth_vein = common.periodic_fbm(u, v, cells * 2, 3, 0.48, rng)
    sand_grit = common.periodic_fbm(u, v, 36, 2, 0.5, rng)

    # Smooth water-worn ovoid pebble domes with secondary smaller stones nestled in gaps
    t_m = np.clip((edge - 0.032) / 0.036, 0.0, 1.0)
    pebble_mask = t_m * t_m * (3.0 - 2.0 * t_m)
    dome = np.power(np.clip(edge * 3.8, 0.0, 1.0), 0.36) * (1.0 - 0.16 * f1)

    t_sub = np.clip((edge_s - 0.05) / 0.04, 0.0, 1.0) * (1.0 - pebble_mask)
    sub_dome = np.power(np.clip(edge_s * 3.6, 0.0, 1.0), 0.42) * 0.42 * t_sub

    h = common.normalize_field(np.maximum(pebble_mask * (0.26 + 0.74 * dome), sub_dome) + (1.0 - np.maximum(pebble_mask, t_sub)) * 0.08 * sand_grit)
    peb_c = pebble_mask * (0.24 + 0.58 * cid + 0.18 * smooth_vein) + t_sub * (0.20 + 0.50 * cid_s) * (1.0 - pebble_mask)
    sand_c = (1.0 - np.clip(pebble_mask + t_sub, 0.0, 1.0)) * (0.08 + 0.18 * sand_grit)
    c = common.normalize_field(peb_c + sand_c)
    rough = np.clip((1.0 - pebble_mask) * 0.86 + pebble_mask * (0.24 + 0.18 * smooth_vein), 0.20, 0.94).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, smooth_vein, stops, contrast=1.18)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=6.8, spec_intensity=0.28), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="river-pebbles",
    kind="texture",
    tileable=True,
    origin="tools/generators/types/river_pebbles.py",
    default_palette="smooth_river_pebbles",
    description="Packed smooth water-worn river pebbles with rounded mineral variation and sandy interstitial gaps.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=6.8,
    height_fn=generate_height,
)
