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
    f1, edge, cid, _, _ = common.periodic_voronoi(u, v, cells, 0.80, rng)
    smooth_vein = common.periodic_fbm(u, v, cells * 2, 3, 0.48, rng)
    sand_grit = common.periodic_fbm(u, v, 36, 2, 0.5, rng)

    # Smooth rounded dome per pebble with recessed sand gaps
    pebble_mask = np.clip((edge - 0.035) / 0.035, 0.0, 1.0)
    dome = np.power(np.clip(edge * 4.2, 0.0, 1.0), 0.36) * (1.0 - 0.18 * f1)

    h = common.normalize_field(pebble_mask * (0.25 + 0.75 * dome) + (1.0 - pebble_mask) * 0.08 * sand_grit)
    c = common.normalize_field(
        np.where(pebble_mask > 0.25, 0.24 + 0.58 * cid + 0.18 * smooth_vein, 0.08 + 0.18 * sand_grit)
    )
    rough = np.where(pebble_mask > 0.35, 0.26 + 0.18 * smooth_vein, 0.88).astype(np.float32)

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
