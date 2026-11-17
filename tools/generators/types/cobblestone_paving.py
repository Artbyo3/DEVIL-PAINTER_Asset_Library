"""Worn Voronoi cobblestone street paving with recessed mortar joints."""

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
    cells = [5, 7, 9, 11][(variant + int(rng.integers(0, 4))) % 4]
    _, edge, cid, _, _ = common.periodic_lloyd_voronoi(u, v, cells, 0.88, rng, lloyd_iters=2)
    rock_detail = common.periodic_fbm(u, v, cells * 2, 5, 0.55, rng)

    warp_edge = edge + (rock_detail - 0.5) * 0.022
    t = np.clip((warp_edge - 0.045) / 0.032, 0.0, 1.0)
    stone_mask = t * t * (3.0 - 2.0 * t)
    cobble = np.power(np.clip((warp_edge - 0.04) * 3.8, 0.0, 1.0), 0.42)
    h = common.normalize_field(stone_mask * cobble * (0.76 + 0.24 * rock_detail))
    stone_c = 0.30 + 0.42 * cid + 0.28 * rock_detail
    mortar_c = 0.12 + 0.18 * rock_detail
    c = common.normalize_field(stone_mask * stone_c + (1.0 - stone_mask) * mortar_c)
    rough = np.clip((1.0 - stone_mask) * 0.88 + stone_mask * (0.36 + 0.26 * rock_detail), 0.25, 0.95).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, rock_detail, stops, contrast=1.16)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=7.2, spec_intensity=0.18), np.asarray(h, dtype=np.float32)


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[0]


def generate_height(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> np.ndarray:
    return _compute(seed=seed, palette=palette, size=size, variant=variant)[1]

SPEC = GeneratorSpec(
    name="cobblestone-paving",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_cobblestone_paving)",
    default_palette="wet_river_cobble",
    description="Worn rounded cobblestone street paving with per-stone tinting and deep mortar joints.",
    render_fn=generate,
    normal_source="native",
    normal_rule="Native physical surface height field (h) used by shade_pbr_surface.",
    normal_strength=7.2,
    height_fn=generate_height,
)
