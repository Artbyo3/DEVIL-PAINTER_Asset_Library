"""Worn Voronoi cobblestone street paving with recessed mortar joints."""

from __future__ import annotations

import numpy as np
from PIL import Image

from .. import common
from ..interface import GeneratorSpec, PaletteArg


def generate(seed: int = 1, palette: PaletteArg = None, size: int = 512, variant: int = 0) -> Image.Image:
    rng = np.random.default_rng(seed)
    u, v = common.make_uv_grid(size)
    cells = [5, 7, 9, 11][(variant + int(rng.integers(0, 4))) % 4]
    _, edge, cid, _, _ = common.periodic_voronoi(u, v, cells, 0.78, rng)
    rock_detail = common.periodic_fbm(u, v, cells * 2, 5, 0.55, rng)

    cobble = np.power(np.clip(edge * 4.5, 0.0, 1.0), 0.42)
    h = common.normalize_field(cobble * (0.75 + 0.25 * rock_detail))
    c = common.normalize_field(np.where(edge < 0.06, 0.12 + 0.2 * rock_detail, 0.30 + 0.40 * cid + 0.30 * rock_detail))
    rough = np.where(edge < 0.06, 0.88, 0.38 + 0.25 * rock_detail).astype(np.float32)

    stops = common.resolve_palette(palette, SPEC.default_palette, rng)
    albedo = common.apply_palette_5stop(c, rock_detail, stops, contrast=1.16)
    return common.shade_pbr_surface(albedo, h, roughness=rough, normal_strength=7.2, spec_intensity=0.18)


SPEC = GeneratorSpec(
    name="cobblestone-paving",
    kind="texture",
    tileable=True,
    origin="p1_work/scripts/custom_generators.py (gen_v2_cobblestone_paving)",
    default_palette="wet_river_cobble",
    description="Worn rounded cobblestone street paving with per-stone tinting and deep mortar joints.",
    render_fn=generate,
)
